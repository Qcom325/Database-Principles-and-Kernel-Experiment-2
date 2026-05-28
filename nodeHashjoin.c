/*-------------------------------------------------------------------------
 *
 * nodeHashjoin.c
 *	  Routines to handle hash join nodes
 *
 * Portions Copyright (c) 1996-2019, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeHashjoin.c
 *
 * PARALLELISM
 *
 * Hash joins can participate in parallel query execution in several ways.  A
 * parallel-oblivious hash join is one where the node is unaware that it is
 * part of a parallel plan.  In this case, a copy of the inner plan is used to
 * build a copy of the hash table in every backend, and the outer plan could
 * either be built from a partial or complete path, so that the results of the
 * hash join are correspondingly either partial or complete.  A parallel-aware
 * hash join is one that behaves differently, coordinating work between
 * backends, and appears as Parallel Hash Join in EXPLAIN output.  A Parallel
 * Hash Join always appears with a Parallel Hash node.
 *
 * Parallel-aware hash joins use the same per-backend state machine to track
 * progress through the hash join algorithm as parallel-oblivious hash joins.
 * In a parallel-aware hash join, there is also a shared state machine that
 * co-operating backends use to synchronize their local state machines and
 * program counters.  The shared state machine is managed with a Barrier IPC
 * primitive.  When all attached participants arrive at a barrier, the phase
 * advances and all waiting participants are released.
 *
 * When a participant begins working on a parallel hash join, it must first
 * figure out how much progress has already been made, because participants
 * don't wait for each other to begin.  For this reason there are switch
 * statements at key points in the code where we have to synchronize our local
 * state machine with the phase, and then jump to the correct part of the
 * algorithm so that we can get started.
 *
 * One barrier called build_barrier is used to coordinate the hashing phases.
 * The phase is represented by an integer which begins at zero and increments
 * one by one, but in the code it is referred to by symbolic names as follows:
 *
 *   PHJ_BUILD_ELECTING              -- initial state
 *   PHJ_BUILD_ALLOCATING            -- one sets up the batches and table 0
 *   PHJ_BUILD_HASHING_INNER         -- all hash the inner rel
 *   PHJ_BUILD_HASHING_OUTER         -- (multi-batch only) all hash the outer
 *   PHJ_BUILD_DONE                  -- building done, probing can begin
 *
 * While in the phase PHJ_BUILD_HASHING_INNER a separate pair of barriers may
 * be used repeatedly as required to coordinate expansions in the number of
 * batches or buckets.  Their phases are as follows:
 *
 *   PHJ_GROW_BATCHES_ELECTING       -- initial state
 *   PHJ_GROW_BATCHES_ALLOCATING     -- one allocates new batches
 *   PHJ_GROW_BATCHES_REPARTITIONING -- all repartition
 *   PHJ_GROW_BATCHES_FINISHING      -- one cleans up, detects skew
 *
 *   PHJ_GROW_BUCKETS_ELECTING       -- initial state
 *   PHJ_GROW_BUCKETS_ALLOCATING     -- one allocates new buckets
 *   PHJ_GROW_BUCKETS_REINSERTING    -- all insert tuples
 *
 * If the planner got the number of batches and buckets right, those won't be
 * necessary, but on the other hand we might finish up needing to expand the
 * buckets or batches multiple times while hashing the inner relation to stay
 * within our memory budget and load factor target.  For that reason it's a
 * separate pair of barriers using circular phases.
 *
 * The PHJ_BUILD_HASHING_OUTER phase is required only for multi-batch joins,
 * because we need to divide the outer relation into batches up front in order
 * to be able to process batches entirely independently.  In contrast, the
 * parallel-oblivious algorithm simply throws tuples 'forward' to 'later'
 * batches whenever it encounters them while scanning and probing, which it
 * can do because it processes batches in serial order.
 *
 * Once PHJ_BUILD_DONE is reached, backends then split up and process
 * different batches, or gang up and work together on probing batches if there
 * aren't enough to go around.  For each batch there is a separate barrier
 * with the following phases:
 *
 *  PHJ_BATCH_ELECTING       -- initial state
 *  PHJ_BATCH_ALLOCATING     -- one allocates buckets
 *  PHJ_BATCH_LOADING        -- all load the hash table from disk
 *  PHJ_BATCH_PROBING        -- all probe
 *  PHJ_BATCH_DONE           -- end
 *
 * Batch 0 is a special case, because it starts out in phase
 * PHJ_BATCH_PROBING; populating batch 0's hash table is done during
 * PHJ_BUILD_HASHING_INNER so we can skip loading.
 *
 * Initially we try to plan for a single-batch hash join using the combined
 * work_mem of all participants to create a large shared hash table.  If that
 * turns out either at planning or execution time to be impossible then we
 * fall back to regular work_mem sized hash tables.
 *
 * To avoid deadlocks, we never wait for any barrier unless it is known that
 * all other backends attached to it are actively executing the node or have
 * already arrived.  Practically, that means that we never return a tuple
 * while attached to a barrier, unless the barrier has reached its final
 * state.  In the slightly special case of the per-batch barrier, we return
 * tuples while in PHJ_BATCH_PROBING phase, but that's OK because we use
 * BarrierArriveAndDetach() to advance it to PHJ_BATCH_DONE without waiting.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "access/htup_details.h"
#include "access/parallel.h"
#include "executor/executor.h"
#include "executor/hashjoin.h"
#include "executor/nodeHash.h"
#include "executor/nodeHashjoin.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "utils/memutils.h"
#include "utils/sharedtuplestore.h"


/*
 * States of the ExecHashJoin state machine
 */
#define HJ_BUILD_HASHTABLE		1
#define HJ_NEED_NEW_TUPLE		2
#define HJ_SCAN_INNER_BUCKET		3
#define HJ_SCAN_OUTER_BUCKET		4
#define HJ_FILL_OUTER_TUPLE		5
#define HJ_FILL_INNER_TUPLES	        6
#define HJ_NEED_NEW_BATCH		7

/* Returns true if doing null-fill on outer relation */
#define HJ_FILL_OUTER(hjstate)	((hjstate)->hj_NullInnerTupleSlot != NULL)
/* Returns true if doing null-fill on inner relation */
#define HJ_FILL_INNER(hjstate)	((hjstate)->hj_NullOuterTupleSlot != NULL)

static TupleTableSlot *ExecHashJoinOuterGetTuple(PlanState *outerNode,
												 HashJoinState *hjstate,
												 uint32 *hashvalue,int symbol);
static TupleTableSlot *ExecParallelHashJoinOuterGetTuple(PlanState *outerNode,
														 HashJoinState *hjstate,
														 uint32 *hashvalue);
static TupleTableSlot *ExecHashJoinGetSavedTuple(HashJoinState *hjstate,
												 BufFile *file,
												 uint32 *hashvalue,
												 TupleTableSlot *tupleSlot);
static bool ExecHashJoinNewBatch(HashJoinState *hjstate);
static bool ExecParallelHashJoinNewBatch(HashJoinState *hjstate);
static void ExecParallelHashJoinPartitionOuter(HashJoinState *node);


/* ----------------------------------------------------------------
 *		ExecHashJoinImpl
 *
 *		This function implements the Hybrid Hashjoin algorithm.  It is marked
 *		with an always-inline attribute so that ExecHashJoin() and
 *		ExecParallelHashJoin() can inline it.  Compilers that respect the
 *		attribute should create versions specialized for parallel == true and
 *		parallel == false with unnecessary branches removed.
 *
 *		Note: the relation we build hash table on is the "inner"
 *			  the other one is "outer".
 * ----------------------------------------------------------------
 */
static pg_attribute_always_inline TupleTableSlot *
ExecHashJoinImpl(PlanState *pstate, bool parallel)
{
	HashJoinState *node = castNode(HashJoinState, pstate);
        HashState  *outer_hashNode;
	HashState  *inner_hashNode;
	ExprState  *joinqual;
	ExprState  *otherqual;
	ExprContext *econtext;
	HashJoinTable inner_hashtable,outer_hashtable;
	TupleTableSlot *outerTupleSlot;
	TupleTableSlot *innerTupleSlot;
	uint32		inner_hashvalue,outer_hashvalue;
	int			batchno;
	
	/*
	 * get information from HashJoin node
	 */
	joinqual = node->js.joinqual;
	otherqual = node->js.ps.qual;
	
	inner_hashNode = (HashState *) innerPlanState(node);
	outer_hashNode = (HashState *) outerPlanState(node);
	
	inner_hashtable = node->hj_InnerHashTable;
	outer_hashtable = node->hj_OuterHashTable;
	
	econtext = node->js.ps.ps_ExprContext;
	

	/*
	 * Reset per-tuple memory context to free any expression evaluation
	 * storage allocated in the previous tuple cycle.
	 */
	ResetExprContext(econtext);
        bool match;
	/*
	 * run the hash join state machine
	 */
	for (;;)
	{
		/*
		 * It's possible to iterate this loop many times before returning a
		 * tuple, in some pathological cases such as needing to move much of
		 * the current batch to a later batch.  So let's check for interrupts
		 * each time through.
		 */
		CHECK_FOR_INTERRUPTS();

		switch (node->hj_JoinState)
		{
			case HJ_BUILD_HASHTABLE:

				/*
				 * First time through: build hash table for inner relation.
				 */
				Assert(inner_hashtable == NULL);
				Assert(outer_hashtable == NULL);

				/*
				 * If the outer relation is completely empty, and it's not
				 * right/full join, we can quit without building the hash
				 * table.  However, for an inner join it is only a win to
				 * check this when the outer relation's startup cost is less
				 * than the projected cost of building the hash table.
				 * Otherwise it's best to build the hash table first and see
				 * if the inner relation is empty.  (When it's a left join, we
				 * should always make this check, since we aren't going to be
				 * able to skip the join on the strength of an empty inner
				 * relation anyway.)
				 *
				 * If we are rescanning the join, we make use of information
				 * gained on the previous scan: don't bother to try the
				 * prefetch if the previous scan found the outer relation
				 * nonempty. This is not 100% reliable since with new
				 * parameters the outer relation might yield different
				 * results, but it's a good heuristic.
				 *
				 * The only way to make the check is to try to fetch a tuple
				 * from the outer plan node.  If we succeed, we have to stash
				 * it away for later consumption by ExecHashJoinOuterGetTuple.
				 */
				node->hj_FirstOuterTupleSlot = NULL;
				node->hj_FirstInnerTupleSlot = NULL;
				
				/*
				 * Create the hash table.  If using Parallel Hash, then
				 * whoever gets here first will create the hash table and any
				 * later arrivals will merely attach to it.
				 */
				inner_hashtable = ExecHashTableCreate(inner_hashNode,
												node->hj_HashOperators,
												node->hj_Collations,
												HJ_FILL_INNER(node));
				node->hj_InnerHashTable = inner_hashtable;
				outer_hashtable = ExecHashTableCreate(outer_hashNode,
												node->hj_HashOperators,
												node->hj_Collations,
												HJ_FILL_INNER(node));
				node->hj_OuterHashTable = outer_hashtable;

				/*
				 * Execute the Hash node, to build the hash table.  If using
				 * Parallel Hash, then we'll try to help hashing unless we
				 * arrived too late.
				 */
				inner_hashNode->hashtable = inner_hashtable;
				outer_hashNode->hashtable = outer_hashtable;
				
				/*
				 * If the inner relation is completely empty, and we're not
				 * doing a left outer join, we can quit without scanning the
				 * outer relation.
				 */
				if (inner_hashtable->totalTuples == 0 && !HJ_FILL_OUTER(node))
					return NULL;
				if (outer_hashtable->totalTuples == 0 && !HJ_FILL_OUTER(node))
					return NULL;

				/*
				 * need to remember whether nbatch has increased since we
				 * began scanning the outer relation
				 */
				inner_hashtable->nbatch_outstart = inner_hashtable->nbatch;
				outer_hashtable->nbatch_outstart = outer_hashtable->nbatch;

				/*
				 * Reset OuterNotEmpty for scan.  (It's OK if we fetched a
				 * tuple above, because ExecHashJoinOuterGetTuple will
				 * immediately set it again.)
				 */
				node->hj_OuterNotEmpty = false;
				node->hj_InnerNotEmpty = false;

				
				node->hj_JoinState = HJ_NEED_NEW_TUPLE;

				/* FALL THRU */

			case HJ_NEED_NEW_TUPLE:

				/*
				 * We don't have an outer tuple, try to get the next one
				 */
				
				outerTupleSlot =
						ExecHashJoinOuterGetTuple(outer_hashNode, node, &outer_hashvalue,1);
				innerTupleSlot =
						ExecHashJoinOuterGetTuple(inner_hashNode, node, &inner_hashvalue,0);

				if (TupIsNull(outerTupleSlot))
				{
				      /* end of batch, or maybe whole join */
				       /*
					*if (HJ_FILL_INNER(node))
					*{
					*	set up to scan for unmatched inner tuples 
					*	ExecPrepHashTableForUnmatched(node);
					*	node->hj_JoinState = HJ_FILL_INNER_TUPLES;
					*}
					*else
				        */
						
					node->hj_JoinState = HJ_SCAN_INNER_BUCKET;
					
				}else{
				    ExecHashTableInsert(node->hj_OuterHashTable, outerTupleSlot, outer_hashvalue);
				}
				
				if (TupIsNull(innerTupleSlot))
				{
				        node->hj_JoinState = HJ_SCAN_OUTER_BUCKET;
					
				}else{
				    ExecHashTableInsert(node->hj_InnerHashTable, innerTupleSlot, inner_hashvalue);
				}
				if (TupIsNull(innerTupleSlot)&&TupIsNull(outerTupleSlot))
				{
				      return NULL;
				}
                                
				econtext->ecxt_outertuple = outerTupleSlot;
				node->hj_MatchedOuter = false;
				econtext->ecxt_innertuple = innerTupleSlot;
				node->hj_MatchedInner = false;

				/*
				 * Find the corresponding bucket for this tuple in the main
				 * hash table or skew hash table.
				 */
				node->hj_OuterCurHashValue = outer_hashvalue;
				ExecHashGetBucketAndBatch(outer_hashtable, outer_hashvalue,
										  &node->hj_OuterCurBucketNo, &batchno);
				node->hj_OuterCurSkewBucketNo = ExecHashGetSkewBucket(outer_hashtable,
																 outer_hashvalue);
				node->hj_OuterCurTuple = NULL;
				node->hj_InnerCurHashValue = inner_hashvalue;
				ExecHashGetBucketAndBatch(inner_hashtable, inner_hashvalue,
										  &node->hj_InnerCurBucketNo, &batchno);
				node->hj_InnerCurSkewBucketNo = ExecHashGetSkewBucket(inner_hashtable,
																 inner_hashvalue);
				node->hj_InnerCurTuple = NULL;

				

				/* OK, let's scan the bucket for matches */
				node->hj_JoinState = HJ_SCAN_INNER_BUCKET;
				continue;

				/* FALL THRU */

			case HJ_SCAN_INNER_BUCKET:
			        if (node->hj_InnerCurTuple == NULL)
                                {
                                    node->hj_InnerCurTuple = inner_hashtable->buckets.unshared[node->hj_InnerCurBucketNo];
                                }
                                
			        match = ExecScanHashBucket(inner_hashNode, node->hj_InnerHashTable,0);
				if (match)
				{        
				      
				        if (joinqual == NULL || ExecQual(joinqual, econtext))
				        {
					    node->hj_MatchedOuter = true;
					    /*
				      	     * This is really only needed if HJ_FILL_INNER(node),
				      	     * but we'll avoid the branch and just set it always.
				      	     */
				      	    HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(node->hj_InnerCurTuple));
					    /*
					     * If we only need to join to the first matching inner
					     * tuple, then consider returning this one, but after that
					     * continue with next outer tuple.
					     */
					     if (otherqual == NULL || ExecQual(otherqual, econtext))
					         return ExecProject(node->js.ps.ps_ProjInfo);
				     	     else
					         InstrCountFiltered2(node, 1);
				        }
				        else
					    InstrCountFiltered1(node, 1);
			      		node->hj_JoinState = HJ_SCAN_INNER_BUCKET;
			     	}else{
			     	        node->hj_JoinState = HJ_SCAN_OUTER_BUCKET;
			     	        continue;
			     	}

			case HJ_SCAN_OUTER_BUCKET:
			        if (node->hj_OuterCurTuple == NULL)
                                {
                                    node->hj_OuterCurTuple = outer_hashtable->buckets.unshared[node->hj_OuterCurBucketNo];
                                }
                                
			        match = ExecScanHashBucket(outer_hashNode, node->hj_OuterHashTable,1);
				if (match)
				{        
				       
				        if (joinqual == NULL || ExecQual(joinqual, econtext))
				        {
					    node->hj_MatchedInner = true;
					    /*
				      	     * This is really only needed if HJ_FILL_INNER(node),
				      	     * but we'll avoid the branch and just set it always.
				      	     */
				      	    HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(node->hj_OuterCurTuple));
					    /*
					     * If we only need to join to the first matching inner
					     * tuple, then consider returning this one, but after that
					     * continue with next outer tuple.
					     */
					     if (otherqual == NULL || ExecQual(otherqual, econtext))
					         return ExecProject(node->js.ps.ps_ProjInfo);
				     	     else
					         InstrCountFiltered2(node, 1);
				        }else
					    InstrCountFiltered1(node, 1);
			      		node->hj_JoinState = HJ_SCAN_OUTER_BUCKET;
			     	}else{
			     	        node->hj_JoinState = HJ_NEED_NEW_TUPLE;
			     	        continue;
			     	}
			
				

			case HJ_FILL_OUTER_TUPLE:

				/*
				 * The current outer tuple has run out of matches, so check
				 * whether to emit a dummy outer-join tuple.  Whether we emit
				 * one or not, the next state is NEED_NEW_OUTER.
				 */
				node->hj_JoinState = HJ_NEED_NEW_TUPLE;

				if (true)
				{
					/*
					 * Generate a fake join tuple with nulls for the inner
					 * tuple, and return it if it passes the non-join quals.
					 */
					econtext->ecxt_innertuple = node->hj_NullInnerTupleSlot;

					if (otherqual == NULL || ExecQual(otherqual, econtext))
						return ExecProject(node->js.ps.ps_ProjInfo);
					else
						InstrCountFiltered2(node, 1);
				}
				break;

			case HJ_FILL_INNER_TUPLES:

				/*
				 * We have finished a batch, but we are doing right/full join,
				 * so any unmatched inner tuples in the hashtable have to be
				 * emitted before we continue to the next batch.
				 */
				 /*
				if (!ExecScanHashTableForUnmatched(node, econtext))
				{
					/* no more unmatched tuples 
					node->hj_JoinState = HJ_NEED_NEW_BATCH;
					continue;
				}
				*/

				/*
				 * Generate a fake join tuple with nulls for the outer tuple,
				 * and return it if it passes the non-join quals.
				 */
				econtext->ecxt_outertuple = node->hj_NullOuterTupleSlot;

				if (otherqual == NULL || ExecQual(otherqual, econtext))
					return ExecProject(node->js.ps.ps_ProjInfo);
				else
					InstrCountFiltered2(node, 1);
				break;

			case HJ_NEED_NEW_BATCH:

				/*
				 * Try to advance to next batch.  Done if there are no more.
				 */
				/*
				if (!ExecHashJoinNewBatch(node))
						return NULL;	/* end of parallel-oblivious join 
				node->hj_JoinState = HJ_NEED_NEW_TUPLE;
				break;
				*/

			default:
				elog(ERROR, "unrecognized hashjoin state: %d",
					 (int) node->hj_JoinState);
		}
	}
}

/* ----------------------------------------------------------------
 *		ExecHashJoin
 *
 *		Parallel-oblivious version.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *			/* return: a tuple or NULL */
ExecHashJoin(PlanState *pstate)
/*

*/
{
	/*
	 * On sufficiently smart compilers this should be inlined with the
	 * parallel-aware branches removed.
	 */
	return ExecHashJoinImpl(pstate, false);
}

/* ----------------------------------------------------------------
 *		ExecParallelHashJoin
 *
 *		Parallel-aware version.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *			/* return: a tuple or NULL */
ExecParallelHashJoin(PlanState *pstate)
{
	/*
	 * On sufficiently smart compilers this should be inlined with the
	 * parallel-oblivious branches removed.
	 */
	return ExecHashJoinImpl(pstate, true);
}

/* ----------------------------------------------------------------
 *		ExecInitHashJoin
 *
 *		Init routine for HashJoin node.
 * ----------------------------------------------------------------
 */
HashJoinState *
ExecInitHashJoin(HashJoin *node, EState *estate, int eflags)
{
	HashJoinState *hjstate;
	Hash	   *outerNode;
	Hash	   *innerNode;
	TupleDesc	outerDesc,
				innerDesc;
	const TupleTableSlotOps *ops;

	/* check for unsupported flags */
	Assert(!(eflags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));

	/*
	 * create state structure
	 */
	hjstate = makeNode(HashJoinState);
	hjstate->js.ps.plan = (Plan *) node;
	hjstate->js.ps.state = estate;
       
	/*
	 * See ExecHashJoinInitializeDSM() and ExecHashJoinInitializeWorker()
	 * where this function may be replaced with a parallel version, if we
	 * managed to launch a parallel query.
	 */
	hjstate->js.ps.ExecProcNode = ExecHashJoin;
	hjstate->js.jointype = node->join.jointype;

	/*
	 * Miscellaneous initialization
	 *
	 * create expression context for node
	 */
	ExecAssignExprContext(estate, &hjstate->js.ps);

	/*
	 * initialize child nodes
	 *
	 * Note: we could suppress the REWIND flag for the inner input, which
	 * would amount to betting that the hash will be a single batch.  Not
	 * clear if this would be a win or not.
	 */
	outerNode = (Hash *) outerPlan(node);
	innerNode = (Hash *) innerPlan(node);

	outerPlanState(hjstate) = ExecInitNode((Plan *) outerNode, estate, eflags);
	outerDesc = ExecGetResultType(outerPlanState(hjstate));
      
	innerPlanState(hjstate) = ExecInitNode((Plan *) innerNode, estate, eflags);
	innerDesc = ExecGetResultType(innerPlanState(hjstate));

	/*
	 * Initialize result slot, type and projection.
	 */
	ExecInitResultTupleSlotTL(&hjstate->js.ps, &TTSOpsVirtual);
	ExecAssignProjectionInfo(&hjstate->js.ps, NULL);

	/*
	 * tuple table initialization
	 */
	ops = ExecGetResultSlotOps(outerPlanState(hjstate), NULL);
	hjstate->hj_OuterTupleSlot = ExecInitExtraTupleSlot(estate, outerDesc,
														ops);
	ops =     ExecGetResultSlotOps(innerPlanState(hjstate), NULL);
	hjstate->hj_InnerTupleSlot = ExecInitExtraTupleSlot(estate, innerDesc,
														ops);
														

	/*
	 * detect whether we need only consider the first matching inner tuple
	 */
	hjstate->js.single_match = (node->join.inner_unique ||
								node->join.jointype == JOIN_SEMI);

	/* set up null tuples for outer joins, if needed */
	switch (node->join.jointype)
	{
		case JOIN_INNER:
		case JOIN_SEMI:
			
		case JOIN_LEFT:
		case JOIN_ANTI:
			hjstate->hj_NullInnerTupleSlot =
				ExecInitNullTupleSlot(estate, innerDesc, &TTSOpsVirtual);
			break;
		case JOIN_RIGHT:
			hjstate->hj_NullOuterTupleSlot =
				ExecInitNullTupleSlot(estate, outerDesc, &TTSOpsVirtual);
			break;
		case JOIN_FULL:
			hjstate->hj_NullOuterTupleSlot =
				ExecInitNullTupleSlot(estate, outerDesc, &TTSOpsVirtual);
			hjstate->hj_NullInnerTupleSlot =
				ExecInitNullTupleSlot(estate, innerDesc, &TTSOpsVirtual);
			break;
		default:
			elog(ERROR, "unrecognized join type: %d",
				 (int) node->join.jointype);
	}

	/*
	 * now for some voodoo.  our temporary tuple slot is actually the result
	 * tuple slot of the Hash node (which is our inner plan).  we can do this
	 * because Hash nodes don't return tuples via ExecProcNode() -- instead
	 * the hash join node uses ExecScanHashBucket() to get at the contents of
	 * the hash table.  -cim 6/9/91
	 */
	{
		HashState  *inner_state = (HashState *) innerPlanState(hjstate);
		TupleTableSlot *inner_slot = inner_state->ps.ps_ResultTupleSlot;

		hjstate->hj_InnerTupleSlot = inner_slot;
		
		HashState  *outer_state = (HashState *) outerPlanState(hjstate);
		TupleTableSlot *outer_slot = outer_state->ps.ps_ResultTupleSlot;

		hjstate->hj_OuterTupleSlot = outer_slot;
	}

	/*
	 * initialize child expressions
	 */
	hjstate->js.ps.qual =
		ExecInitQual(node->join.plan.qual, (PlanState *) hjstate);
	hjstate->js.joinqual =
		ExecInitQual(node->join.joinqual, (PlanState *) hjstate);
	hjstate->hashclauses =
		ExecInitQual(node->hashclauses, (PlanState *) hjstate);

	/*
	 * initialize hash-specific info
	 */
	hjstate->hj_InnerHashTable = NULL;
	hjstate->hj_OuterHashTable = NULL;
	
	hjstate->hj_FirstOuterTupleSlot = NULL;
        hjstate->hj_FirstInnerTupleSlot = NULL;
        
	hjstate->hj_OuterCurHashValue = 0;
	hjstate->hj_InnerCurHashValue = 0;
    
	hjstate->hj_InnerCurBucketNo = 0;
	hjstate->hj_InnerCurSkewBucketNo = INVALID_SKEW_BUCKET_NO;
	hjstate->hj_InnerCurTuple = NULL;
	hjstate->hj_OuterCurBucketNo = 0;
	hjstate->hj_OuterCurSkewBucketNo = INVALID_SKEW_BUCKET_NO;
	hjstate->hj_OuterCurTuple = NULL;

	hjstate->hj_OuterHashKeys = ExecInitExprList(outerNode->hashkeys,
												 (PlanState *) hjstate);
	hjstate->hj_InnerHashKeys = ExecInitExprList(innerNode->hashkeys,
												 (PlanState *) hjstate);
	hjstate->hj_HashOperators = node->hashoperators;
	hjstate->hj_Collations = node->hashcollations;

	hjstate->hj_JoinState = HJ_BUILD_HASHTABLE;
	
	hjstate->hj_MatchedOuter = false;
	hjstate->hj_MatchedInner = false;
	hjstate->hj_OuterNotEmpty = false;
	hjstate->hj_InnerNotEmpty = false;

	return hjstate;
}

/* ----------------------------------------------------------------
 *		ExecEndHashJoin
 *
 *		clean up routine for HashJoin node
 * ----------------------------------------------------------------
 */
void
ExecEndHashJoin(HashJoinState *node)
{
	/*
	 * Free hash table
	 */
	if (node->hj_InnerHashTable)
	{
		ExecHashTableDestroy(node->hj_InnerHashTable);
		node->hj_InnerHashTable = NULL;
	}
        if (node->hj_OuterHashTable)
	{
		ExecHashTableDestroy(node->hj_OuterHashTable);
		node->hj_OuterHashTable = NULL;
	}
	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&node->js.ps);

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(node->js.ps.ps_ResultTupleSlot);
	ExecClearTuple(node->hj_OuterTupleSlot);
	ExecClearTuple(node->hj_InnerTupleSlot);

	/*
	 * clean up subtrees
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));
}

/*
 * ExecHashJoinOuterGetTuple
 *
 *		get the next outer tuple for a parallel oblivious hashjoin: either by
 *		executing the outer plan node in the first pass, or from the temp
 *		files for the hashjoin batches.
 *
 * Returns a null slot if no more outer tuples (within the current batch).
 *
 * On success, the tuple's hash value is stored at *hashvalue --- this is
 * either originally computed, or re-read from the temp file.
 */
static TupleTableSlot *
ExecHashJoinOuterGetTuple(PlanState *outerNode,
						  HashJoinState *hjstate,
						  uint32 *hashvalue,int symbol)
{
	HashJoinTable hashtable;
	int			curbatch;
	TupleTableSlot *slot;
        if (symbol == 0)
        {   
            hashtable = hjstate->hj_InnerHashTable;
            curbatch = hashtable->curbatch;
        }
        else
        {
            hashtable = hjstate->hj_OuterHashTable;
            curbatch = hashtable->curbatch;
        }

	if (curbatch == 0)			/* if it is the first pass */
	{
		/*
		 * Check to see if first outer tuple was already fetched by
		 * ExecHashJoin() and not used yet.
		 */
		if (symbol == 0)
               {
                    slot = hjstate->hj_FirstOuterTupleSlot;
                    if (!TupIsNull(slot))
                        hjstate->hj_FirstOuterTupleSlot = NULL;
                    else
                        slot = ExecProcNode(outerNode);
               }
              else
               {
                    slot = hjstate->hj_FirstInnerTupleSlot; 
                    if (!TupIsNull(slot))
                         hjstate->hj_FirstInnerTupleSlot = NULL;
                   else
                          slot = ExecProcNode(outerNode);
                }

		while (!TupIsNull(slot))
		{
			/*
			 * We have to compute the tuple's hash value.
			 */
			ExprContext *econtext = hjstate->js.ps.ps_ExprContext;

			econtext->ecxt_outertuple = slot;
			List *hashKeys = symbol==1? hjstate->hj_OuterHashKeys : hjstate->hj_InnerHashKeys;

			if (ExecHashGetHashValue(hashtable, econtext,
									 hashKeys,
									 true,	/* outer tuple */
									 HJ_FILL_OUTER(hjstate),
									 hashvalue))
			{
				/* remember outer relation is not empty for possible rescan */
				if (symbol == 0)
                                   hjstate->hj_OuterNotEmpty = true;
                                else
                                   hjstate->hj_InnerNotEmpty = true;
				return slot;
			}

			/*
			 * That tuple couldn't match because of a NULL, so discard it and
			 * continue with the next one.
			 */
			slot = ExecProcNode(outerNode);
		}
	}
	else if (curbatch < hashtable->nbatch)
	{
		BufFile    *file = hashtable->outerBatchFile[curbatch];

		/*
		 * In outer-join cases, we could get here even though the batch file
		 * is empty.
		 */
		if (file == NULL)
			return NULL;

		 if (symbol == 0)
                 {
                     slot = ExecHashJoinGetSavedTuple(hjstate,
                                                 file,
                                                 hashvalue,
                                                 hjstate->hj_OuterTupleSlot);
                  }
                  else
                  {
                     slot = ExecHashJoinGetSavedTuple(hjstate,
                                                 file,
                                                 hashvalue,
                                                 hjstate->hj_InnerTupleSlot);
                  }
		if (!TupIsNull(slot))
			return slot;
	}

	/* End of this batch */
	return NULL;
}

 /*
 * ExecHashJoinOuterGetTuple variant for the parallel case.
 */
static TupleTableSlot *
ExecParallelHashJoinOuterGetTuple(PlanState *outerNode,
								  HashJoinState *hjstate,
								  uint32 *hashvalue)
{
  

	/* End of this batch */
	return NULL;
}

/*
 * ExecHashJoinNewBatch
 *		switch to a new hashjoin batch
 *
 * Returns true if successful, false if there are no more batches.
 */
static bool
ExecHashJoinNewBatch(HashJoinState *hjstate)
{
	return true;
}

/*
 * Choose a batch to work on, and attach to it.  Returns true if successful,
 * false if there are no more batches.
 */
static bool
ExecParallelHashJoinNewBatch(HashJoinState *hjstate)
{
	
	return false;
}

/*
 * ExecHashJoinSaveTuple
 *		save a tuple to a batch file.
 *
 * The data recorded in the file for each tuple is its hash value,
 * then the tuple in MinimalTuple format.
 *
 * Note: it is important always to call this in the regular executor
 * context, not in a shorter-lived context; else the temp file buffers
 * will get messed up.
 */
void
ExecHashJoinSaveTuple(MinimalTuple tuple, uint32 hashvalue,
					  BufFile **fileptr)
{
	return;
}

/*
 * ExecHashJoinGetSavedTuple
 *		read the next tuple from a batch file.  Return NULL if no more.
 *
 * On success, *hashvalue is set to the tuple's hash value, and the tuple
 * itself is stored in the given slot.
 */
static TupleTableSlot *
ExecHashJoinGetSavedTuple(HashJoinState *hjstate,
						  BufFile *file,
						  uint32 *hashvalue,
						  TupleTableSlot *tupleSlot)
{
	return tupleSlot;
}


void
ExecReScanHashJoin(HashJoinState *node)
{
	return;
}

void
ExecShutdownHashJoin(HashJoinState *node)
{
	return;
}

static void
ExecParallelHashJoinPartitionOuter(HashJoinState *hjstate)
{
return;
}

void
ExecHashJoinEstimate(HashJoinState *state, ParallelContext *pcxt)
{
	return;
}

void
ExecHashJoinInitializeDSM(HashJoinState *state, ParallelContext *pcxt)
{
	return;
}

/* ----------------------------------------------------------------
 *		ExecHashJoinReInitializeDSM
 *
 *		Reset shared state before beginning a fresh scan.
 * ----------------------------------------------------------------
 */
void
ExecHashJoinReInitializeDSM(HashJoinState *state, ParallelContext *cxt)
{
	return;
	 
}

void
ExecHashJoinInitializeWorker(HashJoinState *state,
							 ParallelWorkerContext *pwcxt)
{
	return;
}
