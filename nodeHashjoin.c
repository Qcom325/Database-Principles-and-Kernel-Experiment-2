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
 * algorithm so we can get started.
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
#include "executor/tuptable.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "utils/memutils.h"
#include "utils/sharedtuplestore.h"


/*
 * States of the ExecHashJoin state machine
 */
#define HJ_BUILD_HASHTABLE		1
#define HJ_NEED_INNER_TUPLE		2
#define HJ_SCAN_OUTER_BUCKET		3
#define HJ_SCAN_INNER_BUCKET		4
#define HJ_NEED_OUTER_TUPLE		5
#define HJ_FILL_OUTER_TUPLE		6
#define HJ_FILL_INNER_TUPLES	        7
#define HJ_NEED_NEW_BATCH		8

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
 *		Symmetric Hash Join implementation.
 *		Both sides build hash tables incrementally.
 *		Each tuple probes the OTHER hash table upon arrival.
 *
 *		Correctness (no duplicates, no misses):
 *		- Inner arrives first → probes OUTER HT (outer not yet in) → misses
 *		  → inserts into INNER HT → outer arrives → probes INNER HT → finds inner
 *		- Outer arrives first → probes INNER HT (inner already in) → finds inner
 *		  → inserts into OUTER HT → inner arrives next cycle → probes OUTER HT → finds outer
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
	HashJoinTable inner_hashtable,
				outer_hashtable;
	TupleTableSlot *slot;
	uint32		hashvalue;
	int			batchno;
	MemoryContext tupleCxt;
	MemoryContext oldCxt;
	bool		match;

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
	tupleCxt = econtext->ecxt_per_tuple_memory;

	/*
	 * Reset per-tuple memory context to free any expression evaluation
	 * storage allocated in the previous tuple cycle.
	 */
	ResetExprContext(econtext);

	/*
	 * Symmetric Hash Join state machine.
	 *
	 * Strict alternation: inner probes OUTER HT, inserts to INNER HT,
	 * then outer probes INNER HT, inserts to OUTER HT.
	 *
	 * Correctness: each tuple probes the OTHER hash table FIRST,
	 * finding only tuples that arrived in PREVIOUS cycles.
	 * Then it inserts into its OWN hash table for future probes.
	 * Each pair is found exactly once — by whichever tuple arrives second.
	 */
		for (;;)
	{
		CHECK_FOR_INTERRUPTS();

		switch (node->hj_JoinState)
		{
			case HJ_BUILD_HASHTABLE:

				Assert(inner_hashtable == NULL);
				Assert(outer_hashtable == NULL);

				node->hj_FirstOuterTupleSlot = NULL;
				node->hj_FirstInnerTupleSlot = NULL;

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

				inner_hashNode->hashtable = inner_hashtable;
				outer_hashNode->hashtable = outer_hashtable;

				inner_hashtable->nbatch_outstart = inner_hashtable->nbatch;
				outer_hashtable->nbatch_outstart = outer_hashtable->nbatch;

				node->hj_OuterNotEmpty = false;
				node->hj_InnerNotEmpty = false;

				node->hj_JoinState = HJ_NEED_INNER_TUPLE;

				/* FALL THRU */

			/* ----------------------------------------------------------
			 * Read next inner tuple, then probe OUTER hash table.
			 * If inner is exhausted, fall through to NEED_OUTER_TUPLE.
			 * ---------------------------------------------------------- */
		case HJ_NEED_INNER_TUPLE:

			if (node->hj_InnerExhausted)
			{
				/* Inner already exhausted; go read outer */
				node->hj_JoinState = HJ_NEED_OUTER_TUPLE;
				break;
			}

			slot = ExecHashJoinOuterGetTuple(inner_hashNode, node,
											  &hashvalue, 1);

				if (!TupIsNull(slot))
				{
					/* Copy inner tuple into persistent slot */
					oldCxt = MemoryContextSwitchTo(tupleCxt);
					ExecCopySlot(node->hj_InnerTupleSlot, slot);
					MemoryContextSwitchTo(oldCxt);

						 hashvalue,
							/* Set up to probe OUTER hash table */
					econtext->ecxt_outertuple = node->hj_InnerTupleSlot;
					node->hj_InnerCurHashValue = hashvalue;
					ExecHashGetBucketAndBatch(outer_hashtable, hashvalue,
										&node->hj_OuterCurBucketNo, &batchno);
					node->hj_OuterCurSkewBucketNo =
						ExecHashGetSkewBucket(outer_hashtable, hashvalue);
					node->hj_OuterCurTuple = NULL;

					node->hj_JoinState = HJ_SCAN_OUTER_BUCKET;
					continue;
				}

				/* Inner exhausted: mark and fall through to read outer */
				node->hj_InnerExhausted = true;
				/* FALL THRU */

		case HJ_NEED_OUTER_TUPLE:

			if (node->hj_OuterExhausted)
			{
			if (node->hj_InnerExhausted)
			{
				/* Both sides exhausted, check if we need to fill unmatched tuples */
				if (node->js.jointype == JOIN_LEFT ||
					node->js.jointype == JOIN_ANTI ||
					node->js.jointype == JOIN_FULL)
				{
					node->hj_JoinState = HJ_FILL_OUTER_TUPLE;
					ExecPrepHashTableForUnmatched(node, true);
					break;
				}
				else if (node->js.jointype == JOIN_RIGHT)
				{
					node->hj_JoinState = HJ_FILL_INNER_TUPLES;
					ExecPrepHashTableForUnmatched(node, false);
					break;
				}
				else
					return NULL;	/* INNER/SEMI JOIN: done */
			}
			node->hj_JoinState = HJ_NEED_INNER_TUPLE;
			break;
			}

			slot = ExecHashJoinOuterGetTuple(outer_hashNode, node,
											  &hashvalue, 0);

					if (TupIsNull(slot))
				{
					/* Outer exhausted */
					node->hj_OuterExhausted = true;
					if (node->hj_InnerExhausted)
					{
						/* Both sides exhausted, check if we need to fill unmatched tuples */
						if (node->js.jointype == JOIN_LEFT ||
							node->js.jointype == JOIN_ANTI ||
							node->js.jointype == JOIN_FULL)
						{
							node->hj_JoinState = HJ_FILL_OUTER_TUPLE;
							ExecPrepHashTableForUnmatched(node, true);
							break;
						}
						else if (node->js.jointype == JOIN_RIGHT)
						{
							node->hj_JoinState = HJ_FILL_INNER_TUPLES;
							ExecPrepHashTableForUnmatched(node, false);
							break;
						}
						else
							return NULL;  /* INNER/SEMI JOIN: done */
					}

					/* Inner may still have tuples; go process them */
					node->hj_JoinState = HJ_NEED_INNER_TUPLE;
					break;
				}

				/* Copy outer tuple into persistent slot */
				oldCxt = MemoryContextSwitchTo(tupleCxt);
				ExecCopySlot(node->hj_OuterTupleSlot, slot);
				MemoryContextSwitchTo(oldCxt);

				/* Set up to probe INNER hash table */
				econtext->ecxt_outertuple = node->hj_OuterTupleSlot;
				node->hj_OuterCurHashValue = hashvalue;
				ExecHashGetBucketAndBatch(inner_hashtable, hashvalue,
									&node->hj_InnerCurBucketNo, &batchno);
				node->hj_InnerCurSkewBucketNo =
					ExecHashGetSkewBucket(inner_hashtable, hashvalue);
				node->hj_InnerCurTuple = NULL;

				node->hj_JoinState = HJ_SCAN_INNER_BUCKET;
				continue;

			/* ----------------------------------------------------------
			 * Inner tuple probes OUTER hash table (symbol=1).
			 * econtext: ecxt_outertuple = hj_InnerTupleSlot (probe),
			 *           ecxt_innertuple  = matched tuple from outer HT.
			 * ---------------------------------------------------------- */
		case HJ_SCAN_OUTER_BUCKET:

				match = ExecScanHashBucket(node, econtext, 1);

				if (match)
				{
					if (joinqual == NULL || ExecQual(joinqual, econtext))
					{
						node->hj_MatchedInner = true;
						HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(node->hj_OuterCurTuple));

						if (otherqual == NULL || ExecQual(otherqual, econtext))
						{
							return ExecProject(node->js.ps.ps_ProjInfo);
						}
						else
							InstrCountFiltered2(node, 1);
					}
					else
						InstrCountFiltered1(node, 1);

					node->hj_JoinState = HJ_SCAN_OUTER_BUCKET;
				}
				else
				{
					/* Inner finished probing outer HT.
					 * Insert inner into INNER hash table. */
					ExecHashTableInsert(node->hj_InnerHashTable,
										node->hj_InnerTupleSlot,
										node->hj_InnerCurHashValue);
					/* If inner was matched during probing, re-mark it
					 * (ExecHashTableInsert clears match flag). */
					if (node->hj_MatchedInner)
					{
						int bno;
						ExecHashGetBucketAndBatch(node->hj_InnerHashTable,
												node->hj_InnerCurHashValue,
												&bno, &batchno);
						HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(
							node->hj_InnerHashTable->buckets.unshared[bno]));
						node->hj_MatchedInner = false;
					}
					node->hj_JoinState = HJ_NEED_OUTER_TUPLE;
				}
				break;

			/* ----------------------------------------------------------
			 * Outer tuple probes INNER hash table (symbol=0).
			 * econtext: ecxt_outertuple = hj_OuterTupleSlot (probe),
			 *           ecxt_innertuple  = matched tuple from inner HT.
			 * ---------------------------------------------------------- */
			case HJ_SCAN_INNER_BUCKET:

				match = ExecScanHashBucket(node, econtext, 0);

				if (match)
				{
					if (joinqual == NULL || ExecQual(joinqual, econtext))
					{
						node->hj_MatchedOuter = true;
						HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(node->hj_InnerCurTuple));

						if (otherqual == NULL || ExecQual(otherqual, econtext))
						{
							return ExecProject(node->js.ps.ps_ProjInfo);
						}
						else
							InstrCountFiltered2(node, 1);
					}
					else
						InstrCountFiltered1(node, 1);

					node->hj_JoinState = HJ_SCAN_INNER_BUCKET;
				}
				else
				{
					/* Outer finished probing inner HT.
					 * Insert outer into OUTER hash table. */
					ExecHashTableInsert(node->hj_OuterHashTable,
										node->hj_OuterTupleSlot,
										node->hj_OuterCurHashValue);
					/* If outer was matched during probing, re-mark it
					 * (ExecHashTableInsert clears match flag). */
					if (node->hj_MatchedOuter)
					{
						int bno;
						ExecHashGetBucketAndBatch(node->hj_OuterHashTable,
												node->hj_OuterCurHashValue,
												&bno, &batchno);
						HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(
							node->hj_OuterHashTable->buckets.unshared[bno]));
						node->hj_MatchedOuter = false;
					}
					node->hj_JoinState = HJ_NEED_INNER_TUPLE;
				}
				break;

				case HJ_FILL_OUTER_TUPLE:
				/*
				 * For LEFT/ANTI/FULL JOIN: output unmatched outer tuples.
				 * Scan outer hash table for tuples that didn't match any inner tuple.
				 */
				if (ExecScanHashTableForUnmatched(node, econtext, true))
				{
					/* Found an unmatched outer tuple, output with null inner */
					econtext->ecxt_innertuple = node->hj_NullInnerTupleSlot;

					if (otherqual == NULL || ExecQual(otherqual, econtext))
						return ExecProject(node->js.ps.ps_ProjInfo);
					else
						InstrCountFiltered2(node, 1);
				}
				else
				{
					/* No more unmatched outer tuples */
					if (node->js.jointype == JOIN_FULL)
					{
						node->hj_JoinState = HJ_FILL_INNER_TUPLES;
						ExecPrepHashTableForUnmatched(node, false);
					}
					else
						return NULL;	/* LEFT/ANTI JOIN: done */
				}
				break;

			case HJ_FILL_INNER_TUPLES:
				/*
				 * For RIGHT/FULL JOIN: output unmatched inner tuples.
				 * Scan inner hash table for tuples that didn't match any outer tuple.
				 */
				if (ExecScanHashTableForUnmatched(node, econtext, false))
				{
					/* Found an unmatched inner tuple, output with null outer */
					econtext->ecxt_outertuple = node->hj_NullOuterTupleSlot;
					
					if (otherqual == NULL || ExecQual(otherqual, econtext))
						return ExecProject(node->js.ps.ps_ProjInfo);
					else
						InstrCountFiltered2(node, 1);
				}
					else
				{
					return NULL;	/* RIGHT/FULL JOIN: done */
				}
				break;

			case HJ_NEED_NEW_BATCH:
				/* No batching for symmetric hash join */
				break;

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
static TupleTableSlot *
ExecHashJoin(PlanState *pstate)
{
	return ExecHashJoinImpl(pstate, false);
}

/* ----------------------------------------------------------------
 *		ExecParallelHashJoin
 *
 *		Parallel-aware version.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
ExecParallelHashJoin(PlanState *pstate)
{
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

	Assert(!(eflags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));

	hjstate = makeNode(HashJoinState);
	hjstate->js.ps.plan = (Plan *) node;
	hjstate->js.ps.state = estate;

	hjstate->js.ps.ExecProcNode = ExecHashJoin;
	hjstate->js.jointype = node->join.jointype;

	ExecAssignExprContext(estate, &hjstate->js.ps);

	outerNode = (Hash *) outerPlan(node);
	innerNode = (Hash *) innerPlan(node);

	outerPlanState(hjstate) = ExecInitNode((Plan *) outerNode, estate, eflags);
	outerDesc = ExecGetResultType(outerPlanState(hjstate));

	innerPlanState(hjstate) = ExecInitNode((Plan *) innerNode, estate, eflags);
	innerDesc = ExecGetResultType(innerPlanState(hjstate));

	ExecInitResultTupleSlotTL(&hjstate->js.ps, &TTSOpsVirtual);
	ExecAssignProjectionInfo(&hjstate->js.ps, NULL);

	ops = ExecGetResultSlotOps(outerPlanState(hjstate), NULL);
	hjstate->hj_OuterTupleSlot = ExecInitExtraTupleSlot(estate, outerDesc, ops);
	ops = ExecGetResultSlotOps(innerPlanState(hjstate), NULL);
	hjstate->hj_InnerTupleSlot = ExecInitExtraTupleSlot(estate, innerDesc, ops);
	hjstate->hj_InnerMatchSlot = ExecInitExtraTupleSlot(estate, innerDesc, ops);

	ops = ExecGetResultSlotOps(outerPlanState(hjstate), NULL);
	hjstate->hj_OuterMatchSlot = ExecInitExtraTupleSlot(estate, outerDesc, ops);

	hjstate->js.single_match = (node->join.inner_unique ||
								node->join.jointype == JOIN_SEMI);

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
	 * NOTE: In Symmetric Hash Join, we do NOT use the "voodoo" optimization.
	 * Both sides use ExecProcNode incrementally.
	 */

	hjstate->js.ps.qual =
		ExecInitQual(node->join.plan.qual, (PlanState *) hjstate);
	hjstate->js.joinqual =
		ExecInitQual(node->join.joinqual, (PlanState *) hjstate);
	hjstate->hashclauses =
		ExecInitQual(node->hashclauses, (PlanState *) hjstate);

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
	hjstate->hj_InnerExhausted = false;
	hjstate->hj_OuterExhausted = false;

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
	ExecFreeExprContext(&node->js.ps);

	ExecClearTuple(node->js.ps.ps_ResultTupleSlot);
	ExecClearTuple(node->hj_OuterTupleSlot);
	ExecClearTuple(node->hj_InnerTupleSlot);
	ExecClearTuple(node->hj_InnerMatchSlot);

	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));
}

/*
 * ExecHashJoinOuterGetTuple
 *
 *		get the next tuple from either inner or outer plan node.
 *		symbol=0: getting outer tuple (using inner hashtable ref)
 *		symbol=1: getting inner tuple (using outer hashtable ref)
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
	}
	else
	{
		hashtable = hjstate->hj_OuterHashTable;
	}
	curbatch = hashtable->curbatch;

	if (curbatch == 0)
	{
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
			ExprContext *econtext = hjstate->js.ps.ps_ExprContext;

			/*
			 * Both sides' hashkeys use OUTER_VAR (setrefs.c always fixes
			 * Hash node hashkeys with OUTER_VAR), so always place tuple
			 * in ecxt_outertuple.
			 */
			econtext->ecxt_outertuple = slot;

			List *hashKeys = symbol == 0
				? hjstate->hj_OuterHashKeys
				: hjstate->hj_InnerHashKeys;

			if (ExecHashGetHashValue(hashtable, econtext,
									 hashKeys,
									 symbol == 0,
									 HJ_FILL_OUTER(hjstate),
									 hashvalue))
			{
				if (symbol == 0)
					hjstate->hj_OuterNotEmpty = true;
				else
					hjstate->hj_InnerNotEmpty = true;
				return slot;
			}

			slot = ExecProcNode(outerNode);
		}
	}
	else if (curbatch < hashtable->nbatch)
	{
		BufFile    *file = hashtable->outerBatchFile[curbatch];

		if (file == NULL)
			return NULL;

		if (symbol == 0)
		{
			slot = ExecHashJoinGetSavedTuple(hjstate, file, hashvalue,
											 hjstate->hj_OuterTupleSlot);
		}
		else
		{
			slot = ExecHashJoinGetSavedTuple(hjstate, file, hashvalue,
											 hjstate->hj_InnerTupleSlot);
		}
		if (!TupIsNull(slot))
		{
			return slot;
		}
	}

	return NULL;
}

static TupleTableSlot *
ExecParallelHashJoinOuterGetTuple(PlanState *outerNode,
								  HashJoinState *hjstate,
								  uint32 *hashvalue)
{
	return NULL;
}

static bool
ExecHashJoinNewBatch(HashJoinState *hjstate)
{
	return true;
}

static bool
ExecParallelHashJoinNewBatch(HashJoinState *hjstate)
{
	return false;
}

void
ExecHashJoinSaveTuple(MinimalTuple tuple, uint32 hashvalue,
					  BufFile **fileptr)
{
	return;
}

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
