/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>   // For std::memset

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "polybook.h"
#include "search.h"
#include "settings.h"
#include "tbprobe.h"
#include "timeman.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#define load_rlx(x) atomic_load_explicit(&(x), memory_order_relaxed)
#define store_rlx(x,y) atomic_store_explicit(&(x), y, memory_order_relaxed)

SignalsType Signals;
LimitsType Limits;

int TB_Cardinality, TB_CardinalityDTM;
int TB_RootInTB;
int TB_UseRule50;
Depth TB_ProbeDepth;

static Score base_ct;

// Different node types, used as template parameter
enum { NonPV, PV };

// Sizes and phases of the skip blocks, used for distributing search depths
// across the threads
static const int skipSize[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
static const int skipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};

static const int RazorMargin1 = 590;
static const int RazorMargin2 = 604;

INLINE int futility_margin(Depth d, int improving) {
  return (175 - 50 * improving) * d / ONE_PLY;
}

// Margin for pruning capturing moves: almost linear in depth
static const Value CapturePruneMargin[] = {
  0,
  1 * PawnValueEg * 1055 / 1000,
  2 * PawnValueEg * 1042 / 1000,
  3 * PawnValueEg *  963 / 1000,
  4 * PawnValueEg * 1038 / 1000,
  5 * PawnValueEg *  950 / 1000,
  6 * PawnValueEg *  930 / 1000
};

// Futility and reductions lookup tables, initialized at startup
static int FutilityMoveCounts[2][16]; // [improving][depth]
static int Reductions[2][2][64][64];  // [pv][improving][depth][moveNumber]

INLINE Depth reduction(int i, Depth d, int mn, const int NT)
{
  return Reductions[NT][i][min(d / ONE_PLY, 63)][min(mn, 63)] * ONE_PLY;
}

// History and stats update bonus, based on depth
static Value stat_bonus(Depth depth)
{
  int d = depth / ONE_PLY;
  return d > 17 ? 0 : 32 * d * d + 64 * d - 64;
}

// Skill structure is used to implement strength limit
struct Skill {
/*
  Skill(int l) : level(l) {}
  int enabled() const { return level < 20; }
  int time_to_pick(Depth depth) const { return depth / ONE_PLY == 1 + level; }
  Move best_move(size_t multiPV) { return best ? best : pick_best(multiPV); }
  Move pick_best(size_t multiPV);
*/

  int level;
  Move best;
};

//static CounterMoveHistoryStat CounterMoveHistory;

static Value search_PV(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value search_NonPV(Pos *pos, Stack *ss, Value alpha, Depth depth, int cutNode);

static Value qsearch_PV_true(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value qsearch_PV_false(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value qsearch_NonPV_true(Pos *pos, Stack *ss, Value alpha, Depth depth);
static Value qsearch_NonPV_false(Pos *pos, Stack *ss, Value alpha, Depth depth);

static Value value_to_tt(Value v, int ply);
static Value value_from_tt(Value v, int ply);
static void update_pv(Move *pv, Move move, Move *childPv);
static void update_cm_stats(Stack *ss, Piece pc, Square s, int bonus);
static void update_stats(const Pos *pos, Stack *ss, Move move, Move *quiets, int quietsCnt, int bonus);
static void update_capture_stats(const Pos *pos, Move move, Move *captures, int captureCnt, int bonus);
static void check_time(void);
static void stable_sort(RootMove *rm, int num);
static void uci_print_pv(Pos *pos, Depth depth, Value alpha, Value beta);
static int extract_ponder_from_tt(RootMove *rm, Pos *pos);

// search_init() is called during startup to initialize various lookup tables

void search_init(void)
{
  for (int imp = 0; imp <= 1; imp++)
    for (int d = 1; d < 64; ++d)
      for (int mc = 1; mc < 64; ++mc) {
        double r = log(d) * log(mc) / 1.95;

        Reductions[NonPV][imp][d][mc] = ((int)lround(r));
        Reductions[PV][imp][d][mc] = max(Reductions[NonPV][imp][d][mc] - 1, 0);

        // Increase reduction for non-PV nodes when eval is not improving
        if (!imp && r > 1.0)
          Reductions[NonPV][imp][d][mc]++;
      }

  for (int d = 0; d < 16; ++d) {
    FutilityMoveCounts[0][d] = (int)(2.4 + 0.74 * pow(d, 1.78));
    FutilityMoveCounts[1][d] = (int)(5.0 + 1.00 * pow(d, 2.00));
  }
}


// search_clear() resets search state to zero, to obtain reproducible results

void search_clear(void)
{
  if (!settings.ttSize) {
    delayedSettings.clear = true;
    return;
  }

  Time.availableNodes = 0;

  tt_clear();
  for (int i = 0; i < numCmhTables; i++)
    if (cmhTables[i]) {
      stats_clear(cmhTables[i]);
      for (int j = 0; j < 16; j++)
        for (int k = 0; k < 64; k++)
          (*cmhTables[i])[0][0][j][k] = CounterMovePruneThreshold - 1;
    }

  for (int idx = 0; idx < Threads.numThreads; idx++) {
    Pos *pos = Threads.pos[idx];
    stats_clear(pos->counterMoves);
    stats_clear(pos->history);
    stats_clear(pos->captureHistory);
  }

  mainThread.previousScore = VALUE_INFINITE;
  mainThread.previousTimeReduction = 1;
}


// perft() is our utility to verify move generation. All the leaf nodes
// up to the given depth are generated and counted, and the sum is returned.

static uint64_t perft_helper(Pos *pos, Depth depth, uint64_t nodes)
{
  ExtMove *m = (pos->st-1)->endMoves;
  ExtMove *last = pos->st->endMoves = generate_legal(pos, m);
  for (; m < last; m++) {
    do_move(pos, m->move, gives_check(pos, pos->st, m->move));
    if (depth == 0) {
      nodes += generate_legal(pos, last) - last;
    } else
      nodes = perft_helper(pos, depth - ONE_PLY, nodes);
    undo_move(pos, m->move);
  }
  return nodes;
}

uint64_t perft(Pos *pos, Depth depth)
{
  uint64_t cnt, nodes = 0;
  char buf[16];

  ExtMove *m = pos->moveList;
  ExtMove *last = pos->st->endMoves = generate_legal(pos, m);
  for (; m < last; m++) {
    if (depth <= ONE_PLY) {
      cnt = 1;
      nodes++;
    } else {
      do_move(pos, m->move, gives_check(pos, pos->st, m->move));
      if (depth == 2 * ONE_PLY)
        cnt = generate_legal(pos, last) - last;
      else
        cnt = perft_helper(pos, depth - 3 * ONE_PLY, 0);
      nodes += cnt;
      undo_move(pos, m->move);
    }
    printf("%s: %"PRIu64"\n", uci_move(buf, m->move, is_chess960()), cnt);
  }
  return nodes;
}

// mainthread_search() is called by the main thread when the program
// receives the UCI 'go' command. It searches from the root position and
// outputs the "bestmove".

void mainthread_search(void)
{
  Pos *pos = Threads.pos[0];
  int us = pos_stm();
  time_init(us, pos_game_ply());
  tt_new_search();
  char buf[16];
  int playBookMove = 0;

  base_ct = option_value(OPT_CONTEMPT) * PawnValueEg / 100;

  const char *s = option_string_value(OPT_ANALYSIS_CONTEMPT);
  if (Limits.infinite || option_value(OPT_ANALYSE_MODE))
    base_ct =  strcmp(s, "off") == 0 ? 0
             : strcmp(s, "white") == 0 && us == BLACK ? -base_ct
             : strcmp(s, "black") == 0 && us == WHITE ? -base_ct
             : base_ct;

  if (pos->rootMoves->size > 0) {
    Move bookMove = 0;

    if (!Limits.infinite && !Limits.mate)
      bookMove = pb_probe(pos);

    for (int i = 0; i < pos->rootMoves->size; i++)
      if (pos->rootMoves->move[i].pv[0] == bookMove) {
        RootMove tmp = pos->rootMoves->move[0];
        pos->rootMoves->move[0] = pos->rootMoves->move[i];
        pos->rootMoves->move[i] = tmp;
        playBookMove = 1;
        break;
      }

    if (!playBookMove) {
      for (int idx = 1; idx < Threads.numThreads; idx++)
        thread_wake_up(Threads.pos[idx], THREAD_SEARCH);

      thread_search(pos); // Let's start searching!
    }
  }

  // When we reach the maximum depth, we can arrive here without a raise
  // of Signals.stop. However, if we are pondering or in an infinite
  // search, the UCI protocol states that we shouldn't print the best
  // move before the GUI sends a "stop" or "ponderhit" command. We
  // therefore simply wait here until the GUI sends one of those commands
  // (which also raises Signals.stop).
  LOCK(Signals.lock);
  if (!Signals.stop && (Limits.ponder || Limits.infinite)) {
    Signals.sleeping = 1;
    UNLOCK(Signals.lock);
    thread_wait(pos, &Signals.stop);
  } else
    UNLOCK(Signals.lock);

  // Stop the threads if not already stopped
  Signals.stop = 1;

  // Wait until all threads have finished
  if (pos->rootMoves->size > 0) {
    if (!playBookMove) {
      for (int idx = 1; idx < Threads.numThreads; idx++)
        thread_wait_until_sleeping(Threads.pos[idx]);
    }
  } else {
    pos->rootMoves->move[0].pv[0] = 0;
    pos->rootMoves->move[0].pvSize = 1;
    pos->rootMoves->size++;
    printf("info depth 0 score %s\n",
           uci_value(buf, pos_checkers() ? -VALUE_MATE : VALUE_DRAW));
    fflush(stdout);
  }

  // When playing in 'nodes as time' mode, subtract the searched nodes from
  // the available ones before exiting.
  if (Limits.npmsec)
      Time.availableNodes += Limits.inc[us] - threads_nodes_searched();

  // Check if there are threads with a better score than main thread
  Pos *bestThread = pos;
  if (    option_value(OPT_MULTI_PV) == 1
      && !Limits.depth
//      && !Skill(option_value(OPT_SKILL_LEVEL)).enabled()
      &&  pos->rootMoves->move[0].pv[0] != 0)
  {
    for (int idx = 1; idx < Threads.numThreads; idx++) {
      Pos *p = Threads.pos[idx];
      Depth depthDiff = p->completedDepth - bestThread->completedDepth;
      Value scoreDiff = p->rootMoves->move[0].score - bestThread->rootMoves->move[0].score;
      // Select the thread with the best score, always if it is a mate
      if (    scoreDiff > 0
          && (depthDiff >= 0 || p->rootMoves->move[0].score >= VALUE_MATE_IN_MAX_PLY))
        bestThread = p;
    }
  }

  mainThread.previousScore = bestThread->rootMoves->move[0].score;

  // Send new PV when needed
  if (bestThread != pos)
    uci_print_pv(bestThread, bestThread->completedDepth,
                 -VALUE_INFINITE, VALUE_INFINITE);

  printf("bestmove %s", uci_move(buf, bestThread->rootMoves->move[0].pv[0], is_chess960()));

  if (bestThread->rootMoves->move[0].pvSize > 1 || extract_ponder_from_tt(&bestThread->rootMoves->move[0], pos))
    printf(" ponder %s", uci_move(buf, bestThread->rootMoves->move[0].pv[1], is_chess960()));

  printf("\n");
  fflush(stdout);
}


// thread_search() is the main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has
// been consumed, the user stops the search, or the maximum search depth is
// reached.

void thread_search(Pos *pos)
{
  Value bestValue, alpha, beta, delta;
  Move lastBestMove = 0;
  Depth lastBestMoveDepth = DEPTH_ZERO;
  double timeReduction = 1.0;

  Stack *ss = pos->st; // At least the fifth element of the allocated array.
  for (int i = -5; i < 3; i++)
    memset(SStackBegin(ss[i]), 0, SStackSize);
  (ss-1)->endMoves = pos->moveList;

  for (int i = -4; i < 0; i++)
    ss[i].history = &(*pos->counterMoveHistory)[0][0]; // Use as sentinel

  for (int i = 0; i <= MAX_PLY; i++)
    ss[i].ply = i;

  bestValue = delta = alpha = -VALUE_INFINITE;
  beta = VALUE_INFINITE;
  pos->completedDepth = DEPTH_ZERO;

  if (pos->threadIdx == 0) {
    mainThread.failedLow = 0;
    mainThread.bestMoveChanges = 0;
  }

  int multiPV = option_value(OPT_MULTI_PV);
#if 0
  Skill skill(option_value(OPT_SKILL_LEVEL));

  // When playing with strength handicap enable MultiPV search that we will
  // use behind the scenes to retrieve a set of possible moves.
  if (skill.enabled())
      multiPV = std::max(multiPV, (size_t)4);
#endif

  RootMoves *rm = pos->rootMoves;
  multiPV = min(multiPV, rm->size);

  // Iterative deepening loop until requested to stop or the target depth
  // is reached.
  while (   (pos->rootDepth += ONE_PLY) < DEPTH_MAX
         && !Signals.stop
         && !(   Limits.depth
              && pos->threadIdx == 0
              && pos->rootDepth / ONE_PLY > Limits.depth))
  {
    // Distribute search depths across the threads
    if (pos->threadIdx) {
      int i = (pos->threadIdx - 1) % 20;
      if (((pos->rootDepth / ONE_PLY + pos_game_ply() + skipPhase[i]) / skipSize[i]) % 2)
        continue;
    }

    // Age out PV variability metric
    if (pos->threadIdx == 0) {
      mainThread.bestMoveChanges *= 0.517;
      mainThread.failedLow = 0;
    }

    // Save the last iteration's scores before first PV line is searched and
    // all the move scores except the (new) PV are set to -VALUE_INFINITE.
    for (int idx = 0; idx < rm->size; idx++)
      rm->move[idx].previousScore = rm->move[idx].score;

    pos->contempt = pos_stm() == WHITE ?  make_score(base_ct, base_ct / 2)
                                       : -make_score(base_ct, base_ct / 2);

    int PVFirst = 0, PVLast = 0;

    // MultiPV loop. We perform a full root search for each PV line
    for (int PVIdx = 0; PVIdx < multiPV && !Signals.stop; PVIdx++) {
      pos->PVIdx = PVIdx;
      if (PVIdx == PVLast) {
        PVFirst = PVLast;
        for (PVLast++; PVLast < rm->size; PVLast++)
          if (rm->move[PVLast].TBRank != rm->move[PVFirst].TBRank)
            break;
        pos->PVLast = PVLast;
      }

      pos->selDepth = 0;

      // Skip the search if we have a mate value from DTM tables.
      if (abs(rm->move[PVIdx].TBRank) > 1000) {
        bestValue = rm->move[PVIdx].score = rm->move[PVIdx].TBScore;
        alpha = -VALUE_INFINITE;
        beta = VALUE_INFINITE;
        goto skip_search;
      }

      // Reset aspiration window starting size
      if (pos->rootDepth >= 5 * ONE_PLY) {
        Value previousScore = rm->move[PVIdx].previousScore;
        delta = (Value)18;
        alpha = max(previousScore - delta, -VALUE_INFINITE);
        beta  = min(previousScore + delta,  VALUE_INFINITE);

        // Adjust contempt based on root move's previousScore
        int ct = base_ct + 88 * previousScore / (abs(previousScore) + 200);
        pos->contempt = pos_stm() == WHITE ?  make_score(ct, ct / 2)
                                           : -make_score(ct, ct / 2);
      }

      // Start with a small aspiration window and, in the case of a fail
      // high/low, re-search with a bigger window until we're not failing
      // high/low anymore.
      while (1) {
        bestValue = search_PV(pos, ss, alpha, beta, pos->rootDepth);

        // Bring the best move to the front. It is critical that sorting
        // is done with a stable algorithm because all the values but the
        // first and eventually the new best one are set to -VALUE_INFINITE
        // and we want to keep the same order for all the moves except the
        // new PV that goes to the front. Note that in case of MultiPV
        // search the already searched PV lines are preserved.
        stable_sort(&rm->move[PVIdx], PVLast - PVIdx);

        // If search has been stopped, we break immediately. Sorting and
        // writing PV back to TT is safe because RootMoves is still
        // valid, although it refers to the previous iteration.
        if (Signals.stop)
          break;

        // When failing high/low give some update (without cluttering
        // the UI) before a re-search.
        if (   pos->threadIdx == 0
            && multiPV == 1
            && (bestValue <= alpha || bestValue >= beta)
            && time_elapsed() > 3000)
          uci_print_pv(pos, pos->rootDepth, alpha, beta);

        // In case of failing low/high increase aspiration window and
        // re-search, otherwise exit the loop.
        if (bestValue <= alpha) {
          beta = (alpha + beta) / 2;
          alpha = max(bestValue - delta, -VALUE_INFINITE);

          if (pos->threadIdx == 0) {
            mainThread.failedLow = 1;
            Signals.stopOnPonderhit = 0;
          }
        } else if (bestValue >= beta) {
          beta = min(bestValue + delta, VALUE_INFINITE);
        } else
          break;

        delta += delta / 4 + 5;

        assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
      }

      // Sort the PV lines searched so far and update the GUI
      stable_sort(&rm->move[PVFirst], PVIdx - PVFirst + 1);

skip_search:
      if (    pos->threadIdx == 0
          && (Signals.stop || PVIdx + 1 == multiPV || time_elapsed() > 3000))
        uci_print_pv(pos, pos->rootDepth, alpha, beta);
    }

    if (!Signals.stop)
      pos->completedDepth = pos->rootDepth;

    if (rm->move[0].pv[0] != lastBestMove) {
      lastBestMove = rm->move[0].pv[0];
      lastBestMoveDepth = pos->rootDepth;
    }

    // Have we found a "mate in x"?
    if (   Limits.mate
        && bestValue >= VALUE_MATE_IN_MAX_PLY
        && VALUE_MATE - bestValue <= 2 * Limits.mate)
      Signals.stop = 1;

    if (pos->threadIdx != 0)
      continue;

#if 0
    // If skill level is enabled and time is up, pick a sub-optimal best move
    if (skill.enabled() && skill.time_to_pick(thread->rootDepth))
      skill.pick_best(multiPV);
#endif

    // Do we have time for the next iteration? Can we stop searching now?
    if (use_time_management()) {
      if (!Signals.stop && !Signals.stopOnPonderhit) {
        // Stop the search if only one legal move is available, or if all
        // of the available time has been used.
        const int F[] = { mainThread.failedLow,
                          bestValue - mainThread.previousScore };

        int improvingFactor = max(246, min(832, 306 + 119 * F[0] - 6 * F[1]));

        double bestMoveInstability = 1 + mainThread.bestMoveChanges;

        // If the best move is stable over several iterations, reduce time
        // for this move, the longer the move has been stable, the more.
        // Use part of the time gained from a previous stable move for the
        // current move.
        timeReduction = 1;
        for (int i = 3; i < 6; i++)
          if (lastBestMoveDepth * i < pos->completedDepth)
            timeReduction *= 1.25;
        bestMoveInstability *= pow(mainThread.previousTimeReduction, 0.528) / timeReduction;

        if (   rm->size == 1
            || time_elapsed() > time_optimum() * bestMoveInstability * improvingFactor / 581)
        {
          // If we are allowed to ponder do not stop the search now but
          // keep pondering until the GUI sends "ponderhit" or "stop".
          if (Limits.ponder)
            Signals.stopOnPonderhit = 1;
          else
            Signals.stop = 1;
        }
      }
    }
  }

  if (pos->threadIdx != 0)
    return;

  mainThread.previousTimeReduction = timeReduction;

#if 0
  // If skill level is enabled, swap best PV line with the sub-optimal one
  if (skill.enabled())
    std::swap(rm[0], *std::find(rm.begin(),
              rm.end(), skill.best_move(multiPV)));
#endif
}

INLINE Value search_node(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth, int cutNode, const int NT)
{
  const bool PvNode = NT == PV;
  const bool rootNode = PvNode && ss->ply == 0;

  assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(DEPTH_ZERO < depth && depth < DEPTH_MAX);
  assert(!(PvNode && cutNode));

  Move pv[MAX_PLY+1], capturesSearched[32], quietsSearched[64];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, excludedMove, bestMove;
  Depth extension, newDepth;
  Value bestValue, value, ttValue, eval, maxValue;
  int ttHit, inCheck, givesCheck, improving;
  int captureOrPromotion, doFullDepthSearch, moveCountPruning, skipQuiets;
  int ttCapture, pvExact;
  Piece movedPiece;
  int moveCount, captureCount, quietCount;

  // Step 1. Initialize node
  inCheck = !!pos_checkers();
  moveCount = captureCount = quietCount =  ss->moveCount = 0;
  bestValue = -VALUE_INFINITE;
  maxValue = VALUE_INFINITE;

  // Check for the available remaining time
  if (load_rlx(pos->resetCalls)) {
    store_rlx(pos->resetCalls, 0);
    pos->callsCnt = Limits.nodes ? min(1024, Limits.nodes / 1024) : 1024;
  }
  if (--pos->callsCnt <= 0) {
    for (int idx = 0; idx < Threads.numThreads; idx++)
      store_rlx(Threads.pos[idx]->resetCalls, 1);

    check_time();
  }

  // Used to send selDepth info to GUI
  if (PvNode && pos->selDepth < ss->ply)
    pos->selDepth = ss->ply;

  if (!rootNode) {
    // Step 2. Check for aborted search and immediate draw
    if (load_rlx(Signals.stop) || is_draw(pos) || ss->ply >= MAX_PLY)
      return ss->ply >= MAX_PLY && !inCheck ? evaluate(pos) : VALUE_DRAW;

    // Step 3. Mate distance pruning. Even if we mate at the next move our
    // score would be at best mate_in(ss->ply+1), but if alpha is already
    // bigger because a shorter mate was found upward in the tree then
    // there is no need to search because we will never beat the current
    // alpha. Same logic but with reversed signs applies also in the
    // opposite condition of being mated instead of giving mate. In this
    // case return a fail-high score.
    if (PvNode) {
      alpha = max(mated_in(ss->ply), alpha);
      beta = min(mate_in(ss->ply+1), beta);
      if (alpha >= beta)
        return alpha;
      if (pos_rule50_count() >= 3 && alpha < VALUE_DRAW && has_game_cycle(pos)) {
        alpha = VALUE_DRAW;
        if (alpha >= beta)
          return alpha;
      }
    } else {
      if (alpha < mated_in(ss->ply))
        return mated_in(ss->ply);
      if (alpha >= mate_in(ss->ply+1))
        return alpha;
      if (pos_rule50_count() >= 3 && alpha < VALUE_DRAW && has_game_cycle(pos))
        return VALUE_DRAW;
    }
  }

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  (ss+1)->excludedMove = bestMove = 0;
  ss->history = &(*pos->counterMoveHistory)[0][0];
  (ss+2)->killers[0] = (ss+2)->killers[1] = 0;
  Square prevSq = to_sq((ss-1)->currentMove);
  (ss+2)->statScore = 0;

  // Step 4. Transposition table lookup. We don't want the score of a
  // partial search to overwrite a previous full search TT value, so we
  // use a different position key in case of an excluded move.
  excludedMove = ss->excludedMove;
#ifdef BIG_TT
  posKey = pos_key() ^ (Key)excludedMove;
#else
  posKey = pos_key() ^ (Key)((int32_t)excludedMove << 16);
#endif
  tte = tt_probe(posKey, &ttHit);
  ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply) : VALUE_NONE;
  ttMove =  rootNode ? pos->rootMoves->move[pos->PVIdx].pv[0]
          : ttHit    ? tte_move(tte) : 0;

  // At non-PV nodes we check for an early TT cutoff.
  if (  !PvNode
      && ttHit
      && tte_depth(tte) >= depth
      && ttValue != VALUE_NONE // Possible in case of TT access race.
      && (ttValue >= beta ? (tte_bound(tte) & BOUND_LOWER)
                          : (tte_bound(tte) & BOUND_UPPER)))
  {
    // If ttMove is quiet, update move sorting heuristics on TT hit.
    if (ttMove) {
      if (ttValue >= beta) {
        if (!is_capture_or_promotion(pos, ttMove))
          update_stats(pos, ss, ttMove, NULL, 0, stat_bonus(depth));

        // Extra penalty for a quiet TT move in previous ply when it gets
        // refuted.
        if ((ss-1)->moveCount == 1 && !captured_piece())
          update_cm_stats(ss-1, piece_on(prevSq), prevSq,
                          -stat_bonus(depth + ONE_PLY));
      }
      // Penalty for a quiet ttMove that fails low
      else if (!is_capture_or_promotion(pos, ttMove)) {
        Value penalty = -stat_bonus(depth);
        history_update(*pos->history, pos_stm(), ttMove, penalty);
        update_cm_stats(ss, moved_piece(ttMove), to_sq(ttMove), penalty);
      }
    }
    return ttValue;
  }

  // Step 5. Tablebase probe
  if (!rootNode && TB_Cardinality) {
    int piecesCnt = popcount(pieces());

    if (    piecesCnt <= TB_Cardinality
        && (piecesCnt <  TB_Cardinality || depth >= TB_ProbeDepth)
        &&  pos_rule50_count() == 0
        && !can_castle_any())
    {
      int found, wdl = TB_probe_wdl(pos, &found);

      if (found) {
        pos->tbHits++;

        int drawScore = TB_UseRule50 ? 1 : 0;

        value =  wdl < -drawScore ? -VALUE_MATE + MAX_MATE_PLY + 1 + ss->ply
               : wdl >  drawScore ?  VALUE_MATE - MAX_MATE_PLY - 1 - ss->ply
                                  :  VALUE_DRAW + 2 * wdl * drawScore;

        int b =  wdl < -drawScore ? BOUND_UPPER
               : wdl >  drawScore ? BOUND_LOWER : BOUND_EXACT;

        if (    b == BOUND_EXACT
            || (b == BOUND_LOWER ? value >= beta : value <= alpha))
        {
          tte_save(tte, posKey, value_to_tt(value, ss->ply), b,
                   min(DEPTH_MAX - ONE_PLY, depth + 6 * ONE_PLY), 0,
                   VALUE_NONE, tt_generation());
          return value;
        }

        if (piecesCnt <= TB_CardinalityDTM) {
          Value mate = TB_probe_dtm(pos, wdl, &found);
          if (found) {
            mate += wdl > 0 ? -ss->ply : ss->ply;
            tte_save(tte, posKey, value_to_tt(mate, ss->ply), BOUND_EXACT,
                     min(DEPTH_MAX - ONE_PLY, depth + 6 * ONE_PLY), 0,
                     VALUE_NONE, tt_generation());
            return mate;
          }
        }

        if (PvNode) {
          if (b == BOUND_LOWER) {
            bestValue = value;
            if (bestValue > alpha)
              alpha = bestValue;
          } else
            maxValue = value;
        }
      }
    }
  }

  // Step 6. Static evaluation of the position
  if (inCheck) {
    ss->staticEval = VALUE_NONE;
    improving = 0;
    goto moves_loop; // Skip early pruning when in check
  } else if (ttHit) {
    // Never assume anything on values stored in TT
    if ((ss->staticEval = eval = tte_eval(tte)) == VALUE_NONE)
      eval = ss->staticEval = evaluate(pos);

    // Can ttValue be used as a better position evaluation?
    if (ttValue != VALUE_NONE)
      if (tte_bound(tte) & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER))
        eval = ttValue;
  } else {
    eval = ss->staticEval =
    (ss-1)->currentMove != MOVE_NULL ? evaluate(pos)
                                     : -(ss-1)->staticEval + 2 * Tempo;

    tte_save(tte, posKey, VALUE_NONE, BOUND_NONE, DEPTH_NONE, 0,
             ss->staticEval, tt_generation());
  }

  // Step 7. Razoring
  if (  !PvNode
      && depth <= ONE_PLY)
  {
    if (eval + RazorMargin1 <= alpha)
      return qsearch_NonPV_false(pos, ss, alpha, DEPTH_ZERO);
  }
  else if (  !PvNode
           && depth <= 2 * ONE_PLY
           && eval + RazorMargin2 <= alpha)
  {
    Value ralpha = alpha - RazorMargin2;
    Value v = qsearch_NonPV_false(pos, ss, ralpha, DEPTH_ZERO);
    if (v <= ralpha)
      return v;
  }

  improving =   ss->staticEval >= (ss-2)->staticEval
             || (ss-2)->staticEval == VALUE_NONE;

  // Step 8. Futility pruning: child node
  if (   !rootNode
      &&  depth < 7 * ONE_PLY
      &&  eval - futility_margin(depth, improving) >= beta
      &&  eval < VALUE_KNOWN_WIN)  // Do not return unproven wins
    return eval; // - futility_margin(depth); (do not do the right thing)

  // Step 9. Null move search with verification search (is omitted in PV nodes)
  if (   !PvNode
      && (ss-1)->currentMove != MOVE_NULL
      && (ss-1)->statScore < 22500
      &&  eval >= beta
      &&  ss->staticEval >= beta - 36 * depth / ONE_PLY + 225
      && !excludedMove
      &&  pos_non_pawn_material(pos_stm())
      && (ss->ply >= pos->nmpPly || ss->ply % 2 != pos->nmpOdd))
  {
    assert(eval - beta >= 0);

    // Null move dynamic reduction based on depth and value
    Depth R = ((823 + 67 * depth / ONE_PLY) / 256 + min((eval - beta) / PawnValueMg, 3)) * ONE_PLY;

    ss->currentMove = MOVE_NULL;
    ss->history = &(*pos->counterMoveHistory)[0][0];

    do_null_move(pos);
    ss->endMoves = (ss-1)->endMoves;
    Value nullValue =   depth-R < ONE_PLY
                     ? -qsearch_NonPV_false(pos, ss+1, -beta, DEPTH_ZERO)
                     : - search_NonPV(pos, ss+1, -beta, depth-R, !cutNode);
    undo_null_move(pos);

    if (nullValue >= beta) {
      // Do not return unproven mate scores
      if (nullValue >= VALUE_MATE_IN_MAX_PLY)
        nullValue = beta;

      if (  (depth < 12 * ONE_PLY || pos->nmpPly)
          && abs(beta) < VALUE_KNOWN_WIN)
        return nullValue;

      // Do verification search at high depths
      // Disable null move pruning for side to move for the first part of
      // the remaining search tree
      pos->nmpPly = ss->ply + 3 * (depth-R) / (4 * ONE_PLY);
      pos->nmpOdd = ss->ply & 1;

      Value v =  depth-R < ONE_PLY
               ? qsearch_NonPV_false(pos, ss, beta-1, DEPTH_ZERO)
               : search_NonPV(pos, ss, beta-1, depth-R, 0);

      pos->nmpOdd = pos->nmpPly = 0;

      if (v >= beta)
        return nullValue;
    }
  }

  // Step 10. ProbCut
  // If we have a good enough capture and a reduced search returns a value
  // much above beta, we can (almost) safely prune the previous move.
  if (  !PvNode
      && depth >= 5 * ONE_PLY
      && abs(beta) < VALUE_MATE_IN_MAX_PLY)
  {
    Value rbeta = min(beta + 216 - 48 * improving, VALUE_INFINITE);
    Depth rdepth = depth - 4 * ONE_PLY;

    assert(rdepth >= ONE_PLY);

    mp_init_pc(pos, ttMove, rbeta - ss->staticEval);

    int probCutCount = 3;
    while ((move = next_move(pos, 0)) && probCutCount)
      if (is_legal(pos, move)) {
        probCutCount--;
        ss->currentMove = move;
        ss->history = &(*pos->counterMoveHistory)[moved_piece(move)][to_sq(move)];
        givesCheck = gives_check(pos, ss, move);
        do_move(pos, move, givesCheck);

        // Perform a preliminary qsearch to verify that the move holds
        value =   givesCheck
               ? -qsearch_NonPV_true(pos, ss+1, -rbeta, DEPTH_ZERO)
               : -qsearch_NonPV_false(pos, ss+1, -rbeta, DEPTH_ZERO);

        // If the qsearch holds perform the regular search
        if (value >= rbeta)
          value = -search_NonPV(pos, ss+1, -rbeta, rdepth, !cutNode);

        undo_move(pos, move);
        if (value >= rbeta)
          return value;
      }
  }

  // Step 11. Internal iterative deepening
  if (    depth >= 8 * ONE_PLY
      && !ttMove)
  {
    Depth d = (3 * depth / (4 * ONE_PLY) - 2) * ONE_PLY;
    if (PvNode)
      search_PV(pos, ss, alpha, beta, d);
    else
      search_NonPV(pos, ss, alpha, d, cutNode);

    tte = tt_probe(posKey, &ttHit);
    // ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply) : VALUE_NONE;
    ttMove = ttHit ? tte_move(tte) : 0;
  }

moves_loop: // When in check search starts from here.
  ;  // Avoid a compiler warning. A label must be followed by a statement.
  PieceToHistory *cmh  = (ss-1)->history;
  PieceToHistory *fmh  = (ss-2)->history;
  PieceToHistory *fmh2 = (ss-4)->history;

  mp_init(pos, ttMove, depth);
  value = bestValue; // Workaround a bogus 'uninitialized' warning under gcc

  skipQuiets = 0;
  ttCapture = 0;
  pvExact = PvNode && ttHit && tte_bound(tte) == BOUND_EXACT;

  // Step 12. Loop through moves
  // Loop through all pseudo-legal moves until no moves remain or a beta
  // cutoff occurs
  while ((move = next_move(pos, skipQuiets))) {
    assert(move_is_ok(move));

    if (move == excludedMove)
      continue;

    // At root obey the "searchmoves" option and skip moves not listed
    // inRoot Move List. As a consequence any illegal move is also skipped.
    // In MultiPV mode we also skip PV moves which have been already
    // searched.
    if (rootNode) {
      int idx;
      for (idx = pos->PVIdx; idx < pos->PVLast; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move)
          break;
      if (idx == pos->PVLast)
        continue;
    }

    ss->moveCount = ++moveCount;

    if (rootNode && pos->threadIdx == 0 && time_elapsed() > 3000) {
      char buf[16];
      printf("info depth %d currmove %s currmovenumber %d\n",
             depth / ONE_PLY,
             uci_move(buf, move, is_chess960()),
             moveCount + pos->PVIdx);
      fflush(stdout);
    }

    if (PvNode)
      (ss+1)->pv = NULL;

    extension = DEPTH_ZERO;
    captureOrPromotion = is_capture_or_promotion(pos, move);
    movedPiece = moved_piece(move);

    givesCheck = gives_check(pos, ss, move);

    moveCountPruning = depth < 16 * ONE_PLY
                && moveCount >= FutilityMoveCounts[improving][depth / ONE_PLY];

    // Step 13. Singular and Gives Check Extensions

    // Singular extension search. If all moves but one fail low on a search
    // of (alpha-s, beta-s), and just one fails high on (alpha, beta), then
    // that move is singular and should be extended. To verify this we do a
    // reduced search on all the other moves but the ttMove and if the
    // result is lower than ttValue minus a margin then we extend the ttMove.
    if (    depth >= 8 * ONE_PLY
        &&  move == ttMove
        && !rootNode
        && !excludedMove // No recursive singular search
        &&  ttValue != VALUE_NONE
        && (tte_bound(tte) & BOUND_LOWER)
        &&  tte_depth(tte) >= depth - 3 * ONE_PLY
        &&  is_legal(pos, move))
    {
      Value rBeta = max(ttValue - 2 * depth / ONE_PLY, -VALUE_MATE);
//      Value rBeta = min(max(ttValue - 2 * depth / ONE_PLY, -VALUE_MATE), VALUE_KNOWN_WIN);
      Depth d = (depth / (2 * ONE_PLY)) * ONE_PLY;
      ss->excludedMove = move;
      Move cm = ss->countermove;
      Move k1 = ss->mpKillers[0], k2 = ss->mpKillers[1];
      value = search_NonPV(pos, ss, rBeta - 1, d, cutNode);
      ss->excludedMove = 0;

      if (value < rBeta)
        extension = ONE_PLY;

      // The call to search_NonPV with the same value of ss messed up our
      // move picker data. So we fix it.
      mp_init(pos, ttMove, depth);
      ss->stage++;
      ss->countermove = cm; // pedantic
      ss->mpKillers[0] = k1; ss->mpKillers[1] = k2;
    }
    else if (    givesCheck
             && !moveCountPruning
             &&  see_test(pos, move, 0))
      extension = ONE_PLY;

    // Calculate new depth for this move
    newDepth = depth - ONE_PLY + extension;

    // Step 14. Pruning at shallow depth
    if (  !rootNode
        && pos_non_pawn_material(pos_stm())
        && bestValue > VALUE_MATED_IN_MAX_PLY)
    {
      if (   !captureOrPromotion
          && !givesCheck
          && (  !advanced_pawn_push(pos, move)
              || pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK) >= 5000))
      {
        // Move count based pruning
        if (moveCountPruning) {
          skipQuiets = 1;
          continue;
        }

        // Reduced depth of the next LMR search
        int lmrDepth = max(newDepth - reduction(improving, depth, moveCount, NT), DEPTH_ZERO) / ONE_PLY;

        // Countermoves based pruning
        if (   lmrDepth < 3
            && (*cmh )[movedPiece][to_sq(move)] < CounterMovePruneThreshold
            && (*fmh )[movedPiece][to_sq(move)] < CounterMovePruneThreshold)
          continue;

        // Futility pruning: parent node
        if (   lmrDepth < 7
            && !inCheck
            && ss->staticEval + 256 + 200 * lmrDepth <= alpha)
          continue;

        // Prune moves with negative SEE at low depths and below a decreasing
        // threshold at higher depths.
        if (   lmrDepth < 8
            && !extension
            && !see_test(pos, move, -35 * lmrDepth * lmrDepth))
          continue;
      }
      else if (    depth < 7 * ONE_PLY
               && !extension
               && !see_test(pos, move, -CapturePruneMargin[depth / ONE_PLY]))
        continue;
    }

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Check for legality just before making the move
    if (!rootNode && !is_legal(pos, move)) {
      ss->moveCount = --moveCount;
      continue;
    }

    if (move == ttMove && captureOrPromotion)
      ttCapture = 1;

    // Update the current move (this must be done after singular extension
    // search)
    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[movedPiece][to_sq(move)];

    // Step 15. Make the move.
    do_move(pos, move, givesCheck);
    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;

    // Step 16. Reduced depth search (LMR). If the move fails high it will be
    // re-searched at full depth.
    if (    depth >= 3 * ONE_PLY
        &&  moveCount > 1
        && (!captureOrPromotion || moveCountPruning))
    {
      Depth r = reduction(improving, depth, moveCount, NT);

      if (captureOrPromotion) {
        // Increase reduction depending on opponent's stat score
        if (   (ss-1)->statScore >= 0
            && (*pos->captureHistory)[movedPiece][to_sq(move)][type_of_p(captured_piece())] < 0)
          r += ONE_PLY;

        r -= r ? ONE_PLY : DEPTH_ZERO;
      } else {
        // Decrease reduction if opponent's move count is high
        if ((ss-1)->moveCount > 15)
          r -= ONE_PLY;

        // Decrease reduction for exact PV nodes
        if (pvExact)
          r -= ONE_PLY;

        // Increase reduction if ttMove is a capture
        if (ttCapture)
          r += ONE_PLY;

        // Increase reduction for cut nodes
        if (cutNode)
          r += 2 * ONE_PLY;

        // Decrease reduction for moves that escape a capture. Filter out
        // castling moves, because they are coded as "king captures rook" and
        // hence break make_move(). Also use see() instead of see_sign(),
        // because the destination square is empty.
        else if (   type_of_m(move) == NORMAL
                 && !see_test(pos, make_move(to_sq(move), from_sq(move)), 0))
          r -= 2 * ONE_PLY;

        ss->statScore =  (*cmh )[movedPiece][to_sq(move)]
                       + (*fmh )[movedPiece][to_sq(move)]
                       + (*fmh2)[movedPiece][to_sq(move)]
                       + (*pos->history)[pos_stm() ^ 1][from_to(move)]
                       - 4000; // Correction factor.

        // Decrease/increase reduction by comparing with opponent's stat score.
        if (ss->statScore >= 0 && (ss-1)->statScore < 0)
          r -= ONE_PLY;

        else if ((ss-1)->statScore >= 0 && ss->statScore < 0)
          r += ONE_PLY;

        // Decrease/increase reduction for moves with a good/bad history.
        r = max(DEPTH_ZERO, (r / ONE_PLY - ss->statScore / 20000) * ONE_PLY);
      }

      Depth d = max(newDepth - r, ONE_PLY);

      value = -search_NonPV(pos, ss+1, -(alpha+1), d, 1);

      doFullDepthSearch = (value > alpha && d != newDepth);

    } else
      doFullDepthSearch = !PvNode || moveCount > 1;

    // Step 17. Full depth search when LMR is skipped or fails high.
    if (doFullDepthSearch)
        value =  newDepth < ONE_PLY
               ?   givesCheck
                 ? -qsearch_NonPV_true(pos, ss+1, -(alpha+1), DEPTH_ZERO)
                 : -qsearch_NonPV_false(pos, ss+1, -(alpha+1), DEPTH_ZERO)
               : -search_NonPV(pos, ss+1, -(alpha+1), newDepth, !cutNode);

    // For PV nodes only, do a full PV search on the first move or after a fail
    // high (in the latter case search only if value < beta), otherwise let the
    // parent node fail low with value <= alpha and try another move.
    if (PvNode
        && (moveCount == 1 || (value > alpha && (rootNode || value < beta))))
    {
      (ss+1)->pv = pv;
      (ss+1)->pv[0] = 0;

      value =  newDepth < ONE_PLY
             ?   givesCheck
               ? -qsearch_PV_true(pos, ss+1, -beta, -alpha, DEPTH_ZERO)
               : -qsearch_PV_false(pos, ss+1, -beta, -alpha, DEPTH_ZERO)
             : -search_PV(pos, ss+1, -beta, -alpha, newDepth);
    }

    // Step 18. Undo move
    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Step 19. Check for a new best move
    // Finished searching the move. If a stop occurred, the return value of
    // the search cannot be trusted, and we return immediately without
    // updating best move, PV and TT.
    if (load_rlx(Signals.stop))
      return 0;

    if (rootNode) {
      RootMove *rm = NULL;
      for (int idx = 0; idx < pos->rootMoves->size; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move) {
          rm = &pos->rootMoves->move[idx];
          break;
        }

      // PV move or new best move ?
      if (moveCount == 1 || value > alpha) {
        rm->score = value;
        rm->selDepth = pos->selDepth;
        rm->pvSize = 1;

        assert((ss+1)->pv);

        for (Move *m = (ss+1)->pv; *m; ++m)
          rm->pv[rm->pvSize++] = *m;

        // We record how often the best move has been changed in each
        // iteration. This information is used for time management: When
        // the best move changes frequently, we allocate some more time.
        if (moveCount > 1 && pos->threadIdx == 0)
          mainThread.bestMoveChanges++;
      } else
        // All other moves but the PV are set to the lowest value: this is
        // not a problem when sorting because the sort is stable and the
        // move position in the list is preserved - just the PV is pushed up.
        rm->score = -VALUE_INFINITE;
    }

    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (PvNode && !rootNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (PvNode && value < beta) // Update alpha! Always alpha < beta
          alpha = value;
        else {
          assert(value >= beta); // Fail high
          ss->statScore = 0;
          break;
        }
      }
    }

    if (!captureOrPromotion && move != bestMove && quietCount < 64)
      quietsSearched[quietCount++] = move;
    else if (captureOrPromotion && move != bestMove && captureCount < 32)
      capturesSearched[captureCount++] = move;
  }

  // The following condition would detect a stop only after move loop has
  // been completed. But in this case bestValue is valid because we have
  // fully searched our subtree, and we can anyhow save the result in TT.
  /*
  if (Signals.stop)
    return VALUE_DRAW;
  */

  // Step 20. Check for mate and stalemate
  // All legal moves have been searched and if there are no legal moves,
  // it must be a mate or a stalemate. If we are in a singular extension
  // search then return a fail low score.
  if (!moveCount)
    bestValue = excludedMove ? alpha
               :     inCheck ? mated_in(ss->ply) : VALUE_DRAW;
  else if (bestMove) {
    // Quiet best move: update move sorting heuristics.
    if (!is_capture_or_promotion(pos, bestMove))
      update_stats(pos, ss, bestMove, quietsSearched, quietCount,
                   stat_bonus(depth));
    else
      update_capture_stats(pos, bestMove, capturesSearched, captureCount,
                           stat_bonus(depth));

    // Extra penalty for a quiet TT move in previous ply when it gets refuted.
    if ((ss-1)->moveCount == 1 && !captured_piece())
      update_cm_stats(ss-1, piece_on(prevSq), prevSq,
                      -stat_bonus(depth + ONE_PLY));
  }
  // Bonus for prior countermove that caused the fail low.
  else if (   (depth >= 3 * ONE_PLY || PvNode)
           && !captured_piece()
           && move_is_ok((ss-1)->currentMove))
    update_cm_stats(ss-1, piece_on(prevSq), prevSq, stat_bonus(depth));

  if (PvNode && bestValue > maxValue)
     bestValue = maxValue;

  if (!excludedMove)
    tte_save(tte, posKey, value_to_tt(bestValue, ss->ply),
             bestValue >= beta ? BOUND_LOWER :
             PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
             depth, bestMove, ss->staticEval, tt_generation());

  assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

  return bestValue;
}

// search_PV() is the main search function for PV nodes
static Value search_PV(Pos *pos, Stack *ss, Value alpha, Value beta,
    Depth depth)
{
  return search_node(pos, ss, alpha, beta, depth, 0, PV);
}

// search_NonPV is the main search function for non-PV nodes
static Value search_NonPV(Pos *pos, Stack *ss, Value alpha, Depth depth,
    int cutNode)
{
  return search_node(pos, ss, alpha, alpha+1, depth, cutNode, NonPV);
}

// qsearch() is the quiescence search function, which is called by the main
// search function when the remaining depth is zero (or, to be more precise,
// less than ONE_PLY)

INLINE Value qsearch_node(Pos* pos, Stack* ss, Value alpha, Value beta,
    Depth depth, const int NT, const int InCheck)
{
  const bool PvNode = NT == PV;

  assert(InCheck == !!pos_checkers());
  assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(depth <= DEPTH_ZERO);

  Move pv[MAX_PLY+1];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, bestMove;
  Value bestValue, value, ttValue, futilityValue, futilityBase, oldAlpha;
  int ttHit, givesCheck, evasionPrunable;
  Depth ttDepth;
  int moveCount;

  if (PvNode) {
    oldAlpha = alpha; // To flag BOUND_EXACT when eval above alpha and no available moves
    (ss+1)->pv = pv;
    ss->pv[0] = 0;
  }

  bestMove = 0;
  moveCount = 0;

  // Check for an instant draw or if the maximum ply has been reached
  if (is_draw(pos) || ss->ply >= MAX_PLY)
    return ss->ply >= MAX_PLY && !InCheck ? evaluate(pos) : VALUE_DRAW;

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  // Decide whether or not to include checks: this fixes also the type of
  // TT entry depth that we are going to use. Note that in qsearch we use
  // only two types of depth in TT: DEPTH_QS_CHECKS or DEPTH_QS_NO_CHECKS.
  ttDepth = InCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
                                                : DEPTH_QS_NO_CHECKS;

  // Transposition table lookup
  posKey = pos_key();
  tte = tt_probe(posKey, &ttHit);
  ttMove = ttHit ? tte_move(tte) : 0;
  ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply) : VALUE_NONE;

  if (  !PvNode
      && ttHit
      && tte_depth(tte) >= ttDepth
      && ttValue != VALUE_NONE // Only in case of TT access race
      && (ttValue >= beta ? (tte_bound(tte) &  BOUND_LOWER)
                          : (tte_bound(tte) &  BOUND_UPPER)))
    return ttValue;

  // Evaluate the position statically
  if (InCheck) {
    ss->staticEval = VALUE_NONE;
    bestValue = futilityBase = -VALUE_INFINITE;
  } else {
    if (ttHit) {
      // Never assume anything on values stored in TT
      if ((ss->staticEval = bestValue = tte_eval(tte)) == VALUE_NONE)
         ss->staticEval = bestValue = evaluate(pos);

      // Can ttValue be used as a better position evaluation?
      if (ttValue != VALUE_NONE)
        if (tte_bound(tte) & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER))
          bestValue = ttValue;
    } else
      ss->staticEval = bestValue =
      (ss-1)->currentMove != MOVE_NULL ? evaluate(pos)
                                       : -(ss-1)->staticEval + 2 * Tempo;

    // Stand pat. Return immediately if static value is at least beta
    if (bestValue >= beta) {
      if (!ttHit)
        tte_save(tte, posKey, value_to_tt(bestValue, ss->ply),
                 BOUND_LOWER, DEPTH_NONE, 0, ss->staticEval,
                 tt_generation());

      return bestValue;
    }

    if (PvNode && bestValue > alpha)
      alpha = bestValue;

    futilityBase = bestValue + 128;
  }

  // Initialize move picker data for the current position, and prepare
  // to search the moves. Because the depth is <= 0 here, only captures,
  // queen promotions and checks (only if depth >= DEPTH_QS_CHECKS) will
  // be generated.
  mp_init_q(pos, ttMove, depth, to_sq((ss-1)->currentMove));

  // Loop through the moves until no moves remain or a beta cutoff occurs
  while ((move = next_move(pos, 0))) {
    assert(move_is_ok(move));

    givesCheck = gives_check(pos, ss, move);

    moveCount++;

    // Futility pruning
    if (   !InCheck
        && !givesCheck
        &&  futilityBase > -VALUE_KNOWN_WIN
        && !advanced_pawn_push(pos, move)) {
      assert(type_of_m(move) != ENPASSANT); // Due to !advanced_pawn_push

      futilityValue = futilityBase + PieceValue[EG][piece_on(to_sq(move))];

      if (futilityValue <= alpha) {
        bestValue = max(bestValue, futilityValue);
        continue;
      }

      if (futilityBase <= alpha && !see_test(pos, move, 1)) {
        bestValue = max(bestValue, futilityBase);
        continue;
      }
    }

    // Detect non-capture evasions that are candidates to be pruned
    evasionPrunable =    InCheck
                     && (depth != DEPTH_ZERO || moveCount > 2)
                     &&  bestValue > VALUE_MATED_IN_MAX_PLY
                     && !is_capture(pos, move);

    // Don't search moves with negative SEE values
    if (  (!InCheck || evasionPrunable)
        &&  !see_test(pos, move, 0))
      continue;

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Check for legality just before making the move
    if (!is_legal(pos, move)) {
      moveCount--;
      continue;
    }

    ss->currentMove = move;

    // Make and search the move
    do_move(pos, move, givesCheck);
    value =  PvNode
           ?   givesCheck
             ? -qsearch_PV_true(pos, ss+1, -beta, -alpha, depth - ONE_PLY)
             : -qsearch_PV_false(pos, ss+1, -beta, -alpha, depth - ONE_PLY)
           :   givesCheck
             ? -qsearch_NonPV_true(pos, ss+1, -beta, depth - ONE_PLY)
             : -qsearch_NonPV_false(pos, ss+1, -beta, depth - ONE_PLY);
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Check for a new best move
    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        if (PvNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (PvNode && value < beta) { // Update alpha here!
          alpha = value;
          bestMove = move;
        } else { // Fail high
          tte_save(tte, posKey, value_to_tt(value, ss->ply), BOUND_LOWER,
                   ttDepth, move, ss->staticEval, tt_generation());

          return value;
        }
      }
    }
  }

  // All legal moves have been searched. A special case: If we're in check
  // and no legal moves were found, it is checkmate.
  if (InCheck && bestValue == -VALUE_INFINITE)
    return mated_in(ss->ply); // Plies to mate from the root

  tte_save(tte, posKey, value_to_tt(bestValue, ss->ply),
           PvNode && bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
           ttDepth, bestMove, ss->staticEval, tt_generation());

  assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

  return bestValue;
}

static Value qsearch_PV_true(Pos *pos, Stack *ss, Value alpha, Value beta,
    Depth depth)
{
  return qsearch_node(pos, ss, alpha, beta, depth, PV, true);
}

static Value qsearch_PV_false(Pos *pos, Stack *ss, Value alpha, Value beta,
    Depth depth)
{
  return qsearch_node(pos, ss, alpha, beta, depth, PV, false);
}

static Value qsearch_NonPV_true(Pos *pos, Stack *ss, Value alpha, Depth depth)
{
  return qsearch_node(pos, ss, alpha, alpha+1, depth, NonPV, true);
}

static Value qsearch_NonPV_false(Pos *pos, Stack *ss, Value alpha, Depth depth)
{
  return qsearch_node(pos, ss, alpha, alpha+1, depth, NonPV, false);
}

#define rm_lt(m1,m2) ((m1).TBRank != (m2).TBRank ? (m1).TBRank < (m2).TBRank : (m1).score != (m2).score ? (m1).score < (m2).score : (m1).previousScore < (m2).previousScore)

// stable_sort() sorts RootMoves from highest-scoring move to lowest-scoring
// move while preserving order of equal elements.
static void stable_sort(RootMove *rm, int num)
{
  int i, j;

  for (i = 1; i < num; i++)
    if (rm_lt(rm[i - 1], rm[i])) {
      RootMove tmp = rm[i];
      rm[i] = rm[i - 1];
      for (j = i - 1; j > 0 && rm_lt(rm[j - 1], tmp); j--)
        rm[j] = rm[j - 1];
      rm[j] = tmp;
    }
}

// value_to_tt() adjusts a mate score from "plies to mate from the root" to
// "plies to mate from the current position". Non-mate scores are unchanged.
// The function is called before storing a value in the transposition table.

static Value value_to_tt(Value v, int ply)
{
  assert(v != VALUE_NONE);

  return  v >= VALUE_MATE_IN_MAX_PLY  ? v + ply
        : v <= VALUE_MATED_IN_MAX_PLY ? v - ply : v;
}


// value_from_tt() is the inverse of value_to_tt(): It adjusts a mate score
// from the transposition table (which refers to the plies to mate/be mated
// from current position) to "plies to mate/be mated from the root".

static Value value_from_tt(Value v, int ply)
{
  return  v == VALUE_NONE             ? VALUE_NONE
        : v >= VALUE_MATE_IN_MAX_PLY  ? v - ply
        : v <= VALUE_MATED_IN_MAX_PLY ? v + ply : v;
}


// update_pv() adds current move and appends child pv[]

static void update_pv(Move *pv, Move move, Move *childPv)
{
  for (*pv++ = move; childPv && *childPv; )
    *pv++ = *childPv++;
  *pv = 0;
}


// update_cm_stats() updates countermove and follow-up move history.

static void update_cm_stats(Stack *ss, Piece pc, Square s, int bonus)
{
  if (move_is_ok((ss-1)->currentMove))
    cms_update(*(ss-1)->history, pc, s, bonus);

  if (move_is_ok((ss-2)->currentMove))
    cms_update(*(ss-2)->history, pc, s, bonus);

  if (move_is_ok((ss-4)->currentMove))
    cms_update(*(ss-4)->history, pc, s, bonus);
}

// update_capture_stats() updates move sorting heuristics when a new capture
// best move is found

static void update_capture_stats(const Pos *pos, Move move, Move *captures,
    int captureCnt, int bonus)
{
  Piece moved_piece = moved_piece(move);
  int captured = type_of_p(piece_on(to_sq(move)));
  cpth_update(*pos->captureHistory, moved_piece, to_sq(move), captured, bonus);

  // Decrease all the other played capture moves
  for (int i = 0; i < captureCnt; i++) {
    moved_piece = moved_piece(captures[i]);
    captured = type_of_p(piece_on(to_sq(captures[i])));
    cpth_update(*pos->captureHistory, moved_piece, to_sq(captures[i]), captured, -bonus);
  }
}

// update_stats() updates killers, history, countermove and countermove
// plus follow-up move history when a new quiet best move is found.

static void update_stats(const Pos *pos, Stack *ss, Move move, Move *quiets,
    int quietsCnt, int bonus)
{
  if (ss->killers[0] != move) {
    ss->killers[1] = ss->killers[0];
    ss->killers[0] = move;
  }

  int c = pos_stm();
  history_update(*pos->history, c, move, bonus);
  update_cm_stats(ss, moved_piece(move), to_sq(move), bonus);

  if (move_is_ok((ss-1)->currentMove)) {
    Square prevSq = to_sq((ss-1)->currentMove);
    (*pos->counterMoves)[piece_on(prevSq)][prevSq] = move;
  }

  // Decrease all the other played quiet moves
  for (int i = 0; i < quietsCnt; i++) {
    history_update(*pos->history, c, quiets[i], -bonus);
    update_cm_stats(ss, moved_piece(quiets[i]), to_sq(quiets[i]), -bonus);
  }
}

#if 0
// When playing with strength handicap, choose best move among a set of RootMoves
// using a statistical rule dependent on 'level'. Idea by Heinz van Saanen.

  Move Skill::pick_best(size_t multiPV) {

    const RootMoves& rm = Threads.main()->rootMoves;
    static PRNG rng(now()); // PRNG sequence should be non-deterministic

    // RootMoves are already sorted by score in descending order
    Value topScore = rm[0].score;
    int delta = std::min(topScore - rm[multiPV - 1].score, PawnValueMg);
    int weakness = 120 - 2 * level;
    int maxScore = -VALUE_INFINITE;

    // Choose best move. For each move score we add two terms, both dependent on
    // weakness. One is deterministic and bigger for weaker levels, and one is
    // random. Then we choose the move with the resulting highest score.
    for (size_t i = 0; i < multiPV; ++i)
    {
        // This is our magic formula
        int push = (  weakness * int(topScore - rm[i].score)
                    + delta * (rng.rand<unsigned>() % weakness)) / 128;

        if (rm[i].score + push > maxScore)
        {
            maxScore = rm[i].score + push;
            best = rm[i].pv[0];
        }
    }

    return best;
  }
#endif


// check_time() is used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.

static void check_time(void)
{
  TimePoint elapsed = time_elapsed();

  // An engine may not stop pondering until told so by the GUI
  if (Limits.ponder)
    return;

  if (   (use_time_management() && elapsed > time_maximum() - 10)
      || (Limits.movetime && elapsed >= Limits.movetime)
      || (Limits.nodes && threads_nodes_searched() >= Limits.nodes))
        Signals.stop = 1;
}

// uci_print_pv() prints PV information according to the UCI protocol.
// UCI requires that all (if any) unsearched PV lines are sent with a
// previous search score.

static void uci_print_pv(Pos *pos, Depth depth, Value alpha, Value beta)
{
  TimePoint elapsed = time_elapsed() + 1;
  RootMoves *rm = pos->rootMoves;
  int PVIdx = pos->PVIdx;
  int multiPV = min(option_value(OPT_MULTI_PV), rm->size);
  uint64_t nodes_searched = threads_nodes_searched();
  uint64_t tbhits = threads_tb_hits();
  char buf[16];

  for (int i = 0; i < multiPV; i++) {
    int updated = (i <= PVIdx && rm->move[i].score != -VALUE_INFINITE);

    if (depth == ONE_PLY && !updated)
        continue;

    Depth d = updated ? depth : depth - ONE_PLY;
    Value v = updated ? rm->move[i].score : rm->move[i].previousScore;

    int tb = TB_RootInTB && abs(v) < VALUE_MATE - MAX_MATE_PLY;
    if (tb)
      v = rm->move[i].TBScore;

    // An incomplete mate PV may be caused by cutoffs in qsearch() and
    // by TB cutoffs. We try to complete the mate PV if we may be in the
    // latter case.
    if (   abs(v) > VALUE_MATE - MAX_MATE_PLY
        && rm->move[i].pvSize < VALUE_MATE - abs(v)
        && TB_MaxCardinalityDTM > 0)
      TB_expand_mate(pos, &rm->move[i]);

    printf("info depth %d seldepth %d multipv %d score %s",
           d / ONE_PLY, rm->move[i].selDepth + 1, (int)i + 1,
           uci_value(buf, v));

    if (!tb && i == PVIdx)
      printf("%s", v >= beta ? " lowerbound" : v <= alpha ? " upperbound" : "");

    printf(" nodes %"PRIu64" nps %"PRIu64, nodes_searched,
                              nodes_searched * 1000 / elapsed);

    if (elapsed > 1000)
      printf(" hashfull %d", tt_hashfull());

    printf(" tbhits %"PRIu64" time %"PRIi64" pv", tbhits, elapsed);

    for (int idx = 0; idx < rm->move[i].pvSize; idx++)
      printf(" %s", uci_move(buf, rm->move[i].pv[idx], is_chess960()));
    printf("\n");
  }
  fflush(stdout);
}


// extract_ponder_from_tt() is called in case we have no ponder move
// before exiting the search, for instance, in case we stop the search
// during a fail high at root. We try hard to have a ponder move to
// return to the GUI, otherwise in case of 'ponder on' we have nothing
// to think on.

static int extract_ponder_from_tt(RootMove *rm, Pos *pos)
{
  int ttHit;

  assert(rm->pvSize == 1);

  if (!rm->pv[0])
    return 0;

  do_move(pos, rm->pv[0], gives_check(pos, pos->st, rm->pv[0]));
  TTEntry *tte = tt_probe(pos_key(), &ttHit);

  if (ttHit) {
    Move m = tte_move(tte); // Local copy to be SMP safe
    ExtMove list[MAX_MOVES];
    ExtMove *last = generate_legal(pos, list);
    for (ExtMove *p = list; p < last; p++)
      if (p->move == m) {
        rm->pv[rm->pvSize++] = m;
        break;
      }
  }

  undo_move(pos, rm->pv[0]);
  return rm->pvSize > 1;
}

static void TB_rank_root_moves(Pos *pos, RootMoves *rm)
{
  TB_RootInTB = 0;
  TB_UseRule50 = option_value(OPT_SYZ_50_MOVE);
  TB_ProbeDepth = option_value(OPT_SYZ_PROBE_DEPTH) * ONE_PLY;
  TB_Cardinality = option_value(OPT_SYZ_PROBE_LIMIT);
  int dtz_available = 1, dtm_available = 0;

  if (TB_Cardinality > TB_MaxCardinality) {
    TB_Cardinality = TB_MaxCardinality;
    TB_ProbeDepth = DEPTH_ZERO;
  }

  TB_CardinalityDTM =  option_value(OPT_SYZ_USE_DTM)
                     ? min(TB_Cardinality, TB_MaxCardinalityDTM)
                     : 0;

  if (TB_Cardinality >= popcount(pieces()) && !can_castle_any()) {
    // Try to rank moves using DTZ tables.
    TB_RootInTB = TB_root_probe_dtz(pos, rm);

    if (!TB_RootInTB) {
      // DTZ tables are missing.
      dtz_available = 0;

      // Try to rank moves using WDL tables as fallback.
      TB_RootInTB = TB_root_probe_wdl(pos, rm);
    }

    // If ranking was successful, try to obtain mate values from DTM tables.
    if (TB_RootInTB && TB_CardinalityDTM >= popcount(pieces()))
      dtm_available = TB_root_probe_dtm(pos, rm);
  }

  if (TB_RootInTB) { // Ranking was successful.
    // Sort moves according to TB rank.
    stable_sort(rm->move, rm->size);

    // Only probe during search if DTM and DTZ are not available
    // and we are winning.
    if (dtm_available || dtz_available || rm->move[0].TBRank <= 0)
      TB_Cardinality = 0;
  }
  else // Ranking was not successful.
    for (int i = 0; i < rm->size; i++)
      rm->move[i].TBRank = 0;
}


// start_thinking() wakes up the main thread to start a new search,
// then returns immediately.

void start_thinking(Pos *root)
{
  if (Signals.searching)
    thread_wait_until_sleeping(threads_main());

  Signals.stopOnPonderhit = Signals.stop = 0;

  // Generate all legal moves.
  ExtMove list[MAX_MOVES];
  ExtMove *end = generate_legal(root, list);

  // Implement searchmoves option.
  if (Limits.numSearchmoves) {
    ExtMove *p = list;
    for (ExtMove *m = p; m < end; m++)
      for (int i = 0; i < Limits.numSearchmoves; i++)
        if (m->move == Limits.searchmoves[i]) {
          (p++)->move = m->move;
          break;
        }
    end = p;
  }

  RootMoves *moves = Threads.pos[0]->rootMoves;
  moves->size = end - list;
  for (int i = 0; i < moves->size; i++)
    moves->move[i].pv[0] = list[i].move;

  // Rank root moves if root position is a TB position.
  TB_rank_root_moves(root, moves);

  for (int idx = 0; idx < Threads.numThreads; idx++) {
    Pos *pos = Threads.pos[idx];
    pos->selDepth = 0;
    pos->nmpPly = pos->nmpOdd = 0;
    pos->rootDepth = DEPTH_ZERO;
    pos->nodes = pos->tbHits = 0;
    RootMoves *rm = pos->rootMoves;
    rm->size = end - list;
    for (int i = 0; i < rm->size; i++) {
      rm->move[i].pvSize = 1;
      rm->move[i].pv[0] = moves->move[i].pv[0];
      rm->move[i].score = -VALUE_INFINITE;
      rm->move[i].previousScore = -VALUE_INFINITE;
      rm->move[i].selDepth = 0;
      rm->move[i].TBRank = moves->move[i].TBRank;
      rm->move[i].TBScore = moves->move[i].TBScore;
    }
    memcpy(pos, root, offsetof(Pos, moveList));
    // Copy enough of the root State buffer.
    int n = max(5, root->st->pliesFromNull);
    for (int i = 0; i <= n; i++)
      memcpy(&pos->stack[i], &root->st[i - n], StateSize);
    pos->st = pos->stack + n;
    (pos->st-1)->endMoves = pos->moveList;
    pos_set_check_info(pos);
  }

  if (TB_RootInTB)
    Threads.pos[0]->tbHits = end - list;

  Signals.searching = 1;
  thread_wake_up(threads_main(), THREAD_SEARCH);
}
