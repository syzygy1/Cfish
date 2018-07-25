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

// Different node types, used as a template parameter
#define NonPV 0
#define PV 1

// Sizes and phases of the skip blocks, used for distributing search depths
// across the threads
static const int skipSize[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
static const int skipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};

static const int RazorMargin1 = 590;
static const int RazorMargin2 = 604;

INLINE int futility_margin(Depth d, int improving) {
  return (175 - 50 * improving) * d / ONE_PLY;
}

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
  return d > 17 ? 0 : 33 * d * d + 66 * d - 66;
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
//  Move best = 0;
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

  flockfile(stdout);
  printf("bestmove %s", uci_move(buf, bestThread->rootMoves->move[0].pv[0], is_chess960()));

  if (bestThread->rootMoves->move[0].pvSize > 1 || extract_ponder_from_tt(&bestThread->rootMoves->move[0], pos))
    printf(" ponder %s", uci_move(buf, bestThread->rootMoves->move[0].pv[1], is_chess960()));

  printf("\n");
  fflush(stdout);
  funlockfile(stdout);
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
  bool failedLow;

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
    failedLow = false;
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
      if (((pos->rootDepth / ONE_PLY + skipPhase[i]) / skipSize[i]) % 2)
        continue;
    }

    // Age out PV variability metric
    if (pos->threadIdx == 0) {
      mainThread.bestMoveChanges *= 0.517;
      failedLow = false;
    }

    // Save the last iteration's scores before first PV line is searched and
    // all the move scores except the (new) PV are set to -VALUE_INFINITE.
    for (int idx = 0; idx < rm->size; idx++)
      rm->move[idx].previousScore = rm->move[idx].score;

    pos->contempt = pos_stm() == WHITE ?  make_score(base_ct, base_ct / 2)
                                       : -make_score(base_ct, base_ct / 2);

    int pvFirst = 0, pvLast = 0;

    // MultiPV loop. We perform a full root search for each PV line
    for (int pvIdx = 0; pvIdx < multiPV && !Signals.stop; pvIdx++) {
      pos->pvIdx = pvIdx;
      if (pvIdx == pvLast) {
        pvFirst = pvLast;
        for (pvLast++; pvLast < rm->size; pvLast++)
          if (rm->move[pvLast].tbRank != rm->move[pvFirst].tbRank)
            break;
        pos->pvLast = pvLast;
      }

      pos->selDepth = 0;

      // Skip the search if we have a mate value from DTM tables.
      if (abs(rm->move[pvIdx].tbRank) > 1000) {
        bestValue = rm->move[pvIdx].score = rm->move[pvIdx].tbScore;
        alpha = -VALUE_INFINITE;
        beta = VALUE_INFINITE;
        goto skip_search;
      }

      // Reset aspiration window starting size
      if (pos->rootDepth >= 5 * ONE_PLY) {
        Value previousScore = rm->move[pvIdx].previousScore;
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
        stable_sort(&rm->move[pvIdx], pvLast - pvIdx);

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
            failedLow = true;
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
      stable_sort(&rm->move[pvFirst], pvIdx - pvFirst + 1);

skip_search:
      if (    pos->threadIdx == 0
          && (Signals.stop || pvIdx + 1 == multiPV || time_elapsed() > 3000))
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
        const int F[] = { failedLow,
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

// search_PV() is the main search function for PV nodes.
#define NT PV
#include "ntsearch.c"
#undef NT

// search_NonPV is the main search function for non-PV nodes.
#define NT NonPV
#include "ntsearch.c"
#undef NT

// qsearch() is the quiescence search function, which is called by the main
// search function when the remaining depth is zero (or, to be more precise,
// less than ONE_PLY).

#define true 1
#define false 0

#define NT PV
#define InCheck false
#include "qsearch.c"
#undef InCheck
#define InCheck true
#include "qsearch.c"
#undef InCheck
#undef NT
#define NT NonPV
#define InCheck false
#include "qsearch.c"
#undef InCheck
#define InCheck true
#include "qsearch.c"
#undef InCheck
#undef NT

#undef true
#undef false

#define rm_lt(m1,m2) ((m1).tbRank != (m2).tbRank ? (m1).tbRank < (m2).tbRank : (m1).score != (m2).score ? (m1).score < (m2).score : (m1).previousScore < (m2).previousScore)

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

  if (is_capture_or_promotion(pos, move))
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
  int pvIdx = pos->pvIdx;
  int multiPV = min(option_value(OPT_MULTI_PV), rm->size);
  uint64_t nodes_searched = threads_nodes_searched();
  uint64_t tbhits = threads_tb_hits();
  char buf[16];

  flockfile(stdout);
  for (int i = 0; i < multiPV; i++) {
    int updated = (i <= pvIdx && rm->move[i].score != -VALUE_INFINITE);

    if (depth == ONE_PLY && !updated)
        continue;

    Depth d = updated ? depth : depth - ONE_PLY;
    Value v = updated ? rm->move[i].score : rm->move[i].previousScore;

    int tb = TB_RootInTB && abs(v) < VALUE_MATE - MAX_MATE_PLY;
    if (tb)
      v = rm->move[i].tbScore;

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

    if (!tb && i == pvIdx)
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
  funlockfile(stdout);
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
    if (dtm_available || dtz_available || rm->move[0].tbRank <= 0)
      TB_Cardinality = 0;
  }
  else // Ranking was not successful.
    for (int i = 0; i < rm->size; i++)
      rm->move[i].tbRank = 0;
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
      rm->move[i].tbRank = moves->move[i].tbRank;
      rm->move[i].tbScore = moves->move[i].tbScore;
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
