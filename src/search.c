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
#include <string.h>   // For memset

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

static int base_ct;

// Different node types, used as template parameter
enum { NonPV, PV };

static const uint64_t ttHitAverageWindow     = 4096;
static const uint64_t ttHitAverageResolution = 1024;

static const int RazorMargin = 531;

INLINE int futility_margin(Depth d, bool improving) {
  return 217 * (d - improving);
}

// Reductions lookup tables, initialized at startup
static int Reductions[MAX_MOVES]; // [depth or moveNumber]

INLINE Depth reduction(int i, Depth d, int mn)
{
  int r = Reductions[d] * Reductions[mn];
  return ((r + 511) / 1024 + (!i && r > 1007));
}

INLINE int futility_move_count(bool improving, Depth depth)
{
//  return (4 + depth * depth) / (2 - improving);
  return improving ? 4 + depth * depth : (4 + depth * depth) / 2;
}

// History and stats update bonus, based on depth
static Value stat_bonus(Depth depth)
{
  int d = depth;
  return d > 15 ? -8 : 19 * d * d + 155 * d - 132;
}

// Add a small random component to draw evaluations to keep search dynamic
// and to avoid three-fold blindness. (Yucks, ugly hack)
static Value value_draw(Pos *pos)
{
  return VALUE_DRAW + 2 * (pos->nodes & 1) - 1;
}

// Skill structure is used to implement strength limit
struct Skill {
/*
  Skill(int l) : level(l) {}
  int enabled() const { return level < 20; }
  int time_to_pick(Depth depth) const { return depth == 1 + level; }
  Move best_move(size_t multiPV) { return best ? best : pick_best(multiPV); }
  Move pick_best(size_t multiPV);
*/

  int level;
  Move best;
//  Move best = 0;
};

// Breadcrumbs are used to mark nodes as being search by a given thread
static _Atomic uint64_t breadcrumbs[1024];

static Value search_PV(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value search_NonPV(Pos *pos, Stack *ss, Value alpha, Depth depth, bool cutNode);

static Value qsearch_PV_true(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value qsearch_PV_false(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth);
static Value qsearch_NonPV_true(Pos *pos, Stack *ss, Value alpha, Depth depth);
static Value qsearch_NonPV_false(Pos *pos, Stack *ss, Value alpha, Depth depth);

static Value value_to_tt(Value v, int ply);
static Value value_from_tt(Value v, int ply, int r50c);
static void update_pv(Move *pv, Move move, Move *childPv);
static void update_cm_stats(Stack *ss, Piece pc, Square s, int bonus);
static void update_quiet_stats(const Pos *pos, Stack *ss, Move move, int bonus,
    Depth depth);
static void update_capture_stats(const Pos *pos, Move move, Move *captures, int captureCnt, int bonus);
static void check_time(void);
static void stable_sort(RootMove *rm, int num);
static void uci_print_pv(Pos *pos, Depth depth, Value alpha, Value beta);
static int extract_ponder_from_tt(RootMove *rm, Pos *pos);

// search_init() is called during startup to initialize various lookup tables

void search_init(void)
{
  for (int i = 1; i < MAX_MOVES; i++)
    Reductions[i] = (24.8 + log(Threads.numThreads)) * log(i);
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
      for (int chk = 0; chk < 2; chk++)
        for (int c = 0; c < 2; c++)
          for (int j = 0; j < 16; j++)
            for (int k = 0; k < 64; k++)
              (*cmhTables[i])[chk][c][0][0][j][k] = CounterMovePruneThreshold - 1;
    }

  for (int idx = 0; idx < Threads.numThreads; idx++) {
    Pos *pos = Threads.pos[idx];
    stats_clear(pos->counterMoves);
    stats_clear(pos->history);
    stats_clear(pos->captureHistory);
    stats_clear(pos->lowPlyHistory);
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
      nodes = perft_helper(pos, depth - 1, nodes);
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
    if (depth <= 1) {
      cnt = 1;
      nodes++;
    } else {
      do_move(pos, m->move, gives_check(pos, pos->st, m->move));
      if (depth == 2)
        cnt = generate_legal(pos, last) - last;
      else
        cnt = perft_helper(pos, depth - 3, 0);
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
  int us = stm();
  time_init(us, game_ply());
  tt_new_search();
  char buf[16];
  bool playBookMove = false;

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
        playBookMove = true;
        break;
      }

    if (!playBookMove) {
      Threads.pos[0]->bestMoveChanges = 0;
      for (int idx = 1; idx < Threads.numThreads; idx++) {
        Threads.pos[idx]->bestMoveChanges = 0;
        thread_wake_up(Threads.pos[idx], THREAD_SEARCH);
      }

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
           uci_value(buf, checkers() ? -VALUE_MATE : VALUE_DRAW));
    fflush(stdout);
  }

  // When playing in 'nodes as time' mode, subtract the searched nodes from
  // the available ones before exiting.
  if (Limits.npmsec)
    Time.availableNodes += Limits.inc[us] - threads_nodes_searched();

  // Check if there are threads with a better score than main thread
  Pos *bestThread = pos;
  if (    option_value(OPT_MULTI_PV) == 1
      && !playBookMove
      && !Limits.depth
//      && !Skill(option_value(OPT_SKILL_LEVEL)).enabled()
      &&  pos->rootMoves->move[0].pv[0] != 0)
  {
    int i, num = 0, maxNum = min(pos->rootMoves->size, Threads.numThreads);
    Move mvs[maxNum];
    int64_t votes[maxNum];
    Value minScore = pos->rootMoves->move[0].score;
    for (int idx = 1; idx < Threads.numThreads; idx++)
      minScore = min(minScore, Threads.pos[idx]->rootMoves->move[0].score);
    for (int idx = 0; idx < Threads.numThreads; idx++) {
      Pos *p = Threads.pos[idx];
      Move m = p->rootMoves->move[0].pv[0];
      for (i = 0; i < num; i++)
        if (mvs[i] == m) break;
      if (i == num) {
        num++;
        mvs[i] = m;
        votes[i] = 0;
      }
      votes[i] += (p->rootMoves->move[0].score - minScore + 14) * p->completedDepth;
    }
    int64_t bestVote = votes[0];
    for (int idx = 1; idx < Threads.numThreads; idx++) {
      Pos *p = Threads.pos[idx];
      for (i = 0; mvs[i] != p->rootMoves->move[0].pv[0]; i++);
      if (abs(bestThread->rootMoves->move[0].score) >= VALUE_TB_WIN_IN_MAX_PLY) {
        // Make sure we pick the shortest mate
        if (p->rootMoves->move[0].score > bestThread->rootMoves->move[0].score)
          bestThread = p;
      } else if (p->rootMoves->move[0].score >= VALUE_TB_WIN_IN_MAX_PLY
          || (   p->rootMoves->move[0].score > VALUE_TB_LOSS_IN_MAX_PLY
              && votes[i] > bestVote))
      {
        bestVote = votes[i];
        bestThread = p;
      }
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
  Move pv[MAX_PLY + 1];
  Move lastBestMove = 0;
  Depth lastBestMoveDepth = 0;
  double timeReduction = 1.0, totBestMoveChanges = 0;
  int iterIdx = 0;

  Stack *ss = pos->st; // At least the seventh element of the allocated array.
  for (int i = -7; i < 3; i++)
    memset(SStackBegin(ss[i]), 0, SStackSize);
  (ss-1)->endMoves = pos->moveList;

  for (int i = -7; i < 0; i++)
    ss[i].history = &(*pos->counterMoveHistory)[0][0][0][0]; // Use as sentinel

  for (int i = 0; i <= MAX_PLY; i++)
    ss[i].ply = i;
  ss->pv = pv;

  bestValue = delta = alpha = -VALUE_INFINITE;
  beta = VALUE_INFINITE;
  pos->completedDepth = 0;

  if (pos->threadIdx == 0) {
    if (mainThread.previousScore == VALUE_INFINITE)
      for (int i = 0; i < 4; i++)
        mainThread.iterValue[i] = VALUE_ZERO;
    else
      for (int i = 0; i < 4; i++)
        mainThread.iterValue[i] = mainThread.previousScore;
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
  pos->ttHitAverage = ttHitAverageWindow * ttHitAverageResolution / 2;
  int searchAgainCounter = 0;

  // Iterative deepening loop until requested to stop or the target depth
  // is reached.
  while (   ++pos->rootDepth < MAX_PLY
         && !Signals.stop
         && !(   Limits.depth
              && pos->threadIdx == 0
              && pos->rootDepth > Limits.depth))
  {
    // Age out PV variability metric
    if (pos->threadIdx == 0)
      totBestMoveChanges /= 2;

    // Save the last iteration's scores before first PV line is searched and
    // all the move scores except the (new) PV are set to -VALUE_INFINITE.
    for (int idx = 0; idx < rm->size; idx++)
      rm->move[idx].previousScore = rm->move[idx].score;

    pos->contempt = stm() == WHITE ?  make_score(base_ct, base_ct / 2)
                                   : -make_score(base_ct, base_ct / 2);

    int pvFirst = 0, pvLast = 0;

    if (!Threads.increaseDepth)
      searchAgainCounter++;

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
      if (pos->rootDepth >= 4) {
        Value previousScore = rm->move[pvIdx].previousScore;
        delta = 21;
        alpha = max(previousScore - delta, -VALUE_INFINITE);
        beta  = min(previousScore + delta,  VALUE_INFINITE);

        // Adjust contempt based on root move's previousScore
        int ct = base_ct + (102 - base_ct / 2) * previousScore / (abs(previousScore) + 157);
        pos->contempt = stm() == WHITE ?  make_score(ct, ct / 2)
                                       : -make_score(ct, ct / 2);
      }

      // Start with a small aspiration window and, in the case of a fail
      // high/low, re-search with a bigger window until we're not failing
      // high/low anymore.
      int failedHighCnt = 0;
      while (true) {
        Depth adjustedDepth = max(1, pos->rootDepth - failedHighCnt - searchAgainCounter);
        bestValue = search_PV(pos, ss, alpha, beta, adjustedDepth);

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

          failedHighCnt = 0;
          if (pos->threadIdx == 0)
            Signals.stopOnPonderhit = 0;
        } else if (bestValue >= beta) {
          beta = min(bestValue + delta, VALUE_INFINITE);
          failedHighCnt++;
        } else {
          rm->move[pvIdx].bestMoveCount++;
          break;
        }

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
    if (    use_time_management()
        && !Signals.stop
        && !Signals.stopOnPonderhit) {
      // Stop the search if only one legal move is available, or if all
      // of the available time has been used.
      double fallingEval = (332 + 6 * (mainThread.previousScore - bestValue)
                                + 6 * (mainThread.iterValue[iterIdx] - bestValue)) / 704.0;
      fallingEval = clamp(fallingEval, 0.5, 1.5);

      // If the best move is stable over several iterations, reduce time
      // accordingly
      timeReduction =  lastBestMoveDepth + 9 < pos->completedDepth ? 1.94 : 0.91;
      double reduction = (1.41 + mainThread.previousTimeReduction) / (2.27 * timeReduction);

      // Use part of the gained time from a previous stable move for this move
      for (int i = 0; i < Threads.numThreads; i++) {
        totBestMoveChanges += Threads.pos[i]->bestMoveChanges;
        Threads.pos[i]->bestMoveChanges = 0;
      }

      double bestMoveInstability = 1 + totBestMoveChanges / Threads.numThreads;

      if (   rm->size == 1
          || time_elapsed() > time_optimum() * fallingEval * reduction * bestMoveInstability)
      {
        // If we are allowed to ponder do not stop the search now but
        // keep pondering until the GUI sends "ponderhit" or "stop".
        if (Limits.ponder)
          Signals.stopOnPonderhit = 1;
        else
          Signals.stop = 1;
      } else if (Threads.increaseDepth
          && !Limits.ponder
          && time_elapsed() > time_optimum() * fallingEval * reduction * bestMoveInstability * 0.6)
        Threads.increaseDepth = false;
      else
        Threads.increaseDepth = true;
    }

    mainThread.iterValue[iterIdx] = bestValue;
    iterIdx = (iterIdx + 1) & 3;
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

static int best_move_count(Pos *pos, Move move)
{
  int idx;
  for (idx = pos->pvIdx; idx < pos->pvLast; idx++)
    if (pos->rootMoves->move[idx].pv[0] == move)
      break;
  return idx < pos->pvLast ? pos->rootMoves->move[idx].bestMoveCount : 0;
}

// search_node() is the main search function template for both PV
// and non-PV nodes
INLINE Value search_node(Pos *pos, Stack *ss, Value alpha, Value beta,
    Depth depth, bool cutNode, const int NT)
{
  const bool PvNode = NT == PV;
  const bool rootNode = PvNode && ss->ply == 0;

  // Check if we have an upcoming move which draws by repetition, or if the
  // opponent had an alternative move earlier to this position.
  if (   pos->st->pliesFromNull >= 3
      && alpha < VALUE_DRAW
      && !rootNode
      && has_game_cycle(pos, ss->ply))
  {
    alpha = value_draw(pos);
    if (alpha >= beta)
      return alpha;
  }

  // Dive into quiescense search when the depth reaches zero
  if (depth <= 0)
    return  PvNode
          ?   checkers()
            ? qsearch_PV_true(pos, ss, alpha, beta, 0)
            : qsearch_PV_false(pos, ss, alpha, beta, 0)
          :   checkers()
            ? qsearch_NonPV_true(pos, ss, alpha, 0)
            : qsearch_NonPV_false(pos, ss, alpha, 0);

  assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(0 < depth && depth < MAX_PLY);
  assert(!(PvNode && cutNode));

  Move pv[MAX_PLY+1], capturesSearched[32], quietsSearched[64];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, excludedMove, bestMove;
  Depth extension, newDepth;
  Value bestValue, value, ttValue, eval, maxValue;
  int ttHit, ttPv, givesCheck, didLMR;
  bool improving, inCheck, doFullDepthSearch, moveCountPruning;
  bool ttCapture, captureOrPromotion, singularLMR;
  Piece movedPiece;
  int moveCount, captureCount, quietCount;

  // Step 1. Initialize node
  inCheck = checkers();
  moveCount = captureCount = quietCount =  ss->moveCount = 0;
  bestValue = -VALUE_INFINITE;
  maxValue = VALUE_INFINITE;

  // Check for the available remaining time
  if (load_rlx(pos->resetCalls)) {
    store_rlx(pos->resetCalls, false);
    pos->callsCnt = Limits.nodes ? min(1024, Limits.nodes / 1024) : 1024;
  }
  if (--pos->callsCnt <= 0) {
    for (int idx = 0; idx < Threads.numThreads; idx++)
      store_rlx(Threads.pos[idx]->resetCalls, true);

    check_time();
  }

  // Used to send selDepth info to GUI
  if (PvNode && pos->selDepth < ss->ply)
    pos->selDepth = ss->ply;

  if (!rootNode) {
    // Step 2. Check for aborted search and immediate draw
    if (load_rlx(Signals.stop) || is_draw(pos) || ss->ply >= MAX_PLY)
      return  ss->ply >= MAX_PLY && !inCheck ? evaluate(pos)
                                             : value_draw(pos);

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
    } else {
      if (alpha < mated_in(ss->ply))
        return mated_in(ss->ply);
      if (alpha >= mate_in(ss->ply+1))
        return alpha;
    }
  }

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  (ss+1)->excludedMove = bestMove = 0;
  (ss+2)->killers[0] = (ss+2)->killers[1] = 0;
  Square prevSq = to_sq((ss-1)->currentMove);
  if (rootNode)
    (ss+4)->statScore = 0;
  else
    (ss+2)->statScore = 0;

  // Step 4. Transposition table lookup. We don't want the score of a
  // partial search to overwrite a previous full search TT value, so we
  // use a different position key in case of an excluded move.
  excludedMove = ss->excludedMove;
#ifdef BIG_TT
  posKey = key() ^ (Key)excludedMove;
#else
  posKey = key() ^ (Key)((int32_t)excludedMove << 16);
#endif
  tte = tt_probe(posKey, &ttHit);
  ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply, rule50_count()) : VALUE_NONE;
  ttMove =  rootNode ? pos->rootMoves->move[pos->pvIdx].pv[0]
          : ttHit    ? tte_move(tte) : 0;
  ttPv = PvNode ? 4 : (ttHit ? tte_is_pv(tte) : 0);

  if (   ttPv
      && depth > 12
      && ss->ply - 1 < MAX_LPH
      && !captured_piece()
      && move_is_ok((ss-1)->currentMove))
    lph_update(*pos->lowPlyHistory, ss->ply - 1, (ss-1)->currentMove, stat_bonus(depth - 5));

  // pos->ttHitAverage can be used to approximate the running average of ttHit
  pos->ttHitAverage = (ttHitAverageWindow - 1) * pos->ttHitAverage / ttHitAverageWindow + ttHitAverageResolution * !!ttHit;

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
          update_quiet_stats(pos, ss, ttMove, stat_bonus(depth), depth);

        // Extra penalty for a quiet TT or main killer move in previous ply
        // when it gets refuted
        if ((ss-1)->moveCount <= 2 && !captured_piece())
          update_cm_stats(ss-1, piece_on(prevSq), prevSq,
              -stat_bonus(depth + 1));
      }
      // Penalty for a quiet ttMove that fails low
      else if (!is_capture_or_promotion(pos, ttMove)) {
        Value penalty = -stat_bonus(depth);
        history_update(*pos->history, stm(), ttMove, penalty);
        update_cm_stats(ss, moved_piece(ttMove), to_sq(ttMove), penalty);
      }
    }
    if (rule50_count() < 90)
      return ttValue;
  }

  // Step 5. Tablebase probe
  if (!rootNode && TB_Cardinality) {
    int piecesCnt = popcount(pieces());

    if (    piecesCnt <= TB_Cardinality
        && (piecesCnt <  TB_Cardinality || depth >= TB_ProbeDepth)
        &&  rule50_count() == 0
        && !can_castle_any())
    {
      int found, wdl = TB_probe_wdl(pos, &found);

      if (found) {
        pos->tbHits++;

        int drawScore = TB_UseRule50 ? 1 : 0;

        value =  wdl < -drawScore ? VALUE_MATED_IN_MAX_PLY + ss->ply + 1
               : wdl >  drawScore ? VALUE_MATE_IN_MAX_PLY  - ss->ply - 1
                                  : VALUE_DRAW + 2 * wdl * drawScore;

        int b =  wdl < -drawScore ? BOUND_UPPER
               : wdl >  drawScore ? BOUND_LOWER : BOUND_EXACT;

        if (    b == BOUND_EXACT
            || (b == BOUND_LOWER ? value >= beta : value <= alpha))
        {
          tte_save(tte, posKey, value_to_tt(value, ss->ply), ttPv, b,
                   min(MAX_PLY - 1, depth + 6), 0,
                   VALUE_NONE, tt_generation());
          return value;
        }

        if (piecesCnt <= TB_CardinalityDTM) {
          Value mate = TB_probe_dtm(pos, wdl, &found);
          if (found) {
            mate += wdl > 0 ? -ss->ply : ss->ply;
            tte_save(tte, posKey, value_to_tt(mate, ss->ply), ttPv, BOUND_EXACT,
                     min(MAX_PLY - 1, depth + 6), 0,
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
    ss->staticEval = eval = VALUE_NONE;
    improving = false;
    goto moves_loop; // Skip early pruning when in check
  } else if (ttHit) {
    // Never assume anything about values stored in TT
    if ((eval = tte_eval(tte)) == VALUE_NONE)
      eval = evaluate(pos);
    ss->staticEval = eval;

    if (eval == VALUE_DRAW)
      eval = value_draw(pos);

    // Can ttValue be used as a better position evaluation?
    if (ttValue != VALUE_NONE)
      if (tte_bound(tte) & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER))
        eval = ttValue;
  } else {
    if ((ss-1)->currentMove != MOVE_NULL) {
      int bonus = -(ss-1)->statScore / 512;
      ss->staticEval = eval = evaluate(pos) + bonus;
    } else
      ss->staticEval = eval = -(ss-1)->staticEval + 2 * Tempo;

    tte_save(tte, posKey, VALUE_NONE, ttPv, BOUND_NONE, DEPTH_NONE, 0,
             eval, tt_generation());
  }

  // Step 7. Razoring
  if (   !rootNode
      && depth < 2
      && eval <= alpha - RazorMargin)
    return PvNode ? qsearch_PV_false(pos, ss, alpha, beta, 0)
                  : qsearch_NonPV_false(pos, ss, alpha, 0);

  improving =  (ss-2)->staticEval == VALUE_NONE
             ? (ss->staticEval > (ss-4)->staticEval || (ss-4)->staticEval == VALUE_NONE)
             :  ss->staticEval > (ss-2)->staticEval;

  // Step 8. Futility pruning: child node
  if (   !PvNode
      &&  depth < 6
      &&  eval - futility_margin(depth, improving) >= beta
      &&  eval < VALUE_KNOWN_WIN)  // Do not return unproven wins
    return eval; // - futility_margin(depth); (do not do the right thing)

  // Step 9. Null move search with verification search (is omitted in PV nodes)
  if (   !PvNode
      && (ss-1)->currentMove != MOVE_NULL
      && (ss-1)->statScore < 23397
      && eval >= beta
      && eval >= ss->staticEval
      && ss->staticEval >= beta - 32 * depth - 30 * improving + 120 * !!ttPv + 292
      && !excludedMove
      && non_pawn_material_c(stm())
      && (ss->ply >= pos->nmpPly || ss->ply % 2 != pos->nmpOdd))
  {
    assert(eval - beta >= 0);

    // Null move dynamic reduction based on depth and value
    Depth R = ((854 + 68 * depth) / 258 + min((eval - beta) / 192, 3));

    ss->currentMove = MOVE_NULL;
    ss->history = &(*pos->counterMoveHistory)[0][0][0][0];

    do_null_move(pos);
    ss->endMoves = (ss-1)->endMoves;
    Value nullValue = -search_NonPV(pos, ss+1, -beta, depth-R, !cutNode);
    undo_null_move(pos);

    if (nullValue >= beta) {
      // Do not return unproven mate scores
      if (nullValue >= VALUE_TB_WIN_IN_MAX_PLY)
        nullValue = beta;

      if (   (depth < 13 || pos->nmpPly)
          && abs(beta) < VALUE_KNOWN_WIN)
        return nullValue;

      // Do verification search at high depths
      // Disable null move pruning for side to move for the first part of
      // the remaining search tree
      pos->nmpPly = ss->ply + 3 * (depth-R) / 4;
      pos->nmpOdd = ss->ply & 1;

      Value v = search_NonPV(pos, ss, beta-1, depth-R, 0);

      pos->nmpOdd = pos->nmpPly = 0;

      if (v >= beta)
        return nullValue;
    }
  }

  // Step 10. ProbCut
  // If we have a good enough capture and a reduced search returns a value
  // much above beta, we can (almost) safely prune the previous move.
  if (   !PvNode
      &&  depth >= 5
      &&  abs(beta) < VALUE_TB_WIN_IN_MAX_PLY)
  {
    Value rbeta = beta + 189 - 45 * improving;
    Depth rdepth = depth - 4;

    assert(rdepth >= 1);

    mp_init_pc(pos, ttMove, rbeta - ss->staticEval);

    int probCutCount = 2 + 2 * cutNode;
    while (  (move = next_move(pos, 0))
           && probCutCount
           && !(   move == ttMove
                && (tte_bound(tte) & BOUND_LOWER)
                && tte_depth(tte) >= depth - 4
                && ttValue < rbeta))
      if (move != excludedMove && is_legal(pos, move)) {
        assert(is_capture_or_promotion(pos, move));
        assert(depth >= 5);

        captureOrPromotion = true;
        probCutCount--;

        ss->currentMove = move;
        ss->history = &(*pos->counterMoveHistory)[inCheck][captureOrPromotion][moved_piece(move)][to_sq(move)];
        givesCheck = gives_check(pos, ss, move);
        do_move(pos, move, givesCheck);

        // Perform a preliminary qsearch to verify that the move holds
        value =   givesCheck
               ? -qsearch_NonPV_true(pos, ss+1, -rbeta, 0)
               : -qsearch_NonPV_false(pos, ss+1, -rbeta, 0);

        // If the qsearch holds perform the regular search
        if (value >= rbeta)
          value = -search_NonPV(pos, ss+1, -rbeta, rdepth, !cutNode);

        undo_move(pos, move);
        if (value >= rbeta)
          return value;
      }
  }

  // Step 11. Internal iterative deepening
  if (depth >= 7 && !ttMove) {
    if (PvNode)
      search_PV(pos, ss, alpha, beta, depth - 7);
    else
      search_NonPV(pos, ss, alpha, depth - 7, cutNode);

    tte = tt_probe(posKey, &ttHit);
    // ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply) : VALUE_NONE;
    ttMove = ttHit ? tte_move(tte) : 0;
  }

moves_loop: // When in check search starts from here.
  ;  // Avoid a compiler warning. A label must be followed by a statement.
  PieceToHistory *cmh  = (ss-1)->history;
  PieceToHistory *fmh  = (ss-2)->history;
  PieceToHistory *fmh2 = (ss-4)->history;

  mp_init(pos, ttMove, depth, depth > 12 ? ss->ply : MAX_PLY);

  value = bestValue;
  singularLMR = moveCountPruning = false;
  ttCapture = ttMove && is_capture_or_promotion(pos, ttMove);
  bool formerPv = ttPv && !PvNode;

  // Check for a breadcrumb and leave one if none found
  _Atomic uint64_t *crumb = NULL;
  bool marked = false;
  if (ss->ply < 8) {
    crumb = &breadcrumbs[posKey & 1023];
    // The next line assumes there are at most 65535 search threads
    uint64_t v = (posKey & ~0xffffULL) | (pos->threadIdx + 1), expected = 0ULL;
    // If no crumb is in place yet, leave ours
    if (!atomic_compare_exchange_strong_explicit(crumb, &expected, v,
          memory_order_relaxed, memory_order_relaxed))
    {
      // Some crumb was in place already. Its value is now in expected.
      crumb = NULL;
      // Was the crumb is for the same position and was left by another thread?
      v ^= expected;
      if (v != 0 && (v & ~0xffffULL) == 0)
        marked = true;
    }
  }

  // Step 12. Loop through moves
  // Loop through all pseudo-legal moves until no moves remain or a beta
  // cutoff occurs
  while ((move = next_move(pos, moveCountPruning))) {
    assert(move_is_ok(move));

    if (move == excludedMove)
      continue;

    // At root obey the "searchmoves" option and skip moves not listed
    // inRoot Move List. As a consequence any illegal move is also skipped.
    // In MultiPV mode we also skip PV moves which have been already
    // searched.
    if (rootNode) {
      int idx;
      for (idx = pos->pvIdx; idx < pos->pvLast; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move)
          break;
      if (idx == pos->pvLast)
        continue;
    }

    ss->moveCount = ++moveCount;

    if (rootNode && pos->threadIdx == 0 && time_elapsed() > 3000) {
      char buf[16];
      printf("info depth %d currmove %s currmovenumber %d\n",
             depth,
             uci_move(buf, move, is_chess960()),
             moveCount + pos->pvIdx);
      fflush(stdout);
    }

    if (PvNode)
      (ss+1)->pv = NULL;

    extension = 0;
    captureOrPromotion = is_capture_or_promotion(pos, move);
    movedPiece = moved_piece(move);

    givesCheck = gives_check(pos, ss, move);

    // Calculate new depth for this move
    newDepth = depth - 1;

    // Step 13. Pruning at shallow depth
    if (  !rootNode
        && non_pawn_material_c(stm())
        && bestValue > VALUE_TB_LOSS_IN_MAX_PLY)
    {
      // Skip quiet moves if movecount exceeds our FutilityMoveCount threshold
      moveCountPruning = moveCount >= futility_move_count(improving, depth);

      // Reduced depth of the next LMR search
      int lmrDepth = max(newDepth - reduction(improving, depth, moveCount), 0);

      if (   !captureOrPromotion
          && !givesCheck)
      {
        // Countermoves based pruning
        if (   lmrDepth < 4 + ((ss-1)->statScore > 0 || (ss-1)->moveCount == 1)
            && (*cmh )[movedPiece][to_sq(move)] < CounterMovePruneThreshold
            && (*fmh )[movedPiece][to_sq(move)] < CounterMovePruneThreshold)
          continue;

        // Futility pruning: parent node
        if (   lmrDepth < 6
            && !inCheck
            && ss->staticEval + 235 + 172 * lmrDepth <= alpha
            &&  (*cmh )[movedPiece][to_sq(move)]
              + (*fmh )[movedPiece][to_sq(move)]
              + (*fmh2)[movedPiece][to_sq(move)] < 27400)
          continue;

        // Prune moves with negative SEE at low depths and below a decreasing
        // threshold at higher depths.
        if (!see_test(pos, move, -(32 - min(lmrDepth, 18)) * lmrDepth * lmrDepth))
          continue;

      } else {
        // Capture history based pruning when the move doesn't give check
        if (   !givesCheck
            && lmrDepth < 1
            && (*pos->captureHistory)[movedPiece][to_sq(move)][type_of_p(piece_on(to_sq(move)))] < 0)
          continue;

        // See based pruning
        if (!see_test(pos, move, -194 * depth))
          continue;
      }
    }

    // Step 14. Extensions

    // Singular extension search. If all moves but one fail low on a search
    // of (alpha-s, beta-s), and just one fails high on (alpha, beta), then
    // that move is singular and should be extended. To verify this we do a
    // reduced search on all the other moves but the ttMove and if the
    // result is lower than ttValue minus a margin then we extend the ttMove.
    if (    depth >= 6
        &&  move == ttMove
        && !rootNode
        && !excludedMove // No recursive singular search
     /* &&  ttValue != VALUE_NONE implicit in the next condition */
        &&  abs(ttValue) < VALUE_KNOWN_WIN
        && (tte_bound(tte) & BOUND_LOWER)
        &&  tte_depth(tte) >= depth - 3
        &&  is_legal(pos, move))
    {
      Value singularBeta = ttValue - ((formerPv + 4) * depth) / 2;
      Depth singularDepth = (depth - 1 + 3 * formerPv) / 2;
      ss->excludedMove = move;
      Move cm = ss->countermove;
      Move k1 = ss->mpKillers[0], k2 = ss->mpKillers[1];
      value = search_NonPV(pos, ss, singularBeta - 1, singularDepth, cutNode);
      ss->excludedMove = 0;

      if (value < singularBeta) {
        extension = 1;
        singularLMR = true;
      }

      // Multi-cut pruning. Our ttMove is assumed to fail high, and now we
      // failed high also on a reduced search without the ttMove. So we
      // assume that this expected cut-node is not singular, i.e. multiple
      // moves fail high. We therefore prune the whole subtree by returning
      // a soft bound.
      else if (singularBeta >= beta) {
        if (crumb) store_rlx(*crumb, 0);
        return singularBeta;
      }

      // The call to search_NonPV with the same value of ss messed up our
      // move picker data. So we fix it.
      mp_init(pos, ttMove, depth, depth > 12 ? ss->ply : MAX_PLY);
      ss->stage++;
      ss->countermove = cm; // pedantic
      ss->mpKillers[0] = k1; ss->mpKillers[1] = k2;
    }
    else if (    givesCheck
             && (is_discovery_check_on_king(pos, stm() ^ 1, move) || see_test(pos, move, 0)))
      extension = 1;

    // Passed pawn extension
    else if (   move == ss->killers[0]
             && advanced_pawn_push(pos, move)
             && pawn_passed(pos, stm(), to_sq(move)))
      extension = 1;

    // Last captures extension
    else if (   PieceValue[EG][captured_piece()] > PawnValueEg
             && non_pawn_material() <= 2 * RookValueMg)
      extension = 1;

    // Castling extension
    if (type_of_m(move) == CASTLING)
      extension = 1;

    // Late irreversible move extension
    if (   move == ttMove
        && rule50_count() > 80
        && (captureOrPromotion || type_of_p(movedPiece) == PAWN))
      extension = 2;

    // Add extension to new depth
    newDepth += extension;

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Check for legality just before making the move
    if (!rootNode && !is_legal(pos, move)) {
      ss->moveCount = --moveCount;
      continue;
    }

    // Update the current move (this must be done after singular extension
    // search)
    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[inCheck][captureOrPromotion][movedPiece][to_sq(move)];

    // Step 15. Make the move.
    do_move(pos, move, givesCheck);
    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;

    // Step 16. Reduced depth search (LMR). If the move fails high it will be
    // re-searched at full depth.
    if (    depth >= 3
        &&  moveCount > 1 + 2 * rootNode
        && (!rootNode || best_move_count(pos, move) == 0)
        && (   !captureOrPromotion
            || moveCountPruning
            || ss->staticEval + PieceValue[EG][captured_piece()] <= alpha
            || cutNode
            || pos->ttHitAverage < 375 * ttHitAverageResolution * ttHitAverageWindow / 1024))
    {
      Depth r = reduction(improving, depth, moveCount);

      // Decrease reduction if the ttHit runing average is large
      if (pos->ttHitAverage > 500 * ttHitAverageResolution * ttHitAverageWindow / 1024)
        r--;

      // Reduction if other threads are searching this position.
      if (marked)
        r++;

      // Decrease reduction if position is or has been on the PV
      if (ttPv)
        r -= 2;

      if (moveCountPruning && !formerPv)
        r++;

      // Decrease reduction if opponent's move count is high
      if ((ss-1)->moveCount > 14)
        r--;

      // Decrease reduction if ttMove has been singularly extended
      if (singularLMR)
        r -= 1 + formerPv;

      if (!captureOrPromotion) {
        // Increase reduction if ttMove is a capture
        if (ttCapture)
          r++;

        // Increase reduction for cut nodes
        if (cutNode)
          r += 2;

        // Decrease reduction for moves that escape a capture. Filter out
        // castling moves, because they are coded as "king captures rook" and
        // hence break make_move(). Also use see() instead of see_sign(),
        // because the destination square is empty.
        else if (   type_of_m(move) == NORMAL
                 && !see_test(pos, reverse_move(move), 0))
          r -= 2 + !!ttPv;

        ss->statScore =  (*cmh )[movedPiece][to_sq(move)]
                       + (*fmh )[movedPiece][to_sq(move)]
                       + (*fmh2)[movedPiece][to_sq(move)]
                       + (*pos->history)[stm() ^ 1][from_to(move)]
                       - 4926;

        // Decrease/increase reduction by comparing with opponent's stat score.
        if (ss->statScore >= -102 && (ss-1)->statScore < -114)
          r--;

        else if ((ss-1)->statScore >= -116 && ss->statScore < -154)
          r++;

        // Decrease/increase reduction for moves with a good/bad history.
        r -= ss->statScore / 16434;

      } else {

        // Increase reduction for captures/promotions if late move and at
        // low depth
        if (depth < 8 && moveCount > 2)
          r++;

        // Unless giving check, this capture is likely bad
        if (   !givesCheck
            && ss->staticEval + PieceValue[EG][captured_piece()] + 200 * depth <= alpha)
          r++;
      }

      Depth d = clamp(newDepth - r, 1, newDepth);

      value = -search_NonPV(pos, ss+1, -(alpha+1), d, 1);

      doFullDepthSearch = (value > alpha && d != newDepth);
      didLMR = true;
    } else {
      doFullDepthSearch = !PvNode || moveCount > 1;
      didLMR = false;
    }

    // Step 17. Full depth search when LMR is skipped or fails high.
    if (doFullDepthSearch) {
      value = -search_NonPV(pos, ss+1, -(alpha+1), newDepth, !cutNode);

      if (didLMR && !captureOrPromotion) {
        int bonus = value > alpha ?  stat_bonus(newDepth)
                                  : -stat_bonus(newDepth);

        if (move == ss->killers[0])
          bonus += bonus / 4;

        update_cm_stats(ss, movedPiece, to_sq(move), bonus);
      }
    }

    // For PV nodes only, do a full PV search on the first move or after a fail
    // high (in the latter case search only if value < beta), otherwise let the
    // parent node fail low with value <= alpha and try another move.
    if (PvNode
        && (moveCount == 1 || (value > alpha && (rootNode || value < beta))))
    {
      (ss+1)->pv = pv;
      (ss+1)->pv[0] = 0;

      value = -search_PV(pos, ss+1, -beta, -alpha, newDepth);
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
    if (load_rlx(Signals.stop)) {
      if (crumb) store_rlx(*crumb, 0);
      return 0;
    }

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
        if (moveCount > 1)
          pos->bestMoveChanges++;
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

  if (crumb) store_rlx(*crumb, 0);

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
    // Quiet best move: update move sorting heuristics
    if (!is_capture_or_promotion(pos, bestMove)) {
      int bonus = stat_bonus(depth + (bestValue > beta + PawnValueMg));
      update_quiet_stats(pos, ss, bestMove, bonus, depth);

      // Decrease all the other played quiet moves
      for (int i = 0; i < quietCount; i++) {
        history_update(*pos->history, stm(), quietsSearched[i], -bonus);
        update_cm_stats(ss, moved_piece(quietsSearched[i]),
            to_sq(quietsSearched[i]), -bonus);
      }
    }

    update_capture_stats(pos, bestMove, capturesSearched, captureCount,
        stat_bonus(depth + 1));

    // Extra penalty for a quiet TT or main killer move in previous ply
    // when it gets refuted
    if (  ((ss-1)->moveCount == 1 || (ss-1)->currentMove == (ss-1)->killers[0])
        && !captured_piece())
      update_cm_stats(ss-1, piece_on(prevSq), prevSq,
          -stat_bonus(depth + 1));
  }
  // Bonus for prior countermove that caused the fail low
  else if (   (depth >= 3 || PvNode)
           && !captured_piece())
    update_cm_stats(ss-1, piece_on(prevSq), prevSq, stat_bonus(depth));

  if (PvNode && bestValue > maxValue)
     bestValue = maxValue;

  if (!excludedMove && !(rootNode && pos->pvIdx))
    tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), ttPv,
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
    bool cutNode)
{
  return search_node(pos, ss, alpha, alpha+1, depth, cutNode, NonPV);
}

// qsearch_node() is the quiescence search function template, which is
// called by the main search function with zero depth, or recursively with
// further decreasing depth per call.
INLINE Value qsearch_node(Pos *pos, Stack *ss, Value alpha, Value beta,
    Depth depth, const int NT, const bool InCheck)
{
  const bool PvNode = NT == PV;

  assert(InCheck == (bool)checkers());
  assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(depth <= 0);

  Move pv[MAX_PLY+1];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, bestMove;
  Value bestValue, value, ttValue, futilityValue, futilityBase, oldAlpha;
  int ttHit, ttPv, givesCheck;
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
  posKey = key();
  tte = tt_probe(posKey, &ttHit);
  ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply, rule50_count()) : VALUE_NONE;
  ttMove = ttHit ? tte_move(tte) : 0;
  ttPv = ttHit ? tte_is_pv(tte) : 0;

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
      // Never assume anything about values stored in TT
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
        tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), ttPv,
                 BOUND_LOWER, DEPTH_NONE, 0, ss->staticEval,
                 tt_generation());

      return bestValue;
    }

    if (PvNode && bestValue > alpha)
      alpha = bestValue;

    futilityBase = bestValue + 154;
  }

  ss->history = &(*pos->counterMoveHistory)[0][0][0][0];

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

    // Don't search moves with negative SEE values
    if (!InCheck && !see_test(pos, move, 0))
      continue;

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Check for legality just before making the move
    if (!is_legal(pos, move)) {
      moveCount--;
      continue;
    }

    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[InCheck]
                                           [is_capture_or_promotion(pos, move)]
                                           [moved_piece(move)]
                                           [to_sq(move)];

    // Make and search the move
    do_move(pos, move, givesCheck);
    value = PvNode ? givesCheck
                     ? -qsearch_PV_true(pos, ss+1, -beta, -alpha, depth - 1)
                     : -qsearch_PV_false(pos, ss+1, -beta, -alpha, depth - 1)
                   : givesCheck
                     ? -qsearch_NonPV_true(pos, ss+1, -beta, depth - 1)
                     : -qsearch_NonPV_false(pos, ss+1, -beta, depth - 1);
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Check for a new best move
    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (PvNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (PvNode && value < beta) // Update alpha here!
          alpha = value;
        else
          break; // Fail high
      }
    }
  }

  // All legal moves have been searched. A special case: If we're in check
  // and no legal moves were found, it is checkmate.
  if (InCheck && bestValue == -VALUE_INFINITE)
    return mated_in(ss->ply); // Plies to mate from the root

  tte_save(tte, posKey, value_to_tt(bestValue, ss->ply), ttPv,
           bestValue >= beta ? BOUND_LOWER :
           PvNode && bestValue > oldAlpha  ? BOUND_EXACT : BOUND_UPPER,
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

  return  v >= VALUE_TB_WIN_IN_MAX_PLY  ? v + ply
        : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply : v;
}


// value_from_tt() is the inverse of value_to_tt(): It adjusts a mate score
// from the transposition table (which refers to the plies to mate/be mated
// from current position) to "plies to mate/be mated from the root".

static Value value_from_tt(Value v, int ply, int r50c)
{
  if (v == VALUE_NONE)
    return VALUE_NONE;

  if (v >= VALUE_TB_WIN_IN_MAX_PLY) {
    if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 99 - r50c)
      return VALUE_MATE_IN_MAX_PLY - 1;
    return v - ply;
  }

  if (v <= VALUE_TB_LOSS_IN_MAX_PLY) {
    if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 99 - r50c)
      return VALUE_MATED_IN_MAX_PLY + 1;
    return v + ply;
  }

  return v;
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

  if (move_is_ok((ss-6)->currentMove))
    cms_update(*(ss-6)->history, pc, s, bonus);
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

// update_quiet_stats() updates killers, history, countermove and countermove
// plus follow-up move history when a new quiet best move is found.

static void update_quiet_stats(const Pos *pos, Stack *ss, Move move, int bonus,
    Depth depth)
{
  if (ss->killers[0] != move) {
    ss->killers[1] = ss->killers[0];
    ss->killers[0] = move;
  }

  int c = stm();
  history_update(*pos->history, c, move, bonus);
  update_cm_stats(ss, moved_piece(move), to_sq(move), bonus);

  if (type_of_p(moved_piece(move)) != PAWN)
    history_update(*pos->history, c, reverse_move(move), -bonus);

  if (move_is_ok((ss-1)->currentMove)) {
    Square prevSq = to_sq((ss-1)->currentMove);
    (*pos->counterMoves)[piece_on(prevSq)][prevSq] = move;
  }

  if (depth > 12 && ss->ply < MAX_LPH)
    lph_update(*pos->lowPlyHistory, ss->ply, move, stat_bonus(depth - 7));
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
    bool updated = rm->move[i].score != -VALUE_INFINITE;

    if (depth == 1 && !updated)
        continue;

    Depth d = updated ? depth : depth - 1;
    Value v = updated ? rm->move[i].score : rm->move[i].previousScore;

    bool tb = TB_RootInTB && abs(v) < VALUE_MATE_IN_MAX_PLY;
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
           d, rm->move[i].selDepth, i + 1,
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
  TTEntry *tte = tt_probe(key(), &ttHit);

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
  TB_ProbeDepth = option_value(OPT_SYZ_PROBE_DEPTH);
  TB_Cardinality = option_value(OPT_SYZ_PROBE_LIMIT);
  int dtz_available = 1, dtm_available = 0;

  if (TB_Cardinality > TB_MaxCardinality) {
    TB_Cardinality = TB_MaxCardinality;
    TB_ProbeDepth = 0;
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
  Threads.increaseDepth = true;

  for (int i = 0; i < 1024; i++)
    store_rlx(breadcrumbs[i], 0);

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
    pos->rootDepth = 0;
    pos->nodes = pos->tbHits = 0;
    stats_clear(pos->lowPlyHistory);
    RootMoves *rm = pos->rootMoves;
    rm->size = end - list;
    for (int i = 0; i < rm->size; i++) {
      rm->move[i].pvSize = 1;
      rm->move[i].pv[0] = moves->move[i].pv[0];
      rm->move[i].score = -VALUE_INFINITE;
      rm->move[i].previousScore = -VALUE_INFINITE;
      rm->move[i].selDepth = rm->move[i].bestMoveCount = 0;
      rm->move[i].tbRank = moves->move[i].tbRank;
      rm->move[i].tbScore = moves->move[i].tbScore;
    }
    memcpy(pos, root, offsetof(Pos, moveList));
    // Copy enough of the root State buffer.
    int n = max(7, root->st->pliesFromNull);
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
