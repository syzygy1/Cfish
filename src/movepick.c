/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include "movepick.h"
#include "thread.h"

#define HistoryStats_Max ((Value)(1<<28))

// Our insertion sort, which is guaranteed to be stable, as it should be.

INLINE void insertion_sort(Move *m, int *s, int num)
{
  int i, j;

  for (i = 1; i < num; i++) {
    Move tmp = m[i];
    int tmpVal = s[i];
    for (j = i; j != 0 && s[j - 1] < tmpVal; --j) {
      m[j] = m[j-1];
      s[j] = s[j-1];
    }
    m[j] = tmp;
    s[j] = tmpVal;
  }
}

// Our non-stable partition function, the one that Stockfish uses.

INLINE int partition(Move *m, int *s, int num)
{
  int i = 0;

  while (1) {
    while (1)
      if (i == num)
        return i;
      else if (s[i] > 0)
        i++;
      else
        break;
    num--;
    while (1)
      if (i == num)
        return i;
      else if (!(s[num] > 0))
        num--;
      else
        break;
    Move tmp = m[i];
    m[i] = m[num];
    m[num] = tmp;
    int tmpVal = s[i];
    s[i] = s[num];
    s[num] = tmpVal;
    i++;
  }
}

// pick_best() finds the best move in the range (begin, end).

static Move pick_best(Move *begin, Move *end, int *s)
{
  Move *p, *q;

  int best = *s;
  int tmp = best;
  int *r = s++;
  for (p = begin, q = begin + 1; q < end; q++, s++)
    if (*s > best) {
      best = *s;
      p = q;
      r = s;
    }
  Move move = *p;
  *p = *begin;
  *r = tmp;

  return move;
}


// score() assigns a numerical value to each move in a move list. The moves with
// highest values will be picked first.

static void score_captures(const Pos *pos)
{
  Stack *st = pos->st;

  // Winning and equal captures in the main search are ordered by MVV,
  // preferring captures near our home rank.

  int *s = st->curScore = (st-1)->endScore;
  for (Move *m = st->cur; m < st->endMoves; m++)
    *s++ =  PieceValue[MG][piece_on(to_sq(*m))]
          - (Value)(200 * relative_rank_s(pos_stm(), to_sq(*m)));
  st->endScore = s;
}

SMALL
static void score_quiets(const Pos *pos)
{
  Stack *st = pos->st;
  HistoryStats *history = pos->history;
  FromToStats *fromTo = pos->fromTo;

  CounterMoveStats *cm = (st-1)->counterMoves;
  CounterMoveStats *fm = (st-2)->counterMoves;
  CounterMoveStats *f2 = (st-4)->counterMoves;

  CounterMoveStats *tmp = &(*pos->counterMoveHistory)[0][0];
  if (!cm) cm = tmp;
  if (!fm) fm = tmp;
  if (!f2) f2 = tmp;

  uint32_t c = pos_stm();

  int *s = st->endScore = (st-1)->endScore;
  for (Move *m = st->cur; m < st->endMoves; m++) {
    uint32_t move = *m & 4095;
    uint32_t to = move & 63;
    uint32_t from = move >> 6;
    *s++ =  (*history)[piece_on(from)][to]
          + (*cm)[piece_on(from)][to]
          + (*fm)[piece_on(from)][to]
          + (*f2)[piece_on(from)][to]
          + ft_get(*fromTo, c, move);
  }
}

static void score_evasions(const Pos *pos)
{
  Stack *st = pos->st;
  // Try captures ordered by MVV/LVA, then non-captures ordered by
  // history value.

  HistoryStats *history = pos->history;
  FromToStats *fromTo = pos->fromTo;
  uint32_t c = pos_stm();

  int *s = st->endScore;
  for (Move *m = st->cur; m < st->endMoves; m++, s++)
    if (is_capture(pos, *m))
      *s =  PieceValue[MG][piece_on(to_sq(*m))]
          - (Value)type_of_p(moved_piece(*m)) + HistoryStats_Max;
    else
      *s =  (*history)[moved_piece(*m)][to_sq(*m)]
          + ft_get(*fromTo, c, *m);
  st->endScore = s;
}


// next_move() returns the next pseudo-legal move to be searched.

Move next_move(const Pos *pos)
{
  Stack *st = pos->st;
  Move move;

  switch (st->stage) {

  case ST_MAIN_SEARCH: case ST_EVASIONS: case ST_QSEARCH_WITH_CHECKS:
  case ST_QSEARCH_WITHOUT_CHECKS: case ST_PROBCUT:
    st->endMoves = (st-1)->endMoves;
    st->endScore = (st-1)->endScore;
    st->stage++;
    return st->ttMove;

  case ST_CAPTURES_GEN:
    st->endBadCaptures = st->cur = (st-1)->endMoves;
    st->endMoves = generate_captures(pos, st->cur);
    score_captures(pos);
    st->stage++;

  case ST_GOOD_CAPTURES:
    while (st->cur < st->endMoves) {
      move = pick_best(st->cur++, st->endMoves, st->curScore++);
      if (move != st->ttMove) {
        if (see_test(pos, move, 0))
          return move;

        // Losing capture, move it to the beginning of the array.
        *st->endBadCaptures++ = move;
      }
    }
    st->stage++;

    // First killer move.
    move = st->killers[0];
    if (move && move != st->ttMove && is_pseudo_legal(pos, move)
             && !is_capture(pos, move))
      return move;

  case ST_KILLERS:
    st->stage++;
    move = st->killers[1]; // Second killer move.
    if (move && move != st->ttMove && is_pseudo_legal(pos, move)
             && !is_capture(pos, move))
      return move;

  case ST_KILLERS_2:
    st->stage++;
    move = st->countermove;
    if (move && move != st->ttMove && move != st->killers[0]
             && move != st->killers[1] && is_pseudo_legal(pos, move)
             && !is_capture(pos, move))
      return move;

  case ST_QUIET_GEN:
    st->cur = st->endBadCaptures;
    st->endMoves = generate_quiets(pos, st->cur);
    score_quiets(pos);
    if (st->depth < 3 * ONE_PLY) {
      int goodQuiet = partition(st->cur, st->endScore, st->endMoves - st->cur);
      insertion_sort(st->cur, st->endScore, goodQuiet);
    } else
      insertion_sort(st->cur, st->endScore, st->endMoves - st->cur);
    st->stage++;

  case ST_QUIET:
    while (st->cur < st->endMoves) {
      move = *st->cur++;
      if (   move != st->ttMove && move != st->killers[0]
          && move != st->killers[1] && move != st->countermove)
        return move;
    }
    st->stage++;
    st->cur = (st-1)->endMoves; // Return to bad captures.

  case ST_BAD_CAPTURES:
    if (st->cur < st->endBadCaptures)
      return *st->cur++;
    break;

  case ST_ALL_EVASIONS:
    st->cur = (st-1)->endMoves;
    st->endScore = st->curScore = (st-1)->endScore;
    st->endMoves = generate_evasions(pos, st->cur);
    score_evasions(pos);
    st->stage = ST_REMAINING;

    if (st->stage != ST_REMAINING) {
  case ST_QCAPTURES_CHECKS_GEN: case ST_QCAPTURES_NO_CHECKS_GEN:
      st->cur = (st-1)->endMoves;
      st->endMoves = generate_captures(pos, st->cur);
      score_captures(pos);
      st->stage++;
    }

  case ST_QCAPTURES_CHECKS: case ST_REMAINING:
    while (st->cur < st->endMoves) {
      move = pick_best(st->cur++, st->endMoves, st->curScore++);
      if (move != st->ttMove)
        return move;
    }
    if (st->stage != ST_QCAPTURES_CHECKS)
      break;
    st->cur = (st-1)->endMoves;
    st->endMoves = generate_quiet_checks(pos, st->cur);
    st->stage++;

  case ST_CHECKS:
    while (st->cur < st->endMoves) {
      move = *st->cur++;
      if (move != st->ttMove)
        return move;
    }
    break;

  case ST_RECAPTURES_GEN:
    st->cur = (st-1)->endMoves;
    st->endMoves = generate_captures(pos, st->cur);
    score_captures(pos);
    st->stage++;

  case ST_RECAPTURES:
    while (st->cur < st->endMoves) {
      move = pick_best(st->cur++, st->endMoves, st->curScore++);
      if (to_sq(move) == st->recaptureSquare)
        return move;
    }
    break;

  case ST_PROBCUT_GEN:
    st->cur = (st-1)->endMoves;
    st->endMoves = generate_captures(pos, st->cur);
    score_captures(pos);
    st->stage++;

  case ST_PROBCUT_2:
    while (st->cur < st->endMoves) {
      move = pick_best(st->cur++, st->endMoves, st->curScore++);
      if (move != st->ttMove && see_test(pos, move, st->threshold + 1))
        return move;
    }
    break;

  default:
    assume(0);

  }

  return 0;
}

