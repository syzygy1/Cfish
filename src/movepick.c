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

INLINE void insertion_sort(ExtMove *begin, ExtMove *end)
{
  ExtMove tmp, *p, *q;

  for (p = begin + 1; p < end; p++) {
    tmp = *p;
    for (q = p; q != begin && (q-1)->value < tmp.value; --q)
      *q = *(q-1);
    *q = tmp;
  }
}

// Our non-stable partition function, the one that Stockfish uses.

INLINE ExtMove *partition(ExtMove *first, ExtMove *last)
{
  ExtMove tmp;

  while (1) {
    while (1)
      if (first == last)
        return first;
      else if (first->value > 0)
        first++;
      else
        break;
    last--;
    while (1)
      if (first == last)
        return first;
      else if (!(last->value > 0))
        last--;
      else
        break;
    tmp = *first;
    *first = *last;
    *last = tmp;
    first++;
  }
}

// pick_best() finds the best move in the range (begin, end).

static Move pick_best(ExtMove *begin, ExtMove *end)
{
  ExtMove *p, *q;

  for (p = begin, q = begin + 1; q < end; q++)
    if (q->value > p->value)
      p = q;
  Move m = p->move;
  *p = *begin;

  return m;
}


// score() assigns a numerical value to each move in a move list. The moves with
// highest values will be picked first.

static void score_captures(const Pos *pos)
{
  Stack *st = pos->st;

  // Winning and equal captures in the main search are ordered by MVV,
  // preferring captures near our home rank.

  for (ExtMove *m = st->cur; m < st->endMoves; m++)
    m->value =  PieceValue[MG][piece_on(to_sq(m->move))]
              - (Value)(200 * relative_rank_s(pos_stm(), to_sq(m->move)));
}

SMALL
static void score_quiets(const Pos *pos)
{
  Stack *st = pos->st;
  HistoryStats *history = pos->history;
  FromToStats *fromTo = pos->fromTo;

  CounterMoveStats *cmh = (st-1)->counterMoves;
  CounterMoveStats *fmh = (st-2)->counterMoves;
  CounterMoveStats *fmh2 = (st-4)->counterMoves;

  CounterMoveStats *tmp = &(*pos->counterMoveHistory)[0][0];
  if (!cmh) cmh = tmp;
  if (!fmh) fmh = tmp;
  if (!fmh2) fmh2 = tmp;

  uint32_t c = pos_stm();

  for (ExtMove *m = st->cur; m < st->endMoves; m++) {
    uint32_t move = m->move & 4095;
    uint32_t to = move & 63;
    uint32_t from = move >> 6;
    m->value =  (*history)[piece_on(from)][to]
              + (*cmh)[piece_on(from)][to]
              + (*fmh)[piece_on(from)][to]
              + (*fmh2)[piece_on(from)][to]
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

  for (ExtMove *m = st->cur; m < st->endMoves; m++)
    if (is_capture(pos, m->move))
      m->value =  PieceValue[MG][piece_on(to_sq(m->move))]
                - (Value)type_of_p(moved_piece(m->move)) + HistoryStats_Max;
    else
      m->value =  (*history)[moved_piece(m->move)][to_sq(m->move)]
                + ft_get(*fromTo, c, m->move);
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
    st->stage++;
    return st->ttMove;

  case ST_CAPTURES_GEN:
    st->endBadCaptures = st->cur = (st-1)->endMoves;
    st->endMoves = generate_captures(pos, st->cur);
    score_captures(pos);
    st->stage++;

  case ST_GOOD_CAPTURES:
    while (st->cur < st->endMoves) {
      move = pick_best(st->cur++, st->endMoves);
      if (move != st->ttMove) {
        if (see_test(pos, move, 0))
          return move;

        // Losing capture, move it to the beginning of the array.
        (st->endBadCaptures++)->move = move;
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
      ExtMove *goodQuiet = partition(st->cur, st->endMoves);
      insertion_sort(st->cur, goodQuiet);
    } else
      insertion_sort(st->cur, st->endMoves);
    st->stage++;

  case ST_QUIET:
    while (st->cur < st->endMoves) {
      move = (st->cur++)->move;
      if (   move != st->ttMove && move != st->killers[0]
          && move != st->killers[1] && move != st->countermove)
        return move;
    }
    st->stage++;
    st->cur = (st-1)->endMoves; // Return to bad captures.

  case ST_BAD_CAPTURES:
    if (st->cur < st->endBadCaptures)
      return (st->cur++)->move;
    break;

  case ST_ALL_EVASIONS:
    st->cur = (st-1)->endMoves;
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
      move = pick_best(st->cur++, st->endMoves);
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
      move = (st->cur++)->move;
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
      move = pick_best(st->cur++, st->endMoves);
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
      move = pick_best(st->cur++, st->endMoves);
      if (move != st->ttMove && see_test(pos, move, st->threshold))
        return move;
    }
    break;

  default:
    assume(0);

  }

  return 0;
}

