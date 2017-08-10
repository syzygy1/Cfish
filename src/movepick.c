/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

// An insertion sort which sorts moves in descending order up to and
// including a given limit. The order of moves smaller than the limit is
// left unspecified.

INLINE void partial_insertion_sort(ExtMove *begin, ExtMove *end, int limit)
{
  for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; p++)
    if (p->value >= limit) {
      ExtMove tmp = *p, *q;
      *p = *++sortedEnd;
      for (q = sortedEnd; q != begin && (q-1)->value < tmp.value; q--)
        *q = *(q-1);
      *q = tmp;
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
              - (200 * relative_rank_s(pos_stm(), to_sq(m->move)));
}

SMALL
static void score_quiets(const Pos *pos)
{
  Stack *st = pos->st;
  ButterflyHistory *history = pos->history;

  PieceToHistory *cmh = (st-1)->history;
  PieceToHistory *fmh = (st-2)->history;
  PieceToHistory *fmh2 = (st-4)->history;

  uint32_t c = pos_stm();

  for (ExtMove *m = st->cur; m < st->endMoves; m++) {
    uint32_t move = m->move & 4095;
    uint32_t to = move & 63;
    uint32_t from = move >> 6;
    m->value =  (*cmh)[piece_on(from)][to]
              + (*fmh)[piece_on(from)][to]
              + (*fmh2)[piece_on(from)][to]
              + (*history)[c][move];
  }
}

static void score_evasions(const Pos *pos)
{
  Stack *st = pos->st;
  // Try captures ordered by MVV/LVA, then non-captures ordered by
  // stats heuristics.

  ButterflyHistory *history = pos->history;
  uint32_t c = pos_stm();

  for (ExtMove *m = st->cur; m < st->endMoves; m++)
    if (is_capture(pos, m->move))
      m->value =  PieceValue[MG][piece_on(to_sq(m->move))]
                - type_of_p(moved_piece(m->move)) + (1 << 28);
    else
      m->value = (*history)[c][from_to(m->move)];
}


// next_move() returns the next pseudo-legal move to be searched.

Move next_move(const Pos *pos, int skipQuiets)
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
    /* fallthrough */

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
    move = st->mp_killers[0];
    if (move && move != st->ttMove && is_pseudo_legal(pos, move)
             && !is_capture(pos, move))
      return move;
    /* fallthrough */

  case ST_KILLERS:
    st->stage++;
    move = st->mp_killers[1]; // Second killer move.
    if (move && move != st->ttMove && is_pseudo_legal(pos, move)
             && !is_capture(pos, move))
      return move;
    /* fallthrough */

  case ST_KILLERS_2:
    st->stage++;
    move = st->countermove;
    if (move && move != st->ttMove && move != st->mp_killers[0]
             && move != st->mp_killers[1] && is_pseudo_legal(pos, move)
             && !is_capture(pos, move))
      return move;
    /* fallthrough */

  case ST_QUIET_GEN:
    st->cur = st->endBadCaptures;
    st->endMoves = generate_quiets(pos, st->cur);
    score_quiets(pos);
    partial_insertion_sort(st->cur, st->endMoves, -4000 * st->depth / ONE_PLY);
    st->stage++;
    /* fallthrough */

  case ST_QUIET:
    while (    st->cur < st->endMoves
           && (!skipQuiets || st->cur->value >= 0)) {
      move = (st->cur++)->move;
      if (   move != st->ttMove && move != st->mp_killers[0]
          && move != st->mp_killers[1] && move != st->countermove)
        return move;
    }
    st->stage++;
    st->cur = (st-1)->endMoves; // Return to bad captures.
    /* fallthrough */

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
    /* fallthrough */
  case ST_QCAPTURES_CHECKS_GEN: case ST_QCAPTURES_NO_CHECKS_GEN:
      st->cur = (st-1)->endMoves;
      st->endMoves = generate_captures(pos, st->cur);
      score_captures(pos);
      st->stage++;
    }
    /* fallthrough */

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
    /* fallthrough */

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
    /* fallthrough */

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
    /* fallthrough */

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

