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

#ifndef MOVEPICK_H
#define MOVEPICK_H

#include <string.h>   // For memset

#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"

#define stats_clear(s) memset(s, 0, sizeof(*s))

INLINE void hs_update(HistoryStats hs, Piece pc, Square to, Value v)
{
  int w = v >= 0 ? v : -v;
  if (w >= 324)
    return;

  hs[pc][to] -= hs[pc][to] * w / 324;
  hs[pc][to] += ((int)v) * 32;
}

INLINE void cms_update(CounterMoveStats cms, Piece pc, Square to, Value v)
{
  int w = v >= 0 ? v : -v;
  if (w >= 324)
    return;

  cms[pc][to] -= cms[pc][to] * w / 936;
  cms[pc][to] += ((int)v) * 32;
}

INLINE void ft_update(FromToStats ft, int c, Move m, Value v)
{
  int w = v >= 0 ? v : -v;
  if (w >= 324)
    return;

  m &= 4095;
  ft[c][m] -= ft[c][m] * w / 324;
  ft[c][m] += ((int)v) * 32;
}

INLINE Value ft_get(FromToStats ft, int c, Move m)
{
  return ft[c][m & 4095];
}

#define ST_MAIN_SEARCH             0
#define ST_CAPTURES_GEN            1
#define ST_GOOD_CAPTURES           2
#define ST_KILLERS                 3
#define ST_KILLERS_2               4
#define ST_QUIET_GEN               5
#define ST_QUIET                   6
#define ST_BAD_CAPTURES            7

#define ST_EVASIONS                8
#define ST_ALL_EVASIONS            9

#define ST_QSEARCH_WITH_CHECKS     10
#define ST_QCAPTURES_CHECKS_GEN    11
#define ST_QCAPTURES_CHECKS        12
#define ST_CHECKS                  13

#define ST_QSEARCH_WITHOUT_CHECKS  14
#define ST_QCAPTURES_NO_CHECKS_GEN 15
#define ST_REMAINING               16

#define ST_RECAPTURES_GEN          17
#define ST_RECAPTURES              18

#define ST_PROBCUT                 19
#define ST_PROBCUT_GEN             20
#define ST_PROBCUT_2               21

Move next_move(const Pos *pos);

// Initialisation of move picker data.

INLINE void mp_init(const Pos *pos, Move ttm, Depth depth)
{
  assert(depth > DEPTH_ZERO);

  Stack *st = pos->st;

  st->depth = depth;

  Square prevSq = to_sq((st-1)->currentMove);
  st->countermove = (*pos->counterMoves)[piece_on(prevSq)][prevSq];

  st->stage = pos_checkers() ? ST_EVASIONS : ST_MAIN_SEARCH;
  st->ttMove = ttm;
  if (!ttm || !is_pseudo_legal(pos, ttm)) {
    st->stage++;
    st->ttMove = 0;
  }
}

INLINE void mp_init_q(const Pos *pos, Move ttm, Depth depth, Square s)
{
  assert (depth <= DEPTH_ZERO);

  Stack *st = pos->st;

  if (pos_checkers())
    st->stage = ST_EVASIONS;
  else if (depth > DEPTH_QS_NO_CHECKS)
    st->stage = ST_QSEARCH_WITH_CHECKS;
  else if (depth > DEPTH_QS_RECAPTURES)
    st->stage = ST_QSEARCH_WITHOUT_CHECKS;
  else {
    st->stage = ST_RECAPTURES_GEN;
    st->recaptureSquare = s;
    return;
  }

  st->ttMove = ttm;
  if (!ttm || !is_pseudo_legal(pos, ttm)) {
    st->stage++;
    st->ttMove = 0;
  }
}

INLINE void mp_init_pc(const Pos *pos, Move ttm, Value threshold)
{
  assert(!pos_checkers());

  Stack *st = pos->st;

  st->threshold = threshold;

  st->stage = ST_PROBCUT;

  // In ProbCut we generate captures with SEE higher than the given
  // threshold.
  st->ttMove =   ttm && is_pseudo_legal(pos, ttm) && is_capture(pos, ttm)
              && see_test(pos, ttm, threshold) ? ttm : 0;
  if (st->ttMove == 0) st->stage++;
}

#endif

