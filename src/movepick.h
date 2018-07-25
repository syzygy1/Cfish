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

#ifndef MOVEPICK_H
#define MOVEPICK_H

#include <string.h>   // For memset

#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"

#define stats_clear(s) memset(s, 0, sizeof(*s))

static const int CounterMovePruneThreshold = 0;

INLINE void cms_update(PieceToHistory cms, Piece pc, Square to, int v)
{
  cms[pc][to] += v - cms[pc][to] * abs(v) / 29952;
}

INLINE void history_update(ButterflyHistory history, int c, Move m, int v)
{
  m &= 4095;
  history[c][m] += v - history[c][m] * abs(v) / 10692;
}

INLINE void cpth_update(CapturePieceToHistory history, Piece pc, Square to,
                        int captured, int v)
{
  history[pc][to][captured] += v - history[pc][to][captured] * abs(v) / 10692;
}

enum {
  ST_MAIN_SEARCH, ST_CAPTURES_INIT, ST_GOOD_CAPTURES, ST_KILLERS, ST_KILLERS_2,
  ST_QUIET_INIT, ST_QUIET, ST_BAD_CAPTURES,

  ST_EVASION, ST_EVASIONS_INIT, ST_ALL_EVASIONS,

  ST_QSEARCH, ST_QCAPTURES_INIT, ST_QCAPTURES, ST_QCHECKS,

  ST_PROBCUT, ST_PROBCUT_INIT, ST_PROBCUT_2
};

Move next_move(const Pos *pos, int skipQuiets);

// Initialisation of move picker data.

INLINE void mp_init(const Pos *pos, Move ttm, Depth d)
{
  assert(d > DEPTH_ZERO);

  Stack *st = pos->st;

  st->depth = d;

  Square prevSq = to_sq((st-1)->currentMove);
  st->countermove = (*pos->counterMoves)[piece_on(prevSq)][prevSq];
  st->mpKillers[0] = st->killers[0];
  st->mpKillers[1] = st->killers[1];

  st->stage = pos_checkers() ? ST_EVASION : ST_MAIN_SEARCH;
  st->ttMove = ttm;
  if (!ttm || !is_pseudo_legal(pos, ttm)) {
    st->stage++;
    st->ttMove = 0;
  }
}

INLINE void mp_init_q(const Pos *pos, Move ttm, Depth d, Square s)
{
  assert(d <= DEPTH_ZERO);

  Stack *st = pos->st;

  st->stage = pos_checkers() ? ST_EVASION : ST_QSEARCH;
  st->ttMove =   ttm
              && is_pseudo_legal(pos, ttm)
              && (d > DEPTH_QS_RECAPTURES || to_sq(ttm) == s) ? ttm : 0;
  st->stage += (st->ttMove == 0);

  st->depth = d;
  st->recaptureSquare = s;
}

INLINE void mp_init_pc(const Pos *pos, Move ttm, Value th)
{
  assert(!pos_checkers());

  Stack *st = pos->st;

  st->threshold = th;

  st->stage = ST_PROBCUT;

  // In ProbCut we generate captures with SEE higher than the given
  // threshold.
  st->ttMove =   ttm && is_pseudo_legal(pos, ttm) && is_capture(pos, ttm)
              && see_test(pos, ttm, th) ? ttm : 0;
  if (st->ttMove == 0) st->stage++;
}

#endif
