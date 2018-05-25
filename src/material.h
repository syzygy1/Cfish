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

#ifndef MATERIAL_H
#define MATERIAL_H

#include "endgame.h"
#include "misc.h"
#include "position.h"
#include "types.h"

typedef struct Pos Pos;

// MaterialEntry contains various information about a material
// configuration. It contains a material imbalance evaluation, a function
// pointer to a special endgame evaluation function (which in most cases
// is NULL, meaning that the standard evaluation function will be used),
// and scale factors.
//
// The scale factors are used to scale the evaluation score up or down.
// For instance, in KRB vs KR endgames, the score is scaled down by a
// factor of 4, which will result in scores of absolute value less than
// one pawn.

struct MaterialEntry {
  Key key;
  int gamePhase;
  int16_t value;
  uint8_t eval_func;
  uint8_t eval_func_side;
  uint8_t scal_func[2];
  uint8_t factor[2];
};

typedef struct MaterialEntry MaterialEntry;

typedef MaterialEntry MaterialTable[8192];

void material_entry_fill(const Pos *pos, MaterialEntry *e, Key key);

INLINE MaterialEntry *material_probe(const Pos *pos)
{
  Key key = pos_material_key();
  MaterialEntry *e = &pos->materialTable[key >> (64-13)];

  if (unlikely(e->key != key))
    material_entry_fill(pos, e, key);

  return e;
}

INLINE Score material_imbalance(MaterialEntry *me)
{
  return make_score((unsigned)me->value, me->value);
}

INLINE bool material_specialized_eval_exists(MaterialEntry *me)
{
  return me->eval_func != 0;
}

INLINE Value material_evaluate(MaterialEntry *me, const Pos *pos)
{
  return endgame_funcs[me->eval_func](pos, me->eval_func_side);
}

// scale_factor takes a position and a color as input and returns a scale factor
// for the given color. We have to provide the position in addition to the color
// because the scale factor may also be a function which should be applied to
// the position. For instance, in KBP vs K endgames, the scaling function looks
// for rook pawns and wrong-colored bishops.
INLINE int material_scale_factor(MaterialEntry *me, const Pos *pos, int c)
{
  int sf = SCALE_FACTOR_NONE;
  if (me->scal_func[c])
    sf = endgame_funcs[me->scal_func[c]](pos, c);
  return sf != SCALE_FACTOR_NONE ? sf : me->factor[c];
}

#endif

