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

#ifndef NNUE_PURE

#include <assert.h>
#include <string.h>   // For memset

#include "material.h"
#include "position.h"

// Polynomial material imbalance parameters.
#define S(mg,eg) make_score(mg,eg)

static const Score QuadraticOurs[][8] = {
  { S(1419, 1455) },
  { S( 101,   28), S( 37,  39) },
  { S(  57,   64), S(249, 187), S(-49, -62) },
  { S(   0,    0), S(118, 137), S( 10,  27), S(  0,   0) },
  { S( -63,  -68), S( -5,   3), S(100,  81), S(132, 118), S(-246, -244) },
  { S(-210, -211), S( 37,  14), S(147, 141), S(161, 105), S(-158, -174), S(-9, -31) }
};

static const Score QuadraticTheirs[][8] = {
  { 0 },
  { S(  33,  30) },
  { S(  46,  18), S(106,  84) },
  { S(  75,  35), S( 59,  44), S( 60,  15) },
  { S(  26,  35), S(  6,  22), S( 38,  39), S(-12,  -2) },
  { S(  97,  93), S(100, 163), S(-58, -91), S(112, 192), S(276, 225) }
};

#undef S

// Helper used to detect a given material distribution.
INLINE bool is_KXK(const Position *pos, int us)
{
  return  !more_than_one(pieces_c(!us))
        && non_pawn_material_c(us) >= RookValueMg;
}

INLINE bool is_KBPsK(const Position *pos, int us)
{
  return   non_pawn_material_c(us) == BishopValueMg
        && pieces_cp(us, PAWN);
}

INLINE bool is_KQKRPs(const Position *pos, int us) {
  return  !piece_count(us, PAWN)
        && non_pawn_material_c(us) == QueenValueMg
        && piece_count(!us, ROOK) == 1
        && pieces_cp(!us, PAWN);
}

// imbalance() calculates the imbalance by comparing the piece count of each
// piece type for both colors.
static Score imbalance(int us, int pieceCount[][8])
{
  int *pc_us = pieceCount[us];
  int *pc_them = pieceCount[!us];
  Score bonus = SCORE_ZERO;

  // Second-degree polynomial material imbalance, by Tord Romstad
  for (int pt1 = 0; pt1 <= QUEEN; pt1++) {
    if (!pc_us[pt1])
      continue;

    int v = 0;

    for (int pt2 = 0; pt2 <= pt1; pt2++)
      v +=  QuadraticOurs[pt1][pt2] * pc_us[pt2]
          + QuadraticTheirs[pt1][pt2] * pc_them[pt2];

    bonus += pc_us[pt1] * v;
  }

  return bonus;
}

typedef int PieceCountType[2][8];

// material_probe() looks up the current position's material configuration
// in the material hash table. It returns a pointer to the MaterialEntry
// if the position is found. Otherwise a new Entry is computed and stored
// there, so we don't have to recompute all when the same material
// configuration occurs again.

void material_entry_fill(const Position *pos, MaterialEntry *e, Key key)
{
  memset(e, 0, sizeof(MaterialEntry));
  e->key = key;
  e->factor[WHITE] = e->factor[BLACK] = (uint8_t)SCALE_FACTOR_NORMAL;

  Value npm_w = non_pawn_material_c(WHITE);
  Value npm_b = non_pawn_material_c(BLACK);
  Value npm = clamp(npm_w + npm_b, EndgameLimit, MidgameLimit);
  e->gamePhase = ((npm - EndgameLimit) * PHASE_MIDGAME) / (MidgameLimit - EndgameLimit);

  // Look for a specialized evaluation function.
  for (int i = 0; i < NUM_EVAL; i++)
    for (int c = 0; c < 2; c++)
      if (endgame_keys[i][c] == key) {
        e->eval_func = 1 + i;
        e->eval_func_side = c;
        return;
      }

  for (int c = 0; c < 2; c++)
    if (is_KXK(pos, c)) {
      e->eval_func = 10; // EvaluateKXK
      e->eval_func_side = c;
      return;
    }

  // Look for a specialized scaling function.
  for (int i = 0; i < NUM_SCALING; i++)
    for (int c = 0; c < 2; c++)
      if (endgame_keys[NUM_EVAL + i][c] == key) {
        e->scal_func[c] = 11 + i;
        return;
      }

  // We did not find any specialized scaling function, so fall back on
  // generic ones that refer to more than one material distribution. Note
  // that in this case we do not return after setting the function.
  for (int c = 0; c < 2; c++) {
    if (is_KBPsK(pos, c))
      e->scal_func[c] = 17; // ScaleKBPsK

    else if (is_KQKRPs(pos, c))
      e->scal_func[c] = 18; // ScaleKQKRPs
  }

  if (npm_w + npm_b == 0 && pieces_p(PAWN)) { // Only pawns on the board.
    if (!pieces_cp(BLACK, PAWN)) {
      assert(piece_count(WHITE, PAWN) >= 2);

      e->scal_func[WHITE] = 19; // ScaleKPsK
    }
    else if (!pieces_cp(WHITE, PAWN)) {
      assert(piece_count(BLACK, PAWN) >= 2);

      e->scal_func[BLACK] = 19; // ScaleKPsK
    }
    else if (popcount(pieces_p(PAWN)) == 2) { // Each side has one pawn.
      // This is a special case because we set scaling functions
      // for both colors instead of only one.
      e->scal_func[WHITE] = 20; // ScaleKPKP
      e->scal_func[BLACK] = 20; // ScaleKPKP
    }
  }

  // Zero or just one pawn makes it difficult to win, even with a small
  // material advantage. This catches some trivial draws like KK, KBK and
  // KNK and gives a drawish scale factor for cases such as KRKBP and
  // KmmKm (except for KBBKN).
  if (!piece_count(WHITE, PAWN) && npm_w - npm_b <= BishopValueMg)
    e->factor[WHITE] = (uint8_t)(npm_w <  RookValueMg   ? SCALE_FACTOR_DRAW :
                                 npm_b <= BishopValueMg ? 4 : 14);

  if (!piece_count(BLACK, PAWN) && npm_b - npm_w <= BishopValueMg)
    e->factor[BLACK] = (uint8_t)(npm_b <  RookValueMg   ? SCALE_FACTOR_DRAW :
                                 npm_w <= BishopValueMg ? 4 : 14);

  // Evaluate the material imbalance. We use PIECE_TYPE_NONE as a place
  // holder for the bishop pair "extended piece", which allows us to be
  // more flexible in defining bishop pair bonuses.
#define pc(c,p) piece_count_mk(c,p)
  int PieceCount[2][8] = {
    { pc(0, BISHOP) > 1, pc(0, PAWN), pc(0, KNIGHT),
      pc(0, BISHOP)    , pc(0, ROOK), pc(0, QUEEN) },
    { pc(1, BISHOP) > 1, pc(1, PAWN), pc(1, KNIGHT),
      pc(1, BISHOP)    , pc(1, ROOK), pc(1, QUEEN) }
  };
#undef pc
  Score tmp = imbalance(WHITE, PieceCount) - imbalance(BLACK, PieceCount);
  e->score = make_score(mg_value(tmp) / 16, eg_value(tmp) / 16);
}

#else

typedef int make_iso_compilers_happy;

#endif
