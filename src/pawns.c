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

#include "bitboard.h"
#include "pawns.h"
#include "position.h"
#include "thread.h"

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

#define V(v) ((Value)(v))
#define S(mg, eg) make_score(mg, eg)

// Isolated pawn penalty by opposed flag
static const Score Isolated[2] = { S(45, 40), S(30, 27) };

// Backward pawn penalty by opposed flag
static const Score Backward[2] = { S(56, 33), S(41, 19) };

// Unsupported pawn penalty for pawns which are neither isolated or backward,
// by number of pawns it supports [less than 2 / exactly 2].
static const Score Unsupported[2] = { S(17, 8), S(21, 12) };

// Connected pawn bonus by opposed, phalanx, twice supported and rank
static Score Connected[2][2][2][8];

// Doubled pawn penalty
static const Score Doubled = S(18,38);

// Lever bonus by rank
static const Score Lever[8] = {
  S( 0,  0), S( 0,  0), S(0, 0), S(0, 0),
  S(17, 16), S(33, 32), S(0, 0), S(0, 0)
};

  // Weakness of our pawn shelter in front of the king by [distance from edge][rank]
static const Value ShelterWeakness[][8] = {
  { V( 97), V(21), V(26), V(51), V(87), V( 89), V( 99) },
  { V(120), V( 0), V(28), V(76), V(88), V(103), V(104) },
  { V(101), V( 7), V(54), V(78), V(77), V( 92), V(101) },
  { V( 80), V(11), V(44), V(68), V(87), V( 90), V(119) }
};

  // Danger of enemy pawns moving toward our king by [type][distance from edge][rank]
static const Value StormDanger[][4][8] = {
  { { V( 0),  V(  67), V( 134), V(38), V(32) },
    { V( 0),  V(  57), V( 139), V(37), V(22) },
    { V( 0),  V(  43), V( 115), V(43), V(27) },
    { V( 0),  V(  68), V( 124), V(57), V(32) } },
  { { V(20),  V(  43), V( 100), V(56), V(20) },
    { V(23),  V(  20), V(  98), V(40), V(15) },
    { V(23),  V(  39), V( 103), V(36), V(18) },
    { V(28),  V(  19), V( 108), V(42), V(26) } },
  { { V( 0),  V(   0), V(  75), V(14), V( 2) },
    { V( 0),  V(   0), V( 150), V(30), V( 4) },
    { V( 0),  V(   0), V( 160), V(22), V( 5) },
    { V( 0),  V(   0), V( 166), V(24), V(13) } },
  { { V( 0),  V(-283), V(-281), V(57), V(31) },
    { V( 0),  V(  58), V( 141), V(39), V(18) },
    { V( 0),  V(  65), V( 142), V(48), V(32) },
    { V( 0),  V(  60), V( 126), V(51), V(19) } }
};

// Max bonus for king safety. Corresponds to start position with all the pawns
// in front of the king and no enemy pawn on the horizon.
static const Value MaxSafetyBonus = V(258);

#undef S
#undef V

#define Us WHITE
#include "tmplpawns.c"
#undef Us
#define Us BLACK
#include "tmplpawns.c"
#undef Us

// pawn_init() initializes some tables needed by evaluation.

void pawn_init(void)
{
  static const int Seed[8] = { 0, 8, 19, 13, 71, 94, 169, 324 };

  for (int opposed = 0; opposed < 2; opposed++)
    for (int phalanx = 0; phalanx < 2; phalanx++)
      for (int apex = 0; apex < 2; apex++)
        for (int r = RANK_2; r < RANK_8; ++r) {
          int v = (Seed[r] + (phalanx ? (Seed[r + 1] - Seed[r]) / 2 : 0)) >> opposed;
          v += (apex ? v / 2 : 0);
          Connected[opposed][phalanx][apex][r] = make_score(v, v * 5 / 8);
      }
}


// pawns_probe() looks up the current position's pawns configuration in
// the pawns hash table.

PawnEntry *pawn_probe(Pos *pos)
{
  Key key = pos_pawn_key();
  PawnEntry* e = pos->thisThread->pawnTable[key & 16383];

  if (e->key == key)
    return e;

  e->key = key;
  e->score = pawn_evaluate_white(pos, e) - pawn_evaluate_black(pos, e);
  e->asymmetry = popcount(e->semiopenFiles[WHITE] ^ e->semiopenFiles[BLACK]);
  return e;
}

