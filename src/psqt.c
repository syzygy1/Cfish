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

#include "types.h"

Value PieceValue[2][16] = {
{ 0, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg },
{ 0, PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg } };

uint32_t NonPawnPieceValue[16];

#define S(mg, eg) make_score(mg, eg)

// Bonus[PieceType][Square / 2] contains Piece-Square scores. For each
// piece type on a given square a (middlegame, endgame) score pair is
// assigned. Table is defined for files A..D and white side: it is
// symmetric for black side and second half of the files.

static const Score Bonus[][8][4] = {
  {{0}},
  { // Pawn
   { S(  0, 0), S(  0, 0), S(  0, 0), S( 0, 0) },
   { S(-11, 7), S(  6,-4), S(  7, 8), S( 3,-2) },
   { S(-18,-4), S( -2,-5), S( 19, 5), S(24, 4) },
   { S(-17, 3), S( -9, 3), S( 20,-8), S(35,-3) },
   { S( -6, 8), S(  5, 9), S(  3, 7), S(21,-6) },
   { S( -6, 8), S( -8,-5), S( -6, 2), S(-2, 4) },
   { S( -4, 3), S( 20,-9), S( -8, 1), S(-4,18) }
  },
  { // Knight
   { S(-161,-105), S(-96,-82), S(-80,-46), S(-73,-14) },
   { S( -83, -69), S(-43,-54), S(-21,-17), S(-10,  9) },
   { S( -71, -50), S(-22,-39), S(  0, -7), S(  9, 28) },
   { S( -25, -41), S( 18,-25), S( 43,  6), S( 47, 38) },
   { S( -26, -46), S( 16,-25), S( 38,  3), S( 50, 40) },
   { S( -11, -54), S( 37,-38), S( 56, -7), S( 65, 27) },
   { S( -63, -65), S(-19,-50), S(  5,-24), S( 14, 13) },
   { S(-195,-109), S(-67,-89), S(-42,-50), S(-29,-13) }
  },
  { // Bishop
   { S(-44,-58), S(-13,-31), S(-25,-37), S(-34,-19) },
   { S(-20,-34), S( 20, -9), S( 12,-14), S(  1,  4) },
   { S( -9,-23), S( 27,  0), S( 21, -3), S( 11, 16) },
   { S(-11,-26), S( 28, -3), S( 21, -5), S( 10, 16) },
   { S(-11,-26), S( 27, -4), S( 16, -7), S(  9, 14) },
   { S(-17,-24), S( 16, -2), S( 12,  0), S(  2, 13) },
   { S(-23,-34), S( 17,-10), S(  6,-12), S( -2,  6) },
   { S(-35,-55), S(-11,-32), S(-19,-36), S(-29,-17) }
  },
  { // Rook
   { S(-25, 0), S(-16, 0), S(-16, 0), S(-9, 0) },
   { S(-21, 0), S( -8, 0), S( -3, 0), S( 0, 0) },
   { S(-21, 0), S( -9, 0), S( -4, 0), S( 2, 0) },
   { S(-22, 0), S( -6, 0), S( -1, 0), S( 2, 0) },
   { S(-22, 0), S( -7, 0), S(  0, 0), S( 1, 0) },
   { S(-21, 0), S( -7, 0), S(  0, 0), S( 2, 0) },
   { S(-12, 0), S(  4, 0), S(  8, 0), S(12, 0) },
   { S(-23, 0), S(-15, 0), S(-11, 0), S(-5, 0) }
  },
  { // Queen
   { S( 0,-71), S(-4,-56), S(-3,-42), S(-1,-29) },
   { S(-4,-56), S( 6,-30), S( 9,-21), S( 8, -5) },
   { S(-2,-39), S( 6,-17), S( 9, -8), S( 9,  5) },
   { S(-1,-29), S( 8, -5), S(10,  9), S( 7, 19) },
   { S(-3,-27), S( 9, -5), S( 8, 10), S( 7, 21) },
   { S(-2,-40), S( 6,-16), S( 8,-10), S(10,  3) },
   { S(-2,-55), S( 7,-30), S( 7,-21), S( 6, -6) },
   { S(-1,-74), S(-4,-55), S(-1,-43), S( 0,-30) }
  },
  { // King
   { S(267,  0), S(320, 48), S(270, 75), S(195, 84) },
   { S(264, 43), S(304, 92), S(238,143), S(180,132) },
   { S(200, 83), S(245,138), S(176,167), S(110,165) },
   { S(177,106), S(185,169), S(148,169), S(110,179) },
   { S(149,108), S(177,163), S(115,200), S( 66,203) },
   { S(118, 95), S(159,155), S( 84,176), S( 41,174) },
   { S( 87, 50), S(128, 99), S( 63,122), S( 20,139) },
   { S( 63,  9), S( 88, 55), S( 47, 80), S(  0, 90) }
  }
};

#undef S

struct PSQT psqt;

// init() initializes piece-square tables: the white halves of the tables
// are copied from Bonus[] adding the piece value, then the black  halves
// of the tables are initialized by flipping and changing the sign of the
// white scores.

void psqt_init(void) {
  for (int pt = PAWN; pt <= KING; pt++) {
    PieceValue[MG][make_piece(BLACK, pt)] = PieceValue[MG][pt];
    PieceValue[EG][make_piece(BLACK, pt)] = PieceValue[EG][pt];

    Score v = make_score(PieceValue[MG][pt], PieceValue[EG][pt]);

    for (Square s = 0; s < 64; s++) {
      int f = min(file_of(s), FILE_H - file_of(s));
      psqt.psq[make_piece(WHITE, pt)][s] = v + Bonus[pt][rank_of(s)][f];
      psqt.psq[make_piece(BLACK, pt)][s ^ 0x38] = -psqt.psq[make_piece(WHITE, pt)][s];
    }
  }
  union {
    uint16_t val[2];
    uint32_t combi;
  } tmp;
  NonPawnPieceValue[W_PAWN] = NonPawnPieceValue[B_PAWN] = 0;
  for (int pt = KNIGHT; pt < KING; pt++) {
    tmp.val[0] = PieceValue[MG][pt];
    tmp.val[1] = 0;
    NonPawnPieceValue[pt] = tmp.combi;
    tmp.val[0] = 0;
    tmp.val[1] = PieceValue[MG][pt];
    NonPawnPieceValue[pt + 8] = tmp.combi;
  }
}
