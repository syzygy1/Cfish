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

#include "bitboard.h"
#include "misc.h"

#ifndef USE_POPCNT
uint8_t PopCnt16[1 << 16];
#endif
uint8_t SquareDistance[64][64];

static int RookDirs[] = { NORTH, EAST, SOUTH, WEST };
static int BishopDirs[] = { NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST };

static Bitboard sliding_attack(int dirs[], Square sq, Bitboard occupied)
{
  Bitboard attack = 0;

  for (int i = 0; i < 4; i++)
    for (Square s = sq + dirs[i];
         square_is_ok(s) && distance(s, s - dirs[i]) == 1; s += dirs[i])
    {
      attack |= sq_bb(s);
      if (occupied & sq_bb(s))
        break;
    }

  return attack;
}

#if defined(MAGIC_FANCY)
#include "magic-fancy.c"
#elif defined(MAGIC_PLAIN)
#include "magic-plain.c"
#elif defined(MAGIC_BLACK)
#include "magic-black.c"
#elif defined(BMI2_FANCY)
#include "bmi2-fancy.c"
#elif defined(BMI2_PLAIN)
#include "bmi2-plain.c"
#endif

Bitboard SquareBB[64];
Bitboard FileBB[8];
Bitboard RankBB[8];
Bitboard ForwardRanksBB[2][8];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Bitboard DistanceRingBB[64][8];
Bitboard ForwardFileBB[2][64];
Bitboard PassedPawnMask[2][64];
Bitboard PawnAttackSpan[2][64];
Bitboard PseudoAttacks[8][64];
Bitboard PawnAttacks[2][64];

#ifndef PEDANTIC
Bitboard EPMask[16];
Bitboard CastlingPath[64];
int CastlingRightsMask[64];
Square CastlingRookSquare[16];
Key CastlingHash[16];
Bitboard CastlingBits[16];
Score CastlingPSQ[16];
Square CastlingRookFrom[16];
Square CastlingRookTo[16];
#endif

#ifndef USE_POPCNT
// popcount16() counts the non-zero bits using SWAR-Popcount algorithm.

INLINE unsigned popcount16(unsigned u)
{
  u -= (u >> 1) & 0x5555U;
  u = ((u >> 2) & 0x3333U) + (u & 0x3333U);
  u = ((u >> 4) + u) & 0x0F0FU;
  return (u * 0x0101U) >> 8;
}
#endif


// Bitboards::pretty() returns an ASCII representation of a bitboard suitable
// to be printed to standard output. Useful for debugging.

void print_pretty(Bitboard b)
{
  printf("+---+---+---+---+---+---+---+---+\n");

  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f <= 7; f++)
      printf((b & sq_bb(8 * r + f)) ? "| X " : "|   ");

    printf("|\n+---+---+---+---+---+---+---+---+\n");
  }
}


// bitboards_init() initializes various bitboard tables. It is called at
// startup and relies on global objects to be already zero-initialized.

void bitboards_init(void)
{
#ifndef USE_POPCNT
  for (unsigned i = 0; i < (1 << 16); ++i)
    PopCnt16[i] = popcount16(i);
#endif

  for (Square s = 0; s < 64; s++)
    SquareBB[s] = 1ULL << s;

  for (int f = 0; f < 8; f++)
    FileBB[f] = f > FILE_A ? FileBB[f - 1] << 1 : FileABB;

  for (int r = 0; r < 8; r++)
    RankBB[r] = r > RANK_1 ? RankBB[r - 1] << 8 : Rank1BB;

  for (int r = 0; r < 7; r++)
    ForwardRanksBB[WHITE][r] = ~(ForwardRanksBB[BLACK][r + 1] = ForwardRanksBB[BLACK][r] | RankBB[r]);

  for (int c = 0; c < 2; c++)
    for (Square s = 0; s < 64; s++) {
      ForwardFileBB[c][s]  = ForwardRanksBB[c][rank_of(s)] & FileBB[file_of(s)];
      PawnAttackSpan[c][s] = ForwardRanksBB[c][rank_of(s)] & adjacent_files_bb(file_of(s));
      PassedPawnMask[c][s] = ForwardFileBB[c][s] | PawnAttackSpan[c][s];
    }

  for (Square s1 = 0; s1 < 64; s1++)
    for (Square s2 = 0; s2 < 64; s2++)
      if (s1 != s2) {
        SquareDistance[s1][s2] = max(distance_f(s1, s2), distance_r(s1, s2));
        DistanceRingBB[s1][SquareDistance[s1][s2]] |= sq_bb(s2);
      }

#ifndef PEDANTIC
  for (Square s = SQ_A4; s <= SQ_H5; s++)
    EPMask[s - SQ_A4] =  ((sq_bb(s) >> 1) & ~FileHBB)
                       | ((sq_bb(s) << 1) & ~FileABB);
#endif

  int steps[][5] = {
    {0}, { 7, 9 }, { 6, 10, 15, 17 }, {0}, {0}, {0}, { 1, 7, 8, 9 }
  };

  for (int c = 0; c < 2; c++)
    for (int pt = PAWN; pt <= KING; pt++)
      for (int s = 0; s < 64; s++)
        for (int i = 0; steps[pt][i]; i++) {
          Square to = s + (Square)(c == WHITE ? steps[pt][i] : -steps[pt][i]);

          if (square_is_ok(to) && distance(s, to) < 3) {
            if (pt == PAWN)
              PawnAttacks[c][s] |= sq_bb(to);
            else
              PseudoAttacks[pt][s] |= sq_bb(to);
          }
        }

  init_sliding_attacks();

  for (Square s1 = 0; s1 < 64; s1++) {
    PseudoAttacks[QUEEN][s1] = PseudoAttacks[BISHOP][s1] = attacks_bb_bishop(s1, 0);
    PseudoAttacks[QUEEN][s1] |= PseudoAttacks[ROOK][s1] = attacks_bb_rook(s1, 0);

    for (int pt = BISHOP; pt <= ROOK; pt++)
      for (Square s2 = 0; s2 < 64; s2++) {
        if (!(PseudoAttacks[pt][s1] & sq_bb(s2)))
          continue;

        LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0)) | sq_bb(s1) | sq_bb(s2);
        BetweenBB[s1][s2] = attacks_bb(pt, s1, SquareBB[s2]) & attacks_bb(pt, s2, SquareBB[s1]);
      }
  }
}
