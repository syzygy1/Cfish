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

#ifndef BITBOARD_H
#define BITBOARD_H

#include <assert.h>

#include "types.h"

void bitbases_init(void);
unsigned bitbases_probe(Square wksq, Square wpsq, Square bksq, unsigned us);

void bitboards_init(void);
void print_pretty(Bitboard b);

#define AllSquares (~0ULL)
#define DarkSquares  0xAA55AA55AA55AA55ULL
#define LightSquares (~DarkSquares)

#define FileABB 0x0101010101010101ULL
#define FileBBB (FileABB << 1)
#define FileCBB (FileABB << 2)
#define FileDBB (FileABB << 3)
#define FileEBB (FileABB << 4)
#define FileFBB (FileABB << 5)
#define FileGBB (FileABB << 6)
#define FileHBB (FileABB << 7)

#define Rank1BB 0xFFULL
#define Rank2BB (Rank1BB << (8 * 1))
#define Rank3BB (Rank1BB << (8 * 2))
#define Rank4BB (Rank1BB << (8 * 3))
#define Rank5BB (Rank1BB << (8 * 4))
#define Rank6BB (Rank1BB << (8 * 5))
#define Rank7BB (Rank1BB << (8 * 6))
#define Rank8BB (Rank1BB << (8 * 7))

extern uint8_t SquareDistance[64][64];

extern Bitboard SquareBB[64];
extern Bitboard FileBB[8];
extern Bitboard RankBB[8];
extern Bitboard ForwardRanksBB[2][8];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];
extern Bitboard DistanceRingBB[64][8];
extern Bitboard ForwardFileBB[2][64];
extern Bitboard PassedPawnMask[2][64];
extern Bitboard PawnAttackSpan[2][64];
extern Bitboard PseudoAttacks[8][64];
extern Bitboard PawnAttacks[2][64];

#ifndef PEDANTIC
extern Bitboard EPMask[16];
extern Bitboard CastlingPath[64];
extern uint8_t CastlingRightsMask[64];
extern uint8_t CastlingRookSquare[16];
extern uint8_t CastlingToSquare[16]; // To correct the KxR encoding.
extern Key CastlingHash[16];
extern Bitboard CastlingBits[16];
extern Score CastlingPSQ[16];
extern uint8_t CastlingRookFrom[16];
extern uint8_t CastlingRookTo[16];
#endif


INLINE __attribute__((pure)) Bitboard sq_bb(Square s)
{
  return SquareBB[s];
}

#if __x86_64__
INLINE Bitboard inv_sq(Bitboard b, Square s)
{
  __asm__("btcq %1, %0" : "+r" (b) : "r" ((uint64_t)s) : "cc");
  return b;
}
#else
INLINE Bitboard inv_sq(Bitboard b, Square s)
{
  return b ^ sq_bb(s);
}
#endif

INLINE uint64_t more_than_one(Bitboard b)
{
  return b & (b - 1);
}


// rank_bb() and file_bb() return a bitboard representing all the squares on
// the given file or rank.

INLINE Bitboard rank_bb(Rank r)
{
  return RankBB[r];
}

INLINE Bitboard rank_bb_s(Square s)
{
  return RankBB[rank_of(s)];
}

INLINE Bitboard file_bb(File f)
{
  return FileBB[f];
}

INLINE Bitboard file_bb_s(Square s)
{
  return FileBB[file_of(s)];
}


// shift_bb() moves a bitboard one step along direction Direction.
INLINE Bitboard shift_bb(int Direction, Bitboard b)
{
  return  Direction == NORTH  ?  b  << 8
        : Direction == SOUTH  ?  b  >> 8
        : Direction == EAST   ? (b & ~FileHBB) << 1
        : Direction == WEST   ? (b & ~FileABB) >> 1
        : Direction == NORTH_EAST ? (b & ~FileHBB) << 9
        : Direction == SOUTH_EAST ? (b & ~FileHBB) >> 7
        : Direction == NORTH_WEST ? (b & ~FileABB) << 7
        : Direction == SOUTH_WEST ? (b & ~FileABB) >> 9
        : 0;
}


// double_pawn_attacks_bb() returns the pawn attacks for the given color
// from the squares in the given bitboard.

INLINE Bitboard double_pawn_attacks_bb(Bitboard b, const int C)
{
  return C == WHITE ? shift_bb(NORTH_WEST, b) & shift_bb(NORTH_EAST, b)
                    : shift_bb(SOUTH_WEST, b) & shift_bb(SOUTH_EAST, b);
}


// adjacent_files_bb() returns a bitboard representing all the squares
// on the adjacent files of the given one.

INLINE Bitboard adjacent_files_bb(unsigned f)
{
  return shift_bb(EAST, FileBB[f]) | shift_bb(WEST, FileBB[f]);
}


// between_bb() returns a bitboard representing all the squares between
// the two given ones. For instance, between_bb(SQ_C4, SQ_F7) returns a
// bitboard with the bits for square d5 and e6 set. If s1 and s2 are not
// on the same rank, file or diagonal, 0 is returned.

INLINE Bitboard between_bb(Square s1, Square s2)
{
  return BetweenBB[s1][s2];
}


// forward_ranks_bb() returns a bitboard representing all the squares on
// all the ranks in front of the given one, from the point of view of the
// given color. For instance, forward_ranks_bb(BLACK, RANK_3) will return
// the squares on ranks 1 and 2.

INLINE Bitboard forward_ranks_bb(unsigned c, unsigned r)
{
  return ForwardRanksBB[c][r];
}


// forward_file_bb() returns a bitboard representing all the squares
// along the line in front of the given one, from the point of view of
// the given color:
//     ForwardFileBB[c][s] = forward_ranks_bb(c, rank_of(s)) & file_bb(s)

INLINE Bitboard forward_file_bb(unsigned c, Square s)
{
  return ForwardFileBB[c][s];
}


// pawn_attack_span() returns a bitboard representing all the squares
// that can be attacked by a pawn of the given color when it moves along
// its file, starting from the given square:
//     PawnAttackSpan[c][s] = forward_ranks_bb(c, rank_of(s)) & adjacent_files_bb(s);

INLINE Bitboard pawn_attack_span(unsigned c, Square s)
{
  return PawnAttackSpan[c][s];
}


// passed_pawn_mask() returns a bitboard mask which can be used to test
// if a pawn of the given color and on the given square is a passed pawn:
//     PassedPawnMask[c][s] = pawn_attack_span(c, s) | forward_bb(c, s)

INLINE Bitboard passed_pawn_mask(unsigned c, Square s)
{
  return PassedPawnMask[c][s];
}


// aligned() returns true if square s is on the line determined by move m.

INLINE uint64_t aligned(Move m, Square s)
{
  return ((Bitboard *)LineBB)[m & 4095] & sq_bb(s);
}


// distance() functions return the distance between x and y, defined as
// the number of steps for a king in x to reach y. Works with squares,
// ranks, files.

INLINE int distance(Square x, Square y)
{
  return SquareDistance[x][y];
}

INLINE unsigned distance_f(Square x, Square y)
{
  unsigned f1 = file_of(x), f2 = file_of(y);
  return f1 < f2 ? f2 - f1 : f1 - f2;
}

INLINE unsigned distance_r(Square x, Square y)
{
  unsigned r1 = rank_of(x), r2 = rank_of(y);
  return r1 < r2 ? r2 - r1 : r1 - r2;
}

#if defined(MAGIC_FANCY)
#include "magic-fancy.h"
#elif defined(MAGIC_PLAIN)
#include "magic-plain.h"
#elif defined(MAGIC_BLACK)
#include "magic-black.h"
#elif defined(BMI2_FANCY)
#include "bmi2-fancy.h"
#elif defined(BMI2_PLAIN)
#include "bmi2-plain.h"
#endif

INLINE Bitboard attacks_bb(int pt, Square s, Bitboard occupied)
{
  assert(pt != PAWN);

  switch (pt) {
  case BISHOP:
      return attacks_bb_bishop(s, occupied);
  case ROOK:
      return attacks_bb_rook(s, occupied);
  case QUEEN:
      return attacks_bb_bishop(s, occupied) | attacks_bb_rook(s, occupied);
  default:
      return PseudoAttacks[pt][s];
  }
}


// popcount() counts the number of non-zero bits in a bitboard.

INLINE int popcount(Bitboard b)
{
#ifndef USE_POPCNT

  extern uint8_t PopCnt16[1 << 16];
  union { Bitboard bb; uint16_t u[4]; } v = { b };
  return PopCnt16[v.u[0]] + PopCnt16[v.u[1]] + PopCnt16[v.u[2]] + PopCnt16[v.u[3]];

#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)

  return (int)_mm_popcnt_u64(b);

#else // Assumed gcc or compatible compiler

  return __builtin_popcountll(b);

#endif
}


// lsb() and msb() return the least/most significant bit in a non-zero
// bitboard.

#if defined(__GNUC__)

INLINE int lsb(Bitboard b)
{
  assert(b);
  return __builtin_ctzll(b);
}

INLINE int msb(Bitboard b)
{
  assert(b);
  return 63 ^ __builtin_clzll(b);
}

#elif defined(_MSC_VER)

#if defined(_WIN64)

INLINE Square lsb(Bitboard b)
{
  assert(b);
  unsigned long idx;
  _BitScanForward64(&idx, b);
  return (Square) idx;
}

INLINE Square msb(Bitboard b)
{
  assert(b);
  unsigned long idx;
  _BitScanReverse64(&idx, b);
  return (Square) idx;
}

#else

INLINE Square lsb(Bitboard b)
{
  assert(b);
  unsigned long idx;
  if ((uint32_t)b) {
    _BitScanForward(&idx, (uint32_t)b);
    return idx;
  } else {
    _BitScanForward(&idx, (uint32_t)(b >> 32));
    return idx + 32;
  }
}

INLINE Square msb(Bitboard b)
{
  assert(b);
  unsigned long idx;
  if (b >> 32) {
    _BitScanReverse(&idx, (uint32_t)(b >> 32));
    return idx + 32;
  } else {
    _BitScanReverse(&idx, (uint32_t)b);
    return idx;
  }
}

#endif

#else

#error "Compiler not supported."

#endif


// pop_lsb() finds and clears the least significant bit in a non-zero
// bitboard.

INLINE Square pop_lsb(Bitboard* b)
{
  const Square s = lsb(*b);
  *b &= *b - 1;
  return s;
}


// frontmost_sq() and backmost_sq() return the square corresponding to the
// most/least advanced bit relative to the given color.

INLINE Square frontmost_sq(unsigned c, Bitboard b)
{
  return c == WHITE ? msb(b) : lsb(b);
}

INLINE Square  backmost_sq(unsigned c, Bitboard b)
{
  return c == WHITE ? lsb(b) : msb(b);
}

#endif
