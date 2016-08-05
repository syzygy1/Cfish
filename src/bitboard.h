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

#ifndef BITBOARD_H
#define BITBOARD_H

#include <assert.h>

#include "types.h"

void bitbases_init();
int bitbases_probe(Square wksq, Square wpsq, Square bksq, int us);

void bitboards_init();
void print_pretty(Bitboard b);

#define DarkSquares  0xAA55AA55AA55AA55ULL
#define LightSquares 0x55AA55AA55AA55AAULL

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

extern int SquareDistance[64][64];

extern Bitboard SquareBB[64];
extern Bitboard FileBB[8];
extern Bitboard RankBB[8];
extern Bitboard AdjacentFilesBB[8];
extern Bitboard InFrontBB[2][8];
extern Bitboard StepAttacksBB[16][64];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];
extern Bitboard DistanceRingBB[64][8];
extern Bitboard ForwardBB[2][64];
extern Bitboard PassedPawnMask[2][64];
extern Bitboard PawnAttackSpan[2][64];
extern Bitboard PseudoAttacks[8][64];


static inline Bitboard sq_bb(Square s)
{
  return SquareBB[s];
}

static inline uint64_t more_than_one(Bitboard b)
{
  return b & (b - 1);
}


// rank_bb() and file_bb() return a bitboard representing all the squares on
// the given file or rank.

static inline Bitboard rank_bb(int r)
{
  return RankBB[r];
}

static inline Bitboard rank_bb_s(Square s)
{
  return RankBB[rank_of(s)];
}

static inline Bitboard file_bb(int f)
{
  return FileBB[f];
}

static inline Bitboard file_bb_s(Square s)
{
  return FileBB[file_of(s)];
}


// shift_bb() moves a bitboard one step along direction Delta.
static inline Bitboard shift_bb(int Delta, Bitboard b)
{
  return  Delta == DELTA_N  ?  b             << 8 : Delta == DELTA_S  ?  b             >> 8
        : Delta == DELTA_NE ? (b & ~FileHBB) << 9 : Delta == DELTA_SE ? (b & ~FileHBB) >> 7
        : Delta == DELTA_NW ? (b & ~FileABB) << 7 : Delta == DELTA_SW ? (b & ~FileABB) >> 9
        : 0;
}

#define shift_bb_N(b)  ((b) << 8)
#define shift_bb_S(b)  ((b) >> 8)
#define shift_bb_NE(b) (((b) & ~FileHBB) << 9)
#define shift_bb_SE(b) (((b) & ~FileHBB) >> 7)
#define shift_bb_NW(b) (((b) & ~FileABB) << 7)
#define shift_bb_SW(b) (((b) & ~FileABB) >> 9)

// adjacent_files_bb() returns a bitboard representing all the squares
// on the adjacent files of the given one.

static inline Bitboard adjacent_files_bb(int f)
{
  return AdjacentFilesBB[f];
}


// between_bb() returns a bitboard representing all the squares between
// the two given ones. For instance, between_bb(SQ_C4, SQ_F7) returns a
// bitboard with the bits for square d5 and e6 set. If s1 and s2 are not
// on the same rank, file or diagonal, 0 is returned.

static inline Bitboard between_bb(Square s1, Square s2)
{
  return BetweenBB[s1][s2];
}


// in_front_bb() returns a bitboard representing all the squares on all
// the ranks in front of the given one, from the point of view of the
// given color. For instance, in_front_bb(BLACK, RANK_3) will return the
// squares on ranks 1 and 2.

static inline Bitboard in_front_bb(int c, int r)
{
  return InFrontBB[c][r];
}


// forward_bb() returns a bitboard representing all the squares along the
// line in front of the given one, from the point of view of the given
// color:
//        ForwardBB[c][s] = in_front_bb(c, s) & file_bb(s)

static inline Bitboard forward_bb(int c, Square s)
{
  return ForwardBB[c][s];
}


// pawn_attack_span() returns a bitboard representing all the squares
// that can be attacked by a pawn of the given color when it moves along
// its file, starting from the given square:
//       PawnAttackSpan[c][s] = in_front_bb(c, s) & adjacent_files_bb(s);

static inline Bitboard pawn_attack_span(int c, Square s)
{
  return PawnAttackSpan[c][s];
}


// passed_pawn_mask() returns a bitboard mask which can be used to test
// if a pawn of the given color and on the given square is a passed pawn:
//       PassedPawnMask[c][s] = pawn_attack_span(c, s) | forward_bb(c, s)

static inline Bitboard passed_pawn_mask(int c, Square s)
{
  return PassedPawnMask[c][s];
}


// aligned() returns true if the squares s1, s2 and s3 are aligned either
// on a straight or on a diagonal line.

static inline uint64_t aligned(Square s1, Square s2, Square s3)
{
  return LineBB[s1][s2] & sq_bb(s3);
}


// distance() functions return the distance between x and y, defined as
// the number of steps for a king in x to reach y. Works with squares,
// ranks, files.

static inline int distance(Square x, Square y)
{
  return SquareDistance[x][y];
}

static inline int distance_f(Square x, Square y)
{
  int f1 = file_of(x), f2 = file_of(y);
  return f1 < f2 ? f2 - f1 : f1 - f2;
}

static inline int distance_r(Square x, Square y)
{
  int r1 = rank_of(x), r2 = rank_of(y);
  return r1 < r2 ? r2 - r1 : r1 - r2;
}


extern Bitboard RookMasks[64];
extern Bitboard RookMagics[64];
extern unsigned RookShifts[64];
extern Bitboard BishopMasks[64];
extern Bitboard BishopMagics[64];
extern unsigned BishopShifts[64];

// attacks_bb() returns a bitboard representing all the squares attacked
// by a // piece of type Pt (bishop or rook) placed on 's'. The helper
// magic_index() looks up the index using the 'magic bitboards' approach.

static inline unsigned magic_index_bishop(Square s, Bitboard occupied)
{
  if (HasPext)
      return (unsigned)pext(occupied, BishopMasks[s]);

  if (Is64Bit)
      return (unsigned)(((occupied & BishopMasks[s]) * BishopMagics[s])
                           >> BishopShifts[s]);

  unsigned lo = (unsigned)(occupied) & (unsigned)(BishopMasks[s]);
  unsigned hi = (unsigned)(occupied >> 32) & (unsigned)(BishopMasks[s] >> 32);
  return (lo * (unsigned)(BishopMagics[s]) ^ hi * (unsigned)(BishopMagics[s] >> 32)) >> BishopShifts[s];
}

static inline unsigned magic_index_rook(Square s, Bitboard occupied)
{
  if (HasPext)
      return (unsigned)pext(occupied, RookMasks[s]);

  if (Is64Bit)
      return (unsigned)(((occupied & RookMasks[s]) * RookMagics[s])
                           >> RookShifts[s]);

  unsigned lo = (unsigned)(occupied) & (unsigned)(RookMasks[s]);
  unsigned hi = (unsigned)(occupied >> 32) & (unsigned)(RookMasks[s] >> 32);
  return (lo * (unsigned)(RookMagics[s]) ^ hi * (unsigned)(RookMagics[s] >> 32)) >> RookShifts[s];
}

extern Bitboard* RookAttacks[64];
extern Bitboard* BishopAttacks[64];

static inline Bitboard attacks_bb_bishop(Square s, Bitboard occupied)
{
  return BishopAttacks[s][magic_index_bishop(s, occupied)];
}

static inline Bitboard attacks_bb_rook(Square s, Bitboard occupied)
{
  return RookAttacks[s][magic_index_rook(s, occupied)];
}

static inline Bitboard attacks_bb(Piece pc, Square s, Bitboard occupied)
{
  switch (type_of_p(pc)) {
  case BISHOP:
      return attacks_bb_bishop(s, occupied);
  case ROOK:
      return attacks_bb_rook(s, occupied);
  case QUEEN:
      return attacks_bb_bishop(s, occupied) | attacks_bb_rook(s, occupied);
  default:
      return StepAttacksBB[pc][s];
  }
}


// popcount() counts the number of non-zero bits in a bitboard.

static inline int popcount(Bitboard b)
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

static inline Square lsb(Bitboard b)
{
  assert(b);
  return (Square)(__builtin_ctzll(b));
}

static inline Square msb(Bitboard b)
{
  assert(b);
  return (Square)(63 - __builtin_clzll(b));
}

#elif defined(_WIN64) && defined(_MSC_VER)

static inline Square lsb(Bitboard b)
{
  assert(b);
  unsigned long idx;
  _BitScanForward64(&idx, b);
  return (Square) idx;
}

static inline Square msb(Bitboard b)
{
  assert(b);
  unsigned long idx;
  _BitScanReverse64(&idx, b);
  return (Square) idx;
}

#else

#define NO_BSF // Fallback on software implementation for other cases

Square lsb(Bitboard b);
Square msb(Bitboard b);

#endif


// pop_lsb() finds and clears the least significant bit in a non-zero
// bitboard.

static inline Square pop_lsb(Bitboard* b)
{
  const Square s = lsb(*b);
  *b &= *b - 1;
  return s;
}


// frontmost_sq() and backmost_sq() return the square corresponding to the
// most/least advanced bit relative to the given color.

static inline Square frontmost_sq(int c, Bitboard b)
{
  return c == WHITE ? msb(b) : lsb(b);
}

static inline Square  backmost_sq(int c, Bitboard b)
{
  return c == WHITE ? lsb(b) : msb(b);
}

#endif

