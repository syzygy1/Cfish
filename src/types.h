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

#ifndef TYPES_H
#define TYPES_H

#include "config.h"

// When compiling with provided Makefile (e.g. for Linux and OSX),
// configuration is done automatically. To get started type 'make help'.
//
// When Makefile is not used (e.g. with Microsoft Visual Studio) some
// switches need to be set manually:
//
// -DNDEBUG      | Disable debugging mode. Always use this for release.
//
// -DNO_PREFETCH | Disable use of prefetch asm-instruction. You may need
//               | this to run on some very old machines.
//
// -DUSE_POPCNT  | Add runtime support for use of popcnt asm-instruction.
//               | Works only in 64-bit mode and requires hardware with
//               | popcnt support.
//
// -DUSE_PEXT    | Add runtime support for use of pext asm-instruction.
//               | Works only in 64-bit mode and requires hardware with
//               | pext support.

#ifndef NDEBUG
#include <assert.h>
#endif
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define INLINE static inline __attribute__((always_inline))

// Declaring pure functions as pure seems not to help. (Investigate later.)
//#define PURE __attribute__((pure))
#define PURE

#define SMALL __attribute__((optimize("Os")))

// Predefined macros hell:
//
// __GNUC__           Compiler is gcc, Clang or Intel on Linux
// __INTEL_COMPILER   Compiler is Intel
// _MSC_VER           Compiler is MSVC or Intel on Windows
// _WIN32             Building on Windows (any)
// _WIN64             Building on Windows 64 bit

#if defined(_WIN64) && defined(_MSC_VER) // No Makefile used
#  include <intrin.h> // Microsoft header for _BitScanForward64()
#  define IS_64BIT
#endif

#if defined(USE_POPCNT) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
#  include <nmmintrin.h> // Intel and Microsoft header for _mm_popcnt_u64()
#endif

#if !defined(NO_PREFETCH) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
#  include <xmmintrin.h> // Intel and Microsoft header for _mm_prefetch()
#endif

#if defined(USE_PEXT)
#  include <immintrin.h> // Header for _pext_u64() intrinsic
#  define pext(b, m) _pext_u64(b, m)
#else
#  define pext(b, m) (0)
#endif

#ifdef USE_POPCNT
#define HasPopCnt 1
#else
#define HasPopCnt 0
#endif

#ifdef USE_PEXT
#define HasPext 1
#else
#define HasPext 0
#endif

#ifdef IS_64BIT
#define Is64Bit 1
#else
#define Is64Bit 0
#endif

#ifdef NUMA
#define HasNuma 1
#else
#define HasNuma 0
#endif

typedef uint64_t Key;
typedef uint64_t Bitboard;

enum { MAX_MOVES = 256, MAX_PLY = 128 };

// A move needs 16 bits to be stored
//
// bit  0- 5: destination square (from 0 to 63)
// bit  6-11: origin square (from 0 to 63)
// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
// NOTE: EN-PASSANT bit is set only when a pawn can be captured
//
// Null move (MOVE_NULL) is encoded as a2a2.

enum { MOVE_NONE = 0, MOVE_NULL = 65 };

enum { NORMAL, PROMOTION, ENPASSANT, CASTLING };

enum { WHITE, BLACK };

enum { KING_SIDE, QUEEN_SIDE };

enum {
  NO_CASTLING = 0, WHITE_OO = 1, WHITE_OOO = 2,
  BLACK_OO = 4, BLACK_OOO = 8, ANY_CASTLING = 15
};

INLINE int make_castling_right(int c, int s)
{
  return c == WHITE ? s == QUEEN_SIDE ? WHITE_OOO : WHITE_OO
                    : s == QUEEN_SIDE ? BLACK_OOO : BLACK_OO;
}

enum { PHASE_ENDGAME = 0, PHASE_MIDGAME = 128 };
enum { MG, EG };

enum {
  SCALE_FACTOR_DRAW = 0, SCALE_FACTOR_NORMAL = 64,
  SCALE_FACTOR_MAX = 128, SCALE_FACTOR_NONE = 255
};

enum { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT };

enum {
  VALUE_ZERO = 0, VALUE_DRAW = 0,
  VALUE_KNOWN_WIN = 10000, VALUE_MATE = 32000,
  VALUE_INFINITE = 32001, VALUE_NONE = 32002
};

#ifdef LONG_MATES
enum { MAX_MATE_PLY = 600 };
#else
enum { MAX_MATE_PLY = MAX_PLY };
#endif

enum {
  VALUE_MATE_IN_MAX_PLY  = ( VALUE_MATE - MAX_MATE_PLY - MAX_PLY),
  VALUE_MATED_IN_MAX_PLY = (-VALUE_MATE + MAX_MATE_PLY + MAX_PLY)
};

enum {
  PawnValueMg   = 128,   PawnValueEg   = 213,
  KnightValueMg = 782,   KnightValueEg = 865,
  BishopValueMg = 830,   BishopValueEg = 918,
  RookValueMg   = 1289,  RookValueEg   = 1378,
  QueenValueMg  = 2529,  QueenValueEg  = 2687,

  MidgameLimit  = 15258, EndgameLimit = 3915
};

enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };

enum {
  W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
};

enum {
  ONE_PLY = 1,
  DEPTH_ZERO          =  0 * ONE_PLY,
  DEPTH_QS_CHECKS     =  0 * ONE_PLY,
  DEPTH_QS_NO_CHECKS  = -1 * ONE_PLY,
  DEPTH_QS_RECAPTURES = -5 * ONE_PLY,
  DEPTH_NONE = -6 * ONE_PLY,
  DEPTH_MAX = MAX_PLY * ONE_PLY,
};

enum {
  SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
  SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
  SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
  SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
  SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
  SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
  SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
  SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
  SQ_NONE
};

enum {
  NORTH = 8, EAST = 1, SOUTH = -8, WEST = -1,
  NORTH_EAST = NORTH + EAST, SOUTH_EAST = SOUTH + EAST,
  NORTH_WEST = NORTH + WEST, SOUTH_WEST = SOUTH + WEST,
};

enum { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };

enum { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

typedef uint32_t Move;
typedef int32_t Phase;
typedef int32_t Value;
typedef uint32_t Color;
typedef uint32_t Piece;
typedef uint32_t PieceType;
typedef int32_t Depth;
typedef uint32_t Square;
typedef uint32_t File;
typedef uint32_t Rank;

// Score type stores a middlegame and an endgame value in a single integer.
// The endgame value goes in the upper 16 bits, the middlegame value in
// the lower 16 bits.

typedef uint32_t Score;

enum { SCORE_ZERO };

#define make_score(mg,eg) ((((unsigned)(eg))<<16) + (mg))

// Casting an out-of-range value to int16_t is implementation-defined, but
// we assume the implementation does the right thing.
INLINE Value eg_value(Score s)
{
  return (int16_t)((s + 0x8000) >> 16);
}

INLINE Value mg_value(Score s)
{
  return (int16_t)s;
}

/// Division of a Score must be handled separately for each tEerm
INLINE Score score_divide(Score s, int i)
{
  return make_score(mg_value(s) / i, eg_value(s) / i);
}

extern Value PieceValue[2][16];

extern uint32_t NonPawnPieceValue[16];

#define SQUARE_FLIP(s) (sq ^ 0x38)

#define mate_in(ply) ((Value)(VALUE_MATE - (ply)))
#define mated_in(ply) ((Value)(-VALUE_MATE + (ply)))
#define make_square(f,r) ((Square)(((r) << 3) + (f)))
#define make_piece(c,pt) ((Piece)(((c) << 3) + (pt)))
#define type_of_p(p) ((p) & 7)
#define color_of(p) ((p) >> 3)
// since Square is now unsigned, no need to test for s >= SQ_A1
#define square_is_ok(s) ((Square)(s) <= SQ_H8)
#define file_of(s) ((s) & 7)
#define rank_of(s) ((s) >> 3)
#define relative_square(c,s) ((Square)((s) ^ ((c) * 56)))
#define relative_rank(c,r) ((r) ^ ((c) * 7))
#define relative_rank_s(c,s) relative_rank(c,rank_of(s))
#define pawn_push(c) ((c) == WHITE ? 8 : -8)
#define from_sq(m) ((Square)((m)>>6) & 0x3f)
#define to_sq(m) ((Square)((m) & 0x3f))
#define from_to(m) ((m) & 0xfff)
#define type_of_m(m) ((m) >> 14)
#define promotion_type(m) ((((m)>>12) & 3) + KNIGHT)
#define make_move(from,to) ((Move)((to) | ((from) << 6)))
#define make_promotion(from,to,pt) ((Move)((to) | ((from)<<6) | (PROMOTION<<14) | (((pt)-KNIGHT)<<12)))
#define make_enpassant(from,to) ((Move)((to) | ((from)<<6) | (ENPASSANT<<14)))
#define make_castling(from,to) ((Move)((to) | ((from)<<6) | (CASTLING<<14)))
#define move_is_ok(m) (from_sq(m) != to_sq(m))

INLINE int opposite_colors(Square s1, Square s2)
{
  Square s = s1 ^ s2;
  return ((s >> 3) ^ s) & 1;
}

typedef struct Pos Pos;
typedef struct LimitsType LimitsType;
typedef struct RootMove RootMove;
typedef struct RootMoves RootMoves;
typedef struct PawnEntry PawnEntry;
typedef struct MaterialEntry MaterialEntry;

typedef Move CounterMoveStat[16][64];
typedef int16_t PieceToHistory[16][64];
typedef PieceToHistory CounterMoveHistoryStat[16][64];
typedef int16_t ButterflyHistory[2][4096];
typedef int16_t CapturePieceToHistory[16][64][8];

struct ExtMove {
  Move move;
  int value;
};

typedef struct ExtMove ExtMove;

struct PSQT {
  Score psq[16][64];
};

extern struct PSQT psqt;

#ifndef _WIN32
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifdef NDEBUG
#define assume(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
#define assume(x) assert(x)
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#endif
