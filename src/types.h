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
#include <stdint.h>
#include <stdlib.h>
#ifdef __WIN32__
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

#define MAX_MOVES 256
#define MAX_PLY 128

// A move needs 16 bits to be stored
//
// bit  0- 5: destination square (from 0 to 63)
// bit  6-11: origin square (from 0 to 63)
// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
// NOTE: EN-PASSANT bit is set only when a pawn can be captured
//
// Null move (MOVE_NULL) is encoded as a2a2.

#define MOVE_NULL 65

#define NORMAL    0
#define PROMOTION 1
#define ENPASSANT 2
#define CASTLING  3

#define WHITE 0
#define BLACK 1

#define KING_SIDE  0
#define QUEEN_SIDE 1

#define NO_CASTLING  0
#define WHITE_OO     1
#define WHITE_OOO    2
#define BLACK_OO     4
#define BLACK_OOO    8
#define ANY_CASTLING 15

INLINE int make_castling_right(int c, int s)
{
  return c == WHITE ? s == QUEEN_SIDE ? WHITE_OOO : WHITE_OO
                    : s == QUEEN_SIDE ? BLACK_OOO : BLACK_OO;
}

#define PHASE_ENDGAME 0
#define PHASE_MIDGAME 128
#define MG 0
#define EG 1

#define SCALE_FACTOR_DRAW    0
#define SCALE_FACTOR_ONEPAWN 48
#define SCALE_FACTOR_NORMAL  64
#define SCALE_FACTOR_MAX     128
#define SCALE_FACTOR_NONE    255

#define BOUND_NONE  0
#define BOUND_UPPER 1
#define BOUND_LOWER 2
#define BOUND_EXACT 3

#define VALUE_ZERO      0
#define VALUE_DRAW      0
#define VALUE_KNOWN_WIN 10000
#define VALUE_MATE      32000
#define VALUE_INFINITE  32001
#define VALUE_NONE      32002

#define VALUE_MATE_IN_MAX_PLY  (VALUE_MATE - 2 * MAX_PLY)
#define VALUE_MATED_IN_MAX_PLY (-VALUE_MATE + 2 * MAX_PLY)

#define PawnValueMg   171
#define PawnValueEg   240
#define KnightValueMg 764
#define KnightValueEg 848
#define BishopValueMg 826
#define BishopValueEg 891
#define RookValueMg   1282
#define RookValueEg   1373
#define QueenValueMg  2526
#define QueenValueEg  2646

#define MidgameLimit 15258
#define EndgameLimit 3915

#define PAWN   1
#define KNIGHT 2
#define BISHOP 3
#define ROOK   4
#define QUEEN  5
#define KING   6

#define W_PAWN   1
#define W_KNIGHT 2
#define W_BISHOP 3
#define W_ROOK   4
#define W_QUEEN  5
#define W_KING   6

#define B_PAWN   9
#define B_KNIGHT 10
#define B_BISHOP 11
#define B_ROOK   12
#define B_QUEEN  13
#define B_KING   14

#define ONE_PLY 1

#define DEPTH_ZERO          ( 0 * ONE_PLY)
#define DEPTH_QS_CHECKS     ( 0 * ONE_PLY)
#define DEPTH_QS_NO_CHECKS  (-1 * ONE_PLY)
#define DEPTH_QS_RECAPTURES (-5 * ONE_PLY)

#define DEPTH_NONE (-6 * ONE_PLY)
#define DEPTH_MAX  (MAX_PLY * ONE_PLY)

#define SQ_A1 0
#define SQ_B1 1
#define SQ_C1 2
#define SQ_D1 3
#define SQ_E1 4
#define SQ_F1 5
#define SQ_G1 6
#define SQ_H1 7
#define SQ_A2 8
#define SQ_B2 9
#define SQ_C2 10
#define SQ_D2 11
#define SQ_E2 12
#define SQ_F2 13
#define SQ_G2 14
#define SQ_H2 15
#define SQ_A3 16
#define SQ_B3 17
#define SQ_C3 18
#define SQ_D3 19
#define SQ_E3 20
#define SQ_F3 21
#define SQ_G3 22
#define SQ_H3 23
#define SQ_A4 24
#define SQ_B4 25
#define SQ_C4 26
#define SQ_D4 27
#define SQ_E4 28
#define SQ_F4 29
#define SQ_G4 30
#define SQ_H4 31
#define SQ_A5 32
#define SQ_B5 33
#define SQ_C5 34
#define SQ_D5 35
#define SQ_E5 36
#define SQ_F5 37
#define SQ_G5 38
#define SQ_H5 39
#define SQ_A6 40
#define SQ_B6 41
#define SQ_C6 42
#define SQ_D6 43
#define SQ_E6 44
#define SQ_F6 45
#define SQ_G6 46
#define SQ_H6 47
#define SQ_A7 48
#define SQ_B7 49
#define SQ_C7 50
#define SQ_D7 51
#define SQ_E7 52
#define SQ_F7 53
#define SQ_G7 54
#define SQ_H7 55
#define SQ_A8 56
#define SQ_B8 57
#define SQ_C8 58
#define SQ_D8 59
#define SQ_E8 60
#define SQ_F8 61
#define SQ_G8 62
#define SQ_H8 63

#define SQ_NONE 64

#define DELTA_N  8
#define DELTA_E  1
#define DELTA_S -8
#define DELTA_W -1

#define DELTA_NN (DELTA_N + DELTA_N)
#define DELTA_NE (DELTA_N + DELTA_E)
#define DELTA_SE (DELTA_S + DELTA_E)
#define DELTA_SS (DELTA_S + DELTA_S)
#define DELTA_SW (DELTA_S + DELTA_W)
#define DELTA_NW (DELTA_N + DELTA_W)

#define FILE_A 0
#define FILE_B 1
#define FILE_C 2
#define FILE_D 3
#define FILE_E 4
#define FILE_F 5
#define FILE_G 6
#define FILE_H 7

#define RANK_1 0
#define RANK_2 1
#define RANK_3 2
#define RANK_4 3
#define RANK_5 4
#define RANK_6 5
#define RANK_7 6
#define RANK_8 7

// For now we keep the following types, as we might want to change them
// in the future.

typedef uint32_t Move;
typedef int32_t Phase;
typedef int32_t Value;
typedef uint32_t Piece;
typedef int32_t Depth;
typedef uint32_t Square;

// Score type stores a middlegame and an endgame value in a single integer.
// The endgame value goes in the upper 16 bits, the middlegame value in
// the lower 16 bits.

typedef uint32_t Score;

#define SCORE_ZERO 0

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
  int s = (int)(s1) ^ (int)(s2);
  return ((s >> 3) ^ s) & 1;
}

typedef struct Pos Pos;
typedef struct LimitsType LimitsType;
typedef struct RootMoves RootMoves;
typedef struct PawnEntry PawnEntry;
typedef struct MaterialEntry MaterialEntry;

typedef Move CounterMoveStat[16][64];
typedef int PieceToHistory[16][64];
typedef PieceToHistory CounterMoveHistoryStat[16][64];
typedef int ButterflyHistory[2][4096];

struct ExtMove {
  Move move;
  int value;
};

typedef struct ExtMove ExtMove;

struct PSQT {
  Score psq[16][64];
};

extern struct PSQT psqt;

#ifndef __WIN32__
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef __WIN32__
#define FMT_Z "z"
#else
#define FMT_Z "I"
#endif

#ifdef NDEBUG
#define assume(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
#define assume(x) assert(x)
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#endif

