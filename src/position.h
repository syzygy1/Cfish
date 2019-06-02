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

#ifndef POSITION_H
#define POSITION_H

#include <assert.h>
#ifndef _WIN32
#include <pthread.h>
#endif
#include <stdatomic.h>
#include <stddef.h>  // For offsetof()
#include <string.h>

#include "bitboard.h"
#include "types.h"

extern const char PieceToChar[];
extern Key matKey[16];

struct Zob {
  Key psq[16][64];
  Key enpassant[8];
  Key castling[16];
  Key side, noPawns;
};

extern struct Zob zob;

void psqt_init(void);
void zob_init(void);

// Stack struct stores information needed to restore a Pos struct to
// its previous state when we retract a move.

struct Stack {
  // Copied when making a move
  Key pawnKey;
  Key materialKey;
  union {
    struct {
      Score psq;
      union {
        uint16_t nonPawnMaterial[2];
        uint32_t nonPawn;
      };
    };
    uint64_t psqnpm;
  };
  uint8_t castlingRights;
  union {
    struct {
      uint8_t pliesFromNull;
      uint8_t rule50;
    };
    uint16_t plyCounters;
  };

  // Not copied when making a move
  uint8_t capturedPiece;
  uint8_t epSquare;
  Key key;
  Bitboard checkersBB;

  // Original search stack data
  Move* pv;
  PieceToHistory *history;
  uint8_t ply;
  Move currentMove;
  Move excludedMove;
  Move killers[2];
  Value staticEval;
  Value statScore;
  int moveCount;

  // MovePicker data
  Move countermove;
  Depth depth;
  Move ttMove;
  Value threshold;
  Move mpKillers[2];
  uint8_t stage;
  uint8_t recaptureSquare;
  ExtMove *cur, *endMoves, *endBadCaptures;

  // CheckInfo data
  Bitboard blockersForKing[2];
  union {
    struct {
      Bitboard pinnersForKing[2];
    };
    struct {
      Bitboard dummy;           // pinnersForKing[WHITE]
      Bitboard checkSquares[7]; // element 0 is pinnersForKing[BLACK]
    };
  };
  Square ksq;
};

typedef struct Stack Stack;

#define StateCopySize offsetof(Stack, capturedPiece)
#define StateSize offsetof(Stack, pv)
#define SStackBegin(st) (&st.pv)
#define SStackSize (offsetof(Stack, countermove) - offsetof(Stack, pv))


// Pos struct stores information regarding the board representation as
// pieces, side to move, hash keys, castling info, etc. The search uses
// the functions do_move() and undo_move() on a Pos struct to traverse
// the search tree.

struct Pos {
  Stack *st;
  // Board / game representation.
  Bitboard byTypeBB[7]; // no reason to allocate 8 here
  Bitboard byColorBB[2];
  uint32_t sideToMove;
  uint8_t chess960;
  uint8_t board[64];
#ifdef PEDANTIC
  uint8_t pieceCount[16];
  uint8_t pieceList[256];
  uint8_t index[64];
  uint8_t castlingRightsMask[64];
  uint8_t castlingRookSquare[16];
  Bitboard castlingPath[16];
#endif
  Key rootKeyFlip;
  uint16_t gamePly;
  uint8_t hasRepeated;

  ExtMove *moveList;

  // Relevant mainly to the search of the root position.
  RootMoves *rootMoves;
  Stack *stack;
  uint64_t nodes;
  uint64_t tbHits;
  int pvIdx, pvLast;
  int selDepth, nmpPly, nmpOdd;
  Depth rootDepth;
  Depth completedDepth;
  Score contempt;

  // Pointers to thread-specific tables.
  CounterMoveStat *counterMoves;
  ButterflyHistory *history;
  CapturePieceToHistory *captureHistory;
  PawnEntry *pawnTable;
  MaterialEntry *materialTable;
  CounterMoveHistoryStat *counterMoveHistory;

  // Thread-control data.
  uint64_t bestMoveChanges;
  atomic_bool resetCalls;
  int callsCnt;
  int action;
  int threadIdx;
#ifndef _WIN32
  pthread_t nativeThread;
  pthread_mutex_t mutex;
  pthread_cond_t sleepCondition;
#else
  HANDLE nativeThread;
  HANDLE startEvent, stopEvent;
#endif
};

// FEN string input/output
void pos_set(Pos *pos, char *fen, int isChess960);
void pos_fen(const Pos *pos, char *fen);
void print_pos(Pos *pos);

//PURE Bitboard pos_attackers_to_occ(const Pos *pos, Square s, Bitboard occupied);
PURE Bitboard slider_blockers(const Pos *pos, Bitboard sliders, Square s,
                              Bitboard *pinners);

PURE int is_legal(const Pos *pos, Move m);
PURE int is_pseudo_legal(const Pos *pos, Move m);
PURE int gives_check_special(const Pos *pos, Stack *st, Move m);

// Doing and undoing moves
void do_move(Pos *pos, Move m, int givesCheck);
void undo_move(Pos *pos, Move m);
void do_null_move(Pos *pos);
INLINE void undo_null_move(Pos *pos);

// Static exchange evaluation
PURE Value see_sign(const Pos *pos, Move m);
PURE Value see_test(const Pos *pos, Move m, int value);

PURE Key key_after(const Pos *pos, Move m);
PURE int is_draw(const Pos *pos);
PURE bool has_game_cycle(const Pos *pos, int ply);

// Position representation
#define pieces() (pos->byTypeBB[0])
#define pieces_p(p) (pos->byTypeBB[p])
#define pieces_pp(p1,p2) (pos->byTypeBB[p1] | pos->byTypeBB[p2])
#define pieces_c(c) (pos->byColorBB[c])
#define pieces_cp(c,p) (pieces_p(p) & pieces_c(c))
#define pieces_cpp(c,p1,p2) (pieces_pp(p1,p2) & pieces_c(c))
#define piece_on(s) (pos->board[s])
#define ep_square() (pos->st->epSquare)
#define is_empty(s) (!piece_on(s))
#ifdef PEDANTIC
#define piece_count(c,p) (pos->pieceCount[8 * (c) + (p)] - (8*(c)+(p)) * 16)
#define piece_list(c,p) (&pos->pieceList[16 * (8 * (c) + (p))])
#define square_of(c,p) (pos->pieceList[16 * (8 * (c) + (p))])
#define loop_through_pieces(c,p,s) \
  const uint8_t *pl = piece_list(c,p); \
  while ((s = *pl++) != SQ_NONE)
#else
#define piece_count(c,p) (popcount(pieces_cp(c, p)))
#define square_of(c,p) (lsb(pieces_cp(c,p)))
#define loop_through_pieces(c,p,s) \
  Bitboard pcs = pieces_cp(c,p); \
  while (pcs && (s = pop_lsb(&pcs), 1))
#endif
#define piece_count_mk(c, p) (((pos_material_key()) >> (20 * (c) + 4 * (p) + 4)) & 15)

// Castling
#define can_castle_cr(cr) (pos->st->castlingRights & (cr))
#define can_castle_c(c) can_castle_cr((WHITE_OO | WHITE_OOO) << (2 * (c)))
#define can_castle_any() (pos->st->castlingRights)
#ifdef PEDANTIC
#define castling_impeded(cr) (pieces() & pos->castlingPath[cr])
#define castling_rook_square(cr) (pos->castlingRookSquare[cr])
#else
#define castling_impeded(cr) (pieces() & CastlingPath[cr])
#define castling_rook_square(cr) (CastlingRookSquare[cr])
#endif

// Checking
#define pos_checkers() (pos->st->checkersBB)

// Attacks to/from a given square
#define attackers_to_occ(s,occ) pos_attackers_to_occ(pos,s,occ)
#define attackers_to(s) attackers_to_occ(s,pieces())
#define attacks_from_pawn(s,c) (PawnAttacks[c][s])
#define attacks_from_knight(s) (PseudoAttacks[KNIGHT][s])
#define attacks_from_bishop(s) attacks_bb_bishop(s, pieces())
#define attacks_from_rook(s) attacks_bb_rook(s, pieces())
#define attacks_from_queen(s) (attacks_from_bishop(s)|attacks_from_rook(s))
#define attacks_from_king(s) (PseudoAttacks[KING][s])
#define attacks_from(pc,s) attacks_bb(pc,s,pieces())

// Properties of moves
#define moved_piece(m) (piece_on(from_sq(m)))
#define captured_piece() (pos->st->capturedPiece)

// Accessing hash keys
#define pos_key() (pos->st->key)
#define pos_material_key() (pos->st->materialKey)
#define pos_pawn_key() (pos->st->pawnKey)

// Other properties of the position
#define pos_stm() (pos->sideToMove)
#define pos_game_ply() (pos->gamePly)
#define is_chess960() (pos->chess960)
#define pos_nodes_searched() (pos->nodes)
#define pos_rule50_count() (pos->st->rule50)
#define pos_psq_score() (pos->st->psq)
#define pos_non_pawn_material(c) (pos->st->nonPawnMaterial[c])
#define pos_pawns_only() (!pos->st->nonPawn)

INLINE Bitboard blockers_for_king(const Pos *pos, uint32_t c)
{
  return pos->st->blockersForKing[c];
}

INLINE int pawn_passed(const Pos *pos, uint32_t c, Square s)
{
  return !(pieces_cp(c ^ 1, PAWN) & passed_pawn_mask(c, s));
}

INLINE int advanced_pawn_push(const Pos *pos, Move m)
{
  return   type_of_p(moved_piece(m)) == PAWN
        && relative_rank_s(pos_stm(), from_sq(m)) > RANK_4;
}

INLINE int opposite_bishops(const Pos *pos)
{
#if 0
  return   piece_count(WHITE, BISHOP) == 1
        && piece_count(BLACK, BISHOP) == 1
        && opposite_colors(square_of(WHITE, BISHOP), square_of(BLACK, BISHOP));
#elif 0
  return   (pos_material_key() & 0xf0000f0000) == 0x1000010000
        && (pieces_p(BISHOP) & DarkSquares)
        && (pieces_p(BISHOP) & DarkSquares) != pieces_p(BISHOP);
#else
  return   piece_count(WHITE, BISHOP) == 1
        && piece_count(BLACK, BISHOP) == 1
        && (pieces_p(BISHOP) & DarkSquares)
        && (pieces_p(BISHOP) & ~DarkSquares);
#endif
}

INLINE int is_capture_or_promotion(const Pos *pos, Move m)
{
  assert(move_is_ok(m));
  return type_of_m(m) != NORMAL ? type_of_m(m) != CASTLING : !is_empty(to_sq(m));
}

INLINE int is_capture(const Pos *pos, Move m)
{
  // Castling is encoded as "king captures the rook"
  assert(move_is_ok(m));
  return (!is_empty(to_sq(m)) && type_of_m(m) != CASTLING) || type_of_m(m) == ENPASSANT;
}

INLINE int gives_check(const Pos *pos, Stack *st, Move m)
{
  return  type_of_m(m) == NORMAL && !(blockers_for_king(pos, pos_stm() ^ 1) & pieces_c(pos_stm()))
        ? !!(st->checkSquares[type_of_p(moved_piece(m))] & sq_bb(to_sq(m)))
        : gives_check_special(pos, st, m);
}

void pos_set_check_info(Pos *pos);

// undo_null_move is used to undo a null move.

INLINE void undo_null_move(Pos *pos)
{
  assert(!pos_checkers());

  pos->st--;
  pos->sideToMove ^= 1;
}

// Inlining this seems to slow down.
#if 0
// slider_blockers() returns a bitboard of all pieces that are blocking
// attacks on the square 's' from 'sliders'. A piece blocks a slider if
// removing that piece from the board would result in a position where
// square 's' is attacked. Both pinned pieces and discovered check
// candidates are slider blockers and are calculated by calling this
// function.

INLINE Bitboard slider_blockers(const Pos *pos, Bitboard sliders, Square s,
                                Bitboard *pinners)
{
  Bitboard result = 0, snipers;
  *pinners = 0;

  // Snipers are sliders that attack square 's'when a piece removed.
  snipers = (  (PseudoAttacks[ROOK  ][s] & pieces_pp(QUEEN, ROOK))
             | (PseudoAttacks[BISHOP][s] & pieces_pp(QUEEN, BISHOP))) & sliders;

  while (snipers) {
    Square sniperSq = pop_lsb(&snipers);
    Bitboard b = between_bb(s, sniperSq) & pieces();

    if (!more_than_one(b)) {
      result |= b;
      if (b & pieces_c(color_of(piece_on(s))))
        *pinners |= sq_bb(sniperSq);
    }
  }
  return result;
}
#endif

// attackers_to() computes a bitboard of all pieces which attack a given
// square. Slider attacks use the occupied bitboard to indicate occupancy.

INLINE Bitboard pos_attackers_to_occ(const Pos *pos, Square s,
                                     Bitboard occupied)
{
  return  (attacks_from_pawn(s, BLACK)    & pieces_cp(WHITE, PAWN))
        | (attacks_from_pawn(s, WHITE)    & pieces_cp(BLACK, PAWN))
        | (attacks_from_knight(s)         & pieces_p(KNIGHT))
        | (attacks_bb_rook(s, occupied)   & pieces_pp(ROOK,   QUEEN))
        | (attacks_bb_bishop(s, occupied) & pieces_pp(BISHOP, QUEEN))
        | (attacks_from_king(s)           & pieces_p(KING));
}

#endif
