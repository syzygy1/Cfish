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

#ifndef POSITION_H
#define POSITION_H

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>  // For offsetof()
#include <string.h>

#include "bitboard.h"
#include "types.h"

extern Score psqt_psq[2][8][64];
void psqt_init(void);

extern Key zob_exclusion;
void zob_init();

// CheckInfo struct is initialized at constructor time and keeps info used
// to detect if a move gives check.

struct CheckInfo {
  Bitboard dcCandidates;
  Bitboard pinned;
  Bitboard checkSquares[8];
  Square   ksq;
};

typedef struct CheckInfo CheckInfo;

void checkinfo_init(CheckInfo *ci, Pos *pos);


// State struct stores information needed to restore a Position object to
// its previous state when we retract a move.

struct State {
  // Copied when making a move
  Key pawnKey;
  Key materialKey;
  Value nonPawnMaterial[2];
  int castlingRights;
  int rule50;
  int pliesFromNull;
  Score psq;
  Square epSquare;

  // Not copied when making a move
  Key key;
  Bitboard checkersBB;
  int capturedType;
  struct State *previous;
};

typedef struct State State;


// Pos struct stores information regarding the board representation as
// pieces, side to move, hash keys, castling info, etc. The search uses
// the functions do_move() and undo_move() on a Pos struct to traverse
// the search tree.

struct Pos {
  // Board / game representation.
  int board[64];
  Bitboard byTypeBB[7]; // no reason to allocate 8 here
  Bitboard byColorBB[2];
#ifdef PIECELISTS
  int pieceCount[2][8];
  Square pieceList[2][8][16];
  int index[64];
#endif
  int castlingRightsMask[64];
  Square castlingRookSquare[16];
  Bitboard castlingPath[16];
  int sideToMove;
  uint16_t gamePly;
  uint16_t chess960;

  State *st;

  // Relevant mainly to the search of the root position.
  RootMoves *rootMoves;
  State *states;
  uint64_t nodes;
  uint64_t tb_hits;
  int PVIdx;
  int maxPly;
  Depth rootDepth;
  Depth completedDepth;

  // Pointers to thread-specific tables.
  HistoryStats *history;
  MoveStats *counterMoves;
  FromToStats *fromTo;
  PawnEntry *pawnTable;
  MaterialEntry *materialTable;

  // Thread-control data.
  atomic_bool resetCalls;
  int callsCnt;
  int exit, searching;
  int thread_idx;
  pthread_t nativeThread;
  pthread_mutex_t mutex;
  pthread_cond_t sleepCondition;
};

// FEN string input/output
void pos_set(Pos *pos, char *fen, int isChess960);
void pos_fen(Pos *pos, char *fen);
void print_pos(Pos *pos);

Bitboard pos_attackers_to_occ(Pos *pos, Square s, Bitboard occupied);
Bitboard slider_blockers(Pos *pos, Bitboard target, Bitboard sliders, Square s);

int is_legal(Pos *pos, Move m, Bitboard pinned);
int is_pseudo_legal(Pos *pos, Move m);
static int is_capture(Pos *pos, Move m);
static int is_capture_or_promotion(Pos *pos, Move m);
int gives_check(Pos *pos, Move m, const CheckInfo *ci);

// Doing and undoing moves
void do_move(Pos *pos, Move m, int givesCheck);
void undo_move(Pos *pos, Move m);
void do_null_move(Pos *pos);
void undo_null_move(Pos *pos);

// Static exchange evaluation
Value see(Pos *pos, Move m);
Value see_sign(Pos *pos, Move m);

Key key_after(Pos *pos, Move m);
int game_phase(Pos *pos);
int is_draw(Pos *pos);

// Position consistency check, for debugging
int pos_is_ok(Pos *pos, int* failedStep);
//  void flip();

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
#define piece_count(c,p) (pos->pieceCount[c][p])
#define piece_list(c,p) (pos->pieceList[c][p])
#define square_of(c,p) (pos->pieceList[c][p][0])

// Castling
#define can_castle_cr(cr) (pos->st->castlingRights & (cr))
#define can_castle_c(c) can_castle_cr((WHITE_OO | WHITE_OOO) << (2 * (c)))
#define castling_impeded(cr) (pieces() & pos->castlingPath[cr])
#define castling_rook_square(cr) (pos->castlingRookSquare[cr])

// Checking
#define pos_checkers() (pos->st->checkersBB)

// Attacks to/from a given square
#define attackers_to_occ(s,occ) pos_attackers_to_occ(pos,s,occ)
#define attackers_to(s) attackers_to_occ(s,pieces())
#define attacks_from_pawn(s,c) (StepAttacksBB[make_piece(c,PAWN)][s])
#define attacks_from_knight(s) (StepAttacksBB[KNIGHT][s])
#define attacks_from_bishop(s) attacks_bb_bishop(s, pieces())
#define attacks_from_rook(s) attacks_bb_rook(s, pieces())
#define attacks_from_queen(s) (attacks_from_bishop(s)|attacks_from_rook(s))
#define attacks_from_king(s) (StepAttacksBB[KING][s])
#define attacks_from(pc,s) attacks_bb(pc,s,pieces())

// Properties of moves
#define moved_piece(m) (piece_on(from_sq(m)))
#define captured_piece_type() (pos->st->capturedType)

// Accessing hash keys
#define pos_key() (pos->st->key)
#define pos_exclusion_key() (pos_key() ^ zob_exclusion)
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

static inline Bitboard discovered_check_candidates(Pos *pos)
{
  return slider_blockers(pos, pieces_c(pos_stm()), pieces_c(pos_stm()),
                         square_of(pos_stm() ^ 1, KING));
}

static inline Bitboard pinned_pieces(Pos *pos, int c)
{
  return slider_blockers(pos, pieces_c(c), pieces_c(c ^ 1), square_of(c, KING));
}

static inline int pawn_passed(Pos *pos, int c, Square s)
{
  return !(pieces_cp(c ^ 1, PAWN) & passed_pawn_mask(c, s));
}

static inline int advanced_pawn_push(Pos *pos, Move m)
{
  return   type_of_p(moved_piece(m)) == PAWN
        && relative_rank_s(pos_stm(), from_sq(m)) > RANK_4;
}

static inline int opposite_bishops(Pos *pos)
{
  return   piece_count(WHITE, BISHOP) == 1
        && piece_count(BLACK, BISHOP) == 1
        && opposite_colors(square_of(WHITE, BISHOP), square_of(BLACK, BISHOP));
}

static inline int is_capture_or_promotion(Pos *pos, Move m)
{
  assert(move_is_ok(m));
  return type_of_m(m) != NORMAL ? type_of_m(m) != CASTLING : !is_empty(to_sq(m));
}

static inline int is_capture(Pos *pos, Move m)
{
  // Castling is encoded as "king captures the rook"
  assert(move_is_ok(m));
  return (!is_empty(to_sq(m)) && type_of_m(m) != CASTLING) || type_of_m(m) == ENPASSANT;
}

void pos_copy(Pos *dest, Pos *src);

#endif

