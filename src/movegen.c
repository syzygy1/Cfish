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

#if 1
#include "movegen2.c"
#else
#include <assert.h>

#include "movegen.h"
#include "position.h"

static int free_path(Pos *pos, int from, int to, Bitboard enemies)
{
  int d = from > to ? 1 : -1;

  for (int s = to; s != from; s += d)
    if (attackers_to(s) & enemies)
      return 0;

  return 1;
}

// FIXME: squares to be checked can be pregenerated.
// Idem for the castling moves themselves.
// instead of special chess960, check for unpinned rook?
static ExtMove *gen_castling_white(Pos *pos, ExtMove *list)
{
  if (can_castle_cr(WHITE_OO) && !castling_impeded(WHITE_OO)) {
    Square kfrom = square_of(WHITE, KING);
    if (free_path(pos, kfrom, SQ_G1, pieces_c(BLACK))) {
      Square rfrom = castling_rook_square(WHITE_OO);
      (list++)->move = make_castling(kfrom, rfrom);
    }
  }
  if (can_castle_cr(WHITE_OOO) && !castling_impeded(WHITE_OOO)) {
    Square kfrom = square_of(WHITE, KING);
    if (free_path(pos, kfrom, SQ_C1, pieces_c(BLACK)) && (!is_chess960()
                       || !(pieces_cpp(BLACK, ROOK, QUEEN) & sq_bb(SQ_A1)))) {
      Square rfrom = castling_rook_square(WHITE_OOO);
      (list++)->move = make_castling(kfrom, rfrom);
    }
  }

  return list;
}

static ExtMove *gen_castling_black(Pos *pos, ExtMove *list)
{
  if (can_castle_cr(BLACK_OO) && !castling_impeded(BLACK_OO)) {
    Square kfrom = square_of(BLACK, KING);
    if (free_path(pos, kfrom, SQ_G8, pieces_c(WHITE))) {
      Square rfrom = castling_rook_square(BLACK_OO);
      (list++)->move = make_castling(kfrom, rfrom);
    }
  }
  if (can_castle_cr(BLACK_OOO) && !castling_impeded(BLACK_OOO)) {
    Square kfrom = square_of(BLACK, KING);
    if (free_path(pos, kfrom, SQ_C8, pieces_c(WHITE)) && (!is_chess960()
                       || !(pieces_cpp(WHITE, ROOK, QUEEN) & sq_bb(SQ_A8)))) {
      Square rfrom = castling_rook_square(BLACK_OOO);
      (list++)->move = make_castling(kfrom, rfrom);
    }
  }

  return list;
}

static ExtMove *gen_pawn_pushes_white(Pos *pos, ExtMove *list)
{
  Bitboard pawns, b1, b2, empty;

  pawns = pieces_cp(WHITE, PAWN);
  empty = ~pieces();

  // Single and double regular pushes.
  b1 = shift_bb_N(pawns & ~Rank7BB) & empty;
  b2 = shift_bb_N(b1 & Rank3BB) & empty;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to - DELTA_N, to);
  }
  while (b2) {
    Square to = pop_lsb(&b2);
    (list++)->move = make_move(to - DELTA_NN, to);
  }

  // Underpromotions.
  if (pawns & Rank7BB) {
    b1 = shift_bb_NE(pawns & Rank7BB) & pieces_c(BLACK);
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_NE, to, ROOK);
      (list++)->move = make_promotion(to - DELTA_NE, to, BISHOP);
      (list++)->move = make_promotion(to - DELTA_NE, to, KNIGHT);
    }

    b1 = shift_bb_NW(pawns & Rank7BB) & pieces_c(BLACK);
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_NW, to, ROOK);
      (list++)->move = make_promotion(to - DELTA_NW, to, BISHOP);
      (list++)->move = make_promotion(to - DELTA_NW, to, KNIGHT);
    }

    b1 = shift_bb_N(pawns & Rank7BB) & empty;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_N, to, ROOK);
      (list++)->move = make_promotion(to - DELTA_N, to, BISHOP);
      (list++)->move = make_promotion(to - DELTA_N, to, KNIGHT);
    }
  }

  return list;
}

static ExtMove *gen_pawn_pushes_black(Pos *pos, ExtMove *list)
{
  Bitboard pawns, b1, b2, empty;

  pawns = pieces_cp(BLACK, PAWN);
  empty = ~pieces();

  // Single and double regular pushes.
  b1 = shift_bb_S(pawns & ~Rank2BB) & empty;
  b2 = shift_bb_S(b1 & Rank6BB) & empty;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to - DELTA_S, to);
  }
  while (b2) {
    Square to = pop_lsb(&b2);
    (list++)->move = make_move(to - DELTA_SS, to);
  }

  // Underpromotions.
  if (pawns & Rank2BB) {
    b1 = shift_bb_SW(pawns & Rank2BB) & pieces_c(WHITE);
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_SW, to, ROOK);
      (list++)->move = make_promotion(to - DELTA_SW, to, BISHOP);
      (list++)->move = make_promotion(to - DELTA_SW, to, KNIGHT);
    }

    b1 = shift_bb_SE(pawns & Rank2BB) & pieces_c(WHITE);
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_SE, to, ROOK);
      (list++)->move = make_promotion(to - DELTA_SE, to, BISHOP);
      (list++)->move = make_promotion(to - DELTA_SE, to, KNIGHT);
    }

    b1 = shift_bb_S(pawns & Rank2BB) & empty;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_S, to, ROOK);
      (list++)->move = make_promotion(to - DELTA_S, to, BISHOP);
      (list++)->move = make_promotion(to - DELTA_S, to, KNIGHT);
    }
  }

  return list;
}

static ExtMove *gen_pawn_captures_white(Pos *pos, ExtMove *list)
{
  Bitboard pawns, target, b1;

  pawns = pieces_cp(WHITE, PAWN);
  target = pieces_c(BLACK);

  // Queen promotions.
  if (pawns & Rank7BB) {
    b1 = shift_bb_NE(pawns & Rank7BB) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_NE, to, QUEEN);
    }

    b1 = shift_bb_NW(pawns & Rank7BB) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_NW, to, QUEEN);
    }

    b1 = shift_bb_N(pawns & Rank7BB) & ~pieces();
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_N, to, QUEEN);
    }
  }

  // Regular pawn captures.
  b1 = shift_bb_NE(pawns & ~Rank7BB) & target;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to - DELTA_NE, to);
  }

  b1 = shift_bb_NW(pawns & ~Rank7BB) & target;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to - DELTA_NW, to);
  }

  // En passant.
  if (ep_square() != 0) {
     assert(rank_of(ep_square()) == relative_rank(WHITE, RANK_6));

     b1 = pawns & attacks_from_pawn(ep_square(), BLACK);

     assert(b1);

     while (b1)
       (list++)->move = make_enpassant(pop_lsb(&b1), ep_square());
  }

  return list;
}

static ExtMove *gen_pawn_captures_black(Pos *pos, ExtMove *list)
{
  Bitboard pawns, target, b1;

  pawns = pieces_cp(BLACK, PAWN);
  target = pieces_c(WHITE);

  // Queen promotions.
  if (pawns & Rank2BB) {
    b1 = shift_bb_SW(pawns & Rank2BB) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_SW, to, QUEEN);
    }

    b1 = shift_bb_SE(pawns & Rank2BB) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_SE, to, QUEEN);
    }

    b1 = shift_bb_S(pawns & Rank2BB) & ~pieces();
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_S, to, QUEEN);
    }
  }

  // Regular pawn captures.
  b1 = shift_bb_SW(pawns & ~Rank2BB) & target;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to - DELTA_SW, to);
  }

  b1 = shift_bb_SE(pawns & ~Rank2BB) & target;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to - DELTA_SE, to);
  }

  // En passant.
  if (ep_square() != 0) {
     assert(rank_of(ep_square()) == relative_rank(BLACK, RANK_6));

     b1 = pawns & attacks_from_pawn(ep_square(), WHITE);

     assert(b1);

     while (b1)
       (list++)->move = make_enpassant(pop_lsb(&b1), ep_square());
  }

  return list;
}

INLINE ExtMove *gen_pawn_evasions_white(Pos *pos, ExtMove *list,
                                               Bitboard target, Square checksq)
{
  Bitboard pawns, b1, b2;

  pawns = pieces_cp(WHITE, PAWN);

  // Pawn captures of attacking piece.
  b1 = attacks_from_pawn(checksq, BLACK) & pawns;
  while (b1) {
    Square from = pop_lsb(&b1);
    if (checksq < SQ_A8)
      (list++)->move = make_move(from, checksq);
    else {
      (list++)->move = make_promotion(from, checksq, QUEEN);
      (list++)->move = make_promotion(from, checksq, ROOK);
      (list++)->move = make_promotion(from, checksq, BISHOP);
      (list++)->move = make_promotion(from, checksq, KNIGHT);
    }
  }

  if (ep_square() != 0 && ep_square() == checksq + DELTA_N) {
    b1 = attacks_from_pawn(ep_square(), BLACK) & pawns;
    while (b1) {
      Square from = pop_lsb(&b1);
      (list++)->move = make_enpassant(from, ep_square());
    }
  }

  // Interposing single and double pushes.
  target = shift_bb_S(target);
  b1 = target & pawns;
  b2 = shift_bb_S(target & ~pieces()) & pawns & Rank2BB;
  while (b1) {
    Square from = pop_lsb(&b1);
    if (from < SQ_A8)
      (list++)->move = make_move(from, from + DELTA_N);
    else {
      (list++)->move = make_promotion(from, from + DELTA_N, QUEEN);
      (list++)->move = make_promotion(from, from + DELTA_N, ROOK);
      (list++)->move = make_promotion(from, from + DELTA_N, BISHOP);
      (list++)->move = make_promotion(from, from + DELTA_N, KNIGHT);
    }
  }
  while (b2) {
    Square from = pop_lsb(&b2);
    (list++)->move = make_move(from, from + DELTA_NN);
  }

  return list;
}

INLINE ExtMove *gen_pawn_evasions_black(Pos *pos, ExtMove *list,
                                               Bitboard target, Square checksq)
{
  Bitboard pawns, b1, b2;

  pawns = pieces_cp(BLACK, PAWN);

  // Pawn captures of attacking piece.
  b1 = attacks_from_pawn(checksq, WHITE) & pawns;
  while (b1) {
    Square from = pop_lsb(&b1);
    if (checksq > SQ_H1)
      (list++)->move = make_move(from, checksq);
    else {
      (list++)->move = make_promotion(from, checksq, QUEEN);
      (list++)->move = make_promotion(from, checksq, ROOK);
      (list++)->move = make_promotion(from, checksq, BISHOP);
      (list++)->move = make_promotion(from, checksq, KNIGHT);
    }
  }

  if (ep_square() != 0 && ep_square() == checksq + DELTA_S) {
    b1 = attacks_from_pawn(ep_square(), WHITE) & pawns;
    while (b1) {
      Square from = pop_lsb(&b1);
      (list++)->move = make_enpassant(from, ep_square());
    }
  }

  // Interposing single and double pushes.
  target = shift_bb_N(target);
  b1 = target & pawns;
  b2 = shift_bb_N(target & ~pieces()) & pawns & Rank7BB;
  while (b1) {
    Square from = pop_lsb(&b1);
    if (from > SQ_H1)
      (list++)->move = make_move(from, from + DELTA_S);
    else {
      (list++)->move = make_promotion(from, from + DELTA_S, QUEEN);
      (list++)->move = make_promotion(from, from + DELTA_S, ROOK);
      (list++)->move = make_promotion(from, from + DELTA_S, BISHOP);
      (list++)->move = make_promotion(from, from + DELTA_S, KNIGHT);
    }
  }
  while (b2) {
    Square from = pop_lsb(&b2);
    (list++)->move = make_move(from, from + DELTA_SS);
  }

  return list;
}

INLINE ExtMove *gen_pawn_checks_white(Pos *pos, ExtMove *list,
                                             CheckInfo *ci)
{
  Bitboard pawns, target, b1, b2;

  pawns = pieces_cp(WHITE, PAWN);
  target = shift_bb_S(attacks_from_pawn(ci->ksq, BLACK) & ~pieces());

  // Single pawn pushes.
  b1 = target & pawns;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to, to + DELTA_N);
  }

  // Double pawn pushes.
  if (target & Rank3BB) {
    b1 = shift_bb_S(target & ~pieces()) & pawns;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_move(to, to + DELTA_NN);
    }
  }

  // Discovered checks.
  if ((b1 = pawns & ~Rank7BB & ci->dcCandidates)) {
    b1 = shift_bb_N(b1) & ~pieces();
    b2 = shift_bb_N(b1 & Rank3BB) & ~pieces();
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_move(to - DELTA_N, to);
    }
    while (b2) {
      Square to = pop_lsb(&b2);
      (list++)->move = make_move(to - DELTA_NN, to);
    }
  }

  // Knight underpromotions with check.
  pawns &= Rank7BB;
  if (pawns & (target = attacks_from_knight(ci->ksq) & Rank8BB)) {
    b1 = shift_bb_NE(pawns) & pieces_c(BLACK) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_NE, to, KNIGHT);
    }

    b1 = shift_bb_NW(pawns) & pieces_c(BLACK) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_NW, to, KNIGHT);
    }

    b1 = shift_bb_N(pawns) & ~pieces() & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_N, to, KNIGHT);
    }
  }

  return list;
}

INLINE ExtMove *gen_pawn_checks_black(Pos *pos, ExtMove *list,
                                             CheckInfo *ci)
{
  Bitboard pawns, target, b1, b2;

  pawns = pieces_cp(BLACK, PAWN);
  target = shift_bb_N(attacks_from_pawn(ci->ksq, WHITE) & ~pieces());

  // Single pawn pushes.
  b1 = target & pawns;
  while (b1) {
    Square to = pop_lsb(&b1);
    (list++)->move = make_move(to, to + DELTA_S);
  }

  // Double pawn pushes.
  if (target & Rank6BB) {
    b1 = shift_bb_N(target & ~pieces()) & pawns;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_move(to, to + DELTA_SS);
    }
  }

  // Discovered checks.
  if ((b1 = pawns & ~Rank2BB & ci->dcCandidates)) {
    b1 = shift_bb_S(b1) & ~pieces();
    b2 = shift_bb_S(b1 & Rank6BB) & ~pieces();
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_move(to - DELTA_S, to);
    }
    while (b2) {
      Square to = pop_lsb(&b2);
      (list++)->move = make_move(to - DELTA_SS, to);
    }
  }

  // Knight underpromotions with check.
  pawns &= Rank2BB;
  if (pawns & (target = attacks_from_knight(ci->ksq) & Rank1BB)) {
    b1 = shift_bb_SW(pawns) & pieces_c(WHITE) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_SW, to, KNIGHT);
    }

    b1 = shift_bb_SE(pawns) & pieces_c(WHITE) & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_SE, to, KNIGHT);
    }

    b1 = shift_bb_S(pawns) & ~pieces() & target;
    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_promotion(to - DELTA_S, to, KNIGHT);
    }
  }

  return list;
}

static ExtMove *gen_piece_moves(Pos *pos, ExtMove *list, Bitboard source,
                                Bitboard target)
{
  Bitboard b1, b2;
  Square from;

  b1 = pieces_p(KNIGHT) & source;
  while (b1) {
    from = pop_lsb(&b1);
    b2 = attacks_from_knight(from) & target;
    while (b2)
      (list++)->move = make_move(from, pop_lsb(&b2));
  }

  b1 = pieces_pp(BISHOP, QUEEN) & source;
  while (b1) {
    from = pop_lsb(&b1);
    b2 = attacks_from_bishop(from) & target;
    while (b2)
      (list++)->move = make_move(from, pop_lsb(&b2));
  }

  b1 = pieces_pp(ROOK, QUEEN) & source;
  while (b1) {
    from = pop_lsb(&b1);
    b2 = attacks_from_rook(from) & target;
    while (b2)
      (list++)->move = make_move(from, pop_lsb(&b2));
  }

  b1 = pieces_p(KING) & source;
  if (b1) {
    from = lsb(pieces_p(KING) & source);
    b2 = attacks_from_king(from) & target;
    while (b2)
      (list++)->move = make_move(from, pop_lsb(&b2));
  }

  return list;
}


// generate_quiets() generates all pseudo-legal captures and queen
// promotions. Returns a pointer to the end of the move list.

ExtMove *generate_captures(Pos *pos, ExtMove *list)
{
  assert(!pos_checkers());

  if (pos_stm() == WHITE)
    list = gen_pawn_captures_white(pos, list);
  else
    list = gen_pawn_captures_black(pos, list);

  return gen_piece_moves(pos, list, pieces_c(pos_stm()), pieces_c(pos_stm() ^ 1));
}

// generate_quiets() generates all pseudo-legal non-captures and
// underpromotions.

ExtMove *generate_quiets(Pos *pos, ExtMove *list)
{
  assert(!pos_checkers());

  if (pos_stm() == WHITE) {
    list = gen_pawn_pushes_white(pos, list);
    list = gen_castling_white(pos, list);
  } else {
    list = gen_pawn_pushes_black(pos, list);
    list = gen_castling_black(pos, list);
  }

  return gen_piece_moves(pos, list, pieces_c(pos_stm()), ~pieces());
}

// generate_non_evasions() generates all pseudo-legal captures and
// non-captures.

ExtMove *generate_non_evasions(Pos *pos, ExtMove *list)
{
  assert(!pos_checkers());

  if (pos_stm() == WHITE) {
    list = gen_pawn_captures_white(pos, list);
    list = gen_pawn_pushes_white(pos, list);
    list = gen_castling_white(pos, list);
  } else {
    list = gen_pawn_captures_black(pos, list);
    list = gen_pawn_pushes_black(pos, list);
    list = gen_castling_black(pos, list);
  }

  return gen_piece_moves(pos, list, pieces_c(pos_stm()), ~pieces_c(pos_stm()));
}

// generate_quiet_checks() generates all pseudo-legal non-captures and
// knight underpromotions that give check.

ExtMove *generate_quiet_checks(Pos *pos, ExtMove* list)
{
  assert(!pos_checkers());

  CheckInfo ci;
  checkinfo_init(&ci, pos);

  Bitboard dc = ci.dcCandidates;
  Bitboard not_done = pieces_c(pos_stm()) ^ dc;

  while (dc) {
    Square from = pop_lsb(&dc);
    int pt = type_of_p(piece_on(from));

    if (pt == PAWN)
      continue; // Will be generated together with direct pawn checks

    Bitboard b = attacks_from(pt, from) & ~pieces();

    if (pt == KING)
      b &= ~PseudoAttacks[QUEEN][ci.ksq];

    while (b)
      (list++)->move = make_move(from, pop_lsb(&b));
  }

  if (pos_stm() == WHITE)
    list = gen_pawn_checks_white(pos, list, &ci);
  else
    list = gen_pawn_checks_black(pos, list, &ci);

  // Generate castling moves and keep them if they are checks.
  // If two castling moves were generated, at most one of them checks.
  ExtMove *p;
  if (pos_stm() == WHITE)
    p = gen_castling_white(pos, list);
  else
    p = gen_castling_black(pos, list);
  if (list < p) {
    if (gives_check(pos, list->move, &ci))
      list++;
    else if (list + 1 < p && gives_check(pos, (list+1)->move, &ci)) {
      list->move = (list+1)->move;
      list++;
    }
  }

  // Direct quiet checks by knights, bishops, rooks and queens.
  Bitboard b1, b2, target;
  if ((b1 = pieces_p(KNIGHT) & not_done)) {
    target = attacks_from_knight(ci.ksq) & ~pieces();
    while (b1) {
      Square from = pop_lsb(&b1);
      b2 = attacks_from_knight(from) & target;
      while (b2)
        (list++)->move = make_move(from, pop_lsb(&b2));
    }
  }
  if ((b1 = pieces_pp(BISHOP, QUEEN) & not_done)) {
    target = attacks_from_bishop(ci.ksq) & ~pieces();
    while (b1) {
      Square from = pop_lsb(&b1);
      b2 = attacks_from_bishop(from) & target;
      while (b2)
        (list++)->move = make_move(from, pop_lsb(&b2));
    }
  }
  if ((b1 = pieces_pp(ROOK, QUEEN) & not_done)) {
    target = attacks_from_rook(ci.ksq) & ~pieces();
    while (b1) {
      Square from = pop_lsb(&b1);
      b2 = attacks_from_rook(from) & target;
      while (b2)
        (list++)->move = make_move(from, pop_lsb(&b2));
    }
  }

  return list;
}

// generate_evasions() generates all pseudo-legal check evasions.

ExtMove *generate_evasions(Pos *pos, ExtMove *list)
{
  assert(pos_checkers());

  Square ksq = square_of(pos_stm(), KING);
  Bitboard sliderAttacks = 0;
  Bitboard sliders = pos_checkers() & ~pieces_pp(KNIGHT, PAWN);

  // Find all the squares attacked by slider checkers. We will remove them
  // from the king evasions in order to skip known illegal moves.
  while (sliders) {
    Square checksq = pop_lsb(&sliders);
    sliderAttacks |= LineBB[checksq][ksq] ^ sq_bb(checksq);
  }

  // Generate evasions for king, capture and non-capture moves.
  Bitboard b = attacks_from_king(ksq) & ~pieces_c(pos_stm()) & ~sliderAttacks;
  while (b)
    (list++)->move = make_move(ksq, pop_lsb(&b));

  if (more_than_one(pos_checkers()))
    return list; // Double check; only a king move can save the day.

  // Generate blocking evasions or captures of the checking piece.
  Square checksq = lsb(pos_checkers());
  Bitboard target = between_bb(checksq, ksq);

  if (pos_stm() == WHITE)
    list = gen_pawn_evasions_white(pos, list, target, checksq);
  else
    list = gen_pawn_evasions_black(pos, list, target, checksq);

  return gen_piece_moves(pos, list, pieces_c(pos_stm()) & ~pieces_p(KING),
                         target | sq_bb(checksq));
}


// generate_legal() generates all legal moves.

ExtMove *generate_legal(Pos *pos, ExtMove *list)
{
  Bitboard pinned = pinned_pieces(pos, pos_stm());
  Square ksq = square_of(pos_stm(), KING);
  ExtMove* cur = list;

  list = pos_checkers() ? generate_evasions(pos, list)
                        : generate_non_evasions(pos, list);
  while (cur != list)
    if (   (pinned || from_sq(cur->move) == ksq
                               || type_of_m(cur->move) == ENPASSANT)
        && !is_legal(pos, cur->move, pinned))
      cur->move = (--list)->move;
    else
      cur++;

  return list;
}
#endif

