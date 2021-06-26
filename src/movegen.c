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

#include "movegen.h"
#include "position.h"
#include "types.h"

enum { CAPTURES, QUIETS, QUIET_CHECKS, EVASIONS, NON_EVASIONS, LEGAL };


INLINE ExtMove *make_promotions(ExtMove *list, Square to, Square ksq,
    const int Type, const int D)
{
  if (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS) {
    (list++)->move = make_promotion(to - D, to, QUEEN);
    if (attacks_from_knight(to) & sq_bb(ksq))
      (list++)->move = make_promotion(to - D, to, KNIGHT);
  }

  if (Type == QUIETS || Type == EVASIONS || Type == NON_EVASIONS) {
    (list++)->move = make_promotion(to - D, to, ROOK);
    (list++)->move = make_promotion(to - D, to, BISHOP);
    if (!(attacks_from_knight(to) & sq_bb(ksq)))
      (list++)->move = make_promotion(to - D, to, KNIGHT);
  }

  return list;
}


INLINE ExtMove *generate_pawn_moves(const Position *pos, ExtMove *list,
    Bitboard target, const Color Us, const int Type)
{
  // Compute our parametrized parameters at compile time, named according to
  // the point of view of white side.
  const Color    Them     = Us == WHITE ? BLACK      : WHITE;
  const Bitboard TRank8BB = Us == WHITE ? Rank8BB    : Rank1BB;
  const Bitboard TRank7BB = Us == WHITE ? Rank7BB    : Rank2BB;
  const Bitboard TRank3BB = Us == WHITE ? Rank3BB    : Rank6BB;
  const int      Up       = Us == WHITE ? NORTH      : SOUTH;
  const int      Right    = Us == WHITE ? NORTH_EAST : SOUTH_WEST;
  const int      Left     = Us == WHITE ? NORTH_WEST : SOUTH_EAST;

  const Bitboard emptySquares =  Type == QUIETS || Type == QUIET_CHECKS
                               ? target : ~pieces();
  const Bitboard enemies =  Type == EVASIONS ? checkers()
                          : Type == CAPTURES ? target : pieces_c(Them);

  Bitboard pawnsOn7    = pieces_cp(Us, PAWN) &  TRank7BB;
  Bitboard pawnsNotOn7 = pieces_cp(Us, PAWN) & ~TRank7BB;

  // Single and double pawn pushes, no promotions
  if (Type != CAPTURES) {
    Bitboard b1 = shift_bb(Up, pawnsNotOn7)   & emptySquares;
    Bitboard b2 = shift_bb(Up, b1 & TRank3BB) & emptySquares;

    if (Type == EVASIONS) { // Consider only blocking squares
      b1 &= target;
      b2 &= target;
    }

    if (Type == QUIET_CHECKS) {
      Stack *st = pos->st;

      // A quiet check is either a direct check or a discovered check.
      Bitboard dcCandidatePawns = blockers_for_king(pos, Them) & ~file_bb_s(st->ksq);
      b1 &= attacks_from_pawn(st->ksq, Them) | shift_bb(Up, dcCandidatePawns);
      b2 &= attacks_from_pawn(st->ksq, Them) | shift_bb(Up+Up, dcCandidatePawns);
    }

    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_move(to - Up, to);
    }

    while (b2) {
      Square to = pop_lsb(&b2);
      (list++)->move = make_move(to - Up - Up, to);
    }
  }

  // Promotions and underpromotions
  if (pawnsOn7 && (Type != EVASIONS || (target & TRank8BB))) {
    Bitboard b1 = shift_bb(Right, pawnsOn7) & enemies;
    Bitboard b2 = shift_bb(Left , pawnsOn7) & enemies;
    Bitboard b3 = shift_bb(Up   , pawnsOn7) & emptySquares;

    if (Type == EVASIONS)
      b3 &= target;

    while (b1)
      list = make_promotions(list, pop_lsb(&b1), pos->st->ksq, Type, Right);

    while (b2)
      list = make_promotions(list, pop_lsb(&b2), pos->st->ksq, Type, Left);

    while (b3)
      list = make_promotions(list, pop_lsb(&b3), pos->st->ksq, Type, Up);
  }

  // Standard and en-passant captures
  if (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS) {
    Bitboard b1 = shift_bb(Right, pawnsNotOn7) & enemies;
    Bitboard b2 = shift_bb(Left , pawnsNotOn7) & enemies;

    while (b1) {
      Square to = pop_lsb(&b1);
      (list++)->move = make_move(to - Right, to);
    }

    while (b2) {
      Square to = pop_lsb(&b2);
      (list++)->move = make_move(to - Left, to);
    }

    if (ep_square() != 0) {
      assert(rank_of(ep_square()) == relative_rank(Us, RANK_6));

      // An en passant capture can be an evasion only if the checking piece
      // is the double pushed pawn and so is in the target. Otherwise this
      // is a discovered check and we are forced to do otherwise.
      if (Type == EVASIONS && (target & sq_bb(ep_square() + Up)))
        return list;

      b1 = pawnsNotOn7 & attacks_from_pawn(ep_square(), Them);

      assert(b1);

      while (b1)
        (list++)->move = make_enpassant(pop_lsb(&b1), ep_square());
    }
  }

  return list;
}


INLINE ExtMove *generate_moves(const Position *pos, ExtMove *list,
    Bitboard target, const Color Us, const int Pt, const bool Checks)
{
  assert(Pt != KING && Pt != PAWN);

  Bitboard bb = pieces_cp(Us, Pt);

  while (bb) {
    Square from = pop_lsb(&bb);
    Bitboard b = attacks_bb(Pt, from, pieces()) & target;

    if (Checks && (Pt == QUEEN || !(blockers_for_king(pos, !Us) & sq_bb(from))))
      b &= pos->st->checkSquares[Pt];

    while (b)
      (list++)->move = make_move(from, pop_lsb(&b));
  }

  return list;
}


INLINE ExtMove *generate_all(const Position *pos, ExtMove *list, const Color Us,
  const int Type)
{
  const bool Checks = Type == QUIET_CHECKS;
  const Square ksq = square_of(Us, KING);
  Bitboard target;

  if (Type == EVASIONS && more_than_one(checkers()))
    goto kingMoves; // Double check, only a king move can save the day

  target =  Type == EVASIONS     ? between_bb(ksq, lsb(checkers()))
          : Type == NON_EVASIONS ? ~pieces_c(Us)
          : Type == CAPTURES     ? pieces_c(!Us) : ~pieces();

  list = generate_pawn_moves(pos, list, target, Us, Type);
  list = generate_moves(pos, list, target, Us, KNIGHT, Checks);
  list = generate_moves(pos, list, target, Us, BISHOP, Checks);
  list = generate_moves(pos, list, target, Us,   ROOK, Checks);
  list = generate_moves(pos, list, target, Us,  QUEEN, Checks);

kingMoves:
  if (!Checks || blockers_for_king(pos, !Us) & sq_bb(ksq)) {
    Bitboard b = attacks_from(KING, ksq) & (Type == EVASIONS ? ~pieces_c(Us) : target);
    if (Checks)
      b &= ~PseudoAttacks[QUEEN][square_of(!Us, KING)];

    while (b)
      (list++)->move = make_move(ksq, pop_lsb(&b));

    if ((Type == QUIETS || Type == NON_EVASIONS) && can_castle_c(Us)) {
      const int OO = make_castling_right(Us, KING_SIDE);
      if (!castling_impeded(OO) && can_castle_cr(OO))
        (list++)->move = make_castling(ksq, castling_rook_square(OO));

      const int OOO = make_castling_right(Us, QUEEN_SIDE);
      if (!castling_impeded(OOO) && can_castle_cr(OOO))
        (list++)->move = make_castling(ksq, castling_rook_square(OOO));
    }
  }

  return list;
}


// generate_captures() generates all pseudo-legal captures plus queen and
// checking knight promotions.
//
// generate_quiets() generates all pseudo-legal non-captures and
// underpromotions (except checking knight promotions).
//
// generate_evasions() generates all pseudo-legal check evasions
//
// generate_quiet_checks() generates all pseudo-legal non-captures giving
// check, except castling
//
// generate_non_evasions() generates all pseudo-legal captures and
// non-captures.

INLINE ExtMove *generate(const Position *pos, ExtMove *list, const int Type)
{
  assert(Type != LEGAL);
  assert((Type == EVASIONS) == (bool)checkers());

  Color us = stm();

  return us == WHITE ? generate_all(pos, list, WHITE, Type)
                     : generate_all(pos, list, BLACK, Type);
}

// "template" instantiations

NOINLINE ExtMove *generate_captures(const Position *pos, ExtMove *list)
{
  return generate(pos, list, CAPTURES);
}

NOINLINE ExtMove *generate_quiets(const Position *pos, ExtMove *list)
{
  return generate(pos, list, QUIETS);
}

NOINLINE ExtMove *generate_evasions(const Position *pos, ExtMove *list)
{
  return generate(pos, list, EVASIONS);
}

NOINLINE ExtMove *generate_quiet_checks(const Position *pos, ExtMove *list)
{
  return generate(pos, list, QUIET_CHECKS);
}

NOINLINE ExtMove *generate_non_evasions(const Position *pos, ExtMove *list)
{
  return generate(pos, list, NON_EVASIONS);
}


// generate_legal() generates all the legal moves in the given position
NOINLINE ExtMove *generate_legal(const Position *pos, ExtMove *list)
{
  Color us = stm();
  Bitboard pinned = blockers_for_king(pos, us) & pieces_c(us);
  Square ksq = square_of(us, KING);
  ExtMove *cur = list;

  list = checkers() ? generate_evasions(pos, list)
                    : generate_non_evasions(pos, list);
  while (cur != list)
    if (  (  (pinned && pinned & sq_bb(from_sq(cur->move)))
           || from_sq(cur->move) == ksq
           || type_of_m(cur->move) == ENPASSANT)
        && !is_legal(pos, cur->move))
      cur->move = (--list)->move;
    else
      ++cur;

  return list;
}
