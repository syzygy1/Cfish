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

#ifndef NNUE_PURE

#include <assert.h>

#include "bitboard.h"
#include "pawns.h"
#include "position.h"
#include "thread.h"

#define V(v) ((Value)(v))
#define S(mg, eg) make_score(mg, eg)

// Pawn penalties
static const Score Backward      = S( 9, 22);
static const Score Doubled       = S(13, 51);
static const Score DoubledEarly  = S(20,  7);
static const Score Isolated      = S( 3, 15);
static const Score WeakLever     = S( 4, 58);
static const Score WeakUnopposed = S(13, 24);

// Bonus for blocked pawns at 5th or 6th rank
static const Score BlockedPawn[2] = { S(-17, -6), S(-9, 2) };

static const Score BlockedStorm[8] = {
  S(0, 0), S(0, 0), S(75, 78), S(-8, 16), S(-6, 10), S(-6, 6), S(0,2)
};

// Connected pawn bonus
static const int Connected[8] = { 0, 5, 7, 11, 23, 48, 87 };

#undef V
#define V(mg) S(mg,0)
// Strength of pawn shelter for our king by [distance from edge][rank].
// RANK_1 = 0 is used for files where we have no pawn, or pawn is behind
// our king.
static const Score ShelterStrength[4][8] = {
  { V( -5), V( 82), V( 92), V( 54), V( 36), V( 22), V(  28) },
  { V(-44), V( 63), V( 33), V(-50), V(-30), V(-12), V( -62) },
  { V(-11), V( 77), V( 22), V( -6), V( 31), V(  8), V( -45) },
  { V(-39), V(-12), V(-29), V(-50), V(-43), V(-68), V(-164) }
};

// Danger of enemry pawns moving toward our king by [distance from edge][rank].
// RANK_1 = 0 is used for files where the enemy has no pawn or where their
// pawn is behind our king. Note that UnblockedStorm[0][1-2] accommodates
// opponent pawn on edge, likely blocked by our king.
static const Score UnblockedStorm[4][8] = {
  { V( 87), V(-288), V(-168), V( 96), V( 47), V( 44), V( 46) },
  { V( 42), V( -25), V( 120), V( 45), V( 34), V( -9), V( 24) },
  { V( -8), V(  51), V( 167), V( 35), V( -4), V(-16), V(-12) },
  { V(-17), V( -13), V( 100), V(  4), V(  9), V(-16), V(-31) }
};

// KingOnFile[semi-open Us][semi-open Them] contains bonuses/penalties
// for king when the king is on a semi-open or open file.
static const Score KingOnFile[2][2] = {
  { S(-21,10), S(-7, 1) }, { S(0, -3), S(9, -4) }
};

#undef S
#undef V

INLINE Score pawn_evaluate(const Position *pos, PawnEntry *e, const Color Us)
{
  const Color Them  = Us == WHITE ? BLACK : WHITE;
  const int   Up    = Us == WHITE ? NORTH : SOUTH;
  const int   Down  = Us == WHITE ? SOUTH : NORTH;

  Bitboard neighbours, stoppers, doubled, support, phalanx, opposed;
  Bitboard lever, leverPush, blocked;
  Square s;
  bool backward, passed;
  Score score = SCORE_ZERO;

  Bitboard ourPawns   = pieces_cp(Us, PAWN);
  Bitboard theirPawns = pieces_p(PAWN) ^ ourPawns;

  Bitboard doubleAttackThem = pawn_double_attacks_bb(theirPawns, Them);

  e->passedPawns[Us] = 0;
  e->semiopenFiles[Us] = 0xFF;
  e->kingSquares[Us] = SQ_NONE;
  e->pawnAttacks[Us] = e->pawnAttacksSpan[Us] = pawn_attacks_bb(ourPawns, Us);
  e->pawnsOnSquares[Us][BLACK] = popcount(ourPawns & DarkSquares);
  e->pawnsOnSquares[Us][WHITE] = popcount(ourPawns & LightSquares);
  e->blockedCount += popcount(  shift_bb(Up, ourPawns)
                              & (theirPawns | doubleAttackThem));

  // Loop through all pawns of the current color and score each pawn
  loop_through_pieces(Us, PAWN, s) {
    assert(piece_on(s) == make_piece(Us, PAWN));

    int f = file_of(s);
    int r = relative_rank_s(Us, s);
    e->semiopenFiles[Us] &= ~(1 << f);

    // Flag the pawn
    opposed    = theirPawns & forward_file_bb(Us, s);
    blocked    = theirPawns & sq_bb(s + Up);
    stoppers   = theirPawns & passed_pawn_span(Us, s);
    lever      = theirPawns & PawnAttacks[Us][s];
    leverPush  = theirPawns & PawnAttacks[Us][s + Up];
    doubled    = ourPawns   & sq_bb(s - Up);
    neighbours = ourPawns   & adjacent_files_bb(f);
    phalanx    = neighbours & rank_bb_s(s);
    support    = neighbours & rank_bb_s(s - Up);

    if (doubled) {
      // Additional doubled penalty if none of their pawns is fixed
      if (!(ourPawns & shift_bb(Down, theirPawns | pawn_attacks_bb(theirPawns, Them))))
        score -= DoubledEarly;
    }

    // A pawn is backward when it is behind all pawns of the same color on
    // the adjacent files and cannot safely advance.
    backward =  !(neighbours & forward_ranks_bb(Them, rank_of(s + Up)))
              && (leverPush | blocked);

    // Compute additional span if pawn is neither backward nor blocked
    if (!backward && !blocked)
      e->pawnAttacksSpan[Us] |= pawn_attack_span(Us, s);

    // A pawn is passed if one of the three following conditions is true:
    // (a) there are no stoppers except some levers
    // (b) the only stoppers are the leverPush, but we outnumber them
    // (c) there is only one front stopper which can be levered
    //     (Refined in evaluation_passed())
    passed =  !(stoppers ^ lever)
            || (   !(stoppers ^ leverPush)
                && popcount(phalanx) >= popcount(leverPush))
            || (   stoppers == blocked && r >= RANK_5
                && (shift_bb(Up, support) & ~(theirPawns | doubleAttackThem)));

    passed &= !(forward_file_bb(Us, s) & ourPawns);

    // Passed pawns will be properly scored later in evaluation when we have
    // full attack info.
    if (passed)
      e->passedPawns[Us] |= sq_bb(s);

    // Score this pawn
    if (support | phalanx) {
      int v =  Connected[r] * (2 + !!phalanx - !!opposed)
             + 22 * popcount(support);
      score += make_score(v, v * (r - 2) / 4);
    }

    else if (!neighbours) {
      if (    opposed
          && (ourPawns & forward_file_bb(Them, s))
          && !(theirPawns & adjacent_files_bb(f)))
        score -= Doubled;
      else
        score -= Isolated + (!opposed ? WeakUnopposed : 0);
    }

    else if (backward)
      score -= Backward + (!opposed && ((s+1) & 0x06) ? WeakUnopposed : 0);

    if (!support)
      score -=  (doubled ? Doubled : 0)
              + (more_than_one(lever) ? WeakLever : 0);

    if (blocked && r >= RANK_5)
      score += BlockedPawn[r - RANK_5];
  }

  return score;
}


// pawns_probe() looks up the current position's pawns configuration in
// the pawns hash table.

void pawn_entry_fill(const Position *pos, PawnEntry *e, Key key)
{
  e->key = key;
  e->blockedCount = 0;
  e->score = pawn_evaluate(pos, e, WHITE) - pawn_evaluate(pos, e, BLACK);
  e->openFiles = popcount(e->semiopenFiles[WHITE] & e->semiopenFiles[BLACK]);
  e->passedCount = popcount(e->passedPawns[WHITE] | e->passedPawns[BLACK]);
}


// evaluate_shelter() calculates the shelter bonus and the storm penalty
// for a king, by looking at the king file and the two closest files.

INLINE Score evaluate_shelter(const PawnEntry *pe, const Position *pos,
    Square ksq, const Color Us)
{
  const Color Them = Us == WHITE ? BLACK : WHITE;
  
  Bitboard b =  pieces_p(PAWN) & ~forward_ranks_bb(Them, rank_of(ksq));
  Bitboard ourPawns = b & pieces_c(Us) & ~pe->pawnAttacks[Them];
  Bitboard theirPawns = b & pieces_c(Them);
  Score bonus = make_score(5, 5);

  File center = clamp(file_of(ksq), FILE_B, FILE_G);

  for (File f = center - 1; f <= center + 1; f++) {
    b = ourPawns & file_bb(f);
    int ourRank = b ? relative_rank_s(Us, backmost_sq(Us, b)) : 0;

    b = theirPawns & file_bb(f);
    int theirRank = b ? relative_rank_s(Us, frontmost_sq(Them, b)) : 0;

    int d = min(f, FILE_H - f);
    bonus += ShelterStrength[d][ourRank];

    if (ourRank && (ourRank == theirRank - 1)) {
      bonus -= BlockedStorm[theirRank];
    } else
      bonus -= UnblockedStorm[d][theirRank];
  }

  bonus -= KingOnFile[is_on_semiopen_file(pe, Us, ksq)][is_on_semiopen_file(pe, Them, ksq)];

  return bonus;
}


// do_king_safety() calculates a bonus for king safety. It is called only
// when king square changes, which is about 20% of total king_safety() calls.

INLINE Score do_king_safety(PawnEntry *pe, const Position *pos, Square ksq,
    const Color Us)
{
  pe->kingSquares[Us] = ksq;
  pe->castlingRights[Us] = can_castle_c(Us);

  int minPawnDist;

  Bitboard pawns = pieces_cp(Us, PAWN);
  if (!pawns)
    minPawnDist = 6;
  else if (pawns & PseudoAttacks[KING][ksq])
    minPawnDist = 1;
  else for (minPawnDist = 1;
            minPawnDist < 6 && !(DistanceRingBB[ksq][minPawnDist] & pawns);
            minPawnDist++);

  Score shelter = evaluate_shelter(pe, pos, ksq, Us);

  // If we can castle use the bonus after the castling if it is bigger
  if (can_castle_cr(make_castling_right(Us, KING_SIDE))) {
    Score s = evaluate_shelter(pe, pos, relative_square(Us, SQ_G1), Us);
    if (mg_value(s) > mg_value(shelter))
      shelter = s;
  }

  if (can_castle_cr(make_castling_right(Us, QUEEN_SIDE))) {
    Score s = evaluate_shelter(pe, pos, relative_square(Us, SQ_C1), Us);
    if (mg_value(s) > mg_value(shelter))
      shelter = s;
  }

  return shelter - make_score(0, 16 * minPawnDist);
}

// "template" instantiation:
NOINLINE Score do_king_safety_white(PawnEntry *pe, const Position *pos,
    Square ksq)
{
  return do_king_safety(pe, pos, ksq, WHITE);
}

NOINLINE Score do_king_safety_black(PawnEntry *pe, const Position *pos,
    Square ksq)
{
  return do_king_safety(pe, pos, ksq, BLACK);
}

#else

typedef int make_iso_compilers_happy;

#endif
