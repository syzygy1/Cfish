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

#include <assert.h>

#include "bitboard.h"
#include "pawns.h"
#include "position.h"
#include "thread.h"

#define V(v) ((Value)(v))
#define S(mg, eg) make_score(mg, eg)

// Isolated pawn penalty
static const Score Isolated = S(13, 18);

// Backward pawn penalty
static const Score Backward = S(24, 12);

// Connected pawn bonus by opposed, phalanx, #support and rank
static Score Connected[2][2][3][8];

// Doubled pawn penalty
static const Score Doubled = S(18,38);

// Strength of pawn shelter for our king by [distance from edge][rank].
// RANK_1 = 0 is used for files where we have no pawn, or pawn is behind
// our king.
const Value ShelterStrength[4][8] = {
    { V( -9), V(64), V(77), V( 44), V( 4), V( -1), V(-11) },
    { V(-15), V(83), V(51), V(-10), V( 1), V(-10), V(-28) },
    { V(-18), V(84), V(27), V(-12), V(21), V( -7), V(-36) },
    { V( 12), V(79), V(25), V( 19), V( 9), V( -6), V(-33) }
};

// Danger of enemy pawns moving toward our king by
// [type][distance from edge][rank]. For the unopposed and unblocked cases,
// RANK_1 = 0 is used when opponent has no pawn on the given file or
// their pawn is behind our king.
static const Value StormDanger[4][4][8] = {
  { { V( 0),  V(-290), V(-274), V(57), V(41) },  // BlockedByKing
    { V( 0),  V(  60), V( 144), V(39), V(13) },
    { V( 0),  V(  65), V( 141), V(41), V(34) },
    { V( 0),  V(  53), V( 127), V(56), V(14) } },
  { { V( 4),  V(  73), V( 132), V(46), V(31) },  // Unopposed
    { V( 1),  V(  64), V( 143), V(26), V(13) },
    { V( 1),  V(  47), V( 110), V(44), V(24) },
    { V( 0),  V(  72), V( 127), V(50), V(31) } },
  { { V( 0),  V(   0), V(  19), V(23), V( 1) },  // BlockedByPawn
    { V( 0),  V(   0), V(  88), V(27), V( 2) },
    { V( 0),  V(   0), V( 101), V(16), V( 1) },
    { V( 0),  V(   0), V( 111), V(22), V(15) } },
  { { V(22),  V(  45), V( 104), V(62), V( 6) },  // Unblocked
    { V(31),  V(  30), V(  99), V(39), V(19) },
    { V(23),  V(  29), V(  96), V(41), V(15) },
    { V(21),  V(  23), V( 116), V(41), V(15) } }
};

#undef S
#undef V

INLINE Score pawn_evaluate(const Pos *pos, PawnEntry *e, const int Us)
{
  const int Up    = (Us == WHITE ? NORTH      : SOUTH);
  const int Right = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
  const int Left  = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

  Bitboard b, neighbours, stoppers, doubled, supported, phalanx;
  Bitboard lever, leverPush;
  Square s;
  int opposed, backward;
  Score score = SCORE_ZERO;

  Bitboard ourPawns   = pieces_cp(Us, PAWN);
  Bitboard theirPawns = pieces_p(PAWN) ^ ourPawns;

  e->passedPawns[Us] = e->pawnAttacksSpan[Us] = e->weakUnopposed[Us] = 0;
  e->semiopenFiles[Us] = 0xFF;
  e->kingSquares[Us] = SQ_NONE;
  e->pawnAttacks[Us] = shift_bb(Right, ourPawns) | shift_bb(Left, ourPawns);
  e->pawnsOnSquares[Us][BLACK] = popcount(ourPawns & DarkSquares);
  e->pawnsOnSquares[Us][WHITE] = popcount(ourPawns & LightSquares);

  // Loop through all pawns of the current color and score each pawn
  loop_through_pieces(Us, PAWN, s) {
    assert(piece_on(s) == make_piece(Us, PAWN));

    uint32_t f = file_of(s);

    e->semiopenFiles[Us] &= ~(1 << f);
    e->pawnAttacksSpan[Us] |= pawn_attack_span(Us, s);

    // Flag the pawn
    opposed    = !!(theirPawns & forward_file_bb(Us, s));
    stoppers   = theirPawns & passed_pawn_mask(Us, s);
    lever      = theirPawns & PawnAttacks[Us][s];
    leverPush  = theirPawns & PawnAttacks[Us][s + Up];
    doubled    = ourPawns   & sq_bb(s - Up);
    neighbours = ourPawns   & adjacent_files_bb(f);
    phalanx    = neighbours & rank_bb_s(s);
    supported  = neighbours & rank_bb_s(s - Up);

    // A pawn is backward when it is behind all pawns of the same color on the
    // adjacent files and cannot be safely advanced.
    if (!neighbours || lever || relative_rank_s(Us, s) >= RANK_5)
      backward = 0;
    else {
      // Find the backmost rank with neighbours or stoppers
      b = rank_bb_s(backmost_sq(Us, neighbours | stoppers));

      // The pawn is backward when it cannot safely progress to that rank:
      // either there is a stopper in the way on this rank, or there is a
      // stopper on adjacent file which controls the way to that rank.
      backward = !!((b | shift_bb(Up, b & adjacent_files_bb(f))) & stoppers);

      assert(!(backward && (forward_ranks_bb(Us ^ 1, rank_of(s + Up)) & neighbours)));
    }

    // Passed pawns will be properly scored in evaluation because we need
    // full attack info to evaluate them. Include also not passed pawns
    // which could become passed after one or two pawn pushes when they
    // are not attacked more times than defended.
    if (   !(stoppers ^ lever ^ leverPush)
        && !(ourPawns & forward_file_bb(Us, s))
        && popcount(supported) >= popcount(lever) - 1
        && popcount(phalanx)   >= popcount(leverPush))
      e->passedPawns[Us] |= sq_bb(s);

    else if (   stoppers == sq_bb(s + Up)
             && relative_rank_s(Us, s) >= RANK_5)
    {
      b = shift_bb(Up, supported) & ~theirPawns;
      while (b)
        if (!more_than_one(theirPawns & PawnAttacks[Us][pop_lsb(&b)]))
          e->passedPawns[Us] |= sq_bb(s);
    }

    // Score this pawn
    if (supported | phalanx)
      score += Connected[opposed][!!phalanx][popcount(supported)][relative_rank_s(Us, s)];

    else if (!neighbours) {
      score -= Isolated;
      e->weakUnopposed[Us] += !opposed;
    }

    else if (backward) {
      score -= Backward;
      e->weakUnopposed[Us] += !opposed;
    }

    if (doubled && !supported)
      score -= Doubled;
  }

  return score;
}


// pawn_init() initializes some tables needed by evaluation.

void pawn_init(void)
{
  static const int Seed[8] = { 0, 13, 24, 18, 76, 100, 175, 330 };

  for (int opposed = 0; opposed < 2; opposed++)
    for (int phalanx = 0; phalanx < 2; phalanx++)
      for (int support = 0; support <= 2; support++)
        for (int r = RANK_2; r < RANK_8; ++r) {
          int v = 17 * support;
          v += (Seed[r] + (phalanx ? (Seed[r + 1] - Seed[r]) / 2 : 0)) >> opposed;
          Connected[opposed][phalanx][support][r] = make_score(v, v * (r-2) / 4);
      }
}


// pawns_probe() looks up the current position's pawns configuration in
// the pawns hash table.

void pawn_entry_fill(const Pos *pos, PawnEntry *e, Key key)
{
  e->key = key;
  e->score = pawn_evaluate(pos, e, WHITE) - pawn_evaluate(pos, e, BLACK);
  e->openFiles = popcount(e->semiopenFiles[WHITE] & e->semiopenFiles[BLACK]);
  e->asymmetry = popcount(e->passedPawns[WHITE] | e->passedPawns[BLACK]
                        | (e->semiopenFiles[WHITE] ^ e->semiopenFiles[BLACK]));
}


// evaluate_shelter() calculates the shelter bonus and the storm penalty
// for a king, by looking at the king file and the two closest files.

INLINE Value evaluate_shelter(const Pos *pos, Square ksq, const int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const int Down = (Us == WHITE ? SOUTH : NORTH);
  
  enum { BlockedByKing, Unopposed, BlockedByPawn, Unblocked };

  File center = max(FILE_B, min(FILE_G, file_of(ksq)));
  Bitboard b =  pieces_p(PAWN)
              & (forward_ranks_bb(Us, rank_of(ksq)) | rank_bb_s(ksq))
              & (adjacent_files_bb(center) | file_bb(center));
  Bitboard ourPawns = b & pieces_c(Us);
  Bitboard theirPawns = b & pieces_c(Them);
  Value safety = (ourPawns & file_bb_s(ksq)) ? 5 : -5;

  for (File f = center - 1; f <= center + 1; f++) {
    b = ourPawns & file_bb(f);
    Rank rkUs = b ? relative_rank_s(Us, backmost_sq(Us, b)) : RANK_1;

    b = theirPawns & file_bb(f);
    Rank rkThem = b ? relative_rank_s(Us, frontmost_sq(Them, b)) : RANK_1;

    int d = min(f, FILE_H - f);
    safety +=  ShelterStrength[d][rkUs]
             - StormDanger
               [(shift_bb(Down, b) & sq_bb(ksq)) ? BlockedByKing :
                rkUs   == RANK_1                 ? Unopposed :
                rkThem == rkUs + 1               ? BlockedByPawn  : Unblocked]
               [d][rkThem];
  }

  return safety;
}


// do_king_safety() calculates a bonus for king safety. It is called only
// when king square changes, which is about 20% of total king_safety() calls.

INLINE Score do_king_safety(PawnEntry *pe, const Pos *pos, Square ksq,
                                   const int Us)
{
  pe->kingSquares[Us] = ksq;
  pe->castlingRights[Us] = can_castle_c(Us);
  int minKingPawnDistance = 0;

  Bitboard pawns = pieces_cp(Us, PAWN);
  if (pawns)
    while (!(DistanceRingBB[ksq][minKingPawnDistance++] & pawns)) {}

  Value bonus = evaluate_shelter(pos, ksq, Us);

  // If we can castle use the bonus after the castling if it is bigger
  if (can_castle_cr(make_castling_right(Us, KING_SIDE))) {
    Value v = evaluate_shelter(pos, relative_square(Us, SQ_G1), Us);
    bonus = max(bonus, v);
  }

  if (can_castle_cr(make_castling_right(Us, QUEEN_SIDE))) {
    Value v = evaluate_shelter(pos, relative_square(Us, SQ_C1), Us);
    bonus = max(bonus, v);
  }

  return make_score(bonus, -16 * minKingPawnDistance);
}

// "template" instantiation:
Score do_king_safety_white(PawnEntry *pe, const Pos *pos, Square ksq)
{
  return do_king_safety(pe, pos, ksq, WHITE);
}

Score do_king_safety_black(PawnEntry *pe, const Pos *pos, Square ksq)
{
  return do_king_safety(pe, pos, ksq, BLACK);
}
