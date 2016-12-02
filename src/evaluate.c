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
#include <string.h>   // For std::memset

#include "bitboard.h"
#include "evaluate.h"
#include "material.h"
#include "pawns.h"

// Struct EvalInfo contains various information computed and collected
// by the evaluation functions.
struct EvalInfo {
  // attackedBy[color][piece type] is a bitboard representing all squares
  // attacked by a given color and piece type (can be also ALL_PIECES).
  Bitboard attackedBy[2][8];

  // attackedBy2[color] are the squares attacked by 2 pieces of a given
  // color, possibly via x-ray or by one pawn and one piece. Diagonal
  // x-ray through pawn or squares attacked by 2 pawns are not explicitly
  // added.
  Bitboard attackedBy2[2];

  // kingRing[color] is the zone around the king which is considered
  // by the king safety evaluation. This consists of the squares directly
  // adjacent to the king, and the three (or two, for a king on an edge file)
  // squares two ranks in front of the king. For instance, if black's king
  // is on g8, kingRing[BLACK] is a bitboard containing the squares f8, h8,
  // f7, g7, h7, f6, g6 and h6.
  Bitboard kingRing[2];

  // kingAttackersCount[color] is the number of pieces of the given color
  // which attack a square in the kingRing of the enemy king.
  int kingAttackersCount[2];

  // kingAttackersWeight[color] is the sum of the "weights" of the pieces
  // of the given color which attack a square in the kingRing of the enemy
  // king. The weights of the individual piece types are given by the
  // elements in the KingAttackWeights array.
  int kingAttackersWeight[2];

  // kingAdjacentZoneAttacksCount[color] is the number of attacks by the
  // given color to squares directly adjacent to the enemy king. Pieces
  // which attack more than one square are counted multiple times. For
  // instance, if there is a white knight on g5 and black's king is on g8,
  // this white knight adds 2 to kingAdjacentZoneAttacksCount[WHITE].
  int kingAdjacentZoneAttacksCount[2];

  Bitboard pinnedPieces[2];
  MaterialEntry *me;
  PawnEntry *pi;
};

typedef struct EvalInfo EvalInfo;

#define V(v) (Value)(v)
#define S(mg,eg) make_score(mg,eg)

// MobilityBonus[PieceType][attacked] contains bonuses for middle and
// end game, indexed by piece type and number of attacked squares in the
// MobilityArea.
static const Score MobilityBonus[][32] = {
  {0}, {0},
  { S(-75,-76), S(-56,-54), S(- 9,-26), S( -2,-10), S(  6,  5), S( 15, 11), // Knights
    S( 22, 26), S( 30, 28), S( 36, 29) },
  { S(-48,-58), S(-21,-19), S( 16, -2), S( 26, 12), S( 37, 22), S( 51, 42), // Bishops
    S( 54, 54), S( 63, 58), S( 65, 63), S( 71, 70), S( 79, 74), S( 81, 86),
    S( 92, 90), S( 97, 94) },
  { S(-56,-78), S(-25,-18), S(-11, 26), S( -5, 55), S( -4, 70), S( -1, 81), // Rooks
    S(  8,109), S( 14,120), S( 21,128), S( 23,143), S( 31,154), S( 32,160),
    S( 43,165), S( 49,168), S( 59,169) },
  { S(-40,-35), S(-25,-12), S(  2,  7), S(  4, 19), S( 14, 37), S( 24, 55), // Queens
    S( 25, 62), S( 40, 76), S( 43, 79), S( 47, 87), S( 54, 94), S( 56,102),
    S( 60,111), S( 70,116), S( 72,118), S( 73,122), S( 75,128), S( 77,130),
    S( 85,133), S( 94,136), S( 99,140), S(108,157), S(112,158), S(113,161),
    S(118,174), S(119,177), S(123,191), S(128,199) }
};

// Outpost[knight/bishop][supported by pawn] contains bonuses for knights
// and bishops outposts, bigger if outpost piece is supported by a pawn.
static const Score Outpost[][2] = {
  { S(43,11), S(65,20) }, // Knights
  { S(20, 3), S(29, 8) }  // Bishops
};

// ReachableOutpost[knight/bishop][supported by pawn] contains bonuses for
// knights and bishops which can reach an outpost square in one move,
// bigger if outpost square is supported by a pawn.
static const Score ReachableOutpost[][2] = {
  { S(21, 5), S(35, 8) }, // Knights
  { S( 8, 0), S(14, 4) }  // Bishops
};

// RookOnFile[semiopen/open] contains bonuses for each rook when there is
// no friendly pawn on the rook file.
static const Score RookOnFile[2] = { S(20, 7), S(45, 20) };

// ThreatBySafePawn[PieceType] contains bonuses according to which piece
// type is attacked by a pawn which is protected or is not attacked.
static const Score ThreatBySafePawn[8] = {
  S(0, 0), S(0, 0), S(176, 139), S(131, 127), S(217, 218), S(203, 215) };

// Threat[by minor/by rook][attacked PieceType] contains
// bonuses according to which piece type attacks which one.
// Attacks on lesser pieces which are pawn-defended are not considered.
static const Score Threat[2][8] = {
  { S(0, 0), S(0, 33), S(45, 43), S(46, 47), S(72,107), S(48,118) }, // by Minor
  { S(0, 0), S(0, 25), S(40, 62), S(40, 59), S( 0, 34), S(35, 48) }  // by Rook
};

// ThreatByKing[on one/on many] contains bonuses for King attacks on
// pawns or pieces which are not pawn-defended.
static const Score ThreatByKing[2] = { S(3, 62), S(9, 138) };

// Passed[mg/eg][Rank] contains midgame and endgame bonuses for passed pawns.
// We don't use a Score because we process the two components independently.
static const Value Passed[][8] = {
  { V(5), V( 5), V(31), V(73), V(166), V(252) },
  { V(7), V(14), V(38), V(73), V(166), V(252) }
};

// PassedFile[File] contains a bonus according to the file of a passed pawn
static const Score PassedFile[8] = {
  S(  9, 10), S( 2, 10), S( 1, -8), S(-20,-12),
  S(-20,-12), S( 1, -8), S( 2, 10), S( 9, 10)
};

// Assorted bonuses and penalties used by evaluation
static const Score MinorBehindPawn     = S(16,  0);
static const Score BishopPawns         = S( 8, 12);
static const Score RookOnPawn          = S( 8, 24);
static const Score TrappedRook         = S(92,  0);
static const Score CloseEnemies        = S( 7,  0);
static const Score SafeCheck           = S(20, 20);
static const Score OtherCheck          = S(10, 10);
static const Score ThreatByHangingPawn = S(71, 61);
static const Score LooseEnemies        = S( 0, 25);
static const Score WeakQueen           = S(50, 10);
static const Score Hanging             = S(48, 27);
static const Score ThreatByPawnPush    = S(38, 22);
static const Score Unstoppable         = S( 0, 20);
static const Score PawnlessFlank       = S(20, 80);
static const Score HinderPassedPawn    = S( 7,  0);
static const Score ThreatByRank        = S(16,  3);

// Penalty for a bishop on a1/h1 (a8/h8 for black) which is trapped by
// a friendly pawn on b2/g2 (b7/g7 for black). This can obviously only
// happen in Chess960 games.
static const Score TrappedBishopA1H1 = S(50, 50);

#undef S
#undef V

// KingAttackWeights[PieceType] contains king attack weights by piece type
static const int KingAttackWeights[8] = { 0, 0, 78, 56, 45, 11 };

// Penalties for enemy's safe checks
#define QueenContactCheck 997
#define QueenCheck        695
#define RookCheck         638
#define BishopCheck       538
#define KnightCheck       874


// eval_init() initializes king and attack bitboards for a given color
// adding pawn attacks. To be done at the beginning of the evaluation.

INLINE void evalinfo_init(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them = (Us == WHITE ? BLACK   : WHITE);
  const int Down = (Us == WHITE ? DELTA_S : DELTA_N);

  ei->pinnedPieces[Us] = pinned_pieces(pos, Us);
  Bitboard b = ei->attackedBy[Them][KING];
  ei->attackedBy[Them][0] |= b;
  ei->attackedBy[Us][0] |= ei->attackedBy[Us][PAWN] = ei->pi->pawnAttacks[Us];
  ei->attackedBy2[Us] = ei->attackedBy[Us][PAWN] & ei->attackedBy[Us][KING];

  // Init king safety tables only if we are going to use them
  if (pos_non_pawn_material(Us) >= QueenValueMg) {
    ei->kingRing[Them] = b | shift_bb(Down, b);
    b &= ei->attackedBy[Us][PAWN];
    ei->kingAttackersCount[Us] = popcount(b);
    ei->kingAdjacentZoneAttacksCount[Us] = ei->kingAttackersWeight[Us] = 0;
  }
  else
    ei->kingRing[Them] = ei->kingAttackersCount[Us] = 0;
}

// evaluate_piece() assigns bonuses and penalties to the pieces of a given
// color and type.

INLINE Score evaluate_piece(const Pos *pos, EvalInfo *ei, Score *mobility,
                            Bitboard *mobilityArea, const int Us, const int Pt)
{
  Bitboard b, bb;
  Square s;
  Score score = SCORE_ZERO;

  const int Them = (Us == WHITE ? BLACK : WHITE);
  const Bitboard OutpostRanks = (Us == WHITE ? Rank4BB | Rank5BB | Rank6BB
                                             : Rank5BB | Rank4BB | Rank3BB);

  ei->attackedBy[Us][Pt] = 0;

  loop_through_pieces(Us, Pt, s) {
    // Find attacked squares, including x-ray attacks for bishops and rooks
    b = Pt == BISHOP ? attacks_bb_bishop(s, pieces() ^ pieces_cp(Us, QUEEN))
      : Pt == ROOK ? attacks_bb_rook(s, pieces() ^ pieces_cpp(Us, ROOK, QUEEN))
                   : attacks_from(Pt, s);

    if (ei->pinnedPieces[Us] & sq_bb(s))
      b &= LineBB[square_of(Us, KING)][s];

    ei->attackedBy2[Us] |= ei->attackedBy[Us][0] & b;
    ei->attackedBy[Us][0] |= b;
    ei->attackedBy[Us][Pt] |= b;

    if (b & ei->kingRing[Them]) {
      ei->kingAttackersCount[Us]++;
      ei->kingAttackersWeight[Us] += KingAttackWeights[Pt];
      ei->kingAdjacentZoneAttacksCount[Us] += popcount(b & ei->attackedBy[Them][KING]);
    }

    if (Pt == QUEEN)
      b &= ~(  ei->attackedBy[Them][KNIGHT]
             | ei->attackedBy[Them][BISHOP]
             | ei->attackedBy[Them][ROOK]);

    int mob = popcount(b & mobilityArea[Us]);

    mobility[Us] += MobilityBonus[Pt][mob];

    if (Pt == BISHOP || Pt == KNIGHT) {
      // Bonus for outpost squares
      bb = OutpostRanks & ~ei->pi->pawnAttacksSpan[Them];
      if (bb & sq_bb(s))
        score += Outpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & sq_bb(s))];
      else {
        bb &= b & ~pieces_c(Us);
        if (bb)
          score += ReachableOutpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & bb)];
      }

      // Bonus when behind a pawn
      if (    relative_rank_s(Us, s) < RANK_5
          && (pieces_p(PAWN) & sq_bb(s + pawn_push(Us))))
        score += MinorBehindPawn;

      // Penalty for pawns on the same color square as the bishop
      if (Pt == BISHOP)
        score -= BishopPawns * pawns_on_same_color_squares(ei->pi, Us, s);

      // An important Chess960 pattern: A cornered bishop blocked by a friendly
      // pawn diagonally in front of it is a very serious problem, especially
      // when that pawn is also blocked.
      if (   Pt == BISHOP
          && is_chess960()
          && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1))) {
        Square d = pawn_push(Us) + (file_of(s) == FILE_A ? DELTA_E : DELTA_W);
        if (piece_on(s + d) == make_piece(Us, PAWN))
          score -=  piece_on(s + d + pawn_push(Us))             ? TrappedBishopA1H1 * 4
                  : piece_on(s + d + d) == make_piece(Us, PAWN) ? TrappedBishopA1H1 * 2
                                                                : TrappedBishopA1H1;
      }
    }

    if (Pt == ROOK) {
      // Bonus for aligning with enemy pawns on the same rank/file
      if (relative_rank_s(Us, s) >= RANK_5)
        score += RookOnPawn * popcount(pieces_cp(Them, PAWN) & PseudoAttacks[ROOK][s]);

      // Bonus when on an open or semi-open file
      if (semiopen_file(ei->pi, Us, file_of(s)))
        score += RookOnFile[!!semiopen_file(ei->pi, Them, file_of(s))];

      // Penalize when trapped by the king, even more if the king cannot castle
      else if (mob <= 3) {
        Square ksq = square_of(Us, KING);

        if (   ((file_of(ksq) < FILE_E) == (file_of(s) < file_of(ksq)))
            && !semiopen_side(ei->pi, Us, file_of(ksq), file_of(s) < file_of(ksq)))
          score -= (TrappedRook - make_score(mob * 22, 0)) * (1 + !can_castle_c(Us));
      }
    }

    if (Pt == QUEEN) {
      // Penalty if any relative pin or discovered attack against the queen
      Bitboard pinners;
      if (slider_blockers(pos, pieces_cpp(Them, ROOK, BISHOP), s, &pinners))
          score -= WeakQueen;
    }
  }

  return score;
}

// evaluate_pieces() evaluates all pieces in the right order (queens must
// come last). We rely on the inlining compiler to expand all calls to
// evaluate_piece(). No need for C++ templates!

INLINE Score evaluate_pieces(const Pos *pos, EvalInfo *ei, Score *mobility,
                             Bitboard *mobilityArea)
{
  return  evaluate_piece(pos, ei, mobility, mobilityArea, WHITE, KNIGHT)
        - evaluate_piece(pos, ei, mobility, mobilityArea, BLACK, KNIGHT)
        + evaluate_piece(pos, ei, mobility, mobilityArea, WHITE, BISHOP)
        - evaluate_piece(pos, ei, mobility, mobilityArea, BLACK, BISHOP)
        + evaluate_piece(pos, ei, mobility, mobilityArea, WHITE, ROOK)
        - evaluate_piece(pos, ei, mobility, mobilityArea, BLACK, ROOK)
        + evaluate_piece(pos, ei, mobility, mobilityArea, WHITE, QUEEN)
        - evaluate_piece(pos, ei, mobility, mobilityArea, BLACK, QUEEN);
}


// evaluate_king() assigns bonuses and penalties to a king of a given color.

#define WhiteCamp   (Rank1BB | Rank2BB | Rank3BB | Rank4BB | Rank5BB)
#define BlackCamp   (Rank8BB | Rank7BB | Rank6BB | Rank5BB | Rank4BB)
#define QueenSide   (FileABB | FileBBB | FileCBB | FileDBB)
#define CenterFiles (FileCBB | FileDBB | FileEBB | FileFBB)
#define KingSide    (FileEBB | FileFBB | FileGBB | FileHBB)

static const Bitboard KingFlank[2][8] = {
  { QueenSide   & WhiteCamp, QueenSide & WhiteCamp, QueenSide & WhiteCamp, CenterFiles & WhiteCamp,
    CenterFiles & WhiteCamp, KingSide  & WhiteCamp, KingSide  & WhiteCamp, KingSide    & WhiteCamp },
  { QueenSide   & BlackCamp, QueenSide & BlackCamp, QueenSide & BlackCamp, CenterFiles & BlackCamp,
    CenterFiles & BlackCamp, KingSide  & BlackCamp, KingSide  & BlackCamp, KingSide    & BlackCamp },
};

INLINE Score evaluate_king(const Pos *pos, EvalInfo *ei, int Us)
{
  const int Them = (Us == WHITE ? BLACK   : WHITE);
  const int Up = (Us == WHITE ? DELTA_N : DELTA_S);

  Bitboard undefended, b, b1, b2, safe, other;
  int kingDanger;
  const Square ksq = square_of(Us, KING);

  // King shelter and enemy pawns storm
  Score score = Us == WHITE ? king_safety_white(ei->pi, pos, ksq)
                            : king_safety_black(ei->pi, pos, ksq);

  // Main king safety evaluation
  if (ei->kingAttackersCount[Them]) {
    // Find the attacked squares which are defended only by the king...
    undefended =   ei->attackedBy[Them][0]
                &  ei->attackedBy[Us][KING]
                & ~ei->attackedBy2[Us];

    // ... and those which are not defended at all in the larger king ring
    b =  ei->attackedBy[Them][0] & ~ei->attackedBy[Us][0]
       & ei->kingRing[Us] & ~pieces_c(Them);

    // Initialize the 'kingDanger' variable, which will be transformed
    // later into a king danger score. The initial value is based on the
    // number and types of the enemy's attacking pieces, the number of
    // attacked and undefended squares around our king and the quality of
    // the pawn shelter (current 'score' value).
    kingDanger =  min(807, ei->kingAttackersCount[Them] * ei->kingAttackersWeight[Them])
                + 101 * ei->kingAdjacentZoneAttacksCount[Them]
                + 235 * popcount(undefended)
                + 134 * (popcount(b) + !!ei->pinnedPieces[Us])
                - 717 * !pieces_cp(Them, QUEEN)
                -   7 * mg_value(score) / 5 - 5;

    // Analyse the enemy's safe queen contact checks. Firstly, find the
    // undefended squares around the king reachable by the enemy queen...
    b = undefended & ei->attackedBy[Them][QUEEN] & ~pieces_c(Them);

    // ...and keep squares supported by another enemy piece
    kingDanger += QueenContactCheck * popcount(b & ei->attackedBy2[Them]);

    // Analyse the safe enemy's checks which are possible on next move...
    safe  = ~(ei->attackedBy[Us][0] | pieces_c(Them));

    // ... and some other potential checks, only requiring the square to be
    // safe from pawn-attacks, and not being occupied by a blocked pawn.
    other = ~(   ei->attackedBy[Us][PAWN]
              | (pieces_cp(Them, PAWN) & shift_bb(Up, pieces_p(PAWN))));

    b1 = attacks_from_rook(ksq);
    b2 = attacks_from_bishop(ksq);

    // Enemy queen safe checks
    if ((b1 | b2) & ei->attackedBy[Them][QUEEN] & safe) {
      kingDanger += QueenCheck;
      score -= SafeCheck;
    }

    // For other pieces, also consider the square safe if attacked twice,
    // and only defended by a queen.
    safe |=  ei->attackedBy2[Them]
           & ~(ei->attackedBy2[Us] | pieces_c(Them))
           & ei->attackedBy[Us][QUEEN];

    // Enemy rooks safe and other checks
    if (b1 & ei->attackedBy[Them][ROOK] & safe) {
      kingDanger += RookCheck;
      score -= SafeCheck;
    }

    else if (b1 & ei->attackedBy[Them][ROOK] & other)
      score -= OtherCheck;

    // Enemy bishops safe and other checks
    if (b2 & ei->attackedBy[Them][BISHOP] & safe) {
      kingDanger += BishopCheck;
      score -= SafeCheck;
    }

    else if (b2 & ei->attackedBy[Them][BISHOP] & other)
      score -= OtherCheck;

    // Enemy knights safe and other checks
    b = attacks_from_knight(ksq) & ei->attackedBy[Them][KNIGHT];
    if (b & safe) {
      kingDanger += KnightCheck;
      score -= SafeCheck;
    }

    else if (b & other)
      score -= OtherCheck;

    // Compute the king danger score and subtract it from the evaluation.
    // Finally, extract the king danger score from the KingDanger[]
    if (kingDanger > 0)
      score -= make_score(min(kingDanger * kingDanger / 4096, 2 * BishopValueMg), 0);
  }

  // King tropism: firstly, find squares that we attack in the enemy king flank
  uint32_t kf = file_of(ksq);
  b = ei->attackedBy[Them][0] & KingFlank[Us][kf];

  assert(((Us == WHITE ? b << 4 : b >> 4) & b) == 0);
  assert(popcount(Us == WHITE ? b << 4 : b >> 4) == popcount(b));

  // Secondly, add the squares which are attacked twice in that flank and
  // which are not defended by our pawns.
  b =  (Us == WHITE ? b << 4 : b >> 4)
     | (b & ei->attackedBy2[Them] & ~ei->attackedBy[Us][PAWN]);

  score -= CloseEnemies * popcount(b);

  // Penalty when our king is on a pawnless flank.
  if (!(pieces_p(PAWN) & (KingFlank[WHITE][kf] | KingFlank[BLACK][kf])))
    score -= PawnlessFlank;

  return score;
}


// evaluate_threats() assigns bonuses according to the types of the
// attacking and the attacked pieces.

INLINE Score evaluate_threats(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them  = (Us == WHITE ? BLACK    : WHITE);
  const int Up    = (Us == WHITE ? DELTA_N  : DELTA_S);
  const int Left  = (Us == WHITE ? DELTA_NW : DELTA_SE);
  const int Right = (Us == WHITE ? DELTA_NE : DELTA_SW);
  const Bitboard TRank2BB = (Us == WHITE ? Rank2BB  : Rank7BB);
  const Bitboard TRank7BB = (Us == WHITE ? Rank7BB  : Rank2BB);

  enum { Minor, Rook };

  Bitboard b, weak, defended, safeThreats;
  Score score = SCORE_ZERO;

  // Small bonus if the opponent has loose pawns or pieces
  if (  pieces_c(Them) & ~pieces_pp(QUEEN, KING)
      & ~(ei->attackedBy[Us][0] | ei->attackedBy[Them][0]))
    score += LooseEnemies;

  // Non-pawn enemies attacked by a pawn
  weak = pieces_c(Them) & ~pieces_p(PAWN) & ei->attackedBy[Us][PAWN];

  if (weak) {
    b = pieces_cp(Us, PAWN) & ( ~ei->attackedBy[Them][0]
                               | ei->attackedBy[Us][0]);

    safeThreats = (shift_bb(Right, b) | shift_bb(Left, b)) & weak;

    if (weak ^ safeThreats)
      score += ThreatByHangingPawn;

    while (safeThreats)
      score += ThreatBySafePawn[piece_on(pop_lsb(&safeThreats)) - 8 * Them];
  }

  // Non-pawn enemies defended by a pawn
  defended = pieces_c(Them) & ~pieces_p(PAWN) & ei->attackedBy[Them][PAWN];

  // Enemies not defended by a pawn and under our attack
  weak =   pieces_c(Them)
        & ~ei->attackedBy[Them][PAWN]
        &  ei->attackedBy[Us][0];

  // Add a bonus according to the kind of attacking pieces
  if (defended | weak) {
    b = (defended | weak) & (ei->attackedBy[Us][KNIGHT] | ei->attackedBy[Us][BISHOP]);
    while (b) {
      Square s = pop_lsb(&b);
      score += Threat[Minor][piece_on(s) - 8 * Them];
      if (piece_on(s) != make_piece(Them, PAWN))
        score += ThreatByRank * relative_rank_s(Them, s);
    }

    b = (pieces_cp(Them, QUEEN) | weak) & ei->attackedBy[Us][ROOK];
    while (b) {
      Square s = pop_lsb(&b);
      score += Threat[Rook ][piece_on(s) - 8 * Them];
      if (piece_on(s) != make_piece(Them, PAWN))
        score += ThreatByRank * relative_rank_s(Them, s);
    }

    score += Hanging * popcount(weak & ~ei->attackedBy[Them][0]);

    b = weak & ei->attackedBy[Us][KING];
    if (b)
      score += ThreatByKing[!!more_than_one(b)];
  }

  // Bonus if some pawns can safely push and attack an enemy piece
  b = pieces_cp(Us, PAWN) & ~TRank7BB;
  b = shift_bb(Up, b | (shift_bb(Up, b & TRank2BB) & ~pieces()));

  b &=  ~pieces()
      & ~ei->attackedBy[Them][PAWN]
      & (ei->attackedBy[Us][0] | ~ei->attackedBy[Them][0]);

  b =  (shift_bb(Left, b) | shift_bb(Right, b))
     &  pieces_c(Them)
     & ~ei->attackedBy[Us][PAWN];

  score += ThreatByPawnPush * popcount(b);

  return score;
}


// evaluate_passed_pawns() evaluates the passed pawns of the given color.

INLINE Score evaluate_passed_pawns(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);

  Bitboard b, bb, squaresToQueen, defendedSquares, unsafeSquares;
  Score score = SCORE_ZERO;

  b = ei->pi->passedPawns[Us];

  while (b) {
    Square s = pop_lsb(&b);

    assert(pawn_passed(pos, Us, s));
    assert(!(pieces_p(PAWN) & forward_bb(Us, s)));

    bb = forward_bb(Us, s) & (ei->attackedBy[Them][0] | pieces_c(Them));
    score -= HinderPassedPawn * popcount(bb);

    int r = relative_rank_s(Us, s) - RANK_2;
    int rr = r * (r - 1);

    Value mbonus = Passed[MG][r], ebonus = Passed[EG][r];

    if (rr) {
      Square blockSq = s + pawn_push(Us);

      // Adjust bonus based on the king's proximity
      ebonus +=  distance(square_of(Them, KING), blockSq) * 5 * rr
               - distance(square_of(Us, KING), blockSq) * 2 * rr;

      // If blockSq is not the queening square then consider also a second push
      if (relative_rank_s(Us, blockSq) != RANK_8)
        ebonus -= distance(square_of(Us, KING), blockSq + pawn_push(Us)) * rr;

      // If the pawn is free to advance, then increase the bonus
      if (is_empty(blockSq)) {
        // If there is a rook or queen attacking/defending the pawn from behind,
        // consider all the squaresToQueen. Otherwise consider only the squares
        // in the pawn's path attacked or occupied by the enemy.
        defendedSquares = unsafeSquares = squaresToQueen = forward_bb(Us, s);

        bb = forward_bb(Them, s) & pieces_pp(ROOK, QUEEN) & attacks_from_rook(s);

        if (!(pieces_c(Us) & bb))
          defendedSquares &= ei->attackedBy[Us][0];

        if (!(pieces_c(Them) & bb))
          unsafeSquares &= ei->attackedBy[Them][0] | pieces_c(Them);

        // If there aren't any enemy attacks, assign a big bonus. Otherwise
        // assign a smaller bonus if the block square isn't attacked.
        int k = !unsafeSquares ? 18 : !(unsafeSquares & sq_bb(blockSq)) ? 8 : 0;

        // If the path to the queen is fully defended, assign a big bonus.
        // Otherwise assign a smaller bonus if the block square is defended.
        if (defendedSquares == squaresToQueen)
          k += 6;

        else if (defendedSquares & sq_bb(blockSq))
          k += 4;

        mbonus += k * rr, ebonus += k * rr;
      }
      else if (pieces_c(Us) & sq_bb(blockSq))
        mbonus += rr + r * 2, ebonus += rr + r * 2;
    } // rr != 0

    score += make_score(mbonus, ebonus) + PassedFile[file_of(s)];
  }

  // Add the scores to the middlegame and endgame eval
  return score;
}


// evaluate_space() computes the space evaluation for a given side. The
// space evaluation is a simple bonus based on the number of safe squares
// available for minor pieces on the central four files on ranks 2--4. Safe
// squares one, two or three squares behind a friendly pawn are counted
// twice. Finally, the space bonus is multiplied by a weight. The aim is to
// improve play on game opening.

INLINE Score evaluate_space(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const Bitboard SpaceMask =
    Us == WHITE ? (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank2BB | Rank3BB | Rank4BB)
                : (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank7BB | Rank6BB | Rank5BB);

  // Find the safe squares for our pieces inside the area defined by
  // SpaceMask. A square is unsafe if it is attacked by an enemy
  // pawn, or if it is undefended and attacked by an enemy piece.
  Bitboard safe =   SpaceMask
                 & ~pieces_cp(Us, PAWN)
                 & ~ei->attackedBy[Them][PAWN]
                 & (ei->attackedBy[Us][0] | ~ei->attackedBy[Them][0]);

  // Find all squares which are at most three squares behind some friendly pawn
  Bitboard behind = pieces_cp(Us, PAWN);
  behind |= (Us == WHITE ? behind >>  8 : behind <<  8);
  behind |= (Us == WHITE ? behind >> 16 : behind << 16);

  // Since SpaceMask[Us] is fully on our half of the board...
  assert((unsigned)(safe >> (Us == WHITE ? 32 : 0)) == 0);

  // ...count safe + (behind & safe) with a single popcount
  int bonus = popcount((Us == WHITE ? safe << 32 : safe >> 32) | (behind & safe));
  bonus = min(16, bonus);
  int weight = popcount(pieces_c(Us)) - 2 * ei->pi->openFiles;

  return make_score(bonus * weight * weight / 18, 0);
}


// evaluate_initiative() computes the initiative correction value for the
// position, i.e., second order bonus/malus based on the known
// attacking/defending status of the players.

// Since only eg is involved, we return a Value and not a Score.
INLINE Value evaluate_initiative(const Pos *pos, int asymmetry, Value eg)
{
  int kingDistance =  distance_f(square_of(WHITE, KING), square_of(BLACK, KING))
                    - distance_r(square_of(WHITE, KING), square_of(BLACK, KING));
  int pawns = popcount(pieces_p(PAWN));

  // Compute the initiative bonus for the attacking side
  int initiative = 8 * (asymmetry + kingDistance - 15) + 12 * pawns;

  // Now apply the bonus: note that we find the attacking side by extracting
  // the sign of the endgame value, and that we carefully cap the bonus so
  // that the endgame score will never be divided by more than two.
  Value value = ((eg > 0) - (eg < 0)) * max(initiative, -abs(eg / 2));

//  return make_score(0, value);
  return value;
}

// evaluate_scale_factor() computes the scale factor for the winning side

INLINE int evaluate_scale_factor(const Pos *pos, EvalInfo *ei, Value eg)
{
  int strongSide = eg > VALUE_DRAW ? WHITE : BLACK;
  int sf = material_scale_factor(ei->me, pos, strongSide);

  // If we don't already have an unusual scale factor, check for certain
  // types of endgames, and use a lower scale for those.
  if (    ei->me->gamePhase < PHASE_MIDGAME
      && (sf == SCALE_FACTOR_NORMAL || sf == SCALE_FACTOR_ONEPAWN)) {
    if (opposite_bishops(pos)) {
      // Endgame with opposite-colored bishops and no other pieces
      // (ignoring pawns) is almost a draw, in case of KBP vs KB, it is
      // even more a draw.
      if (   pos_non_pawn_material(WHITE) == BishopValueMg
          && pos_non_pawn_material(BLACK) == BishopValueMg)
        sf = more_than_one(pieces_p(PAWN)) ? 31 : 9;

      // Endgame with opposite-colored bishops, but also other pieces. Still
      // a bit drawish, but not as drawish as with only the two bishops.
      else
        sf = 46;
    }
    // Endings where weaker side can place his king in front of the opponent's
    // pawns are drawish.
    else if (    abs(eg) <= BishopValueEg
             &&  piece_count(strongSide, PAWN) <= 2
             && !pawn_passed(pos, strongSide ^ 1, square_of(strongSide ^ 1, KING)))
      sf = 37 + 7 * piece_count(strongSide, PAWN);
  }

  return sf;
}


// evaluate() is the main evaluation function. It returns a static evaluation
// of the position from the point of view of the side to move.

Value evaluate(const Pos *pos)
{
  assert(!pos_checkers());

  Score mobility[2] = { SCORE_ZERO, SCORE_ZERO };
  EvalInfo ei;

  // Probe the material hash table
  ei.me = material_probe(pos);

  // If we have a specialized evaluation function for the current material
  // configuration, call it and return.
  if (material_specialized_eval_exists(ei.me))
    return material_evaluate(ei.me, pos);

  // Initialize score by reading the incrementally updated scores included
  // in the position struct (material + piece square tables) and the
  // material imbalance. Score is computed internally from the white point
  // of view.
  Score score = pos_psq_score() + material_imbalance(ei.me);

  // Probe the pawn hash table
  ei.pi = pawn_probe(pos);
  score += ei.pi->score;

  // Initialize attack and king safety bitboards.
  ei.attackedBy[WHITE][0] = ei.attackedBy[BLACK][0] = 0;
  ei.attackedBy[WHITE][KING] = attacks_from_king(square_of(WHITE, KING));
  ei.attackedBy[BLACK][KING] = attacks_from_king(square_of(BLACK, KING));
  evalinfo_init(pos, &ei, WHITE);
  evalinfo_init(pos, &ei, BLACK);

  // Pawns blocked or on ranks 2 and 3 will be excluded from the mobility area
  Bitboard blockedPawns[] = {
    pieces_cp(WHITE, PAWN) & (shift_bb_S(pieces()) | Rank2BB | Rank3BB),
    pieces_cp(BLACK, PAWN) & (shift_bb_N(pieces()) | Rank7BB | Rank6BB)
  };

  // Do not include in mobility area squares protected by enemy pawns, or
  // occupied by our blocked pawns or king.
  Bitboard mobilityArea[] = {
    ~(ei.attackedBy[BLACK][PAWN] | blockedPawns[WHITE] | pieces_cp(WHITE, KING)),
    ~(ei.attackedBy[WHITE][PAWN] | blockedPawns[BLACK] | pieces_cp(BLACK, KING))
  };

  // Evaluate all pieces but king and pawns
  score += evaluate_pieces(pos, &ei, mobility, mobilityArea);
  score += mobility[WHITE] - mobility[BLACK];

  // Evaluate kings after all other pieces because we need full attack
  // information when computing the king safety evaluation.
  score +=  evaluate_king(pos, &ei, WHITE)
          - evaluate_king(pos, &ei, BLACK);

  // Evaluate tactical threats, we need full attack information including king
  score +=  evaluate_threats(pos, &ei, WHITE)
          - evaluate_threats(pos, &ei, BLACK);

  // Evaluate passed pawns, we need full attack information including king
  score +=  evaluate_passed_pawns(pos, &ei, WHITE)
          - evaluate_passed_pawns(pos, &ei, BLACK);

  // If both sides have only pawns, score for potential unstoppable pawns
  if (pos_pawns_only()) {
    Bitboard b;
    if ((b = ei.pi->passedPawns[WHITE]) != 0)
      score += Unstoppable * relative_rank_s(WHITE, frontmost_sq(WHITE, b));

    if ((b = ei.pi->passedPawns[BLACK]) != 0)
      score -= Unstoppable * relative_rank_s(BLACK, frontmost_sq(BLACK, b));
  }

  // Evaluate space for both sides, only during opening
  if (pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK) >= 12222)
      score +=  evaluate_space(pos, &ei, WHITE)
              - evaluate_space(pos, &ei, BLACK);

  // Evaluate position potential for the winning side
  //  score += evaluate_initiative(pos, ei.pi->asymmetry, eg_value(score));
  int eg = eg_value(score);
  eg += evaluate_initiative(pos, ei.pi->asymmetry, eg);

  // Evaluate scale factor for the winning side
  //int sf = evaluate_scale_factor(pos, &ei, eg_value(score));
  int sf = evaluate_scale_factor(pos, &ei, eg);

  // Interpolate between a middlegame and a (scaled by 'sf') endgame score
  //  Value v =  mg_value(score) * ei.me->gamePhase
  //           + eg_value(score) * (PHASE_MIDGAME - ei.me->gamePhase) * sf / SCALE_FACTOR_NORMAL;
  Value v =  mg_value(score) * ei.me->gamePhase
           + eg * (PHASE_MIDGAME - ei.me->gamePhase) * sf / SCALE_FACTOR_NORMAL;

  v /= PHASE_MIDGAME;

  return (pos_stm() == WHITE ? v : -v) + Tempo; // Side to move point of view
}

