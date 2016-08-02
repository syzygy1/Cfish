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

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

// Trace

#define MATERIAL  8
#define IMBALANCE 9
#define MOBILITY  10
#define THREAT    11
#define PASSED    12
#define SPACE     13
#define TOTAL     14
#define TERM_NB   15

#if 0
static double scores[TERM_NB][2][2];

static inline double to_cp(Value v)
{
  return ((double)v) / PawnValueEg;
}

static void add_c(int idx, int c, Score s) {
  scores[idx][c][MG] = to_cp(mg_value(s));
  scores[idx][c][EG] = to_cp(eg_value(s));
}

static void add(int idx, Score w, Score b) {
  add(idx, WHITE, w);
  add(idx, BLACK, b);
}
#endif

#if 0
static void print_term(Term t);

std::ostream& operator<<(std::ostream& os, Term t) {

  if (t == MATERIAL || t == IMBALANCE || t == Term(PAWN) || t == TOTAL)
      os << "  ---   --- |   ---   --- | ";
  else
      os << std::setw(5) << scores[t][WHITE][MG] << " "
         << std::setw(5) << scores[t][WHITE][EG] << " | "
         << std::setw(5) << scores[t][BLACK][MG] << " "
         << std::setw(5) << scores[t][BLACK][EG] << " | ";

  os << std::setw(5) << scores[t][WHITE][MG] - scores[t][BLACK][MG] << " "
     << std::setw(5) << scores[t][WHITE][EG] - scores[t][BLACK][EG] << " \n";

  return os;
}
#endif

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
#define S(mg,eg) make_score((unsigned)(mg),(unsigned)(eg))

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
static const Score Threat[][8] = {
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
static const Score SafeCheck           = S(20, 20);
static const Score OtherCheck          = S(10, 10);
static const Score ThreatByHangingPawn = S(71, 61);
static const Score LooseEnemies        = S( 0, 25);
static const Score WeakQueen           = S(35,  0);
static const Score Hanging             = S(48, 27);
static const Score ThreatByPawnPush    = S(38, 22);
static const Score Unstoppable         = S( 0, 20);

// Penalty for a bishop on a1/h1 (a8/h8 for black) which is trapped by
// a friendly pawn on b2/g2 (b7/g7 for black). This can obviously only
// happen in Chess960 games.
static const Score TrappedBishopA1H1 = S(50, 50);

#undef S
#undef V

// King danger constants and variables. The king danger scores are looked-up
// in KingDanger[]. Various little "meta-bonuses" measuring the strength
// of the enemy attack are added up into an integer, which is used as an
// index to KingDanger[].
static Score KingDanger[512];

// KingAttackWeights[PieceType] contains king attack weights by piece type
static const int KingAttackWeights[8] = { 0, 0, 7, 5, 4, 1 };

// Penalties for enemy's safe checks
#define QueenContactCheck 89
#define QueenCheck        62
#define RookCheck         57
#define BishopCheck       48
#define KnightCheck       78


#define Us WHITE
#include "tmpleval.c"
#undef Us
#define Us BLACK
#include "tmpleval.c"
#undef Us


// evaluate_initiative() computes the initiative correction value for the
// position, i.e., second order bonus/malus based on the known
// attacking/defending status of the players.
Score evaluate_initiative(Pos *pos, int asymmetry, Value eg)
{
  int kingDistance =  distance_f(square_of(WHITE, KING), square_of(BLACK, KING))
                    - distance_r(square_of(WHITE, KING), square_of(BLACK, KING));
  int pawns = popcount(pieces_p(PAWN));

  // Compute the initiative bonus for the attacking side
  int initiative = 8 * (asymmetry + kingDistance - 15) + 12 * pawns;

  // Now apply the bonus: note that we find the attacking side by extracting
  // the sign of the endgame value, and that we carefully cap the bonus so
  // that the endgame score will never be divided by more than two.
  int value = ((eg > 0) - (eg < 0)) * max(initiative, -abs(eg / 2));

  return make_score(0, value);
}

// evaluate_scale_factor() computes the scale factor for the winning side
int evaluate_scale_factor(Pos *pos, EvalInfo *ei, Value eg)
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
             &&  ei->pi->pawnSpan[strongSide] <= 1
             && !pawn_passed(pos, strongSide ^ 1, square_of(strongSide ^ 1, KING)))
      sf = ei->pi->pawnSpan[strongSide] ? 51 : 37;
  }

  return sf;
}


// evaluate() is the main evaluation function. It returns a static evaluation
// of the position from the point of view of the side to move.

//template<bool DoTrace>
Value evaluate(Pos *pos)
{
  assert(!pos_checkers());

  EvalInfo ei;
  Score score, mobility[2] = { SCORE_ZERO, SCORE_ZERO };

  // Initialize score by reading the incrementally updated scores included in
  // the position object (material + piece square tables). Score is computed
  // internally from the white point of view.
  score = pos_psq_score();

  // Probe the material hash table
  ei.me = material_probe(pos);
  score += material_imbalance(ei.me);

  // If we have a specialized evaluation function for the current material
  // configuration, call it and return.
  if (material_specialized_eval_exists(ei.me))
    return material_evaluate(ei.me, pos);

  // Probe the pawn hash table
  ei.pi = pawn_probe(pos);
  score += ei.pi->score;

  // Initialize attack and king safety bitboards
  ei.attackedBy[WHITE][0] = ei.attackedBy[BLACK][0] = 0;
  ei.attackedBy[WHITE][KING] = attacks_from_king(square_of(WHITE, KING));
  ei.attackedBy[BLACK][KING] = attacks_from_king(square_of(BLACK, KING));
  eval_init_white(pos, &ei);
  eval_init_black(pos, &ei);

  // Pawns blocked or on ranks 2 and 3 will be excluded from the mobility area
  Bitboard blockedPawns[] = {
    pieces_cp(WHITE, PAWN) & (shift_bb_S(pieces()) | Rank2BB | Rank3BB),
    pieces_cp(BLACK, PAWN) & (shift_bb_N(pieces()) | Rank7BB | Rank6BB)
  };

  // Do not include in mobility area squares protected by enemy pawns, or occupied
  // by our blocked pawns or king.
  Bitboard mobilityArea[] = {
    ~(ei.attackedBy[BLACK][PAWN] | blockedPawns[WHITE] | pieces_cp(WHITE, KING)),
    ~(ei.attackedBy[WHITE][PAWN] | blockedPawns[BLACK] | pieces_cp(BLACK, KING))
  };

  // Evaluate all pieces but king and pawns
  score +=  evaluate_pieces_white(pos, &ei, mobility, mobilityArea)
          - evaluate_pieces_black(pos, &ei, mobility, mobilityArea);
  score += mobility[WHITE] - mobility[BLACK];

  // Evaluate kings after all other pieces because we need full attack
  // information when computing the king safety evaluation.
  score +=  evaluate_king_white(pos, &ei)
          - evaluate_king_black(pos, &ei);

  // Evaluate tactical threats, we need full attack information including king
  score +=  evaluate_threats_white(pos, &ei)
          - evaluate_threats_black(pos, &ei);

  // Evaluate passed pawns, we need full attack information including king
  score +=  evaluate_passed_pawns_white(pos, &ei)
          - evaluate_passed_pawns_black(pos, &ei);

  // If both sides have only pawns, score for potential unstoppable pawns
  if (!pos_non_pawn_material(WHITE) && !pos_non_pawn_material(BLACK)) {
    Bitboard b;
    if ((b = ei.pi->passedPawns[WHITE]) != 0)
      score += Unstoppable * relative_rank_s(WHITE, frontmost_sq(WHITE, b));

    if ((b = ei.pi->passedPawns[BLACK]) != 0)
      score -= Unstoppable * relative_rank_s(BLACK, frontmost_sq(BLACK, b));
  }

  // Evaluate space for both sides, only during opening
  if (pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK) >= 12222)
      score +=  evaluate_space_white(pos, &ei)
              - evaluate_space_black(pos, &ei);

  // Evaluate position potential for the winning side
  score += evaluate_initiative(pos, ei.pi->asymmetry, eg_value(score));

  // Evaluate scale factor for the winning side
  int sf = evaluate_scale_factor(pos, &ei, eg_value(score));

  // Interpolate between a middlegame and a (scaled by 'sf') endgame score
  Value v =  mg_value(score) * ei.me->gamePhase
           + eg_value(score) * (PHASE_MIDGAME - ei.me->gamePhase) * sf / SCALE_FACTOR_NORMAL;

  v /= PHASE_MIDGAME;

#if 0
  // In case of tracing add all remaining individual evaluation terms
  if (DoTrace)
  {
      Trace::add(MATERIAL, pos.psq_score());
      Trace::add(IMBALANCE, ei.me->imbalance());
      Trace::add(PAWN, ei.pi->pawns_score());
      Trace::add(MOBILITY, mobility[WHITE], mobility[BLACK]);
      Trace::add(SPACE, evaluate_space<WHITE>(pos, &ei)
                      , evaluate_space<BLACK>(pos, &ei));
      Trace::add(TOTAL, score);
  }
#endif

  return (pos_stm() == WHITE ? v : -v) + Tempo; // Side to move point of view
}

#if 0

// eval_trace() is like evaluate(), but prints the detailed descriptions
// and values of each evaluation term to stdout. Useful for debugging.

void eval_trace(Pos *pos)
{
  memset(scores, 0, sizeof(scores));

  Value v = evaluate<true>(pos);
  v = pos->stm == WHITE ? v : -v; // White's point of view

  printf("      Eval term |    White    |    Black    |    Total    \n"
         "                |   MG    EG  |   MG    EG  |   MG    EG  \n"
         "----------------+-------------+-------------+-------------\n"
         "       Material | %s"
         "      Imbalance | %s"
         "          Pawns | %s"
         "        Knights | %s"
         "         Bishop | %s"
         "          Rooks | %s"
         "         Queens | %s"
         "       Mobility | %s"
         "    King safety | %s"
         "        Threats | %s"
         "   Passed pawns | %s"
         "          Space | %s"
         "----------------+-------------+-------------+-------------\n"
         "          Total | %s"
         Term(MATERIAL), Term(IMBALANCE), Term(PAWN), Term(KNIGHT),
         Term(BISHOP), Term(ROOK), Term(QUEEN), Term(MOBILITY),
         Term(KING), Term(THREAT), Term(PASSED), Term(SPACE), Term(TOTAL));

  printf("\nTotal Evaluation: %d (white side)\n", to_cp(v));
}
#endif


// eval_init() computes evaluation weights, usually at startup.

void eval_init() {
  int MaxSlope = 322;
  int Peak = 47410;
  int t = 0;

  for (int i = 0; i < 400; ++i) {
    t = min(Peak, min(i * i - 16, t + MaxSlope));
    KingDanger[i] = make_score(t * 268 / 7700, 0);
  }
}

