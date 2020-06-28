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
#include "evaluate.h"
#include "material.h"
#include "pawns.h"

// Struct EvalInfo contains various information computed and collected
// by the evaluation functions.
struct EvalInfo {
  MaterialEntry *me;
  PawnEntry *pe;
  Bitboard mobilityArea[2];

  // attackedBy[color][piece type] is a bitboard representing all squares
  // attacked by a given color and piece type. A special "piece type" which
  // is also calculated is ALL_PIECES.
  Bitboard attackedBy[2][8];

  // attackedBy2[color] are the squares attacked by 2 pieces of a given
  // color, possibly via x-ray or by one pawn and one piece. Diagonal
  // x-ray through pawn or squares attacked by 2 pawns are not explicitly
  // added.
  Bitboard attackedBy2[2];

  // kingRing[color] are the squares adjacent to the king plus some other
  // very near squares, depending on king position.
  Bitboard kingRing[2];

  // kingAttackersCount[color] is the number of pieces of the given color
  // which attack a square in the kingRing of the enemy king.
  int kingAttackersCount[2];

  // kingAttackersWeight[color] is the sum of the "weights" of the pieces
  // of the given color which attack a square in the kingRing of the enemy
  // king. The weights of the individual piece types are given by the
  // elements in the KingAttackWeights array.
  int kingAttackersWeight[2];

  // kingAttacksCount[color] is the number of attacks by the given color to
  // squares directly adjacent to the enemy king. Pieces which attack more
  // than one square are counted multiple times. For instance, if there is
  // a white knight on g5 and black's king is on g8, this white knight adds
  // 2 to kingAttacksCount[WHITE].
  int kingAttacksCount[2];
};

typedef struct EvalInfo EvalInfo;

#define Center      ((FileDBB | FileEBB) & (Rank4BB | Rank5BB))
#define QueenSide   (FileABB | FileBBB | FileCBB | FileDBB)
#define CenterFiles (FileCBB | FileDBB | FileEBB | FileFBB)
#define KingSide    (FileEBB | FileFBB | FileGBB | FileHBB)

static const Bitboard KingFlank[8] = {
  QueenSide ^ FileDBB, QueenSide, QueenSide, CenterFiles,
  CenterFiles, KingSide, KingSide, KingSide ^ FileEBB
};

// Thresholds for lazy and space evaluation
enum { LazyThreshold = 1400, SpaceThreshold = 12222 };

// KingAttackWeights[PieceType] contains king attack weights by piece type
static const int KingAttackWeights[8] = { 0, 0, 81, 52, 44, 10 };

// Penalties for enemy's safe checks
enum {
  QueenSafeCheck  = 780,
  RookSafeCheck   = 1080,
  BishopSafeCheck = 635,
  KnightSafeCheck = 790
};

#define V(v) (Value)(v)
#define S(mg,eg) make_score(mg,eg)

// MobilityBonus[PieceType-2][attacked] contains bonuses for middle and
// end game, indexed by piece type and number of attacked squares in the
// mobility area.
static const Score MobilityBonus[4][32] = {
  // Knights
  { S(-62,-81), S(-53,-56), S(-12,-30), S( -4,-14), S(  3,  8), S( 13, 15),
    S( 22, 23), S( 28, 27), S( 33, 33) },
  // Bishops
  { S(-48,-59), S(-20,-23), S( 16, -3), S( 26, 13), S( 38, 24), S( 51, 42),
    S( 55, 54), S( 63, 57), S( 63, 65), S( 68, 73), S( 81, 78), S( 81, 86),
    S( 91, 88), S( 98, 97) },
  // Rooks
  { S(-58,-76), S(-27,-18), S(-15, 28), S(-10, 55), S( -5, 69), S( -2, 82),
    S(  9,112), S( 16,118), S( 30,132), S( 29,142), S( 32,155), S( 38,165),
    S( 46,166), S( 48,169), S( 58,171) },
  // Queens
  { S(-39,-36), S(-21,-15), S(  3,  8), S(  3, 18), S( 14, 34), S( 22, 54),
    S( 28, 61), S( 41, 73), S( 43, 79), S( 48, 92), S( 56, 94), S( 60,104),
    S( 60,113), S( 66,120), S( 67,123), S( 70,126), S( 71,133), S( 73,136),
    S( 79,140), S( 88,143), S( 88,148), S( 99,166), S(102,170), S(102,175),
    S(106,184), S(109,191), S(113,206), S(116,212) }
};

// RookOnFile[semiopen/open] contains bonuses for each rook when there is
// no friendly pawn on the rook file.
static const Score RookOnFile[2] = { S(21, 4), S(47, 25) };

// ThreatByMinor/ByRook[attacked PieceType] contains bonuses according to
// which piece type attacks which one. Attacks on lesser pieces which are
// pawn defended are not considered.
static const Score ThreatByMinor[8] = {
  S(0, 0), S(6, 32), S(59, 41), S(79, 56), S(90,119), S(79,161)
};

static const Score ThreatByRook[8] = {
  S(0, 0), S(3, 44), S(38, 71), S(38, 61), S( 0, 38), S(51, 38)
};

// PassedRank[mg/eg][Rank] contains midgame and endgame bonuses for passed
// pawns. We don't use a Score because we process the two components
// independently.
static const Value PassedRank[][8] = {
  { V(0), V(10), V(17), V(15), V(62), V(168), V(276) },
  { V(0), V(28), V(33), V(41), V(72), V(177), V(260) }
};

// PassedFile[File] contains a bonus according to the file of a passed pawn
static const Score PassedFile[8] = {
  S( 0,  0), S(11,  8), S(22, 16), S(33, 24),
  S(33, 24), S(22, 16), S(11,  8), S( 0,  0)
};

// Assorted bonuses and penalties used by evaluation
static const Score BishopPawns        = S(  3,  7);
static const Score CorneredBishop     = S( 50, 50);
static const Score FlankAttacks       = S(  8,  0);
static const Score Hanging            = S( 69, 36);
static const Score KingProtector      = S(  7,  8);
static const Score KnightOnQueen      = S( 16, 12);
static const Score LongDiagonalBishop = S( 45,  0);
static const Score MinorBehindPawn    = S( 18,  3);
static const Score Outpost            = S( 30, 21);
static const Score PawnlessFlank      = S( 17, 95);
static const Score RestrictedPiece    = S(  7,  7);
static const Score ReachableOutpost   = S( 32, 10);
static const Score RookOnQueenFile    = S(  7,  6);
static const Score SliderOnQueen      = S( 59, 18);
static const Score ThreatByKing       = S( 24, 89);
static const Score ThreatByPawnPush   = S( 48, 39);
static const Score ThreatBySafePawn   = S(173, 94);
static const Score TrappedRook        = S( 52, 10);
static const Score WeakQueen          = S( 49, 15);

#undef S
#undef V

// eval_init() initializes king and attack bitboards for a given color
// adding pawn attacks. To be done at the beginning of the evaluation.

INLINE void evalinfo_init(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const int Down = (Us == WHITE ? SOUTH : NORTH);
  const Bitboard LowRanks = (Us == WHITE ? Rank2BB | Rank3BB
                                         : Rank7BB | Rank6BB);

  const Square ksq = square_of(Us, KING);

  Bitboard dblAttackByPawn = pawn_double_attacks_bb(pieces_cp(Us, PAWN), Us);

  // Find our pawns on the first two ranks, and those which are blocked
  Bitboard b = pieces_cp(Us, PAWN) & (shift_bb(Down, pieces()) | LowRanks);

  // Squares occupied by those pawns, by our king or queen, by blockers to
  // attacks on our king or controlled by enemy pawns are excluded from the
  // mobility area
  ei->mobilityArea[Us] = ~(b | pieces_cpp(Us, KING, QUEEN) | blockers_for_king(pos, Us) | ei->pe->pawnAttacks[Them]);

  // Initialise attackedBy[] for kings and pawns
  b = ei->attackedBy[Us][KING] = attacks_from_king(square_of(Us, KING));
  ei->attackedBy[Us][PAWN] = ei->pe->pawnAttacks[Us];
  ei->attackedBy[Us][0] = b | ei->attackedBy[Us][PAWN];
  ei->attackedBy2[Us]   = (b & ei->attackedBy[Us][PAWN]) | dblAttackByPawn;

  // Init our king safety tables only if we are going to use them
  Square s = make_square(clamp(file_of(ksq), FILE_B, FILE_G),
                         clamp(rank_of(ksq), RANK_2, RANK_7));
  ei->kingRing[Us] = PseudoAttacks[KING][s] | sq_bb(s);

  ei->kingAttackersCount[Them] = popcount(ei->kingRing[Us] & ei->pe->pawnAttacks[Them]);
  ei->kingRing[Us] &= ~dblAttackByPawn;
  ei->kingAttacksCount[Them] = ei->kingAttackersWeight[Them] = 0;
}

// evaluate_piece() assigns bonuses and penalties to the pieces of a given
// color and type.

INLINE Score evaluate_piece(const Pos *pos, EvalInfo *ei, Score *mobility,
                            const int Us, const int Pt)
{
  const int Them  = (Us == WHITE ? BLACK      : WHITE);
  const int Down  = (Us == WHITE ? SOUTH      : NORTH);
  const Bitboard OutpostRanks = (Us == WHITE ? Rank4BB | Rank5BB | Rank6BB
                                             : Rank5BB | Rank4BB | Rank3BB);

  Bitboard b, bb;
  Square s;
  Score score = SCORE_ZERO;

  ei->attackedBy[Us][Pt] = 0;

  loop_through_pieces(Us, Pt, s) {
    // Find attacked squares, including x-ray attacks for bishops and rooks
    b = Pt == BISHOP ? attacks_bb_bishop(s, pieces() ^ pieces_p(QUEEN))
      : Pt == ROOK ? attacks_bb_rook(s,
                              pieces() ^ pieces_p(QUEEN) ^ pieces_cp(Us, ROOK))
                   : attacks_from(Pt, s);

    if (blockers_for_king(pos, Us) & sq_bb(s))
      b &= LineBB[square_of(Us, KING)][s];

    ei->attackedBy2[Us] |= ei->attackedBy[Us][0] & b;
    ei->attackedBy[Us][0] |= b;
    ei->attackedBy[Us][Pt] |= b;

    if (b & ei->kingRing[Them]) {
      ei->kingAttackersCount[Us]++;
      ei->kingAttackersWeight[Us] += KingAttackWeights[Pt];
      ei->kingAttacksCount[Us] += popcount(b & ei->attackedBy[Them][KING]);
    }

    int mob = popcount(b & ei->mobilityArea[Us]);

    mobility[Us] += MobilityBonus[Pt - 2][mob];

    if (Pt == BISHOP || Pt == KNIGHT) {
      // Bonus for outpost squares
      bb = OutpostRanks & ei->attackedBy[Us][PAWN] & ~ei->pe->pawnAttacksSpan[Them];
      if (bb & sq_bb(s))
        score += Outpost * (Pt == KNIGHT ? 2 : 1);

      else if (Pt == KNIGHT && bb & b & ~pieces_c(Us))
        score += ReachableOutpost;

      // Knight and Bishop bonus for being right behind a pawn
      if (shift_bb(Down, pieces_p(PAWN)) & sq_bb(s))
        score += MinorBehindPawn;

      // Penalty if the minor is far from the king
      score -= KingProtector * distance(s, square_of(Us, KING));

      if (Pt == BISHOP) {
        // Penalty according to number of pawns on the same color square as
        // the bishop, bigger when the center files are blocked with pawns
        Bitboard blocked = pieces_cp(Us, PAWN) & shift_bb(Down, pieces());

        score -= BishopPawns * pawns_on_same_color_squares(ei->pe, Us, s)
                             * (1 + popcount(blocked & CenterFiles));

        // Bonus for bishop on a long diagonal which can "see" both center
        // squares
        if (more_than_one(attacks_bb_bishop(s, pieces_p(PAWN)) & Center))
          score += LongDiagonalBishop;

        // An important Chess960 pattern: a cornered bishop blocked by a
        // friendly pawn diagonally in front of it is a very serious problem,
        // especially when that pawn is also blocked.
        if (   is_chess960()
            && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1)))
        {
          Square d = pawn_push(Us) + (file_of(s) == FILE_A ? EAST : WEST);
          if (piece_on(s + d) == make_piece(Us, PAWN))
            score -=  piece_on(s + d + pawn_push(Us))             ? CorneredBishop * 4
                    : piece_on(s + d + d) == make_piece(Us, PAWN) ? CorneredBishop * 2
                                                                  : CorneredBishop;
        }
      }
    }

    if (Pt == ROOK) {
      // Bonus for rook on the same file as a queen
      if (file_bb_s(s) & pieces_p(QUEEN))
        score += RookOnQueenFile;

      // Bonus for rook on an open or semi-open file
      if (semiopen_file(ei->pe, Us, file_of(s)))
        score += RookOnFile[!!semiopen_file(ei->pe, Them, file_of(s))];

      // Penalty when trapped by the king, even more if the king cannot castle
      else if (mob <= 3) {
        File kf = file_of(square_of(Us, KING));

        if ((kf < FILE_E) == (file_of(s) < kf))
          score -= TrappedRook * (1 + !can_castle_c(Us));
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

INLINE Score evaluate_pieces(const Pos *pos, EvalInfo *ei, Score *mobility)
{
  return  evaluate_piece(pos, ei, mobility, WHITE, KNIGHT)
        - evaluate_piece(pos, ei, mobility, BLACK, KNIGHT)
        + evaluate_piece(pos, ei, mobility, WHITE, BISHOP)
        - evaluate_piece(pos, ei, mobility, BLACK, BISHOP)
        + evaluate_piece(pos, ei, mobility, WHITE, ROOK)
        - evaluate_piece(pos, ei, mobility, BLACK, ROOK)
        + evaluate_piece(pos, ei, mobility, WHITE, QUEEN)
        - evaluate_piece(pos, ei, mobility, BLACK, QUEEN);
}


// evaluate_king() assigns bonuses and penalties to a king of a given color.

INLINE Score evaluate_king(const Pos *pos, EvalInfo *ei, Score *mobility,
                           int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const Bitboard Camp = (   Us == WHITE
                         ? AllSquares ^ Rank6BB ^ Rank7BB ^ Rank8BB
                         : AllSquares ^ Rank1BB ^ Rank2BB ^ Rank3BB);

  const Square ksq = square_of(Us, KING);
  Bitboard weak, b, b1, b2, b3, safe, unsafeChecks = 0;
  int kingDanger = 0;

  // King shelter and enemy pawns storm
  Score score = Us == WHITE ? king_safety_white(ei->pe, pos, ksq)
                            : king_safety_black(ei->pe, pos, ksq);

  // Attacked squares defended at most once by our queen or king
  weak =  ei->attackedBy[Them][0]
        & ~ei->attackedBy2[Us]
        & (   ei->attackedBy[Us][KING] | ei->attackedBy[Us][QUEEN]
           | ~ei->attackedBy[Us][0]);


  // Analyse the safe enemy's checks which are possible on next move
  safe  = ~pieces_c(Them);
  safe &= ~ei->attackedBy[Us][0] | (weak & ei->attackedBy2[Them]);

  b1 = attacks_bb_rook(ksq, pieces() ^ pieces_cp(Us, QUEEN));
  b2 = attacks_bb_bishop(ksq, pieces() ^ pieces_cp(Us, QUEEN));

  // Enemy rooks checks
  Bitboard RookCheck =  b1
                      & safe
                      & ei->attackedBy[Them][ROOK];
  if (RookCheck)
    kingDanger += RookSafeCheck;
  else
    unsafeChecks |= b1 & ei->attackedBy[Them][ROOK];

  Bitboard QueenCheck =  (b1 | b2)
                       & ei->attackedBy[Them][QUEEN]
                       & safe
                       & ~ei->attackedBy[Us][QUEEN]
                       & ~RookCheck;
  if (QueenCheck)
    kingDanger += QueenSafeCheck;

  // Enemy bishops checks: we count them only if they are from squares from
  // which we can't give a queen check, because queen checks are more valuable.
  Bitboard BishopCheck =  b2
                        & ei->attackedBy[Them][BISHOP]
                        & safe
                        & ~QueenCheck;
  if (BishopCheck)
    kingDanger += BishopSafeCheck;
  else
    unsafeChecks |= b2 & ei->attackedBy[Them][BISHOP];

  // Enemy knights checks
  b = attacks_from_knight(ksq) & ei->attackedBy[Them][KNIGHT];
  if (b & safe)
    kingDanger += KnightSafeCheck;
  else
    unsafeChecks |= b;

  // Find the squares that opponent attacks in our king flank, the squares
  // which they attack twice in that flank, and the squares that we defend.
  b1 = ei->attackedBy[Them][0] & KingFlank[file_of(ksq)] & Camp;
  b2 = b1 & ei->attackedBy2[Them];
  b3 = ei->attackedBy[Us][0] & KingFlank[file_of(ksq)] & Camp;

  int kingFlankAttack = popcount(b1) + popcount(b2);
  int kingFlankDefense = popcount(b3);

  kingDanger +=  ei->kingAttackersCount[Them] * ei->kingAttackersWeight[Them]
               + 185 * popcount(ei->kingRing[Us] & weak)
               + 148 * popcount(unsafeChecks)
               +  98 * popcount(blockers_for_king(pos, Us))
               +  69 * ei->kingAttacksCount[Them]
               +   3 * kingFlankAttack * kingFlankAttack / 8
               +       mg_value(mobility[Them] - mobility[Us])
               - 873 * !pieces_cp(Them, QUEEN)
               - 100 * !!(ei->attackedBy[Us][KNIGHT] & ei->attackedBy[Us][KING])
               -   6 * mg_value(score) / 8
               -   4 * kingFlankDefense
               +  37;

  // Transform the kingDanger units into a Score, and subtract it from
  // the evaluation
  if (kingDanger > 100)
    score -= make_score(kingDanger * kingDanger / 4096, kingDanger / 16);

  // Penalty when our king is on a pawnless flank
  if (!(pieces_p(PAWN) & KingFlank[file_of(ksq)]))
    score -= PawnlessFlank;

  // Penalty if king flank is under attack, potentially moving toward the king
  score -= FlankAttacks * kingFlankAttack;

  return score;
}


// evaluate_threats() assigns bonuses according to the types of the
// attacking and the attacked pieces.

INLINE Score evaluate_threats(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them  = (Us == WHITE ? BLACK      : WHITE);
  const int Up    = (Us == WHITE ? NORTH      : SOUTH);
  const Bitboard TRank3BB = (Us == WHITE ? Rank3BB  : Rank6BB);

  enum { Minor, Rook };

  Bitboard b, weak, defended, stronglyProtected, safe;
  Score score = SCORE_ZERO;

  // Non-pawn enemies
  Bitboard nonPawnEnemies = pieces_c(Them) & ~pieces_p(PAWN);

  // Squares strongly protected by the opponent, either because they attack the
  // square with a pawn or because they attack the square twice and we don't.
  stronglyProtected =  ei->attackedBy[Them][PAWN]
                     | (ei->attackedBy2[Them] & ~ei->attackedBy2[Us]);

  // Non-pawn enemies, strongly protected
  defended = nonPawnEnemies & stronglyProtected;

  // Enemies not strongly protected and under our attack
  weak = pieces_c(Them) & ~stronglyProtected & ei->attackedBy[Us][0];

  // Add a bonus according to the kind of attacking pieces
  if (defended | weak) {
    b = (defended | weak) & (ei->attackedBy[Us][KNIGHT] | ei->attackedBy[Us][BISHOP]);
    while (b)
      score += ThreatByMinor[piece_on(pop_lsb(&b)) - 8 * Them];

    b = weak & ei->attackedBy[Us][ROOK];
    while (b)
      score += ThreatByRook[piece_on(pop_lsb(&b)) - 8 * Them];

    // Bonus for king attacks on pawns or pieces which are not pawn-defended
    if (weak & ei->attackedBy[Us][KING])
      score += ThreatByKing;

    b =  ~ei->attackedBy[Them][0]
       | (nonPawnEnemies & ei->attackedBy2[Us]);
    score += Hanging * popcount(weak & b);
  }

  // Bonus for restricting their piece moves
  b =   ei->attackedBy[Them][0]
     & ~stronglyProtected
     &  ei->attackedBy[Us][0];
  score += RestrictedPiece * popcount(b);

  // Protected or unattacked squares
  safe = ~ei->attackedBy[Them][0] | ei->attackedBy[Us][0];

  // Bonus for attacking enemy pieces with our relatively safe pawns
  b = pieces_cp(Us, PAWN) & safe;
  b = pawn_attacks_bb(b, Us) & nonPawnEnemies;
  score += ThreatBySafePawn * popcount(b);

  // Find the squares reachable by a single pawn push
  b  = shift_bb(Up, pieces_cp(Us, PAWN)) & ~pieces();
  b |= shift_bb(Up, b & TRank3BB) & ~pieces();

  // Keep only those squares which are relatively safe
  b &= ~ei->attackedBy[Them][PAWN] & safe;

  // Bonus for safe pawn threats on the next move
  b = pawn_attacks_bb(b, Us) & nonPawnEnemies;
  score += ThreatByPawnPush * popcount(b);

  // Bonus for impending threats against enemy queen
  if (piece_count(Them, QUEEN) == 1) {
    Square s = square_of(Them, QUEEN);
    safe = ei->mobilityArea[Us] & ~stronglyProtected;

    b = ei->attackedBy[Us][KNIGHT] & attacks_from_knight(s);

    score += KnightOnQueen * popcount(b & safe);

    b =  (ei->attackedBy[Us][BISHOP] & attacks_from_bishop(s))
       | (ei->attackedBy[Us][ROOK  ] & attacks_from_rook(s));

    score += SliderOnQueen * popcount(b & safe & ei->attackedBy2[Us]);
  }

  return score;
}


// Helper function
INLINE int capped_distance(Square s1, Square s2)
{
  return min(distance(s1, s2), 5);
}

// evaluate_passed_pawns() evaluates the passed pawns and candidate passed
// pawns of the given color.

INLINE Score evaluate_passed_pawns(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const int Up   = (Us == WHITE ? NORTH : SOUTH);

  Bitboard b, bb, squaresToQueen, unsafeSquares;
  Score score = SCORE_ZERO;

  b = ei->pe->passedPawns[Us];

  while (b) {
    Square s = pop_lsb(&b);

    assert(!(pieces_cp(Them, PAWN) & forward_file_bb(Us, s + Up)));

    int r = relative_rank_s(Us, s);

    Value mbonus = PassedRank[MG][r], ebonus = PassedRank[EG][r];

    if (r > RANK_3) {
      int w = 5 * r - 13;
      Square blockSq = s + Up;

      // Adjust bonus based on the king's proximity
      ebonus += ( (capped_distance(square_of(Them, KING), blockSq) * 19) / 4
                 - capped_distance(square_of(Us, KING), blockSq) * 2 ) * w;

      // If blockSq is not the queening square then consider also a second push
      if (r != RANK_7)
        ebonus -= capped_distance(square_of(Us, KING), blockSq + Up) * w;

      // If the pawn is free to advance, then increase the bonus
      if (is_empty(blockSq)) {
        // If there is a rook or queen attacking/defending the pawn from behind,
        // consider all the squaresToQueen. Otherwise consider only the squares
        // in the pawn's path attacked or occupied by the enemy.
        squaresToQueen = forward_file_bb(Us, s);
        unsafeSquares = passed_pawn_span(Us, s);

        bb = forward_file_bb(Them, s) & pieces_pp(ROOK, QUEEN);

        if (!(pieces_c(Them) & bb))
          unsafeSquares &= ei->attackedBy[Them][0];

        // If there are no enemy attacks on passed pawn span, assign a big
        // bonus. Otherwise, assign a smaller bonus if the path to queen is
        // not attacked and an even smaller bonus if it is attacked but
        // block square is not.
        int k =  !unsafeSquares                    ? 35
               : !(unsafeSquares & squaresToQueen) ? 20
               : !(unsafeSquares & sq_bb(blockSq)) ? 9 : 0;

        // Assign a larger bonus if the block square is defended
        if ((pieces_c(Us) & bb) || (ei->attackedBy[Us][0] & sq_bb(blockSq)))
          k += 5;

        mbonus += k * w, ebonus += k * w;
      }
    } // w != 0

    // Scale down bonus for candidate passers which need more than one
    // push to become passed or have a pawn in front of them.
    if (   !pawn_passed(pos, Us, s + Up)
        || (pieces_p(PAWN) & sq_bb(s + Up)))
    {
      mbonus /= 2;
      ebonus /= 2;
    }

    score += make_score(mbonus, ebonus) - PassedFile[file_of(s)];
  }

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
  const int Down = (Us == WHITE ? SOUTH : NORTH);
  const Bitboard SpaceMask =
    Us == WHITE ? (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank2BB | Rank3BB | Rank4BB)
                : (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank7BB | Rank6BB | Rank5BB);

  // Find the available squares for our pieces inside the SpaceMask area
  Bitboard safe =   SpaceMask
                 & ~pieces_cp(Us, PAWN)
                 & ~ei->attackedBy[Them][PAWN];

  // Find all squares which are at most three squares behind some friendly pawn
  Bitboard behind = pieces_cp(Us, PAWN);
  behind |= shift_bb(Down, behind);
  behind |= shift_bb(Down + Down, behind);

  int bonus = popcount(safe) + popcount(behind & safe & ~ei->attackedBy[Them][0]);
  int weight = popcount(pieces_c(Us)) - 1;
  Score score = make_score(bonus * weight * weight / 16, 0);

  return score;
}


// evaluate_initiative() computes the initiative correction value for the
// position, i.e., second order bonus/malus based on the known
// attacking/defending status of the players.

// Since only eg is involved, we return a Value and not a Score.
INLINE Value evaluate_initiative(const Pos *pos, int passedCount, Score score)
{
  Value mg = mg_value(score);
  Value eg = eg_value(score);

  int outflanking =  distance_f(square_of(WHITE, KING), square_of(BLACK, KING))
                   - distance_r(square_of(WHITE, KING), square_of(BLACK, KING));

  bool infiltration =   rank_of(square_of(WHITE, KING)) > RANK_4
                     || rank_of(square_of(BLACK, KING)) < RANK_5;

  bool bothFlanks = (pieces_p(PAWN) & QueenSide) && (pieces_p(PAWN) & KingSide);

  bool almostUnwinnable =   !passedCount
                         &&  outflanking < 0
                         && !bothFlanks;

  // Compute the initiative bonus for the attacking side
  int initiative =   9 * passedCount
                  + 11 * popcount(pieces_p(PAWN))
                  +  9 * outflanking
                  + 12 * infiltration
                  + 21 * bothFlanks
                  + 51 * !non_pawn_material()
                  - 43 * almostUnwinnable
                  - 100;

  // Now apply the bonus: note that we find the attacking side by extracting
  // the sign of the midgame or endgame values, and that we carefully cap the
  // bonus so that the midgame and endgame scores will never change sign after
  // the bonus.
  int u = ((mg > 0) - (mg < 0)) * max(min(initiative + 50, 0), -abs(mg));
  int v = ((eg > 0) - (eg < 0)) * max(initiative, -abs(eg));

  return make_score(u, v);
}

// evaluate_scale_factor() computes the scale factor for the winning side

INLINE int evaluate_scale_factor(const Pos *pos, EvalInfo *ei, Value eg)
{
  int strongSide = eg > VALUE_DRAW ? WHITE : BLACK;
  int sf = material_scale_factor(ei->me, pos, strongSide);

  // If scale is not already specific, scale down via general heuristics
  if (sf == SCALE_FACTOR_NORMAL) {
    if (   opposite_bishops(pos)
        && non_pawn_material() == 2 * BishopValueMg)
      sf = 22;
    else
      sf = min(sf, 36 + (opposite_bishops(pos) ? 2 : 7) * piece_count(strongSide, PAWN));
    sf = max(0, sf - (rule50_count() - 12) / 4);
  }

  return sf;
}


// evaluate() is the main evaluation function. It returns a static evaluation
// of the position from the point of view of the side to move.

Value evaluate(const Pos *pos)
{
  assert(!checkers());

  Score mobility[2] = { SCORE_ZERO, SCORE_ZERO };
  Value v;
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
  Score score = psq_score() + material_imbalance(ei.me) + pos->contempt;

  // Probe the pawn hash table
  ei.pe = pawn_probe(pos);
  score += ei.pe->score;

  // Early exit if score is high
  v = (mg_value(score) + eg_value(score)) / 2;
  if (abs(v) > LazyThreshold + non_pawn_material() / 64)
    return stm() == WHITE ? v : -v;

  // Initialize attack and king safety bitboards.
  evalinfo_init(pos, &ei, WHITE);
  evalinfo_init(pos, &ei, BLACK);

  // Evaluate all pieces but king and pawns
  score += evaluate_pieces(pos, &ei, mobility);
  score += mobility[WHITE] - mobility[BLACK];

  // Evaluate kings after all other pieces because we need full attack
  // information when computing the king safety evaluation.
  score +=  evaluate_king(pos, &ei, mobility, WHITE)
          - evaluate_king(pos, &ei, mobility, BLACK);

  // Evaluate tactical threats, we need full attack information including king
  score +=  evaluate_threats(pos, &ei, WHITE)
          - evaluate_threats(pos, &ei, BLACK);

  // Evaluate passed pawns, we need full attack information including king
  score +=  evaluate_passed_pawns(pos, &ei, WHITE)
          - evaluate_passed_pawns(pos, &ei, BLACK);

  // Evaluate space for both sides, only during opening
  if (non_pawn_material() >= SpaceThreshold)
      score +=  evaluate_space(pos, &ei, WHITE)
              - evaluate_space(pos, &ei, BLACK);

  // Evaluate position potential for the winning side
  score += evaluate_initiative(pos, ei.pe->passedCount, score);

  // Evaluate scale factor for the winning side
  //int sf = evaluate_scale_factor(pos, &ei, eg_value(score));
  int sf = evaluate_scale_factor(pos, &ei, eg_value(score));

  // Interpolate between a middlegame and a (scaled by 'sf') endgame score
  v =  mg_value(score) * ei.me->gamePhase
     + eg_value(score) * (PHASE_MIDGAME - ei.me->gamePhase) * sf / SCALE_FACTOR_NORMAL;

  v /= PHASE_MIDGAME;

  return (stm() == WHITE ? v : -v) + Tempo; // Side to move point of view
}
