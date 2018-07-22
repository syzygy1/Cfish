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

  // kingRing[color] is the zone around the king which is considered
  // by the king safety evaluation. This consists of the squares directly
  // adjacent to the king, and (only for a king on its first rank) the
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
  QueenSide, QueenSide, QueenSide, CenterFiles,
  CenterFiles, KingSide, KingSide, KingSide
};

// Thresholds for lazy and space evaluation
enum { LazyThreshold = 1500, SpaceThreshold = 12222 };

// KingAttackWeights[PieceType] contains king attack weights by piece type
static const int KingAttackWeights[8] = { 0, 0, 77, 55, 44, 10 };

// Penalties for enemy's safe checks
enum {
  QueenSafeCheck  = 780,
  RookSafeCheck   = 880,
  BishopSafeCheck = 435,
  KnightSafeCheck = 790
};

#define V(v) (Value)(v)
#define S(mg,eg) make_score(mg,eg)

// MobilityBonus[PieceType-2][attacked] contains bonuses for middle and
// end game, indexed by piece type and number of attacked squares in the
// mobility area.
static const Score MobilityBonus[4][32] = {
  // Knights
  { S(-75,-76), S(-57,-54), S( -9,-28), S( -2,-10), S(  6,  5), S( 14, 12),
    S( 22, 26), S( 29, 29), S( 36, 29) },
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

// Outpost[knight/bishop][supported by pawn] contains bonuses for minors
// if they can reach an outpost square, bigger if that square is supported0
// by a pawn. If the minor occupies an outpost square, then score is doubled.
static const Score Outpost[][2] = {
  { S(22, 6), S(36,12) }, // Knight
  { S( 9, 2), S(15, 5) }  // Bishop
};

// RookOnFile[semiopen/open] contains bonuses for each rook when there is
// no friendly pawn on the rook file.
static const Score RookOnFile[2] = { S(20, 7), S(45, 20) };

// ThreatByMinor/ByRook[attacked PieceType] contains bonuses according to
// which piece type attacks which one. Attacks on lesser pieces which are
// pawn defended are not considered.
static const Score ThreatByMinor[8] = {
  S(0, 0), S(0, 31), S(39, 42), S(57, 44), S(68,112), S(47,120)
};

static const Score ThreatByRook[8] = {
  S(0, 0), S(0, 24), S(38, 71), S(38, 61), S( 0, 38), S(36, 38)
};

// PassedRank[mg/eg][Rank] contains midgame and endgame bonuses for passed
// pawns. We don't use a Score because we process the two components
// independently.
static const Value PassedRank[][8] = {
  { V(0), V( 5), V(12), V(10), V(57), V(163), V(271) },
  { V(0), V(18), V(23), V(31), V(62), V(167), V(250) }
};

// PassedFile[File] contains a bonus according to the file of a passed pawn
static const Score PassedFile[8] = {
  S( -1,  7), S( 0,  9), S(-9, -8), S(-30,-14),
  S(-30,-14), S(-9, -8), S( 0,  9), S( -1,  7)
};

// Rank-dependent factor for a passed-pawn bonus
static const int PassedDanger[8] = { 0, 0, 0, 3, 7, 11, 20 };

// Assorted bonuses and penalties used by evaluation
static const Score BishopPawns        = S(  3,  7);
static const Score CloseEnemies       = S(  6,  0);
static const Score Connectivity       = S(  3,  1);
static const Score CorneredBishop     = S( 50, 50);
static const Score Hanging            = S( 52, 30);
static const Score HinderPassedPawn   = S(  4,  0);
static const Score KingProtector      = S(  6,  6);
static const Score KnightOnQueen      = S( 21, 11);
static const Score LongDiagonalBishop = S( 22,  0);
static const Score MinorBehindPawn    = S( 16,  0);
static const Score Overload           = S( 10,  5);
static const Score PawnlessFlank      = S( 20, 80);
static const Score RookOnPawn         = S(  8, 24);
static const Score SliderOnQueen      = S( 42, 21);
static const Score ThreatByKing       = S( 23, 76);
static const Score ThreatByPawnPush   = S( 45, 40);
static const Score ThreatByRank       = S( 16,  3);
static const Score ThreatBySafePawn   = S(173,102);
static const Score TrappedRook        = S( 92,  0);
static const Score WeakQueen          = S( 50, 10);
static const Score WeakUnopposedPawn  = S(  5, 29);

#undef S
#undef V

// eval_init() initializes king and attack bitboards for a given color
// adding pawn attacks. To be done at the beginning of the evaluation.

INLINE void evalinfo_init(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const int Up   = (Us == WHITE ? NORTH : SOUTH);
  const int Down = (Us == WHITE ? SOUTH : NORTH);
  const Bitboard LowRanks = (Us == WHITE ? Rank2BB | Rank3BB
                                         : Rank7BB | Rank6BB);

  // Find our pawns on the first two ranks, and those which are blocked
  Bitboard b = pieces_cp(Us, PAWN) & (shift_bb(Down, pieces()) | LowRanks);

  // Squares occupied by those pawns, by our king or queen, or controlled by
  // enemy pawns are excluded from the mobility area
  ei->mobilityArea[Us] = ~(b | pieces_cpp(Us, KING, QUEEN) | ei->pe->pawnAttacks[Them]);

  // Initialise the attackedBy bitboards for kings and pawns
  b = ei->attackedBy[Us][KING] = attacks_from_king(square_of(Us, KING));
  ei->attackedBy[Us][PAWN] = ei->pe->pawnAttacks[Us];
  ei->attackedBy[Us][0] = b | ei->attackedBy[Us][PAWN];
  ei->attackedBy2[Us]   = b & ei->attackedBy[Us][PAWN];

  // Init our king safety tables only if we are going to use them
  if (pos_non_pawn_material(Them) >= RookValueMg + KnightValueMg) {
    ei->kingRing[Us] = b;

    if (relative_rank_s(Us, square_of(Us, KING)) == RANK_1)
      ei->kingRing[Us] |= shift_bb(Up, b);

    if (file_of(square_of(Us, KING)) == FILE_H)
      ei->kingRing[Us] |= shift_bb(WEST, ei->kingRing[Us]);

    else if (file_of(square_of(Us, KING)) == FILE_A)
      ei->kingRing[Us] |= shift_bb(EAST, ei->kingRing[Us]);

    ei->kingAttackersCount[Them] = popcount(ei->kingRing[Us] & ei->pe->pawnAttacks[Them]);
    ei->kingAttacksCount[Them] = ei->kingAttackersWeight[Them] = 0;
  }
  else
    ei->kingRing[Us] = ei->kingAttackersCount[Them] = 0;
}

// evaluate_piece() assigns bonuses and penalties to the pieces of a given
// color and type.

INLINE Score evaluate_piece(const Pos *pos, EvalInfo *ei, Score *mobility,
                            const int Us, const int Pt)
{
  const int Them = (Us == WHITE ? BLACK : WHITE);
  const int Down = (Us == WHITE ? SOUTH : NORTH);
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
      bb = OutpostRanks & ~ei->pe->pawnAttacksSpan[Them];
      if (bb & sq_bb(s))
        score += Outpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & sq_bb(s))] * 2;
      else {
        bb &= b & ~pieces_c(Us);
        if (bb)
          score += Outpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & bb)];
      }

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
        if (more_than_one(Center & (attacks_bb_bishop(s, pieces_p(PAWN)) | sq_bb(s))))
          score += LongDiagonalBishop;
      }

      // An important Chess960 pattern: A cornered bishop blocked by a friendly
      // pawn diagonally in front of it is a very serious problem, especially
      // when that pawn is also blocked.
      if (   Pt == BISHOP
          && is_chess960()
          && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1))) {
        Square d = pawn_push(Us) + (file_of(s) == FILE_A ? EAST : WEST);
        if (piece_on(s + d) == make_piece(Us, PAWN))
          score -=  piece_on(s + d + pawn_push(Us))             ? CorneredBishop * 4
                  : piece_on(s + d + d) == make_piece(Us, PAWN) ? CorneredBishop * 2
                                                                : CorneredBishop;
      }
    }

    if (Pt == ROOK) {
      // Bonus for aligning with enemy pawns on the same rank/file
      if (relative_rank_s(Us, s) >= RANK_5)
        score += RookOnPawn * popcount(pieces_cp(Them, PAWN) & PseudoAttacks[ROOK][s]);

      // Bonus when on an open or semi-open file
      if (semiopen_file(ei->pe, Us, file_of(s)))
        score += RookOnFile[!!semiopen_file(ei->pe, Them, file_of(s))];

      // Penalty when trapped by the king, even more if the king cannot castle
      else if (mob <= 3) {
        File kf = file_of(square_of(Us, KING));

        if ((kf < FILE_E) == (file_of(s) < kf))
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
  Bitboard weak, b, b1, b2, safe, unsafeChecks;

  // King shelter and enemy pawns storm
  Score score = Us == WHITE ? king_safety_white(ei->pe, pos, ksq)
                            : king_safety_black(ei->pe, pos, ksq);

  // Main king safety evaluation
  if (ei->kingAttackersCount[Them] > (1 - piece_count(Them, QUEEN))) {
    // Attacked squares defended at most once by our queen or king
    weak =  ei->attackedBy[Them][0]
          & ~ei->attackedBy2[Us]
          & (   ei->attackedBy[Us][KING] | ei->attackedBy[Us][QUEEN]
             | ~ei->attackedBy[Us][0]);

    int kingDanger = 0;
    unsafeChecks = 0;

    // Analyse the safe enemy's checks which are possible on next move
    safe  = ~pieces_c(Them);
    safe &= ~ei->attackedBy[Us][0] | (weak & ei->attackedBy2[Them]);

    b1 = attacks_bb_rook(ksq, pieces() ^ pieces_cp(Us, QUEEN));
    b2 = attacks_bb_bishop(ksq, pieces() ^ pieces_cp(Us, QUEEN));

    // Enemy queen safe checks
    if ((b1 | b2) & ei->attackedBy[Them][QUEEN] & safe & ~ei->attackedBy[Us][QUEEN])
      kingDanger += QueenSafeCheck;

    // For minors and rooks, also consider the square safe if attacked twice
    // and only defended by our queen.
    safe |=  ei->attackedBy2[Them]
           & ~(ei->attackedBy2[Us] | pieces_c(Them))
           & ei->attackedBy[Us][QUEEN];

    b1 &= ei->attackedBy[Them][ROOK];
    b2 &= ei->attackedBy[Them][BISHOP];

    // Enemy rooks checks
    if (b1 & safe)
      kingDanger += RookSafeCheck;
    else
      unsafeChecks |= b1;

    // Enemy bishops checks
    if (b2 & safe)
      kingDanger += BishopSafeCheck;
    else
      unsafeChecks |= b2;

    // Enemy knights checks
    b = attacks_from_knight(ksq) & ei->attackedBy[Them][KNIGHT];
    if (b & safe)
      kingDanger += KnightSafeCheck;
    else
      unsafeChecks |= b;

    // Unsafe or occupied checking squares will also be considered, as long
    // the square is in the attacker's mobility area.
    unsafeChecks &= ei->mobilityArea[Them];

    kingDanger +=  ei->kingAttackersCount[Them] * ei->kingAttackersWeight[Them]
                 +  69 * ei->kingAttacksCount[Them]
                 + 185 * popcount(ei->kingRing[Us] & weak)
                 + 129 * popcount(blockers_for_king(pos, Us) | unsafeChecks)
                 - 873 * !pieces_cp(Them, QUEEN)
                 -   6 * mg_value(score) / 8
                 -   2;

    // Transform the kingDanger units into a Score, and subtract it from
    // the evaluation.
    if (kingDanger > 0) {
      int mobilityDanger = mg_value(mobility[Them] - mobility[Us]);
      kingDanger = max(0, kingDanger + mobilityDanger);
      score -= make_score(kingDanger * kingDanger / 4096, kingDanger / 16);
    }
  }

  // King tropism: firstly, find squares that we attack in the enemy king flank
  File kf = file_of(ksq);
  b = ei->attackedBy[Them][0] & KingFlank[kf] & Camp;

  assert(((Us == WHITE ? b << 4 : b >> 4) & b) == 0);
  assert(popcount(Us == WHITE ? b << 4 : b >> 4) == popcount(b));

  // Secondly, add the squares which are attacked twice in that flank and
  // which are not defended by our pawns.
  b =  (Us == WHITE ? b << 4 : b >> 4)
     | (b & ei->attackedBy2[Them] & ~ei->attackedBy[Us][PAWN]);

  score -= CloseEnemies * popcount(b);

  // Penalty when our king is on a pawnless flank.
  if (!(pieces_p(PAWN) & KingFlank[kf]))
    score -= PawnlessFlank;

  return score;
}


// evaluate_threats() assigns bonuses according to the types of the
// attacking and the attacked pieces.

INLINE Score evaluate_threats(const Pos *pos, EvalInfo *ei, const int Us)
{
  const int Them  = (Us == WHITE ? BLACK      : WHITE);
  const int Up    = (Us == WHITE ? NORTH      : SOUTH);
  const int Left  = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);
  const int Right = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
  const Bitboard TRank3BB = (Us == WHITE ? Rank3BB  : Rank6BB);

  enum { Minor, Rook };

  Bitboard b, weak, defended, stronglyProtected, safeThreats;
  Score score = SCORE_ZERO;

  // Non-pawn enemies
  Bitboard nonPawnEnemies = pieces_c(Them) & ~pieces_p(PAWN);

  // Our safe or protected pawns
  b = pieces_cp(Us, PAWN) & (~ei->attackedBy[Them][0] | ei->attackedBy[Us][0]);

  safeThreats = (shift_bb(Right, b) | shift_bb(Left, b)) & nonPawnEnemies;
  score += ThreatBySafePawn * popcount(safeThreats);

  // Squares strongly protected by the opponent, either because they attack the
  // square with a pawn or because they attack the square twice and we don't.
  stronglyProtected =  ei->attackedBy[Them][PAWN]
                     | (ei->attackedBy2[Them] & ~ei->attackedBy2[Us]);

  // Non-pawn enemies, strongly protected
  defended =  (pieces_c(Them) ^ pieces_cp(Them, PAWN))
            & stronglyProtected;

  // Enemies not strongly protected and under our attack
  weak =   pieces_c(Them)
        & ~stronglyProtected
        &  ei->attackedBy[Us][0];

  // Add a bonus according to the kind of attacking pieces
  if (defended | weak) {
    b = (defended | weak) & (ei->attackedBy[Us][KNIGHT] | ei->attackedBy[Us][BISHOP]);
    while (b) {
      Square s = pop_lsb(&b);
      score += ThreatByMinor[piece_on(s) - 8 * Them];
      if (piece_on(s) != make_piece(Them, PAWN))
        score += ThreatByRank * relative_rank_s(Them, s);
    }

    b = (pieces_cp(Them, QUEEN) | weak) & ei->attackedBy[Us][ROOK];
    while (b) {
      Square s = pop_lsb(&b);
      score += ThreatByRook[piece_on(s) - 8 * Them];
      if (piece_on(s) != make_piece(Them, PAWN))
        score += ThreatByRank * relative_rank_s(Them, s);
    }

    // Bonus for king attacks on pawns or pieces which are not pawn-defended
    if (weak & ei->attackedBy[Us][KING])
      score += ThreatByKing;

    // Bonus for overload (non-pawn enemies attacked and defended exactly once)
    b =  nonPawnEnemies
       & ei->attackedBy[Us][0]
       & ei->attackedBy[Them][0] & ~ei->attackedBy2[Them];
    score += Overload * popcount(b);

    score += Hanging * popcount(weak & ~ei->attackedBy[Them][0]);
  }

  // Bonus for opponent unopposed weak pawns
  if (pieces_cpp(Us, ROOK, QUEEN))
    score += WeakUnopposedPawn * ei->pe->weakUnopposed[Them];

  // Find the squares reachable by a single pawn push
  b  = shift_bb(Up, pieces_cp(Us, PAWN)) & ~pieces();
  b |= shift_bb(Up, b & TRank3BB) & ~pieces();

  // Keep only those squares which are not completely unsafe
  b &=  ~pieces()
      & ~ei->attackedBy[Them][PAWN]
      & (ei->attackedBy[Us][0] | ~ei->attackedBy[Them][0]);

  // Add a bonus for each new pawn threat from those squares
  b =  (shift_bb(Left, b) | shift_bb(Right, b))
     &  pieces_c(Them)
     & ~ei->attackedBy[Us][PAWN];

  score += ThreatByPawnPush * popcount(b);

  // Bonus for impending threats against enemy queen
  if (piece_count(Them, QUEEN) == 1) {
    Square s = square_of(Them, QUEEN);
    safeThreats = ei->mobilityArea[Us] & ~stronglyProtected;

    b = ei->attackedBy[Us][KNIGHT] & attacks_from_knight(s);

    score += KnightOnQueen * popcount(b & safeThreats);

    b =  (ei->attackedBy[Us][BISHOP] & attacks_from_bishop(s))
       | (ei->attackedBy[Us][ROOK  ] & attacks_from_rook(s));

    score += SliderOnQueen * popcount(b & safeThreats & ei->attackedBy2[Us]);
  }

  // Bonus for protected knights, bishops, rooks and queens
  b = (pieces_c(Us) ^ pieces_cpp(Us, PAWN, KING)) & ei->attackedBy[Us][0];
  score += Connectivity * popcount(b);

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

  Bitboard b, bb, squaresToQueen, defendedSquares, unsafeSquares;
  Score score = SCORE_ZERO;

  b = ei->pe->passedPawns[Us];

  while (b) {
    Square s = pop_lsb(&b);

    assert(!(pieces_cp(Them, PAWN) & forward_file_bb(Us, s + Up)));

    bb = forward_file_bb(Us, s) & pieces_c(Them);
    score -= HinderPassedPawn * popcount(bb);

    int r = relative_rank_s(Us, s);
    int w = PassedDanger[r];

    Value mbonus = PassedRank[MG][r], ebonus = PassedRank[EG][r];

    if (w) {
      Square blockSq = s + Up;

      // Adjust bonus based on the king's proximity
      ebonus +=  capped_distance(square_of(Them, KING), blockSq) * 5 * w
               - capped_distance(square_of(Us, KING), blockSq) * 2 * w;

      // If blockSq is not the queening square then consider also a second push
      if (r != RANK_7)
        ebonus -= capped_distance(square_of(Us, KING), blockSq + Up) * w;

      // If the pawn is free to advance, then increase the bonus
      if (is_empty(blockSq)) {
        // If there is a rook or queen attacking/defending the pawn from behind,
        // consider all the squaresToQueen. Otherwise consider only the squares
        // in the pawn's path attacked or occupied by the enemy.
        defendedSquares = unsafeSquares = squaresToQueen = forward_file_bb(Us, s);

        bb = forward_file_bb(Them, s) & pieces_pp(ROOK, QUEEN) & attacks_from_rook(s);

        if (!(pieces_c(Us) & bb))
          defendedSquares &= ei->attackedBy[Us][0];

        if (!(pieces_c(Them) & bb))
          unsafeSquares &= ei->attackedBy[Them][0] | pieces_c(Them);

        // If there aren't any enemy attacks, assign a big bonus. Otherwise
        // assign a smaller bonus if the block square isn't attacked.
        int k = !unsafeSquares ? 20 : !(unsafeSquares & sq_bb(blockSq)) ? 9 : 0;

        // If the path to the queen is fully defended, assign a big bonus.
        // Otherwise assign a smaller bonus if the block square is defended.
        if (defendedSquares == squaresToQueen)
          k += 6;

        else if (defendedSquares & sq_bb(blockSq))
          k += 4;

        mbonus += k * w, ebonus += k * w;
      }
      else if (pieces_c(Us) & sq_bb(blockSq))
        mbonus += w + r * 2, ebonus += w + r * 2;
    } // w != 0

    // Scale down bonus for candidate passers which need more than one
    // push to become passed or have a pawn in front of them.
    if (   !pawn_passed(pos, Us, s + Up)
        || (pieces_p(PAWN) & forward_file_bb(Us, s)))
    {
      mbonus /= 2;
      ebonus /= 2;
    }

    score += make_score(mbonus, ebonus) + PassedFile[file_of(s)];
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
  const Bitboard SpaceMask =
    Us == WHITE ? (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank2BB | Rank3BB | Rank4BB)
                : (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank7BB | Rank6BB | Rank5BB);

  // Find the available squares for our pieces inside the SpaceMask area
  Bitboard safe =   SpaceMask
                 & ~pieces_cp(Us, PAWN)
                 & ~ei->attackedBy[Them][PAWN];

  // Find all squares which are at most three squares behind some friendly pawn
  Bitboard behind = pieces_cp(Us, PAWN);
  behind |= (Us == WHITE ? behind >>  8 : behind <<  8);
  behind |= (Us == WHITE ? behind >> 16 : behind << 16);

  // Since SpaceMask[Us] is fully on our half of the board...
  assert((unsigned)(safe >> (Us == WHITE ? 32 : 0)) == 0);

  // ...count safe + (behind & safe) with a single popcount.
  int bonus = popcount((Us == WHITE ? safe << 32 : safe >> 32) | (behind & safe));
  int weight = popcount(pieces_c(Us)) - 2 * ei->pe->openFiles;

  return make_score(bonus * weight * weight / 16, 0);
}


// evaluate_initiative() computes the initiative correction value for the
// position, i.e., second order bonus/malus based on the known
// attacking/defending status of the players.

// Since only eg is involved, we return a Value and not a Score.
INLINE Value evaluate_initiative(const Pos *pos, int asymmetry, Value eg)
{
  int outflanking =  distance_f(square_of(WHITE, KING), square_of(BLACK, KING))
                   - distance_r(square_of(WHITE, KING), square_of(BLACK, KING));
  bool bothFlanks = (pieces_p(PAWN) & QueenSide) && (pieces_p(PAWN) & KingSide);

  // Compute the initiative bonus for the attacking side
  int initiative =    8 * asymmetry
                  +  12 * popcount(pieces_p(PAWN))
                  +  12 * outflanking
                  +  16 * bothFlanks
                  +  48 * !(pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK))
                  - 136;

  // Now apply the bonus: note that we find the attacking side by extracting
  // the sign of the endgame value, and that we carefully cap the bonus so
  // that the endgame score will never change sign after the bonus.
  Value value = ((eg > 0) - (eg < 0)) * max(initiative, -abs(eg));

//  return make_score(0, value);
  return value;
}

// evaluate_scale_factor() computes the scale factor for the winning side

INLINE int evaluate_scale_factor(const Pos *pos, EvalInfo *ei, Value eg)
{
  int strongSide = eg > VALUE_DRAW ? WHITE : BLACK;
  int sf = material_scale_factor(ei->me, pos, strongSide);

  // If scale is not already specific, scale down via general heuristics
  if (sf == SCALE_FACTOR_NORMAL) {
    if (   opposite_bishops(pos)
        && pos_non_pawn_material(WHITE) == BishopValueMg
        && pos_non_pawn_material(BLACK) == BishopValueMg)
      return 31;
    else
      return min(40 + (opposite_bishops(pos) ? 2 : 7) * piece_count(strongSide, PAWN), sf);
  }

  return sf;
}


// evaluate() is the main evaluation function. It returns a static evaluation
// of the position from the point of view of the side to move.

Value evaluate(const Pos *pos)
{
  assert(!pos_checkers());

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
  Score score = pos_psq_score() + material_imbalance(ei.me) + pos->contempt;

  // Probe the pawn hash table
  ei.pe = pawn_probe(pos);
  score += ei.pe->score;

  // Early exit if score is high
  v = (mg_value(score) + eg_value(score)) / 2;
  if (abs(v) > LazyThreshold)
    return pos_stm() == WHITE ? v : -v;

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
  if (pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK) >= SpaceThreshold)
      score +=  evaluate_space(pos, &ei, WHITE)
              - evaluate_space(pos, &ei, BLACK);

  // Evaluate position potential for the winning side
  //  score += evaluate_initiative(pos, ei.pi->asymmetry, eg_value(score));
  int eg = eg_value(score);
  eg += evaluate_initiative(pos, ei.pe->asymmetry, eg);

  // Evaluate scale factor for the winning side
  //int sf = evaluate_scale_factor(pos, &ei, eg_value(score));
  int sf = evaluate_scale_factor(pos, &ei, eg);

  // Interpolate between a middlegame and a (scaled by 'sf') endgame score
  //  Value v =  mg_value(score) * ei.me->gamePhase
  //           + eg_value(score) * (PHASE_MIDGAME - ei.me->gamePhase) * sf / SCALE_FACTOR_NORMAL;
  v =  mg_value(score) * ei.me->gamePhase
     + eg * (PHASE_MIDGAME - ei.me->gamePhase) * sf / SCALE_FACTOR_NORMAL;

  v /= PHASE_MIDGAME;

  return (pos_stm() == WHITE ? v : -v) + Tempo; // Side to move point of view
}
