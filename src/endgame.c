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

#include "bitboard.h"
#include "endgame.h"
#include "movegen.h"
#include "position.h"

// Table used to drive the king towards the edge of the board
// in KX vs K and KQ vs KR endgames.
static const int PushToEdges[64] = {
  100, 90, 80, 70, 70, 80, 90, 100,
   90, 70, 60, 50, 50, 60, 70,  90,
   80, 60, 40, 30, 30, 40, 60,  80,
   70, 50, 30, 20, 20, 30, 50,  70,
   70, 50, 30, 20, 20, 30, 50,  70,
   80, 60, 40, 30, 30, 40, 60,  80,
   90, 70, 60, 50, 50, 60, 70,  90,
  100, 90, 80, 70, 70, 80, 90, 100
};

// Table used to drive the king towards a corner square of the
// right color in KBN vs K endgames.
static const int PushToCorners[64] = {
  6400, 6080, 5760, 5440, 5120, 4800, 4480, 4160,
  6080, 5760, 5440, 5120, 4800, 4480, 4160, 4480,
  5760, 5440, 4960, 4480, 4480, 4000, 4480, 4800,
  5440, 5120, 4480, 3840, 3520, 4480, 4800, 5120,
  5120, 4800, 4480, 3520, 3840, 4480, 5120, 5440,
  4800, 4480, 4000, 4480, 4480, 4960, 5440, 5760,
  4480, 4160, 4480, 4800, 5120, 5440, 5760, 6080,
  4160, 4480, 4800, 5120, 5440, 5760, 6080, 6400
};

// Tables used to drive a piece towards or away from another piece
static const int PushClose[8] = { 0, 0, 100, 80, 60, 40, 20, 10 };
static const int PushAway [8] = { 0, 5, 20, 40, 60, 80, 90, 100 };

// Pawn Rank based scaling factors used in KRPPKRP endgame
static const int KRPPKRPScaleFactors[8] = { 0, 9, 10, 14, 21, 44, 0, 0 };

#ifndef NDEBUG
static bool verify_material(const Pos *pos, int c, Value npm, int pawnsCnt)
{
  return   pos_non_pawn_material(c) == npm
        && piece_count(c, PAWN) == pawnsCnt;
}
#endif

// Map the square as if strongSide is white and strongSide's only pawn
// is on the left half of the board.
static Square normalize(const Pos *pos, unsigned strongSide, Square sq)
{
  assert(piece_count(strongSide, PAWN) == 1);

  if (file_of(square_of(strongSide, PAWN)) >= FILE_E)
    sq ^= 0x07;

  if (strongSide == BLACK)
    sq ^= 0x38;

  return sq;
}


// Compute material key from an endgame code string.

static Key calc_key(const char *code, int c)
{
  Key key = 0;
  int color = c << 3;

  for (; *code; code++)
    for (int i = 1;; i++)
      if (*code == PieceToChar[i]) {
        key += matKey[i ^ color];
        break;
      }

  return key;
}

static EgFunc EvaluateKPK, EvaluateKNNK, EvaluateKNNKP, EvaluateKBNK,
              EvaluateKRKP, EvaluateKRKB, EvaluateKRKN, EvaluateKQKP,
              EvaluateKQKR, EvaluateKXK;

static EgFunc ScaleKNPK, ScaleKNPKB, ScaleKRPKR, ScaleKRPKB,
              ScaleKBPKB, ScaleKBPKN, ScaleKBPPKB, ScaleKRPPKRP,
              ScaleKBPsK, ScaleKQKRPs, ScaleKPKP, ScaleKPsK;

EgFunc *endgame_funcs[NUM_EVAL + NUM_SCALING + 6] = {
  NULL,
// Entries 1-10 are evaluation functions.
  &EvaluateKPK,    // 1
  &EvaluateKNNK,   // 2
  &EvaluateKNNKP,  // 3
  &EvaluateKBNK,   // 4
  &EvaluateKRKP,   // 5
  &EvaluateKRKB,   // 6
  &EvaluateKRKN,   // 7
  &EvaluateKQKP,   // 8
  &EvaluateKQKR,   // 9
  &EvaluateKXK,    // 10
// Entries 11-22 are scaling functions.
  &ScaleKNPK,      // 11
  &ScaleKNPKB,     // 12
  &ScaleKRPKR,     // 13
  &ScaleKRPKB,     // 14
  &ScaleKBPKB,     // 15
  &ScaleKBPKN,     // 16
  &ScaleKBPPKB,    // 17
  &ScaleKRPPKRP,   // 18
  &ScaleKBPsK,     // 19
  &ScaleKQKRPs,    // 20
  &ScaleKPsK,      // 21
  &ScaleKPKP       // 22
};

Key endgame_keys[NUM_EVAL + NUM_SCALING][2];

static const char *endgame_codes[NUM_EVAL + NUM_SCALING] = {
  // Codes for evaluation functions 1-9.
  "KPk", "KNNk", "KNNkp", "KBNk", "KRkp", "KRkb", "KRkn", "KQkp", "KQkr",
  // Codes for scaling functions 11-18.
  "KNPk", "KNPkb", "KRPkr", "KRPkb", "KBPkb", "KBPkn", "KBPPkb", "KRPPkrp"
};

void endgames_init(void)
{
  for (int i = 0; i < NUM_EVAL + NUM_SCALING; i++) {
    endgame_keys[i][WHITE] = calc_key(endgame_codes[i], WHITE);
    endgame_keys[i][BLACK] = calc_key(endgame_codes[i], BLACK);
  }
}


// Mate with KX vs K. This function is used to evaluate positions with
// king and plenty of material vs a lone king. It simply gives the
// attacking side a bonus for driving the defending king towards the edge
// of the board, and for keeping the distance between the two kings small.
static Value EvaluateKXK(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, weakSide, VALUE_ZERO, 0));
  assert(!pos_checkers()); // Eval is never called when in check

  // Stalemate detection with lone king
  if (pos_stm() == weakSide) {
    ExtMove list[MAX_MOVES];
    if (generate_legal(pos, list) == list)
      return VALUE_DRAW;
  }

  Square winnerKSq = square_of(strongSide, KING);
  Square loserKSq = square_of(weakSide, KING);

  Value result =  pos_non_pawn_material(strongSide)
                + piece_count(strongSide, PAWN) * PawnValueEg
                + PushToEdges[loserKSq]
                + PushClose[distance(winnerKSq, loserKSq)];

  Bitboard bb = pieces_c(strongSide);
  if (   (pieces_pp(QUEEN, ROOK) & bb)
      || ((pieces_p(BISHOP) & bb) && (pieces_p(KNIGHT) & bb))
      || (   (pieces_p(BISHOP) & bb & DarkSquares)
          && (pieces_p(BISHOP) & bb & LightSquares)))
    result = min(result + VALUE_KNOWN_WIN, VALUE_MATE_IN_MAX_PLY - 1);

  return strongSide == pos_stm() ? result : -result;
}


// Mate with KBN vs K. This is similar to KX vs K, but we have to drive the
// defending king towards a corner square of the right color.
static Value EvaluateKBNK(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, KnightValueMg + BishopValueMg, 0));
  assert(verify_material(pos, weakSide, VALUE_ZERO, 0));

  Square winnerKSq = square_of(strongSide, KING);
  Square loserKSq = square_of(weakSide, KING);
  Square bishopSq = lsb(pieces_p(BISHOP));

  // kbnk_mate_table() tries to drive toward corners A1 or H8. If we have a
  // bishop that cannot reach the above squares, we flip the kings in order
  // to drive the enemy toward corners A8 or H1.
  if (opposite_colors(bishopSq, SQ_A1)) {
    winnerKSq = winnerKSq ^ 0x38;
    loserKSq  = loserKSq ^ 0x38;
  }

  Value result =  VALUE_KNOWN_WIN
                + PushClose[distance(winnerKSq, loserKSq)]
                + PushToCorners[loserKSq];

  return strongSide == pos_stm() ? result : -result;
}


// KP vs K. This endgame is evaluated with the help of a bitbase.
static Value EvaluateKPK(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, VALUE_ZERO, 1));
  assert(verify_material(pos, weakSide, VALUE_ZERO, 0));

  // Assume strongSide is white and the pawn is on files A-D
  Square wksq = normalize(pos, strongSide, square_of(strongSide, KING));
  Square bksq = normalize(pos, strongSide, square_of(weakSide, KING));
  Square psq  = normalize(pos, strongSide, lsb(pieces_p(PAWN)));

  unsigned us = strongSide == pos_stm() ? WHITE : BLACK;

  if (!bitbases_probe(wksq, psq, bksq, us))
    return VALUE_DRAW;

  Value result = VALUE_KNOWN_WIN + PawnValueEg + (Value)(rank_of(psq));

  return strongSide == pos_stm() ? result : -result;
}


// KR vs KP. This is a somewhat tricky endgame to evaluate precisely without
// a bitbase. The function below returns drawish scores when the pawn is
// far advanced with support of the king, while the attacking king is far
// away.
static Value EvaluateKRKP(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, RookValueMg, 0));
  assert(verify_material(pos, weakSide, VALUE_ZERO, 1));

  Square wksq = relative_square(strongSide, square_of(strongSide, KING));
  Square bksq = relative_square(strongSide, square_of(weakSide, KING));
  Square rsq  = relative_square(strongSide, lsb(pieces_p(ROOK)));
  Square psq  = relative_square(strongSide, lsb(pieces_p(PAWN)));

  Square queeningSq = make_square(file_of(psq), RANK_1);
  Value result;

  // If the stronger side's king is in front of the pawn, it's a win
  if (forward_file_bb(WHITE, wksq) & sq_bb(psq))
    result = RookValueEg - distance(wksq, psq);

  // If the weaker side's king is too far from the pawn and the rook,
  // it's a win.
  else if (   distance(bksq, psq) >= 3 + (pos_stm() == weakSide)
           && distance(bksq, rsq) >= 3)
    result = RookValueEg - distance(wksq, psq);

  // If the pawn is far advanced and supported by the defending king,
  // the position is drawish
  else if (   rank_of(bksq) <= RANK_3
           && distance(bksq, psq) == 1
           && rank_of(wksq) >= RANK_4
           && distance(wksq, psq) > 2 + (pos_stm() == strongSide))
    result = (Value)(80) - 8 * distance(wksq, psq);

  else
    result =  (Value)(200) - 8 * (  distance(wksq, psq + SOUTH)
                                  - distance(bksq, psq + SOUTH)
                                  - distance(psq, queeningSq));

  return strongSide == pos_stm() ? result : -result;
}


// KR vs KB. This is very simple, and always returns drawish scores.  The
// score is slightly bigger when the defending king is close to the edge.
static Value EvaluateKRKB(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, RookValueMg, 0));
  assert(verify_material(pos, weakSide, BishopValueMg, 0));

  Value result = (Value)PushToEdges[square_of(weakSide, KING)];
  return strongSide == pos_stm() ? result : -result;
}


// KR vs KN. The attacking side has slightly better winning chances than
// in KR vs KB, particularly if the king and the knight are far apart.
static Value EvaluateKRKN(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, RookValueMg, 0));
  assert(verify_material(pos, weakSide, KnightValueMg, 0));

  Square bksq = square_of(weakSide, KING);
  Square bnsq = lsb(pieces_p(KNIGHT));
  Value result = (Value)PushToEdges[bksq] + PushAway[distance(bksq, bnsq)];
  return strongSide == pos_stm() ? result : -result;
}


// KQ vs KP. In general, this is a win for the stronger side, but there are a
// few important exceptions. A pawn on 7th rank and on the A,C,F or H files
// with a king positioned next to it can be a draw, so in that case, we only
// use the distance between the kings.
static Value EvaluateKQKP(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, QueenValueMg, 0));
  assert(verify_material(pos, weakSide, VALUE_ZERO, 1));

  Square winnerKSq = square_of(strongSide, KING);
  Square loserKSq = square_of(weakSide, KING);
  Square pawnSq = lsb(pieces_p(PAWN));

  Value result = (Value)PushClose[distance(winnerKSq, loserKSq)];

  if (   relative_rank_s(weakSide, pawnSq) != RANK_7
      || distance(loserKSq, pawnSq) != 1
      || !((FileABB | FileCBB | FileFBB | FileHBB) & sq_bb(pawnSq)))
    result += QueenValueEg - PawnValueEg;

  return strongSide == pos_stm() ? result : -result;
}


// KQ vs KR.  This is almost identical to KX vs K:  We give the attacking
// king a bonus for having the kings close together, and for forcing the
// defending king towards the edge. If we also take care to avoid null
// move for the defending side in the search, this is usually sufficient
// to win KQ vs KR.
static Value EvaluateKQKR(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, QueenValueMg, 0));
  assert(verify_material(pos, weakSide, RookValueMg, 0));

  Square winnerKSq = square_of(strongSide, KING);
  Square loserKSq = square_of(weakSide, KING);

  Value result =  QueenValueEg
                - RookValueEg
                + PushToEdges[loserKSq]
                + PushClose[distance(winnerKSq, loserKSq)];

  return strongSide == pos_stm() ? result : -result;
}


// KNN vs KP. Simply push the opposing king to the corner.
static Value EvaluateKNNKP(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, 2 * KnightValueMg, 0));
  assert(verify_material(pos, weakSide, VALUE_ZERO, 1));

  Value result =  2 * KnightValueEg
                - PawnValueEg
                + PushToEdges[square_of(weakSide, KING)];

  return strongSide == pos_stm() ? result : -result;
}


// Some cases of trivial draws.
Value EvaluateKNNK(const Pos *pos, unsigned strongSide)
{
  // Avoid compiler warnings about unused variables.
  (void)pos, (void)strongSide;

  return VALUE_DRAW;
}


// KB and one or more pawns vs K. It checks for draws with rook pawns
// and a bishop of the wrong color. If such a draw is detected,
// SCALE_FACTOR_DRAW is returned. If not, the return value is
// SCALE_FACTOR_NONE, i.e. no scaling will be used.
int ScaleKBPsK(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(pos_non_pawn_material(strongSide) == BishopValueMg);
  assert(pieces_cp(strongSide, PAWN));

  // No assertions about the material of weakSide, because we want draws to
  // be detected even when the weaker side has some pawns.

  Bitboard pawns = pieces_cp(strongSide, PAWN);
  File pawnsFile = file_of(lsb(pawns));

  // All pawns are on a single rook file?
  if (    (pawnsFile == FILE_A || pawnsFile == FILE_H)
      && !(pawns & ~file_bb(pawnsFile))) {

    Square bishopSq = square_of(strongSide, BISHOP);
    Square queeningSq = relative_square(strongSide, make_square(pawnsFile, RANK_8));
    Square kingSq = square_of(weakSide, KING);

    if (   opposite_colors(queeningSq, bishopSq)
        && distance(queeningSq, kingSq) <= 1)
      return SCALE_FACTOR_DRAW;
  }

  // If all the pawns are on the same B or G file, then it's potentially a draw
  if (    (pawnsFile == FILE_B || pawnsFile == FILE_G)
      && !(pieces_p(PAWN) & ~file_bb(pawnsFile))
      && pos_non_pawn_material(weakSide) == 0
      && piece_count(weakSide, PAWN)) {

    // Get weakSide pawn that is closest to the home rank
    Square weakPawnSq = backmost_sq(weakSide, pieces_cp(weakSide, PAWN));

    Square strongKingSq = square_of(strongSide, KING);
    Square weakKingSq = square_of(weakSide, KING);
    Square bishopSq = square_of(strongSide, BISHOP);

    // There is potential for a draw if our pawn is blocked on the 7th rank,
    // the bishop cannot attack it or they only have one pawn left.
    if (   relative_rank_s(strongSide, weakPawnSq) == RANK_7
        && (pieces_cp(strongSide, PAWN) & sq_bb(weakPawnSq + pawn_push(weakSide)))
        && (opposite_colors(bishopSq, weakPawnSq) || piece_count(strongSide, PAWN) == 1)) {

      unsigned strongKingDist = distance(weakPawnSq, strongKingSq);
      unsigned weakKingDist = distance(weakPawnSq, weakKingSq);

      // It is a draw if the weak king is on its back two ranks, within 2
      // squares of the blocking pawn and the strong king is not closer.
      // (I think this rule fails only in practically unreachable
      // positions such as 5k1K/6p1/6P1/8/8/3B4/8/8 w and positions where
      // qsearch will immediately correct the problem such as
      // 8/4k1p1/6P1/1K6/3B4/8/8/8 w)
      if (   relative_rank_s(strongSide, weakKingSq) >= RANK_7
          && weakKingDist <= 2
          && weakKingDist <= strongKingDist)
        return SCALE_FACTOR_DRAW;
    }
  }

  return SCALE_FACTOR_NONE;
}


// KQ vs KR and one or more pawns. It tests for fortress draws with a rook
// on the third rank defended by a pawn.
static int ScaleKQKRPs(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, QueenValueMg, 0));
  assert(piece_count(weakSide, ROOK) == 1);
  assert(pieces_cp(weakSide, PAWN));

  Square kingSq = square_of(weakSide, KING);
  Square rsq = lsb(pieces_p(ROOK));

  if (    relative_rank_s(weakSide, kingSq) <= RANK_2
      &&  relative_rank_s(weakSide, square_of(strongSide, KING)) >= RANK_4
      &&  relative_rank_s(weakSide, rsq) == RANK_3
      && (  pieces_p(PAWN)
          & attacks_from_king(kingSq)
          & attacks_from_pawn(rsq, strongSide)))
    return SCALE_FACTOR_DRAW;

  return SCALE_FACTOR_NONE;
}


// KRP vs KR. This function knows a handful of the most important classes
// of drawn positions, but is far from perfect. It would probably be a
// good idea to add more knowledge in the future.
//
// It would also be nice to rewrite the actual code for this function,
// which is mostly copied from Glaurung 1.x, and is not very pretty.
static int ScaleKRPKR(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, RookValueMg, 1));
  assert(verify_material(pos, weakSide,   RookValueMg, 0));

  // Assume strongSide is white and the pawn is on files A-D.
  Square wksq = normalize(pos, strongSide, square_of(strongSide, KING));
  Square bksq = normalize(pos, strongSide, square_of(weakSide, KING));
  Square wrsq = normalize(pos, strongSide, square_of(strongSide, ROOK));
  Square wpsq = normalize(pos, strongSide, lsb(pieces_p(PAWN)));
  Square brsq = normalize(pos, strongSide, square_of(weakSide, ROOK));

  File f = file_of(wpsq);
  Rank r = rank_of(wpsq);
  Square queeningSq = make_square(f, RANK_8);
  signed tempo = (pos_stm() == strongSide);

  // If the pawn is not too far advanced and the defending king defends
  // the queening square, use the third-rank defence.
  if (   r <= RANK_5
      && distance(bksq, queeningSq) <= 1
      && wksq <= SQ_H5
      && (rank_of(brsq) == RANK_6 || (r <= RANK_3 && rank_of(wrsq) != RANK_6)))
    return SCALE_FACTOR_DRAW;

  // The defending side saves a draw by checking from behind in case the
  // pawn has advanced to the 6th rank with the king behind.
  if (   r == RANK_6
      && distance(bksq, queeningSq) <= 1
      && rank_of(wksq) + tempo <= RANK_6
      && (rank_of(brsq) == RANK_1 || (!tempo && distance_f(brsq, wpsq) >= 3)))
    return SCALE_FACTOR_DRAW;

  if (   r >= RANK_6
      && bksq == queeningSq
      && rank_of(brsq) == RANK_1
      && (!tempo || distance(wksq, wpsq) >= 2))
    return SCALE_FACTOR_DRAW;

  // White pawn on a7 and rook on a8 is a draw if black's king is on g7
  // or h7 and the black rook is behind the pawn.
  if (   wpsq == SQ_A7
      && wrsq == SQ_A8
      && (bksq == SQ_H7 || bksq == SQ_G7)
      && file_of(brsq) == FILE_A
      && (rank_of(brsq) <= RANK_3 || file_of(wksq) >= FILE_D || rank_of(wksq) <= RANK_5))
    return SCALE_FACTOR_DRAW;

  // If the defending king blocks the pawn and the attacking king is too
  // far away, it is a draw.
  if (   r <= RANK_5
      && bksq == wpsq + NORTH
      && distance(wksq, wpsq) - tempo >= 2
      && distance(wksq, brsq) - tempo >= 2)
    return SCALE_FACTOR_DRAW;

  // Pawn on the 7th rank supported by the rook from behind usually wins
  // if the attacking king is closer to the queening square than the
  // defending king, and the defending king cannot gain tempi by
  // threatening the attacking rook.
  if (   r == RANK_7
      && f != FILE_A
      && file_of(wrsq) == f
      && wrsq != queeningSq
      && (distance(wksq, queeningSq) < distance(bksq, queeningSq) - 2 + tempo)
      && (distance(wksq, queeningSq) < distance(bksq, wrsq) + tempo))
    return SCALE_FACTOR_MAX - 2 * distance(wksq, queeningSq);

  // Similar to the above, but with the pawn further back
  if (   f != FILE_A
      && file_of(wrsq) == f
      && wrsq < wpsq
      && (distance(wksq, queeningSq) < distance(bksq, queeningSq) - 2 + tempo)
      && (distance(wksq, wpsq + NORTH) < distance(bksq, wpsq + NORTH) - 2 + tempo)
      && (  distance(bksq, wrsq) + tempo >= 3
          || (    distance(wksq, queeningSq) < distance(bksq, wrsq) + tempo
              && (distance(wksq, wpsq + NORTH) < distance(bksq, wrsq) + tempo))))
    return  SCALE_FACTOR_MAX
          - 8 * distance(wpsq, queeningSq)
          - 2 * distance(wksq, queeningSq);

  // If the pawn is not far advanced and the defending king is somewhere in
  // the pawn's path, it's probably a draw.
  if (r <= RANK_4 && bksq > wpsq) {
    if (file_of(bksq) == file_of(wpsq))
      return 10;
    if (   distance_f(bksq, wpsq) == 1
        && distance(wksq, bksq) > 2)
      return 24 - 2 * distance(wksq, bksq);
  }
  return SCALE_FACTOR_NONE;
}

static int ScaleKRPKB(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, RookValueMg, 1));
  assert(verify_material(pos, weakSide, BishopValueMg, 0));

  // Test for a rook pawn
  if (pieces_p(PAWN) & (FileABB | FileHBB)) {
    Square ksq = square_of(weakSide, KING);
    Square bsq = lsb(pieces_p(BISHOP));
    Square psq = lsb(pieces_p(PAWN));
    Rank rk = relative_rank_s(strongSide, psq);
    Square push = pawn_push(strongSide);

    // If the pawn is on the 5th rank and the pawn (currently) is on
    // the same color square as the bishop then there is a chance of
    // a fortress. Depending on the king position give a moderate
    // reduction or a stronger one if the defending king is near the
    // corner but not trapped there.
    if (rk == RANK_5 && !opposite_colors(bsq, psq)) {
      int d = distance(psq + 3 * push, ksq);

      if (d <= 2 && !(d == 0 && ksq == square_of(strongSide, KING) + 2 * push))
        return 24;
      else
        return 48;
    }

    // When the pawn has moved to the 6th rank we can be fairly sure
    // it is drawn if the bishop attacks the square in front of the
    // pawn from a reasonable distance and the defending king is near
    // the corner
    if (   rk == RANK_6
        && distance(psq + 2 * push, ksq) <= 1
        && (PseudoAttacks[BISHOP][bsq] & sq_bb(psq + push))
        && distance_f(bsq, psq) >= 2)
      return 8;
  }

  return SCALE_FACTOR_NONE;
}

// KRPP vs KRP. There is just a single rule: if the stronger side has no
// passed pawns and the defending king is actively placed, the position
// is drawish.
static int ScaleKRPPKRP(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, RookValueMg, 2));
  assert(verify_material(pos, weakSide,   RookValueMg, 1));

#ifdef PEDANTIC
  Square wpsq1 = piece_list(strongSide, PAWN)[0];
  Square wpsq2 = piece_list(strongSide, PAWN)[1];
#else
  Square wpsq1 = lsb(pieces_cp(strongSide, PAWN));
  Square wpsq2 = msb(pieces_cp(strongSide, PAWN));
#endif
  Square bksq = square_of(weakSide, KING);

  // Does the stronger side have a passed pawn?
  if (pawn_passed(pos, strongSide, wpsq1) || pawn_passed(pos, strongSide, wpsq2))
    return SCALE_FACTOR_NONE;

  Rank r = max(relative_rank_s(strongSide, wpsq1), relative_rank_s(strongSide, wpsq2));

  if (   distance_f(bksq, wpsq1) <= 1
      && distance_f(bksq, wpsq2) <= 1
      && relative_rank_s(strongSide, bksq) > r) {
    assert(r > RANK_1 && r < RANK_7);
    return KRPPKRPScaleFactors[r];
  }
  return SCALE_FACTOR_NONE;
}


// K and two or more pawns vs K. There is just a single rule here: If all
// pawns are on the same rook file and are blocked by the defending king,
// it is a draw.
static int ScaleKPsK(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(pos_non_pawn_material(strongSide) == 0);
  assert(piece_count(strongSide, PAWN) >= 2);
  assert(verify_material(pos, weakSide, VALUE_ZERO, 0));

  Square ksq = square_of(weakSide, KING);
  Bitboard pawns = pieces_cp(strongSide, PAWN);

  // If all pawns are ahead of the king, on a single rook file and
  // the king is within one file of the pawns, it's a draw.
  if (   !(pawns & ~forward_ranks_bb(weakSide, rank_of(ksq)))
      && !((pawns & ~FileABB) && (pawns & ~FileHBB))
      &&  distance_f(ksq, lsb(pawns)) <= 1)
    return SCALE_FACTOR_DRAW;

  return SCALE_FACTOR_NONE;
}


// KBP vs KB. There are two rules: if the defending king is somewhere
// along the path of the pawn, and the square of the king is not of the
// same color as the stronger side's bishop, it is a draw. If the two
// bishops have opposite color, it's almost always a draw.
static int ScaleKBPKB(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, BishopValueMg, 1));
  assert(verify_material(pos, weakSide,   BishopValueMg, 0));

  Square pawnSq = lsb(pieces_p(PAWN));
  Square strongBishopSq = square_of(strongSide, BISHOP);
  Square weakBishopSq = square_of(weakSide, BISHOP);
  Square weakKingSq = square_of(weakSide, KING);

  // Case 1: Defending king blocks the pawn, and cannot be driven away
  if (   file_of(weakKingSq) == file_of(pawnSq)
      && relative_rank_s(strongSide, pawnSq) < relative_rank_s(strongSide, weakKingSq)
      && (   opposite_colors(weakKingSq, strongBishopSq)
          || relative_rank_s(strongSide, weakKingSq) <= RANK_6))
    return SCALE_FACTOR_DRAW;

  // Case 2: Opposite colored bishops
  if (opposite_colors(strongBishopSq, weakBishopSq))
    return SCALE_FACTOR_DRAW;

  return SCALE_FACTOR_NONE;
}


// KBPP vs KB. It detects a few basic draws with opposite-colored bishops.
static int ScaleKBPPKB(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, BishopValueMg, 2));
  assert(verify_material(pos, weakSide,   BishopValueMg, 0));

  Square wbsq = square_of(strongSide, BISHOP);
  Square bbsq = square_of(weakSide, BISHOP);

  if (!opposite_colors(wbsq, bbsq))
    return SCALE_FACTOR_NONE;

  Square ksq = square_of(weakSide, KING);
#ifdef PEDANTIC
  Square psq1 = piece_list(strongSide, PAWN)[0];
  Square psq2 = piece_list(strongSide, PAWN)[1];
#else
  Square psq1 = lsb(pieces_cp(strongSide, PAWN));
  Square psq2 = msb(pieces_cp(strongSide, PAWN));
#endif
  int r1 = rank_of(psq1);
  int r2 = rank_of(psq2);
  Square blockSq1, blockSq2;

  if (relative_rank_s(strongSide, psq1) > relative_rank_s(strongSide, psq2)) {
    blockSq1 = psq1 + pawn_push(strongSide);
    blockSq2 = make_square(file_of(psq2), rank_of(psq1));
  } else {
    blockSq1 = psq2 + pawn_push(strongSide);
    blockSq2 = make_square(file_of(psq1), rank_of(psq2));
  }

  switch (distance_f(psq1, psq2)) {
  case 0:
    // Both pawns are on the same file. It is an easy draw if the defender
    // firmly controls some square in the frontmost pawn's path.
    if (   file_of(ksq) == file_of(blockSq1)
        && relative_rank_s(strongSide, ksq) >= relative_rank_s(strongSide, blockSq1)
        && opposite_colors(ksq, wbsq))
      return SCALE_FACTOR_DRAW;
    else
      return SCALE_FACTOR_NONE;

  case 1:
    // Pawns on adjacent files. It is a draw if the defender firmly controls
    // the square in front of the frontmost pawn's path, and the square
    // diagonally behind this square on the file of the other pawn.
    if (   ksq == blockSq1
        && opposite_colors(ksq, wbsq)
        && (   bbsq == blockSq2
            || (attacks_from_bishop(blockSq2) & pieces_cp(weakSide, BISHOP))
            || distance(r1, r2) >= 2))
      return SCALE_FACTOR_DRAW;

    else if (   ksq == blockSq2
             && opposite_colors(ksq, wbsq)
             && (   bbsq == blockSq1
                 || (attacks_from_bishop(blockSq1)
                                         & pieces_cp(weakSide, BISHOP))))
      return SCALE_FACTOR_DRAW;
    else
      return SCALE_FACTOR_NONE;

  default:
    // The pawns are not on the same file or adjacent files. No scaling.
    return SCALE_FACTOR_NONE;
  }
}


// KBP vs KN. There is a single rule: If the defending king is somewhere
// along the path of the pawn, and the square of the king is not of the
// same color as the stronger side's bishop, it is a draw.
static int ScaleKBPKN(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, BishopValueMg, 1));
  assert(verify_material(pos, weakSide, KnightValueMg, 0));

  Square pawnSq = lsb(pieces_p(PAWN));
  Square strongBishopSq = lsb(pieces_p(BISHOP));
  Square weakKingSq = square_of(weakSide, KING);

  if (   file_of(weakKingSq) == file_of(pawnSq)
      && relative_rank_s(strongSide, pawnSq) < relative_rank_s(strongSide, weakKingSq)
      && (   opposite_colors(weakKingSq, strongBishopSq)
          || relative_rank_s(strongSide, weakKingSq) <= RANK_6))
    return SCALE_FACTOR_DRAW;

  return SCALE_FACTOR_NONE;
}


// KNP vs K. There is a single rule: if the pawn is a rook pawn on the
// 7th rank and the defending king prevents the pawn from advancing, the
// position is drawn.
static int ScaleKNPK(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, KnightValueMg, 1));
  assert(verify_material(pos, weakSide, VALUE_ZERO, 0));

  // Assume strongSide is white and the pawn is on files A-D
  Square pawnSq     = normalize(pos, strongSide, lsb(pieces_p(PAWN)));
  Square weakKingSq = normalize(pos, strongSide, square_of(weakSide, KING));

  if (pawnSq == SQ_A7 && distance(SQ_A8, weakKingSq) <= 1)
    return SCALE_FACTOR_DRAW;

  return SCALE_FACTOR_NONE;
}


// KNP vs KB. If knight can block bishop from taking pawn, it is a win.
// Otherwise the position is drawn.
static int ScaleKNPKB(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  Square pawnSq = lsb(pieces_p(PAWN));
  Square bishopSq = lsb(pieces_p(BISHOP));
  Square weakKingSq = square_of(weakSide, KING);

  // King needs to get close to promoting pawn to prevent knight from blocking.
  // Rules for this are very tricky, so just approximate.
  if (forward_file_bb(strongSide, pawnSq) & attacks_from_bishop(bishopSq))
    return distance(weakKingSq, pawnSq);

  return SCALE_FACTOR_NONE;
}


// KP vs KP. This is done by removing the weakest side's pawn and probing
// the KP vs K bitbase: If the weakest side has a draw without the pawn,
// it probably has at least a draw with the pawn as well. The exception
// is when the stronger side's pawn is far advanced and not on a rook
// file; in this case it is often possible to win
// (e.g. 8/4k3/3p4/3P4/6K1/8/8/8 w - - 0 1).
static int ScaleKPKP(const Pos *pos, unsigned strongSide)
{
  unsigned weakSide = strongSide ^ 1;

  assert(verify_material(pos, strongSide, VALUE_ZERO, 1));
  assert(verify_material(pos, weakSide,   VALUE_ZERO, 1));

  // Assume strongSide is white and the pawn is on files A-D
  Square wksq = normalize(pos, strongSide, square_of(strongSide, KING));
  Square bksq = normalize(pos, strongSide, square_of(weakSide, KING));
  Square psq  = normalize(pos, strongSide, square_of(strongSide, PAWN));

  unsigned us = strongSide == pos_stm() ? WHITE : BLACK;

  // If the pawn has advanced to the fifth rank or further, and is not a
  // rook pawn, it is too dangerous to assume that it is at least a draw.
  if (rank_of(psq) >= RANK_5 && file_of(psq) != FILE_A)
    return SCALE_FACTOR_NONE;

  // Probe the KPK bitbase with the weakest side's pawn removed. If it is
  // a draw, it is probably at least a draw even with the pawn.
  return bitbases_probe(wksq, psq, bksq, us) ? SCALE_FACTOR_NONE : SCALE_FACTOR_DRAW;
}
