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
#include "types.h"

// There are 24 possible pawn squares: the first 4 files and ranks from 2 to 7
enum { MAX_INDEX = 2*24*64*64 };

// Each uint32_t stores results of 32 positions, one per bit
static uint32_t KPKBitbase[MAX_INDEX / 32];

// A KPK bitbase index is an integer in [0, IndexMax] range
//
// Information is mapped in a way that minimizes the number of iterations:
//
// bit  0- 5: white king square (from SQ_A1 to SQ_H8)
// bit  6-11: black king square (from SQ_A1 to SQ_H8)
// bit    12: side to move (WHITE or BLACK)
// bit 13-14: white pawn file (from FILE_A to FILE_D)
// bit 15-17: white pawn RANK_7 - rank
//            (from RANK_7 - RANK_7 to RANK_7 - RANK_2)
static unsigned bb_index(unsigned us, Square bksq, Square wksq, Square psq)
{
  return wksq | (bksq << 6) | (us << 12) | (file_of(psq) << 13) | ((RANK_7 - rank_of(psq)) << 15);
}

enum { RES_INVALID = 0, RES_UNKNOWN = 1, RES_DRAW = 2, RES_WIN = 4 };

unsigned bitbases_probe(Square wksq, Square wpsq, Square bksq, unsigned us)
{
  assert(file_of(wpsq) <= FILE_D);

  unsigned idx = bb_index(us, bksq, wksq, wpsq);
  return KPKBitbase[idx / 32] & (1 << (idx & 0x1F));
}

static uint8_t initial(unsigned idx)
{
  int ksq[2] = { (idx >> 0) & 0x3f, (idx >> 6) & 0x3f };
  int us     = (idx >> 12) & 0x01;
  int psq    = make_square((idx >> 13) & 0x03, RANK_7 - ((idx >> 15) & 0x07));

  // Check if two pieces are on the same square or if a king can be captured
  if (   distance(ksq[WHITE], ksq[BLACK]) <= 1
      || ksq[WHITE] == psq
      || ksq[BLACK] == psq
      || (us == WHITE && (PawnAttacks[WHITE][psq] & sq_bb(ksq[BLACK]))))
    return RES_INVALID;

  // Immediate win if a pawn can be promoted without getting captured
  if (   us == WHITE
      && rank_of(psq) == RANK_7
      && ksq[us] != psq + NORTH
      && (    distance(ksq[us ^ 1], psq + NORTH) > 1
          || (PseudoAttacks[KING][ksq[us]] & sq_bb((psq + NORTH)))))
    return RES_WIN;

  // Immediate draw if it is a stalemate or a king captures undefended pawn
  if (   us == BLACK
      && (  !(PseudoAttacks[KING][ksq[us]] & ~(PseudoAttacks[KING][ksq[us ^ 1]] | PawnAttacks[us ^ 1][psq]))
          || (PseudoAttacks[KING][ksq[us]] & sq_bb(psq) & ~PseudoAttacks[KING][ksq[us ^ 1]])))
    return RES_DRAW;

  // Position will be classified later
  return RES_UNKNOWN;
}

static uint8_t classify(uint8_t *db, unsigned idx)
{
  int ksq[2] = { (idx >> 0) & 0x3f, (idx >> 6) & 0x3f };
  int us     = (idx >> 12) & 0x01;
  int psq    = make_square((idx >> 13) & 0x03, RANK_7 - ((idx >> 15) & 0x07));

  // White to move: If one move leads to a position classified as WIN, the
  // result of the current position is WIN. If all moves lead to positions
  // classified as DRAW, the current position is classified as DRAW,
  // otherwise the current position is classified as UNKNOWN.
  //
  // Black to move: If one move leads to a position classified as DRAW, the
  // result of the current position is DRAW. If all moves lead to positions
  // classified as WIN, the position is classified as WIN, otherwise the
  // current position is classified as UNKNOWN.

  int them = us ^ 1;
  int good = (us == WHITE ? RES_WIN : RES_DRAW);
  int bad  = (us == WHITE ? RES_DRAW : RES_WIN);

  uint8_t r = RES_INVALID;
  Bitboard b = PseudoAttacks[KING][ksq[us]];

  while (b)
    r |= us == WHITE ? db[bb_index(them, ksq[them]  , pop_lsb(&b), psq)]
                     : db[bb_index(them, pop_lsb(&b), ksq[them]  , psq)];

  if (us == WHITE) {
    if (rank_of(psq) < RANK_7)      // Single push
      r |= db[bb_index(them, ksq[them], ksq[us], psq + NORTH)];

    if (   rank_of(psq) == RANK_2   // Double push
        && psq + NORTH != ksq[us]
        && psq + NORTH != ksq[them])
      r |= db[bb_index(them, ksq[them], ksq[us], psq + NORTH + NORTH)];
  }

  return db[idx] = r & good  ? good  : r & RES_UNKNOWN ? RES_UNKNOWN : bad;
}

void bitbases_init()
{
  uint8_t *db = malloc(MAX_INDEX);
  unsigned idx, repeat = 1;

  // Initialize db with known win / draw positions
  for (idx = 0; idx < MAX_INDEX; idx++)
    db[idx] = initial(idx);

  // Iterate through the positions until none of the unknown positions can be
  // changed to either wins or draws (15 cycles needed).
  while (repeat)
    for (repeat = idx = 0; idx < MAX_INDEX; idx++)
      repeat |= (db[idx] == RES_UNKNOWN && classify(db, idx) != RES_UNKNOWN);

  // Map 32 results into one KPKBitbase[] entry
  for (idx = 0; idx < MAX_INDEX; ++idx)
      if (db[idx] == RES_WIN)
          KPKBitbase[idx / 32] |= 1UL << (idx & 0x1F);

  free(db);
}
