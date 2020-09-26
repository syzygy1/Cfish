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

#ifndef SEARCH_H
#define SEARCH_H

#include "misc.h"
#include "position.h"
#include "thread.h"
#include "types.h"

// RootMove struct is used for moves at the root of the tree. For each root
// move we store a score and a PV (really a refutation in the case of moves
// which fail low). Score is normally set at -VALUE_INFINITE for all non-pv
// moves.

struct RootMove {
  int pvSize;
  Value score;
  Value previousScore;
  int selDepth;
  int tbRank;
  Value tbScore;
  Move pv[MAX_PLY];
};

typedef struct RootMove RootMove;

struct RootMoves {
  int size;
  RootMove move[MAX_MOVES];
};

typedef struct RootMoves RootMoves;

/// LimitsType struct stores information sent by GUI about available time to
/// search the current move, maximum depth/time, if we are in analysis mode or
/// if we have to ponder while it's our opponent's turn to move.

struct LimitsType {
  int time[2];
  int inc[2];
  int npmsec;
  int movestogo;
  int depth;
  int movetime;
  int mate;
  bool infinite;
  uint64_t nodes;
  TimePoint startTime;
  int numSearchmoves;
  Move searchmoves[MAX_MOVES];
};

typedef struct LimitsType LimitsType;

extern LimitsType Limits;

INLINE int use_time_management(void)
{
  return Limits.time[WHITE] || Limits.time[BLACK];
}

void search_init(void);
void search_clear(void);
uint64_t perft(Position *pos, Depth depth);
void start_thinking(Position *pos, bool ponderMode);

#endif
