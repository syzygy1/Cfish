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

#ifndef MOVEPICK_H
#define MOVEPICK_H

#include <string.h>   // For memset

#include "movegen.h"
#include "position.h"
#include "search.h"
#include "types.h"

#define max(a,b) ((a) > (b) ? (a) : (b))


// The Stats struct stores moves statistics. According to the template
// parameter the class can store History and Countermoves. History records
// how often different moves have been successful or unsuccessful during
// the current search and is used for reduction and move ordering decisions.
// Countermoves store the move that refute a previous one. Entries are
// stored using only the moving piece and destination square, hence two
// moves with different origin but same destination and piece will be
// considered identical.

// note: it's just a 16x64 table.
/*
template<typename T, bool CM = false>
struct Stats {

  static const Value Max = Value(1 << 28);

  const T* operator[](Piece pc) const { return table[pc]; }
  T* operator[](Piece pc) { return table[pc]; }
  void clear() { std::memset(table, 0, sizeof(table)); }

  void update(Piece pc, Square to, Move m) { table[pc][to] = m; }

  void update(Piece pc, Square to, Value v) {

    if (abs(int(v)) >= 324)
        return;

    table[pc][to] -= table[pc][to] * abs(int(v)) / (CM ? 936 : 324);
    table[pc][to] += int(v) * 32;
  }

private:
  T table[16][64];
};

typedef Stats<Move> MoveStats;
typedef Stats<Value, false> HistoryStats;
typedef Stats<Value,  true> CounterMoveStats;
typedef Stats<CounterMoveStats> CounterMoveHistoryStats;
*/

typedef Move MoveStats[16][64];
typedef Value HistoryStats[16][64];
typedef Value CounterMoveStats[16][64];
typedef CounterMoveStats CounterMoveHistoryStats[16][64];

#define stats_clear(s) memset(s, 0, sizeof(*s))

static inline void hs_update(HistoryStats hs, Piece pc, Square to, Value v)
{
  int w = v >= 0 ? v : -v;
  if (w >= 324)
    return;

  hs[pc][to] -= hs[pc][to] * w / 324;
  hs[pc][to] += ((int)v) * 32;
}

static inline void cms_update(CounterMoveStats cms, Piece pc, Square to, Value v)
{
  int w = v >= 0 ? v : -v;
  if (w >= 324)
    return;

  cms[pc][to] -= cms[pc][to] * w / 936;
  cms[pc][to] += ((int)v) * 32;
}

// MovePicker struct is used to pick one pseudo legal move at a time from
// the current position. The most important method is next_move(), which
// returns a new pseudo legal move each time it is called, until there
// are no moves left, when MOVE_NONE is returned. In order to improve the
// efficiency of the alpha beta algorithm, MovePicker attempts to return
// the moves which are most likely to get a cut-off first.

struct MovePicker {
  Pos *pos;
  Stack *ss;
  Move countermove;
  Depth depth;
  Move ttMove;
  ExtMove killers[3];
  Square recaptureSquare;
  Value threshold;
  int stage;
  ExtMove *cur, *endMoves, *endBadCaptures;
  ExtMove moves[MAX_MOVES];
};

typedef struct MovePicker MovePicker;

void mp_init(MovePicker *mp, Pos *pos, Move ttm, Depth depth, Stack *ss);
void mp_init_q(MovePicker *mp, Pos *pos, Move ttm, Depth depth, Square s);
void mp_init_pc(MovePicker *mp, Pos *pos, Move ttm, Value threshold);
Move next_move(MovePicker *mp);

#endif

