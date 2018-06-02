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

#include <float.h>
#include <math.h>

#include "search.h"
#include "timeman.h"
#include "uci.h"

struct TimeManagement Time; // Our global time management struct

typedef enum { OptimumTime, MaxTime } TimeType;

// Plan time management at most this many moves ahead
static const int MoveHorizon = 50;
// When in trouble, we can step over reserved time with this ratio
static const double MaxRatio = 7.3;
// But we must not steal time from remaining moves over this ratio
static const double StealRatio = 0.34;

// move_importance() is a skew-logistic function based on naive statistical
// analysis of "how many games are still undecided after n half moves".
// The game is considered "undecided" as long as neither side has >275cp
// advantage. Data was extracted from the CCRL game database with some
// simple filtering criteria.

static double move_importance(int ply) {
  const double XScale = 6.85;
  const double XShift = 64.5;
  const double Skew   = 0.171;

  return pow((1 + exp((ply - XShift) / XScale)), -Skew) + DBL_MIN;
}

static int remaining(int myTime, int movesToGo, int ply, int slowMover,
                     const TimeType T)
{
  const double TMaxRatio   = (T == OptimumTime ? 1 : MaxRatio);
  const double TStealRatio = (T == OptimumTime ? 0 : StealRatio);

  double moveImportance = (move_importance(ply) * slowMover) / 100;
  double otherMovesImportance = 0;

  for (int i = 1; i < movesToGo; i++)
    otherMovesImportance += move_importance(ply + 2 * i);

  double ratio1 = (TMaxRatio * moveImportance) /
                  (TMaxRatio * moveImportance + otherMovesImportance);
  double ratio2 = (moveImportance + TStealRatio * otherMovesImportance) /
                  (moveImportance + otherMovesImportance);

  return myTime * min(ratio1, ratio2);
}

// tm_init() is called at the beginning of the search and calculates the
// allowed thinking time out of the time control and current game ply. We
// support four different kinds of time controls, set in 'Limits'.
//
//  inc == 0 && movestogo == 0 means: x basetime  [sudden death!]
//  inc == 0 && movestogo != 0 means: x moves in y minutes
//  inc >  0 && movestogo == 0 means: x basetime + z increment
//  inc >  0 && movestogo != 0 means: x moves in y minutes + z increment

void time_init(int us, int ply)
{
  int minThinkingTime = option_value(OPT_MIN_THINK_TIME);
  int moveOverhead    = option_value(OPT_MOVE_OVERHEAD);
  int slowMover       = option_value(OPT_SLOW_MOVER);
  int npmsec          = option_value(OPT_NODES_TIME);

  // If we have to play in 'nodes as time' mode, then convert from time
  // to nodes, and use resulting values in time management formulas.
  // WARNING: Given npms (nodes per millisecond) must be much lower then
  // the real engine speed to avoid time losses.
  if (npmsec) {
    if (!Time.availableNodes) // Only once at game start
      Time.availableNodes = npmsec * Limits.time[us]; // Time is in msec

    // Convert from millisecs to nodes
    Limits.time[us] = (int)Time.availableNodes;
    Limits.inc[us] *= npmsec;
    Limits.npmsec = npmsec;
  }

  Time.startTime = Limits.startTime;
  Time.optimumTime = Time.maximumTime = max(Limits.time[us], minThinkingTime);

  const int MaxMTG = Limits.movestogo ? min(Limits.movestogo, MoveHorizon)
                                      : MoveHorizon;

  // We calculate optimum time usage for different hypothetical "moves to go"
  // values and choose the minimum of calculated search time values. sUsually
  // the greatest hypMTG gives the minimum values.
  for (int hypMTG = 1; hypMTG <= MaxMTG; hypMTG++) {
    // Calculate thinking time for hypothetical "moves to go" value
    int hypMyTime =  Limits.time[us]
                   + Limits.inc[us] * (hypMTG - 1)
                   - moveOverhead * (2 + min(hypMTG, 40));

    hypMyTime = max(hypMyTime, 0);

    int t1 =  minThinkingTime
            + remaining(hypMyTime, hypMTG, ply, slowMover, OptimumTime);
    int t2 =  minThinkingTime
            + remaining(hypMyTime, hypMTG, ply, slowMover, MaxTime);

    Time.optimumTime = min(t1, Time.optimumTime);
    Time.maximumTime = min(t2, Time.maximumTime);
  }

  if (option_value(OPT_PONDER))
    Time.optimumTime += Time.optimumTime / 4;
}
