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

// tm_init() is called at the beginning of the search and calculates the
// time bounds allowed for the current game ply. We currently support:
// 1) x basetime (+z increment)
// 2) x moves in y seconds (+z increment)

void time_init(Color us, int ply)
{
  int moveOverhead    = option_value(OPT_MOVE_OVERHEAD);
  int slowMover       = option_value(OPT_SLOW_MOVER);
  int npmsec          = option_value(OPT_NODES_TIME);

  // optScale is a percentage of available time to use for the current move.
  // maxScale is a multiplier applied to optimumTime.
  double optScale, maxScale;

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

  // Maximum move horizon of 50 moves
  int mtg = Limits.movestogo ? min(Limits.movestogo, 50) : 50;

  // Make sure that timeLeft > 0 since we may use it as a divisor
  TimePoint timeLeft = max(1, Limits.time[us] + Limits.inc[us] * (mtg - 1) - moveOverhead * (2 + mtg));

  // A user may scale time usage by setting UCI option "Slow Mover".
  // Default is 100 and changing this value will probably lose Elo.
  timeLeft = slowMover * timeLeft / 100;

  // x basetime (+z increment)
  // If there is a healthy increment, timeLeft can exceed actual available
  // game time for the current move, so also cap to 20% of available game time.
  if (Limits.movestogo == 0) {
    optScale = min(0.0084 + pow(ply + 3.0, 0.5) * 0.0042,
                    0.2 * Limits.time[us] / (double)timeLeft);
    maxScale = min(7.0, 4.0 + ply / 12.0);
  }
  // x moves in y seconds (+z increment)
  else {
    optScale = min((0.8 + ply / 120.0) / mtg,
                     0.8 * Limits.time[us] / (double)timeLeft);
    maxScale = min(6.3, 1.5 + 0.11 * mtg);
  }

  // Never use more than 80% of the available time for this move
  Time.optimumTime = optScale * timeLeft;
  Time.maximumTime = min(0.8 * Limits.time[us] - moveOverhead, maxScale * Time.optimumTime);

  if (use_time_management()) {
    int strength = log(max(1, (int)(Time.optimumTime * Threads.numThreads  / 10))) * 60;
    Time.tempoNNUE = clamp((strength + 264) / 24, 18, 30);
  } else
    Time.tempoNNUE = 28; // default for no time given

  if (option_value(OPT_PONDER))
    Time.optimumTime += Time.optimumTime / 4;
}
