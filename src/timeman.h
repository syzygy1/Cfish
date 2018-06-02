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

#ifndef TIMEMAN_H
#define TIMEMAN_H

#include "misc.h"
#include "search.h"
#include "thread.h"

// The TimeManagement class computes the optimal time to think depending on
// the maximum available time, the game move number and other parameters.

struct TimeManagement {
  TimePoint startTime;
  int optimumTime;
  int maximumTime;
  int64_t availableNodes;
};

extern struct TimeManagement Time;

void time_init(int us, int ply);

#define time_optimum() Time.optimumTime
#define time_maximum() Time.maximumTime

INLINE TimePoint time_elapsed(void)
{
  return Limits.npmsec ? (int64_t)threads_nodes_searched()
                       : now() - Time.startTime;
}

#endif
