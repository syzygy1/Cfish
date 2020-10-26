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

#include <inttypes.h>
#include <stdio.h>
#include <string.h>   // For memset
#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "bitboard.h"
#include "numa.h"
#include "settings.h"
#include "thread.h"
#include "tt.h"
#include "types.h"
#include "uci.h"

TranspositionTable TT; // Our global transposition table

// tt_free() frees the allocated transposition table memory.

void tt_free(void)
{
  if (TT.table)
    free_memory(&TT.alloc);
  TT.table = NULL;
}


// tt_allocate() allocates the transposition table, measured in megabytes.

void tt_allocate(size_t mbSize)
{
  TT.clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);
  size_t size = TT.clusterCount * sizeof(Cluster);

  TT.table = NULL;
  if (settings.largePages) {
    TT.table = allocate_memory(size, true, &TT.alloc);
#if !defined(__linux__)
    if (!TT.table)
      printf("info string Unable to allocate large pages for the "
             "transposition table.\n");
    else
      printf("info string Transposition table allocated using large pages.\n");
    fflush(stdout);
#endif
  }
  if (!TT.table)
    TT.table = allocate_memory(size, false, &TT.alloc);
  if (!TT.table)
    goto failed;

  // Clear the TT table to page in the memory immediately. This avoids
  // an initial slow down during the first second or minutes of the search.
  tt_clear();
  return;

failed:
  fprintf(stderr, "Failed to allocate %"PRIu64"MB for "
                  "transposition table.\n", (uint64_t)mbSize);
  exit(EXIT_FAILURE);
}


// tt_clear() initialises the entire transposition table to zero.

void tt_clear(void)
{
  // We let search threads clear the table in parallel. In NUMA mode,
  // this has the beneficial effect of spreading the TT over all nodes.

  if (TT.table) {
    for (int idx = 0; idx < Threads.numThreads; idx++)
      thread_wake_up(Threads.pos[idx], THREAD_TT_CLEAR);
    for (int idx = 0; idx < Threads.numThreads; idx++)
      thread_wait_until_sleeping(Threads.pos[idx]);
  }
}

void tt_clear_worker(int idx)
{
  // Find out which part of the TT this thread should clear.
  // To each thread we assign a number of 2MB blocks.

  size_t total = TT.clusterCount * sizeof(Cluster);
  size_t slice = (total + Threads.numThreads - 1) / Threads.numThreads;
  size_t blocks = (slice + (2 * 1024 * 1024) - 1) / (2 * 1024 * 1024);
  size_t begin = idx * blocks * (2 * 1024 * 1024);
  size_t end = begin + blocks * (2 * 1024 * 1024);
  begin = min(begin, total);
  end = min(end, total);

  // Now clear that part
  memset((uint8_t *)TT.table + begin, 0, end - begin);
}


// tt_probe() looks up the current position in the transposition table.
// It returns true and a pointer to the TTEntry if the position is found.
// Otherwise, it returns false and a pointer to an empty or least valuable
// TTEntry to be replaced later. The replace value of an entry is
// calculated as its depth minus 8 times its relative age. TTEntry t1 is
// considered more valuable than TTEntry t2 if its replace value is greater
// than that of t2.

TTEntry *tt_probe(Key key, bool *found)
{
  TTEntry *tte = tt_first_entry(key);
  uint16_t key16 = key; // Use the low 16 bits as key inside the cluster

  for (int i = 0; i < ClusterSize; i++)
    if (tte[i].key16 == key16 || !tte[i].depth8) {
//      if ((tte[i].genBound8 & 0xF8) != TT.generation8 && tte[i].key16)
      tte[i].genBound8 = TT.generation8 | (tte[i].genBound8 & 0x7); // Refresh
      *found = tte[i].depth8;
      return &tte[i];
    }

  // Find an entry to be replaced according to the replacement strategy
  TTEntry *replace = tte;
  for (int i = 1; i < ClusterSize; i++)
    // Due to our packed storage format for generation and its cyclic
    // nature we add 263 (256 is the modulus plus 7 to keep the unrelated
    // lowest three bits from affecting the result) to calculate the entry
    // age correctly even after generation8 overflows into the next cycle.
    if ( replace->depth8 - ((263 + TT.generation8 - replace->genBound8) & 0xF8)
        >  tte[i].depth8 - ((263 + TT.generation8 -   tte[i].genBound8) & 0xF8))
      replace = &tte[i];

  *found = false;
  return replace;
}


// Returns an approximation of the hashtable occupation during a search. The
// hash is x permill full, as per UCI protocol.

int tt_hashfull(void)
{
  int cnt = 0;
  for (int i = 0; i < 1000 / ClusterSize; i++) {
    const TTEntry *tte = &TT.table[i].entry[0];
    for (int j = 0; j < ClusterSize; j++)
      cnt += tte[j].depth8 && (tte[j].genBound8 & 0xf8) == TT.generation8;
  }
  return cnt * 1000 / (ClusterSize * (1000 / ClusterSize));
}
