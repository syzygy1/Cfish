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

#define _GNU_SOURCE

#include <string.h>   // For std::memset
#include <stdio.h>
#if defined(__GNUC__) && !defined(__WIN32__)
#include <sys/mman.h>
#endif

#include "bitboard.h"
#include "numa.h"
#include "settings.h"
#include "tt.h"
#include "types.h"
#include "uci.h"

TranspositionTable TT; // Our global transposition table

// tt_free() frees the allocated transposition table memory.

void tt_free(void)
{
#if (defined(__GNUC__) && defined(__WIN32__)) || defined(_WIN32)
  if (TT.mem)
    VirtualFree(TT.mem, 0, MEM_RELEASE);
#else
  if (TT.mem)
    munmap(TT.mem, TT.alloc_size);
#endif
  TT.mem = NULL;
}


// tt_allocate() allocates the transposition table, measured in 
// megabytes.

void tt_allocate(size_t mbSize)
{
  size_t count = ((size_t)1) << msb((mbSize * 1024 * 1024) / sizeof(Cluster));

  TT.mask = count - 1;

  size_t size = count * sizeof(Cluster);

#if (defined(__GNUC__) && defined(__WIN32__)) || defined(_WIN32)

  TT.mem = NULL;
  if (settings.large_pages) {
    size_t page_size = large_page_minimum;
    size_t lp_size = (size + page_size - 1) & ~(page_size - 1);
    TT.mem = VirtualAlloc(NULL, lp_size,
                          MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                          PAGE_READWRITE);
    if (!TT.mem)
      printf("info string Unable to allocate large pages for the "
             "transposition table.\n");
    else
      printf("info string Transposition table allocated using large pages.\n");
    fflush(stdout);
  }

  if (!TT.mem)
    TT.mem = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!TT.mem)
    goto failed;
  TT.table = (Cluster *)TT.mem;

#else /* Unix */

  size_t alignment = settings.large_pages ? (1ULL << 21) : 1;
  size_t alloc_size = size + alignment - 1;

#if defined(__APPLE__) && defined(VM_FLAGS_SUPERPAGE_SIZE_2MB)

  if (settings.large_pages) {
    TT.mem = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
    if (!TT.mem)
      printf("info string Unable to allocate large pages for the "
             "transposition table.\n");
    else
      printf("info string Transposition table allocated using large pages.\n");
    fflush(stdout);
  }
  if (!TT.mem)
    TT.mem = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#else

  TT.mem = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

#endif

  TT.alloc_size = alloc_size;
  TT.table = (Cluster *)(  (((uintptr_t)TT.mem) + alignment - 1)
                         & ~(alignment - 1));
  if (!TT.mem)
    goto failed;

#ifdef NUMA
  // Interleave the shared transposition table across all nodes.
  // Create an interleave mask of the nodes on which threads are
  // actually running?
  if (settings.numa_enabled)
    numa_interleave_memory(TT.table, count * sizeof(Cluster), settings.mask);
#endif

#ifdef __linux__
#ifdef MADV_HUGEPAGE

  // Advise the kernel to allocate large pages.
  if (settings.large_pages)
    madvise(TT.table, count * sizeof(Cluster), MADV_HUGEPAGE);

#endif
#endif

#endif

  return;


failed:
  fprintf(stderr, "Failed to allocate %" FMT_Z "uMB for "
                  "transposition table.\n", mbSize);
  exit(EXIT_FAILURE);
}


// tt_clear() overwrites the entire transposition table with zeros. It
// is called whenever the table is resized, or when the user asks the
// program to clear the table (from the UCI interface).

void tt_clear(void)
{
  if (TT.table)
    memset(TT.table, 0, (TT.mask + 1) * sizeof(Cluster));
}


// tt_probe() looks up the current position in the transposition table.
// It returns true and a pointer to the TTEntry if the position is found.
// Otherwise, it returns false and a pointer to an empty or least valuable
// TTEntry to be replaced later. The replace value of an entry is
// calculated as its depth minus 8 times its relative age. TTEntry t1 is
// considered more valuable than TTEntry t2 if its replace value is greater
// than that of t2.

TTEntry *tt_probe(Key key, int *found)
{
  TTEntry *tte = tt_first_entry(key);
  uint16_t key16 = key >> 48; // Use the high 16 bits as key inside the cluster

  for (int i = 0; i < ClusterSize; i++)
    if (!tte[i].key16 || tte[i].key16 == key16) {
      if ((tte[i].genBound8 & 0xFC) != TT.generation8 && tte[i].key16)
        tte[i].genBound8 = (uint8_t)(TT.generation8 | tte_bound(&tte[i])); // Refresh
      *found = (int)tte[i].key16;
      return &tte[i];
    }

  // Find an entry to be replaced according to the replacement strategy
  TTEntry* replace = tte;
  for (int i = 1; i < ClusterSize; i++)
    // Due to our packed storage format for generation and its cyclic
    // nature we add 259 (256 is the modulus plus 3 to keep the lowest
    // two bound bits from affecting the result) to calculate the entry
    // age correctly even after generation8 overflows into the next cycle.
    if (  replace->depth8 - ((259 + TT.generation8 - replace->genBound8) & 0xFC) * 2
        >   tte[i].depth8 - ((259 + TT.generation8 -   tte[i].genBound8) & 0xFC) * 2)
      replace = &tte[i];

  *found = 0;
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
      if ((tte[j].genBound8 & 0xFC) == TT.generation8)
        cnt++;
  }
  return cnt;
}

