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

#ifndef TT_H
#define TT_H

#include "misc.h"
#include "types.h"

// TTEntry struct is the 10 bytes transposition table entry, defined as below:
//
// key        16 bit
// depth       8 bit
// generation  5 bit
// pv node     1 bit
// bound type  2 bit
// move       16 bit
// value      16 bit
// eval value 16 bit

struct TTEntry {
  uint16_t key16;
  uint8_t  depth8;
  uint8_t  genBound8;
  uint16_t move16;
  int16_t  value16;
  int16_t  eval16;
};

typedef struct TTEntry TTEntry;

// A TranspositionTable consists of a power of 2 number of clusters and
// each cluster consists of ClusterSize number of TTEntry. Each non-empty
// entry contains information of exactly one position. The size of a
// cluster should divide the size of a cache line size, to ensure that
// clusters never cross cache lines. This ensures best cache performance,
// as the cacheline is prefetched, as soon as possible.

enum { CacheLineSize = 64, ClusterSize = 3 };

struct Cluster {
  TTEntry entry[ClusterSize];
  char padding[2]; // Align to a divisor of the cache line size
};

typedef struct Cluster Cluster;

struct TranspositionTable {
  size_t clusterCount;
  Cluster *table;
  alloc_t alloc;
  uint8_t generation8; // Size must be not bigger than TTEntry::genBound8
};

typedef struct TranspositionTable TranspositionTable;

extern TranspositionTable TT;

INLINE void tte_save(TTEntry *tte, Key k, Value v, bool pv, int b, Depth d,
    Move m, Value ev)
{
  // Preserve any existing move for the same position
  if (m || (uint16_t)k != tte->key16)
    tte->move16 = (uint16_t)m;

  // Don't overwrite more valuable entries
  if (  (uint16_t)k != tte->key16
      || d - DEPTH_OFFSET > tte->depth8 - 4
      || b == BOUND_EXACT)
  {
    assert(d > DEPTH_OFFSET && d < 256 + DEPTH_OFFSET);

    tte->key16     = (uint16_t)k;
    tte->depth8    = (uint8_t)(d - DEPTH_OFFSET);
    tte->genBound8 = (uint8_t)(TT.generation8 | ((uint8_t)pv << 2) | b);
    tte->value16   = (int16_t)v;
    tte->eval16    = (int16_t)ev;
  }
}

INLINE Move tte_move(TTEntry *tte)
{
  return tte->move16;
}

INLINE Value tte_value(TTEntry *tte)
{
  return tte->value16;
}

INLINE Value tte_eval(TTEntry *tte)
{
  return tte->eval16;
}

INLINE Depth tte_depth(TTEntry *tte)
{
  return tte->depth8 + DEPTH_OFFSET;
}

INLINE bool tte_is_pv(TTEntry *tte)
{
  return tte->genBound8 & 0x4;
}

INLINE int tte_bound(TTEntry *tte)
{
  return tte->genBound8 & 0x3;
}

void tt_free(void);

INLINE void tt_new_search(void)
{
  TT.generation8 += 8; // Lower 3 bits are used by PvNode and Bound
}

INLINE TTEntry *tt_first_entry(Key key)
{
  return &TT.table[mul_hi64(key, TT.clusterCount)].entry[0];
}

TTEntry *tt_probe(Key key, bool *found);
int tt_hashfull(void);
void tt_allocate(size_t mbSize);
void tt_clear(void);
void tt_clear_worker(int idx);

#endif
