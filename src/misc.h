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

#ifndef MISC_H
#define MISC_H

#include <assert.h>
#include <stdio.h>
#ifndef __WIN32__
#include <pthread.h>
#endif
#include <stdatomic.h>
#include <sys/time.h>

#include "types.h"

void print_engine_info(int to_uci);

// prefetch() preloads the given address in L1/L2 cache. This is
// a non-blocking function that doesn't stall the CPU waiting for data
// to be loaded from memory, which can be quite slow.

INLINE void prefetch(void *addr)
{
#ifndef NO_PREFETCH

#if defined(__INTEL_COMPILER)
// This hack prevents prefetches from being optimized away by
// Intel compiler. Both MSVC and gcc seem not be affected by this.
  __asm__ ("");
#endif

#if defined(__INTEL_COMPILER) || defined(_MSC_VER)
  _mm_prefetch((char*)addr, _MM_HINT_T0);
#else
  __builtin_prefetch(addr);
#endif
#endif
}

void start_logger(const char *fname);

void dbg_hit_on(int b);
void dbg_hit_on_cond(int c, int b);
void dbg_mean_of(int v);
void dbg_print();

typedef uint64_t TimePoint; // A value in milliseconds

INLINE TimePoint now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1000 * (uint64_t)tv.tv_sec + (uint64_t)tv.tv_usec / 1000;
}

#ifndef __WIN32__
extern pthread_mutex_t io_mutex;
#define IO_LOCK   pthread_mutex_lock(&io_mutex)
#define IO_UNLOCK pthread_mutex_unlock(&io_mutex)
#else
ssize_t getline(char **lineptr, size_t *n, FILE *stream);
extern HANDLE io_mutex;
#define IO_LOCK WaitForSingleObject(io_mutex, INFINITE)
#define IO_UNLOCK ReleaseMutex(io_mutex)
int large_pages_supported(void);
extern size_t large_page_minimum;
#endif

struct PRNG
{
  uint64_t s;
};

typedef struct PRNG PRNG;

void prng_init(PRNG *rng, uint64_t seed);
uint64_t prng_rand(PRNG *rng);
uint64_t prng_sparse_rand(PRNG *rng);

#endif

