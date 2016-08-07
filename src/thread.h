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

#ifndef THREAD_H
#define THREAD_H

#include <stdatomic.h>
#include <pthread.h>

#include "material.h"
#include "movepick.h"
#include "pawns.h"
#include "position.h"
#include "search.h"
#include "thread_win32.h"

#define MAX_THREADS 128


// Thread struct keeps together all the thread-related stuff. We also use
// per-thread pawn and material hash tables so that once we get a pointer to an
// entry its life time is unlimited and we don't have to care about someone
// changing the entry under our feet.

struct Thread {
  pthread_t nativeThread;
  pthread_mutex_t mutex;
  pthread_cond_t sleepCondition;
  int exit, searching;

  PawnTable *pawnTable;
  MaterialTable *materialTable;
  size_t idx, PVIdx;
  int maxPly, callsCnt;

  Pos rootPos;
  RootMoves *rootMoves;
  Depth rootDepth;
  HistoryStats *history;
  MoveStats *counterMoves;
  FromToStats *fromTo;
  Depth completedDepth;
  atomic_bool resetCalls;
};

typedef struct Thread Thread;

Thread *thread_create(int idx);
void thread_search(Thread *thread ); // virtual?
void thread_idle_loop(Thread *thread);
void thread_start_searching(Thread *thread, int resume);
void thread_wait_for_search_finished(Thread *thread);
void thread_wait(Thread *thread, atomic_bool *b);


// MainThread is a derived class with a specific overload for the main thread

struct MainThread {
  int easyMovePlayed, failedLow;
  double bestMoveChanges;
  Value previousScore;
};

typedef struct MainThread MainThread;

extern MainThread mainThread;

void mainthread_search();


// ThreadPool struct handles all the threads-related stuff like init,
// starting, parking and, most importantly, launching a thread. All the
// access to threads data is done through this class.

struct ThreadPool {
  Thread *thread[MAX_THREADS];
  size_t num_threads;
//  StateListPtr setupStates; // ??
};

typedef struct ThreadPool ThreadPool;

void threads_init(void);
void threads_exit(void);
void threads_start_thinking(Pos *pos, State *, LimitsType *);
void threads_read_uci_options(void);
uint64_t threads_nodes_searched(void);

extern ThreadPool Threads;

static inline Thread *threads_main(void)
{
  return Threads.thread[0];
}

#endif

