/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

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

#include <assert.h>

#include "material.h"
#include "movegen.h"
#include "movepick.h"
#include "numa.h"
#include "pawns.h"
#include "search.h"
#include "settings.h"
#include "thread.h"
#include "uci.h"
#include "tbprobe.h"

// Global objects
ThreadPool Threads;
MainThread mainThread;
CounterMoveHistoryStat **cmh_tables = NULL;
int num_cmh_tables = 0;

// thread_init() is where a search thread starts and initialises itself.

void thread_init(void *arg)
{
  int idx = (intptr_t)arg;

  int node;
  if (settings.numa_enabled)
    node = bind_thread_to_numa_node(idx);
  else
    node = 0;
  if (node >= num_cmh_tables) {
    int old = num_cmh_tables;
    num_cmh_tables = node + 16;
    cmh_tables = realloc(cmh_tables,
                         num_cmh_tables * sizeof(CounterMoveHistoryStat *));
    while (old < num_cmh_tables)
      cmh_tables[old++] = NULL;
  }
  if (!cmh_tables[node]) {
    if (settings.numa_enabled)
      cmh_tables[node] = numa_alloc(sizeof(CounterMoveHistoryStat));
    else
      cmh_tables[node] = calloc(sizeof(CounterMoveHistoryStat), 1);
    for (int j = 0; j < 16; j++)
      for (int k = 0; k < 64; k++)
        (*cmh_tables[node])[0][0][j][k] = VALUE_ZERO - 1;
  }

  Pos *pos;

  if (settings.numa_enabled) {
    pos = numa_alloc(sizeof(Pos));
    pos->pawnTable = numa_alloc(PAWN_ENTRIES * sizeof(PawnEntry));
    pos->materialTable = numa_alloc(8192 * sizeof(MaterialEntry));
    pos->counterMoves = numa_alloc(sizeof(CounterMoveStat));
    pos->history = numa_alloc(sizeof(ButterflyHistory));
    pos->rootMoves = numa_alloc(sizeof(RootMoves));
    pos->stack = numa_alloc((MAX_PLY + 110) * sizeof(Stack));
    pos->moveList = numa_alloc(10000 * sizeof(ExtMove));
  } else {
    pos = calloc(sizeof(Pos), 1);
    pos->pawnTable = calloc(PAWN_ENTRIES * sizeof(PawnEntry), 1);
    pos->materialTable = calloc(8192 * sizeof(MaterialEntry), 1);
    pos->counterMoves = calloc(sizeof(CounterMoveStat), 1);
    pos->history = calloc(sizeof(ButterflyHistory), 1);
    pos->rootMoves = calloc(sizeof(RootMoves), 1);
    pos->stack = calloc((MAX_PLY + 110) * sizeof(Stack), 1);
    pos->moveList = calloc(10000 * sizeof(ExtMove), 1);
  }
  pos->thread_idx = idx;
  pos->counterMoveHistory = cmh_tables[node];

  atomic_store(&pos->resetCalls, 0);
  pos->exit = 0;
  pos->selDepth = pos->callsCnt = 0;

#ifndef __WIN32__  // linux

  pthread_mutex_init(&pos->mutex, NULL);
  pthread_cond_init(&pos->sleepCondition, NULL);

  Threads.pos[idx] = pos;

  pthread_mutex_lock(&Threads.mutex);
  Threads.initializing = 0;
  pthread_cond_signal(&Threads.sleepCondition);
  pthread_mutex_unlock(&Threads.mutex);

#else // Windows

  pos->startEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  pos->stopEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

  Threads.pos[idx] = pos;

  SetEvent(Threads.event);

#endif

  thread_idle_loop(pos);
}

// thread_create() launches a new thread.

void thread_create(int idx)
{
#ifndef __WIN32__

  pthread_t thread;

  Threads.initializing = 1;
  pthread_mutex_lock(&Threads.mutex);
  pthread_create(&thread, NULL, (void*(*)(void*))thread_init,
                 (void *)(intptr_t)idx);
  while (Threads.initializing)
    pthread_cond_wait(&Threads.sleepCondition, &Threads.mutex);
  pthread_mutex_unlock(&Threads.mutex);

#else

  HANDLE *thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)thread_init, (void *)(intptr_t)idx, 0 , NULL);
  WaitForSingleObject(Threads.event, INFINITE);

#endif

  Threads.pos[idx]->nativeThread = thread;
}


// thread_destroy() waits for thread termination before returning.

void thread_destroy(Pos *pos)
{
#ifndef __WIN32__
  pthread_mutex_lock(&pos->mutex);
  pos->exit = 1;
  pthread_cond_signal(&pos->sleepCondition);
  pthread_mutex_unlock(&pos->mutex);
  pthread_join(pos->nativeThread, NULL);
  pthread_cond_destroy(&pos->sleepCondition);
  pthread_mutex_destroy(&pos->mutex);
#else
  pos->exit = 1;
  SetEvent(pos->startEvent);
  WaitForSingleObject(pos->nativeThread, INFINITE);
  CloseHandle(pos->startEvent);
  CloseHandle(pos->stopEvent);
#endif

  if (settings.numa_enabled) {
    numa_free(pos->pawnTable, PAWN_ENTRIES * sizeof(PawnEntry));
    numa_free(pos->materialTable, 8192 * sizeof(MaterialEntry));
    numa_free(pos->counterMoves, sizeof(CounterMoveStat));
    numa_free(pos->history, sizeof(ButterflyHistory));
    numa_free(pos->rootMoves, sizeof(RootMoves));
    numa_free(pos->stack, (MAX_PLY + 110) * sizeof(Stack));
    numa_free(pos->moveList, 10000 * sizeof(ExtMove));
    numa_free(pos, sizeof(Pos));
  } else {
    free(pos->pawnTable);
    free(pos->materialTable);
    free(pos->counterMoves);
    free(pos->history);
    free(pos->rootMoves);
    free(pos->stack);
    free(pos->moveList);
    free(pos);
  }
}


// thread_wait_for_search_finished() waits on sleep condition until
// not searching.

void thread_wait_for_search_finished(Pos *pos)
{
#ifndef __WIN32__
  pthread_mutex_lock(&pos->mutex);
  while (pos->searching)
    pthread_cond_wait(&pos->sleepCondition, &pos->mutex);
  pthread_mutex_unlock(&pos->mutex);
#else
  WaitForSingleObject(pos->stopEvent, INFINITE);
#endif

  if (pos->thread_idx == 0)
    Signals.searching = 0;
}


// thread_wait() waits on sleep condition until condition is true.

void thread_wait(Pos *pos, atomic_bool *condition)
{
#ifndef __WIN32__
  pthread_mutex_lock(&pos->mutex);
  while (!atomic_load(condition))
    pthread_cond_wait(&pos->sleepCondition, &pos->mutex);
  pthread_mutex_unlock(&pos->mutex);
#else
  (void)condition;
  WaitForSingleObject(pos->startEvent, INFINITE);
#endif
}


// thread_start_searching() wakes up the thread that will start the search.

void thread_start_searching(Pos *pos, int resume)
{
#ifndef __WIN32__
  pthread_mutex_lock(&pos->mutex);

  if (!resume)
    pos->searching = 1;

  pthread_cond_signal(&pos->sleepCondition);
  pthread_mutex_unlock(&pos->mutex);
#else
  (void)resume;
  SetEvent(pos->startEvent);
#endif
}


// thread_idle_loop() is where the thread is parked when it has no work to do.

void thread_idle_loop(Pos *pos)
{
#ifndef __WIN32__

  pthread_mutex_lock(&pos->mutex);
  while (1) {

    while (!pos->searching && !pos->exit) {
      pthread_cond_signal(&pos->sleepCondition); // Wake up any waiting thread
      pthread_cond_wait(&pos->sleepCondition, &pos->mutex);
    }

    pthread_mutex_unlock(&pos->mutex);

    if (pos->exit)
      break;

    if (pos->thread_idx == 0)
      mainthread_search();
    else
      thread_search(pos);

    pthread_mutex_lock(&pos->mutex);
    pos->searching = 0;
  }

#else

  while (1) {
    WaitForSingleObject(pos->startEvent, INFINITE);

    if (pos->exit)
      break;

    if (pos->thread_idx == 0)
      mainthread_search();
    else
      thread_search(pos);

    SetEvent(pos->stopEvent);
  }

#endif
}


// threads_init() creates and launches requested threads that will go
// immediately to sleep. We cannot use a constructor because Threads is a
// static object and we need a fully initialized engine at this point due to
// allocation of Endgames in the Thread constructor.

void threads_init(void)
{
#ifndef __WIN32__
  pthread_mutex_init(&Threads.mutex, NULL);
  pthread_cond_init(&Threads.sleepCondition, NULL);
#else
  io_mutex = CreateMutex(NULL, FALSE, NULL);
  Threads.event = CreateEvent(NULL, FALSE, FALSE, NULL);
#endif

#ifdef NUMA
  numa_init();
#endif

  Threads.num_threads = 1;
  thread_create(0);
}


// threads_exit() terminates threads before the program exits. Cannot be
// done in destructor because threads must be terminated before deleting
// any static objects while still in main().

void threads_exit(void)
{
  threads_set_number(0);

#ifndef __WIN32__
  pthread_cond_destroy(&Threads.sleepCondition);
  pthread_mutex_destroy(&Threads.mutex);
#else
  CloseHandle(io_mutex);
  CloseHandle(Threads.event);
#endif

#ifdef NUMA
  numa_exit();
#endif
}


// threads_set_number() creates/destroys threads to match the requested
// number.

void threads_set_number(int num)
{
  while (Threads.num_threads < num)
    thread_create(Threads.num_threads++);

  while (Threads.num_threads > num)
    thread_destroy(Threads.pos[--Threads.num_threads]);

  if (num == 0 && num_cmh_tables > 0) {
    for (int i = 0; i < num_cmh_tables; i++)
      if (cmh_tables[i]) {
        if (settings.numa_enabled)
          numa_free(cmh_tables[i], sizeof(CounterMoveHistoryStat));
        else
          free(cmh_tables[i]);
      }
    free(cmh_tables);
    cmh_tables = NULL;
    num_cmh_tables = 0;
  }

  if (num == 0)
    Signals.searching = 0;
}


// threads_nodes_searched() returns the number of nodes searched.

uint64_t threads_nodes_searched(void)
{
  uint64_t nodes = 0;
  for (int idx = 0; idx < Threads.num_threads; idx++)
    nodes += Threads.pos[idx]->nodes;
  return nodes;
}


// threads_tb_hits() returns the number of TB hits.

uint64_t threads_tb_hits(void)
{
  uint64_t hits = 0;
  for (int idx = 0; idx < Threads.num_threads; idx++)
    hits += Threads.pos[idx]->tb_hits;
  return hits;
}

