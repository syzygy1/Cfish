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

#include <assert.h>

#include "material.h"
#include "movegen.h"
#include "movepick.h"
#include "pawns.h"
#include "search.h"
#include "thread.h"
#include "uci.h"
#include "tbprobe.h"

// Global objects
ThreadPool Threads;
MainThread mainThread;

// thread_create() launches the thread and then waits until it goes to sleep
// in idle_loop().

Pos *thread_create(int idx)
{
  Pos *pos = malloc(sizeof(Pos));
  pos->thread_idx = idx;

  pos->pawnTable = calloc(16384, sizeof(PawnEntry));
  pos->materialTable = calloc(8192, sizeof(MaterialEntry));
  pos->history = malloc(sizeof(HistoryStats));
  pos->counterMoves = malloc(sizeof(MoveStats));
  pos->fromTo = malloc(sizeof(FromToStats));

  pos->rootMoves = malloc(sizeof(RootMoves));
  pos->stack = malloc((5 + MAX_PLY + 10) * sizeof(Stack));
  pos->stack += 5;
  pos->moveList = malloc(10000 * sizeof(ExtMove));

  stats_clear(pos->history);
  stats_clear(pos->counterMoves);
  stats_clear(pos->fromTo);

  atomic_store(&pos->resetCalls, 0);
  pos->exit = 0;
  pos->maxPly = pos->callsCnt = 0;

#ifndef __WIN32__
  pthread_mutex_init(&pos->mutex, NULL);
  pthread_cond_init(&pos->sleepCondition, NULL);

  pthread_mutex_lock(&pos->mutex);
  pos->searching = 1;
  pthread_create(&pos->nativeThread, NULL, (void*(*)(void*))thread_idle_loop, pos);
  while (pos->searching)
    pthread_cond_wait(&pos->sleepCondition, &pos->mutex);
  pthread_mutex_unlock(&pos->mutex);
#else
  pos->startEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  pos->stopEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  pos->nativeThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)thread_idle_loop, pos, 0, NULL);
#endif

  return pos;
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
#else
  pos->exit = 1;
  SetEvent(pos->startEvent);
  WaitForSingleObject(pos->nativeThread, INFINITE);
#endif

  free(pos->pawnTable);
  free(pos->materialTable);
  free(pos->history);
  free(pos->counterMoves);
  free(pos->fromTo);

  free(pos->rootMoves);
  free(pos->stack - 5);
  free(pos->moveList);

#ifndef __WIN32__
  pthread_cond_destroy(&pos->sleepCondition);
  pthread_mutex_destroy(&pos->mutex);
#else
#endif
  
  free(pos);
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
  while (!pos->exit) {
#ifndef __WIN32__
    pthread_mutex_lock(&pos->mutex);

    pos->searching = 0;

    while (!pos->searching && !pos->exit) {
      pthread_cond_signal(&pos->sleepCondition); // Wake up any waiting thread
      pthread_cond_wait(&pos->sleepCondition, &pos->mutex);
    }

    pthread_mutex_unlock(&pos->mutex);

    if (!pos->exit) {
      if (pos->thread_idx == 0)
        mainthread_search();
      else
        thread_search(pos);
    }
#else
//    pos->searching = 0;
    WaitForSingleObject(pos->startEvent, INFINITE);
    if (!pos->exit) {
      if (pos->thread_idx == 0)
        mainthread_search();
      else
        thread_search(pos);
    }
    SetEvent(pos->stopEvent);
#endif
  }
}


// threads_init() creates and launches requested threads that will go
// immediately to sleep. We cannot use a constructor because Threads is a
// static object and we need a fully initialized engine at this point due to
// allocation of Endgames in the Thread constructor.

void threads_init(void)
{
#ifdef __WIN32__
  io_mutex = CreateMutex(NULL, FALSE, NULL);
#endif

  Threads.num_threads = 1;
  Threads.pos[0] = thread_create(0);
}


// threads_exit() terminates threads before the program exits. Cannot be
// done in destructor because threads must be terminated before deleting
// any static objects while still in main().

void threads_exit(void)
{
  while (Threads.num_threads > 0)
    thread_destroy(Threads.pos[--Threads.num_threads]);

#ifdef __WIN32__
  CloseHandle(io_mutex);
#endif
}


// threads_read_uci_options() updates internal threads parameters from
// the corresponding UCI options and creates/destroys threads to match
// requested number. Thread objects are dynamically allocated.

void threads_read_uci_options(void)
{
  size_t requested = option_value(OPT_THREADS);

  assert(requested > 0);

  while (Threads.num_threads < requested) {
    Threads.pos[Threads.num_threads] = thread_create(Threads.num_threads);
    Threads.num_threads++;
  }

  while (Threads.num_threads > requested)
    thread_destroy(Threads.pos[--Threads.num_threads]);
}


// threads_nodes_searched() returns the number of nodes searched.

uint64_t threads_nodes_searched(void)
{
  uint64_t nodes = 0;
  for (size_t idx = 0; idx < Threads.num_threads; idx++)
    nodes += Threads.pos[idx]->nodes;
  return nodes;
}


// threads_tb_hits() returns the number of TB hits.

uint64_t threads_tb_hits(void)
{
  uint64_t hits = 0;
  for (size_t idx = 0; idx < Threads.num_threads; idx++)
    hits += Threads.pos[idx]->tb_hits;
  return hits;
}


// threads_start_thinking() wakes up the main thread sleeping in
// idle_loop() and starts a new search, then returns immediately.

void threads_start_thinking(Pos *root, LimitsType *limits)
{
  if (Signals.searching)
    thread_wait_for_search_finished(threads_main());

  Signals.stopOnPonderhit = Signals.stop = 0;
  Limits = *limits;

  ExtMove list[MAX_MOVES];
  ExtMove *end = generate_legal(root, list);

  end = TB_filter_root_moves(root, list, end);

  ExtMove *p = list;
  for (ExtMove *m = p; m < end; m++) {
    int i;
    for (i = 0; i < limits->num_searchmoves; i++)
      if (m->move == limits->searchmoves[i])
        break;
    if (i == limits->num_searchmoves)
      *p++ = *m;
  }
  end = p;

  for (size_t idx = 0; idx < Threads.num_threads; idx++) {
    Pos *pos = Threads.pos[idx];
    pos->maxPly = 0;
    pos->rootDepth = DEPTH_ZERO;
    pos->nodes = pos->tb_hits = 0;
    RootMoves *rm = pos->rootMoves;
    rm->size = end - list;
    for (size_t i = 0; i < rm->size; i++) {
      rm->move[i].pv[0] = list[i].move;
      rm->move[i].score = -VALUE_INFINITE;
      rm->move[i].previousScore = -VALUE_INFINITE;
    }
    pos_copy(pos, root);
  }

  if (TB_RootInTB)
    Threads.pos[0]->tb_hits = end - list;

  Signals.searching = 1;
  thread_start_searching(threads_main(), 0);
}

