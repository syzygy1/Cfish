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

#include "movegen.h"
#include "search.h"
#include "thread.h"
#include "uci.h"
#include "tbprobe.h"

// Global objects
ThreadPool Threads;
MainThread mainThread;

// thread_create() launches the thread and then waits until it goes to sleep
// in idle_loop().

Thread *thread_create(int idx)
{
  Thread *th = malloc(sizeof(Thread));
  th->idx = idx;

  th->pawnTable = malloc(sizeof(PawnTable));
  th->materialTable = malloc(sizeof(MaterialTable));
  th->history = malloc(sizeof(HistoryStats));
  th->counterMoves = malloc(sizeof(MoveStats));
  th->rootMoves = malloc(sizeof(RootMoves));

  stats_clear(th->history);
  stats_clear(th->counterMoves);

  atomic_store(&th->resetCalls, 0);
  th->exit = 0;
  th->maxPly = th->callsCnt = 0;

  pthread_mutex_init(&th->mutex, NULL);
  pthread_cond_init(&th->sleepCondition, NULL);

  pthread_mutex_lock(&th->mutex);
  th->searching = 1;
  pthread_create(&th->nativeThread, NULL, (void*(*)(void*))thread_idle_loop, th);
  while (th->searching)
    pthread_cond_wait(&th->sleepCondition, &th->mutex);
  pthread_mutex_unlock(&th->mutex);

  return th;
}


// thread_destroy() waits for thread termination before returning.

void thread_destroy(Thread *th)
{
  pthread_mutex_lock(&th->mutex);
  th->exit = 1;
  pthread_cond_signal(&th->sleepCondition);
  pthread_mutex_unlock(&th->mutex);
  pthread_join(th->nativeThread, NULL);

  free(th->pawnTable);
  free(th->materialTable);
  free(th->history);
  free(th->counterMoves);
  free(th->rootMoves);
  
  free(th);
}


// thread_wait_for_search_finished() waits on sleep condition until
// not searching.

void thread_wait_for_search_finished(Thread *th)
{
  pthread_mutex_lock(&th->mutex);
  while (th->searching)
    pthread_cond_wait(&th->sleepCondition, &th->mutex);
  pthread_mutex_unlock(&th->mutex);
}


// thread_wait() waits on sleep condition until condition is true.

void thread_wait(Thread *th, atomic_bool *condition)
{
  pthread_mutex_lock(&th->mutex);
  while (!atomic_load(condition))
    pthread_cond_wait(&th->sleepCondition, &th->mutex);
  pthread_mutex_unlock(&th->mutex);
}


// thread_start_searching() wakes up the thread that will start the search.

void thread_start_searching(Thread *th, int resume)
{
  pthread_mutex_lock(&th->mutex);

  if (!resume)
    th->searching = 1;

  pthread_cond_signal(&th->sleepCondition);
  pthread_mutex_unlock(&th->mutex);
}


// thread_idle_loop() is where the thread is parked when it has no work to do.

void thread_idle_loop(Thread *th)
{
  while (!th->exit) {
    pthread_mutex_lock(&th->mutex);

    th->searching = 0;

    while (!th->searching && !th->exit) {
      pthread_cond_signal(&th->sleepCondition); // Wake up any waiting thread
      pthread_cond_wait(&th->sleepCondition, &th->mutex);
    }

    pthread_mutex_unlock(&th->mutex);

    if (!th->exit)
      thread_search(th);
  }
}


// threads_init() creates and launches requested threads that will go
// immediately to sleep. We cannot use a constructor because Threads is a
// static object and we need a fully initialized engine at this point due to
// allocation of Endgames in the Thread constructor.

void threads_init(void)
{
  Threads.num_threads = 1;
  Threads.thread[0] = thread_create(0);
//  threads_read_uci_options();
}


// threads_exit() terminates threads before the program exits. Cannot be
// done in destructor because threads must be terminated before deleting
// any static objects while still in main().

void threads_exit(void)
{
  while (Threads.num_threads > 0)
    thread_destroy(Threads.thread[--Threads.num_threads]);
}


// threads_read_uci_options() updates internal threads parameters from
// the corresponding UCI options and creates/destroys threads to match
// requested number. Thread objects are dynamically allocated.

void threads_read_uci_options(void)
{
  size_t requested = option_value(OPT_THREADS);

  assert(requested > 0);

  while (Threads.num_threads < requested) {
    Threads.thread[Threads.num_threads] = thread_create(Threads.num_threads);
    Threads.num_threads++;
  }

  while (Threads.num_threads > requested)
    thread_destroy(Threads.thread[--Threads.num_threads]);
}


// threads_nodes_searched() returns the number of nodes searched.

uint64_t threads_nodes_searched(void)
{
  uint64_t nodes = 0;
  for (size_t idx = 0; idx < Threads.num_threads; idx++)
    nodes += Threads.thread[idx]->rootPos.nodes;
  return nodes;
}


// threads_start_thinking() wakes up the main thread sleeping in
// idle_loop() and starts a new search, then returns immediately.

void threads_start_thinking(Pos *pos, State *states, LimitsType *limits)
{
  thread_wait_for_search_finished(threads_main());

  Signals.stopOnPonderhit = Signals.stop = 0;
  Limits = *limits;
  RootMoves rootMoves;

  ExtMove list[MAX_MOVES];
  ExtMove *end = generate_legal(pos, list);
  for (ExtMove *m = list; m < end; m++) {
    int i;
    for (i = 0; i < limits->num_searchmoves; i++)
      if (m->move == limits->searchmoves[i])
        break;
    if (i == limits->num_searchmoves) {
      rootMoves.move[rootMoves.size].pv[0] = m->move;
      rootMoves.move[rootMoves.size].score = -VALUE_INFINITE;
      rootMoves.move[rootMoves.size].previousScore = -VALUE_INFINITE;
      rootMoves.size++;
    }
  }

  TB_filter_root_moves(pos, &rootMoves);

#if 0
  // After ownership transfer 'states' becomes empty, so if we stop the search
  // and call 'go' again without setting a new position states.get() == NULL.
  assert(states.get() || setupStates.get());

  if (states.get())
    setupStates = std::move(states); // Ownership transfer, states is now empty

  State tmp = setupStates->back();
#endif

//  State tmp = *states;
  State tmp = *(pos->st);

  char fen[80];
  pos_fen(pos, fen);

  for (size_t idx = 0; idx < Threads.num_threads; idx++) {
    Thread *th = Threads.thread[idx];
    th->maxPly = 0;
    th->rootDepth = DEPTH_ZERO;
    *(th->rootMoves) = rootMoves;
    pos_set(&th->rootPos, fen, is_chess960(), states, th);
  }

  *states = tmp; // Restore st->previous, cleared by Position::set()

  mainthread_search();
}

