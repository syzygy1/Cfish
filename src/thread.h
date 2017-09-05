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
#if defined(__GNUC__) && !defined(__WIN32__)
#include <pthread.h>
#else
#include <windows.h>
#endif

#include "types.h"

#define MAX_THREADS 512

#if defined(__GNUC) && !defined(__WIN32__)
#define LOCK_T pthread_mutex_t
#define LOCK_INIT(x) pthread_mutex_init(&(x), NULL)
#define LOCK_DESTROY(x) pthread_mutex_destroy(&(x))
#define LOCK(x) pthread_mutex_lock(&(x))
#define UNLOCK(x) pthread_mutex_unlock(&(x))
#else
#define LOCK_T HANDLE
#define LOCK_INIT(x) do { x = CreateMutex(NULL, FALSE, NULL); } while (0)
#define LOCK_DESTROY(x) CloseHandle(x)
#define LOCK(x) WaitForSingleObject(x, INFINITE)
#define UNLOCK(x) ReleaseMutex(x)
#endif

void thread_init(void *arg);
void thread_create(int idx);
void thread_search(Pos *pos);
void thread_idle_loop(Pos *pos);
void thread_start_searching(Pos *pos, int resume);
void thread_wait_for_search_finished(Pos *pos);
void thread_wait(Pos *pos, atomic_bool *b);


// MainThread struct seems to exist mostly for easy move.

struct MainThread {
  int easyMovePlayed, failedLow;
  double bestMoveChanges;
  Value previousScore;
};

typedef struct MainThread MainThread;

extern MainThread mainThread;

void mainthread_search(void);


// ThreadPool struct handles all the threads-related stuff like init,
// starting, parking and, most importantly, launching a thread. All the
// access to threads data is done through this class.

struct ThreadPool {
  Pos *pos[MAX_THREADS];
  int num_threads;
#if defined(__GNUC__) && !defined(__WIN32__)
  pthread_mutex_t mutex;
  pthread_cond_t sleepCondition;
  int initializing;
#else
  HANDLE event;
#endif
};

typedef struct ThreadPool ThreadPool;

void threads_init(void);
void threads_exit(void);
void threads_start_thinking(Pos *pos, LimitsType *);
void threads_set_number(int num);
uint64_t threads_nodes_searched(void);
uint64_t threads_tb_hits(void);

extern ThreadPool Threads;

INLINE Pos *threads_main(void)
{
  return Threads.pos[0];
}

CounterMoveHistoryStat **cmh_tables;
int num_cmh_tables;

#endif

