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

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#ifdef __WIN32__
#include <windows.h>
#endif

#include "misc.h"
#include "thread.h"

// Version number. If Version is left empty, then compile date in the format
// DD-MM-YY and show in engine_info.
char Version[] = "";

#ifndef __WIN32__
pthread_mutex_t io_mutex = PTHREAD_MUTEX_INITIALIZER;
#else
HANDLE io_mutex;
#endif

#if 0
// Our fancy logging facility. The trick here is to replace cin.rdbuf() and
// cout.rdbuf() with two Tie objects that tie cin and cout to a file stream. We
/ can toggle the logging of std::cout and std:cin at runtime whilst preserving
// usual I/O functionality, all without changing a single line of code!
// Idea from http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

struct Tie: public streambuf { // MSVC requires split streambuf for cin and cout

  Tie(streambuf* b, streambuf* l) : buf(b), logBuf(l) {}

  int sync() { return logBuf->pubsync(), buf->pubsync(); }
  int overflow(int c) { return log(buf->sputc((char)c), "<< "); }
  int underflow() { return buf->sgetc(); }
  int uflow() { return log(buf->sbumpc(), ">> "); }

  streambuf *buf, *logBuf;

  int log(int c, const char* prefix) {

    static int last = '\n'; // Single log file

    if (last == '\n')
        logBuf->sputn(prefix, 3);

    return last = logBuf->sputc((char)c);
  }
};

class Logger {

  Logger() : in(cin.rdbuf(), file.rdbuf()), out(cout.rdbuf(), file.rdbuf()) {}
 ~Logger() { start(""); }

  ofstream file;
  Tie in, out;

public:
  static void start(const std::string& fname) {

    static Logger l;

    if (!fname.empty() && !l.file.is_open())
    {
        l.file.open(fname, ifstream::out);
        cin.rdbuf(&l.in);
        cout.rdbuf(&l.out);
    }
    else if (fname.empty() && l.file.is_open())
    {
        cout.rdbuf(l.out.buf);
        cin.rdbuf(l.in.buf);
        l.file.close();
    }
  }
};
#endif

static char months[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
static char date[] = __DATE__;

// print engine_info() prints the full name of the current Stockfish version.
// This will be either "Stockfish <Tag> DD-MM-YY" (where DD-MM-YY is the
// date when the program was compiled) or "Stockfish <Version>", depending
// on whether Version is empty.

void print_engine_info(int to_uci)
{
  char my_date[64];

  printf("Cfish %s", Version);

  if (strlen(Version) == 0) {
    int day, month, year;

    strcpy(my_date, date);
    char *str = strtok(my_date, " "); // month
    for (month = 1; strncmp(str, &months[3 * month - 3], 3) != 0; month++);
    str = strtok(NULL, " "); // day
    day = atoi(str);
    str = strtok(NULL, " "); // year
    year = atoi(str);

    printf("%02d%02d%02d", day, month, year % 100);
  }

  printf("%s%s%s%s\n", Is64Bit ? " 64" : ""
                     , HasPext ? " BMI2" : (HasPopCnt ? " POPCNT" : "")
                     , HasNuma ? " NUMA" : ""
                     , to_uci ? "\nid author T. Romstad, M. Costalba, "
                                "J. Kiiski, G. Linscott"
                              : " by Syzygy based on Stockfish");
  fflush(stdout);
}


// Debug functions used mainly to collect run-time statistics
static int64_t hits[2], means[2];

void dbg_hit_on(int b)
{
  hits[0]++;
  if (b) hits[1]++;
}

void dbg_hit_on_cond(int c, int b)
{
  if (c) dbg_hit_on(b);
}

void dbg_mean_of(int v)
{
  means[0]++;
  means[1] += v;
}

void dbg_print(void)
{
  if (hits[0])
    fprintf(stderr, "Total %"PRIu64" Hits %"PRIu64" hit rate (%%%"PRIu64")\n",
                    hits[0], hits[1] , 100 * hits[1] / hits[0]);

  if (means[0])
    fprintf(stderr, "Total %"PRIu64" Mean %f\n", means[0],
                    (double)means[1] / means[0]);
}

#if 0
/// Trampoline helper to avoid moving Logger to misc.h
void start_logger(const std::string& fname) { Logger::start(fname); }
#endif

void start_logger(const char *fname)
{
  (void)fname;
}

// xorshift64star Pseudo-Random Number Generator
// This class is based on original code written and dedicated
// to the public domain by Sebastiano Vigna (2014).
// It has the following characteristics:
//
//  -  Outputs 64-bit numbers
//  -  Passes Dieharder and SmallCrush test batteries
//  -  Does not require warm-up, no zeroland to escape
//  -  Internal state is a single 64-bit integer
//  -  Period is 2^64 - 1
//  -  Speed: 1.60 ns/call (Core i7 @3.40GHz)
//
// For further analysis see
//   <http://vigna.di.unimi.it/ftp/papers/xorshift.pdf>

void prng_init(PRNG *rng, uint64_t seed)
{
  rng->s = seed;
}

uint64_t prng_rand(PRNG *rng)
{
  uint64_t s = rng->s;

  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  rng->s = s;

  return s * 2685821657736338717LL;
}

uint64_t prng_sparse_rand(PRNG *rng)
{
  uint64_t r1 = prng_rand(rng);
  uint64_t r2 = prng_rand(rng);
  uint64_t r3 = prng_rand(rng);
  return r1 & r2 & r3;
}

#ifdef __WIN32__
ssize_t getline(char **lineptr, size_t *n, FILE *stream)
{
  if (*n == 0)
    *lineptr = malloc(*n = 100);

  char c = 0;
  size_t i = 0;
  while ((c = getc(stream)) != EOF) {
    (*lineptr)[i++] = c;
    if (i == *n)
      *lineptr = realloc(*lineptr, *n += 100);
    if (c == '\n') break;
  }
  (*lineptr)[i] = 0;
  return i;
}
#endif

#ifdef __WIN32__
typedef SIZE_T (WINAPI *GLPM)(void);
size_t large_page_minimum;

int large_pages_supported(void)
{
  GLPM imp_GetLargePageMinimum =
             (GLPM)GetProcAddress(GetModuleHandle("kernel32.dll"),
                                  "GetLargePageMinimum");
  if (!imp_GetLargePageMinimum)
    return 0;

  if ((large_page_minimum = imp_GetLargePageMinimum()) == 0)
    return 0;

  LUID priv_luid;
  if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &priv_luid))
    return 0;

  HANDLE token;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &token))
    return 0;

  TOKEN_PRIVILEGES token_privs;
  token_privs.PrivilegeCount = 1;
  token_privs.Privileges[0].Luid = priv_luid;
  token_privs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
  if (!AdjustTokenPrivileges(token, FALSE, &token_privs, 0, NULL, NULL))
    return 0;

  return 1;
}
#endif

