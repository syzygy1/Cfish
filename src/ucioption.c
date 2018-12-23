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
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "misc.h"
#include "numa.h"
#include "polybook.h"
#include "search.h"
#include "settings.h"
#include "tbprobe.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

// 'On change' actions, triggered by an option's value change
static void on_clear_hash(Option *opt)
{
  (void)opt;

  if (settings.ttSize)
    search_clear();
}

static void on_hash_size(Option *opt)
{
  delayedSettings.ttSize = opt->value;
}

static void on_numa(Option *opt)
{
#ifdef NUMA
  read_numa_nodes(opt->valString);
#else
  (void)opt;
#endif
}

static void on_threads(Option *opt)
{
  delayedSettings.numThreads = opt->value;
}

static void on_tb_path(Option *opt)
{
  TB_init(opt->valString);
}

static void on_large_pages(Option *opt)
{
  delayedSettings.largePages = opt->value;
}

static void on_book_file(Option *opt)
{
  pb_init(opt->valString);
}

static void on_best_book_move(Option *opt)
{
  pb_set_best_book_move(opt->value);
}

static void on_book_depth(Option *opt)
{
  pb_set_book_depth(opt->value);
}

#ifdef IS_64BIT
#ifdef BIG_TT
#define MAXHASHMB (1024 * 1024)
#else
#define MAXHASHMB 131072
#endif
#else
#define MAXHASHMB 2048
#endif

static Option optionsMap[] = {
  { "Contempt", OPT_TYPE_SPIN, 24, -100, 100, NULL, NULL, 0, NULL },
  { "Analysis Contempt", OPT_TYPE_COMBO, 0, 0, 0,
    "Off var Off var White var Black", NULL, 0, NULL },
  { "Threads", OPT_TYPE_SPIN, 1, 1, MAX_THREADS, NULL, on_threads, 0, NULL },
  { "Hash", OPT_TYPE_SPIN, 16, 1, MAXHASHMB, NULL, on_hash_size, 0, NULL },
  { "Clear Hash", OPT_TYPE_BUTTON, 0, 0, 0, NULL, on_clear_hash, 0, NULL },
  { "Ponder", OPT_TYPE_CHECK, 0, 0, 0, NULL, NULL, 0, NULL },
  { "MultiPV", OPT_TYPE_SPIN, 1, 1, 500, NULL, NULL, 0, NULL },
  { "Skill Level", OPT_TYPE_SPIN, 20, 0, 20, NULL, NULL, 0, NULL },
  { "Move Overhead", OPT_TYPE_SPIN, 30, 0, 5000, NULL, NULL, 0, NULL },
  { "Minimum Thinking Time", OPT_TYPE_SPIN, 20, 0, 5000, NULL, NULL, 0, NULL },
  { "Slow Mover", OPT_TYPE_SPIN, 84, 10, 1000, NULL, NULL, 0, NULL },
  { "nodestime", OPT_TYPE_SPIN, 0, 0, 10000, NULL, NULL, 0, NULL },
  { "UCI_AnalyseMode", OPT_TYPE_CHECK, 0, 0, 0, NULL, NULL, 0, NULL },
  { "UCI_Chess960", OPT_TYPE_CHECK, 0, 0, 0, NULL, NULL, 0, NULL },
  { "SyzygyPath", OPT_TYPE_STRING, 0, 0, 0, "<empty>", on_tb_path, 0, NULL },
  { "SyzygyProbeDepth", OPT_TYPE_SPIN, 1, 1, 100, NULL, NULL, 0, NULL },
  { "Syzygy50MoveRule", OPT_TYPE_CHECK, 1, 0, 0, NULL, NULL, 0, NULL },
  { "SyzygyProbeLimit", OPT_TYPE_SPIN, 7, 0, 7, NULL, NULL, 0, NULL },
  { "SyzygyUseDTM", OPT_TYPE_CHECK, 1, 0, 0, NULL, NULL, 0, NULL },
  { "BookFile", OPT_TYPE_STRING, 0, 0, 0, "<empty>", on_book_file, 0, NULL },
  { "BestBookMove", OPT_TYPE_CHECK, 1, 0, 0, NULL, on_best_book_move, 0, NULL },
  { "BookDepth", OPT_TYPE_SPIN, 255, 1, 255, NULL, on_book_depth, 0, NULL },
  { "LargePages", OPT_TYPE_CHECK, 1, 0, 0, NULL, on_large_pages, 0, NULL },
  { "NUMA", OPT_TYPE_STRING, 0, 0, 0, "all", on_numa, 0, NULL },
  { NULL }
};

// options_init() initializes the UCI options to their hard-coded default
// values.

void options_init()
{
  char *s;
  size_t len;

#ifdef NUMA
  // On a non-NUMA machine, disable the NUMA option to diminish confusion.
  if (!numaAvail)
    optionsMap[OPT_NUMA].type = OPT_TYPE_DISABLED;
#else
  optionsMap[OPT_NUMA].type = OPT_TYPE_DISABLED;
#endif
#ifdef _WIN32
  // Disable the LargePages option if the machine does not support it.
  if (!large_pages_supported())
    optionsMap[OPT_LARGE_PAGES].type = OPT_TYPE_DISABLED;
#endif
#if defined(__linux__) && !defined(MADV_HUGEPAGE)
  optionsMap[OPT_LARGE_PAGES].type = OPT_TYPE_DISABLED;
#endif
  optionsMap[OPT_SKILL_LEVEL].type = OPT_TYPE_DISABLED;
  if (sizeof(size_t) < 8) {
    optionsMap[OPT_SYZ_PROBE_LIMIT].def = 5;
    optionsMap[OPT_SYZ_PROBE_LIMIT].maxVal = 5;
  }
  for (Option *opt = optionsMap; opt->name != NULL; opt++) {
    if (opt->type == OPT_TYPE_DISABLED)
      continue;
    switch (opt->type) {
    case OPT_TYPE_CHECK:
    case OPT_TYPE_SPIN:
      opt->value = opt->def;
    case OPT_TYPE_BUTTON:
      break;
    case OPT_TYPE_STRING:
      opt->valString = malloc(strlen(opt->defString) + 1);
      strcpy(opt->valString, opt->defString);
      break;
    case OPT_TYPE_COMBO:
      s = strstr(opt->defString, " var");
      len = strlen(opt->defString) - strlen(s);
      opt->valString = malloc(len + 1);
      strncpy(opt->valString, opt->defString, len);
      opt->valString[len] = 0;
      for (s = opt->valString; *s; s++)
        *s = tolower(*s);
      break;
    }
    if (opt->onChange)
      opt->onChange(opt);
  }
}

void options_free(void)
{
  for (Option *opt = optionsMap; opt->name != NULL; opt++)
    if (opt->type == OPT_TYPE_STRING)
      free(opt->valString);
}

static const char *optTypeStr[] =
{
  "check", "spin", "button", "string", "combo"
};

// print_options() prints all options in the format required by the
// UCI protocol.

void print_options(void)
{
  for (Option *opt = optionsMap; opt->name != NULL; opt++) {
    if (opt->type == OPT_TYPE_DISABLED)
      continue;
    printf("option name %s type %s", opt->name, optTypeStr[opt->type]);
    switch (opt->type) {
    case OPT_TYPE_CHECK:
      printf(" default %s", opt->def ? "true" : "false");
      break;
    case OPT_TYPE_SPIN:
      printf(" default %d min %d max %d", opt->def, opt->minVal, opt->maxVal);
    case OPT_TYPE_BUTTON:
      break;
    case OPT_TYPE_STRING:
    case OPT_TYPE_COMBO:
      printf(" default %s", opt->defString);
      break;
    }
    printf("\n");
  }
  fflush(stdout);
}

int option_value(int optIdx)
{
  return optionsMap[optIdx].value;
}

const char *option_string_value(int optIdx)
{
  return optionsMap[optIdx].valString;
}

void option_set_value(int optIdx, int value)
{
  Option *opt = &optionsMap[optIdx];

  opt->value = value;
  if (opt->onChange)
    opt->onChange(opt);
}

int option_set_by_name(char *name, char *value)
{
  for (Option *opt = optionsMap; opt->name != NULL; opt++) {
    if (opt->type == OPT_TYPE_DISABLED)
      continue;
    if (strcasecmp(opt->name, name) == 0) {
      int val;
      switch (opt->type) {
      case OPT_TYPE_CHECK:
        if (strcmp(value, "true") == 0)
          opt->value = 1;
        else if (strcmp(value, "false") == 0)
          opt->value = 0;
        else
          return 1;
        break;
      case OPT_TYPE_SPIN:
        val = atoi(value);
        if (val < opt->minVal || val > opt->maxVal)
          return 1;
        opt->value = val;
      case OPT_TYPE_BUTTON:
        break;
      case OPT_TYPE_STRING:
        free(opt->valString);
        opt->valString = malloc(strlen(value) + 1);
        strcpy(opt->valString, value);
        break;
      case OPT_TYPE_COMBO:
        free(opt->valString);
        opt->valString = malloc(strlen(value) + 1);
        strcpy(opt->valString, value);
        for (char *s = opt->valString; *s; s++)
          *s = tolower(*s);
      }
      if (opt->onChange)
        opt->onChange(opt);
      return 1;
    }
  }

  return 0;
}
