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

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#ifndef __WIN32__
#include <sys/mman.h>
#endif

#include "misc.h"
#include "numa.h"
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

  if (settings.tt_size)
    search_clear();
}

static void on_hash_size(Option *opt)
{
  delayed_settings.tt_size = opt->value;
}

static void on_logger(Option *opt)
{
  start_logger(opt->val_string);
}

static void on_numa(Option *opt)
{
#ifdef NUMA
  read_numa_nodes(opt->val_string);
#else
  (void)opt;
#endif
}

static void on_threads(Option *opt)
{
  delayed_settings.num_threads = opt->value;
}

static void on_tb_path(Option *opt)
{
  TB_init(opt->val_string);
}

static void on_largepages(Option *opt)
{
  delayed_settings.large_pages = opt->value;
}

#ifdef IS_64BIT
#define MAXHASHMB (1024 * 1024)
#else
#define MAXHASHMB 2048
#endif

static Option options_map[] = {
  { "Debug Log File", OPT_TYPE_STRING, 0, 0, 0, "<empty>", on_logger, 0, NULL },
  { "Contempt", OPT_TYPE_SPIN, 0, -100, 100, NULL, NULL, 0, NULL },
  { "Threads", OPT_TYPE_SPIN, 1, 1, 128, NULL, on_threads, 0, NULL },
  { "Hash", OPT_TYPE_SPIN, 16, 1, MAXHASHMB, NULL, on_hash_size, 0, NULL },
  { "Clear Hash", OPT_TYPE_BUTTON, 0, 0, 0, NULL, on_clear_hash, 0, NULL },
  { "Ponder", OPT_TYPE_CHECK, 0, 0, 0, NULL, NULL, 0, NULL },
  { "MultiPV", OPT_TYPE_SPIN, 1, 1, 500, NULL, NULL, 0, NULL },
  { "Repetition Fix", OPT_TYPE_CHECK, 0, 0, 0, NULL, NULL, 0, NULL },
  { "Skill Level", OPT_TYPE_SPIN, 20, 0, 20, NULL, NULL, 0, NULL },
  { "Move Overhead", OPT_TYPE_SPIN, 30, 0, 5000, NULL, NULL, 0, NULL },
  { "Minimum Thinking Time", OPT_TYPE_SPIN, 20, 0, 5000, NULL, NULL, 0, NULL },
  { "Slow Mover", OPT_TYPE_SPIN, 89, 10, 1000, NULL, NULL, 0, NULL },
  { "nodestime", OPT_TYPE_SPIN, 0, 0, 10000, NULL, NULL, 0, NULL },
  { "UCI_Chess960", OPT_TYPE_CHECK, 0, 0, 0, NULL, NULL, 0, NULL },
  { "SyzygyPath", OPT_TYPE_STRING, 0, 0, 0, "<empty>", on_tb_path, 0, NULL },
  { "SyzygyProbeDepth", OPT_TYPE_SPIN, 1, 1, 100, NULL, NULL, 0, NULL },
  { "Syzygy50MoveRule", OPT_TYPE_CHECK, 1, 0, 0, NULL, NULL, 0, NULL },
  { "SyzygyProbeLimit", OPT_TYPE_SPIN, 6, 0, 6, NULL, NULL, 0, NULL },
  { "LargePages", OPT_TYPE_CHECK, 1, 0, 0, NULL, on_largepages, 0, NULL },
  { "NUMA", OPT_TYPE_STRING, 0, 0, 0, "all", on_numa, 0, NULL },
  { NULL }
};


// options_init() initializes the UCI options to their hard-coded default
// values.

void options_init()
{
#ifdef NUMA
  // On a non-NUMA machine, disable the NUMA option to diminish confusion.
  if (!numa_avail)
    options_map[OPT_NUMA].type = OPT_TYPE_DISABLED;
#else
  options_map[OPT_NUMA].type = OPT_TYPE_DISABLED;
#endif
#ifdef __WIN32__
  // Disable the LargePages option if the machine does not support it.
  if (!large_pages_supported())
    options_map[OPT_LARGE_PAGES].type = OPT_TYPE_DISABLED;
#endif
#ifdef __linux__
#ifndef MADV_HUGEPAGE
  options_map[OPT_LARGE_PAGES].type = OPT_TYPE_DISABLED;
#endif
#endif
  // Disable Repetition Fix for now, since it has not been implemented yet.
  options_map[OPT_REP_FIX].type = OPT_TYPE_DISABLED;
  for (Option *opt = options_map; opt->name != NULL; opt++) {
    if (opt->type == OPT_TYPE_DISABLED)
      continue;
    switch (opt->type) {
    case OPT_TYPE_CHECK:
    case OPT_TYPE_SPIN:
      opt->value = opt->def;
    case OPT_TYPE_BUTTON:
      break;
    case OPT_TYPE_STRING:
      opt->val_string = malloc(strlen(opt->def_string) + 1);
      strcpy(opt->val_string, opt->def_string);
      break;
    }
    if (opt->on_change)
      opt->on_change(opt);
  }
}

void options_free(void)
{
  for (Option *opt = options_map; opt->name != NULL; opt++)
    if (opt->type == OPT_TYPE_STRING)
      free(opt->val_string);
}

static char *opt_type_str[] =
{
  "check", "spin", "button", "string"
};

// print_options() priints all options in the format required by the
// UCI protocol.

void print_options(void)
{
  for (Option *opt = options_map; opt->name != NULL; opt++) {
    if (opt->type == OPT_TYPE_DISABLED)
      continue;
    printf("option name %s type %s", opt->name, opt_type_str[opt->type]);
    switch (opt->type) {
    case OPT_TYPE_CHECK:
      printf(" default %s", opt->value ? "true" : "false");
      break;
    case OPT_TYPE_SPIN:
      printf(" default %d min %d max %d", opt->def, opt->min_val, opt->max_val);
    case OPT_TYPE_BUTTON:
      break;
    case OPT_TYPE_STRING:
      printf(" default %s", opt->def_string);
      break;
    }
    printf("\n");
  }
  fflush(stdout);
}

int option_value(int opt_idx)
{
  return options_map[opt_idx].value;
}

char *option_string_value(int opt_idx)
{
  return options_map[opt_idx].val_string;
}

void option_set_value(int opt_idx, int value)
{
  Option *opt = &options_map[opt_idx];

  opt->value = value;
  if (opt->on_change)
    opt->on_change(opt);
}

int option_set_by_name(char *name, char *value)
{
  for (Option *opt = options_map; opt->name != NULL; opt++) {
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
        if (val < opt->min_val || val > opt->max_val)
          return 1;
        opt->value = val;
      case OPT_TYPE_BUTTON:
        break;
      case OPT_TYPE_STRING:
        free(opt->val_string);
        opt->val_string = malloc(strlen(value) + 1);
        strcpy(opt->val_string, value);
        break;
      }
      if (opt->on_change)
        opt->on_change(opt);
      return 1;
    }
  }

  return 0;
}

