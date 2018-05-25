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

#ifndef UCI_H
#define UCI_H

#include <string.h>

#include "types.h"

struct Option;
typedef struct Option Option;

typedef void (*OnChange)(Option *);

enum {
  OPT_TYPE_CHECK, OPT_TYPE_SPIN, OPT_TYPE_BUTTON, OPT_TYPE_STRING,
  OPT_TYPE_COMBO, OPT_TYPE_DISABLED
};

enum {
  OPT_CONTEMPT,
  OPT_ANALYSIS_CONTEMPT,
  OPT_THREADS,
  OPT_HASH,
  OPT_CLEAR_HASH,
  OPT_PONDER,
  OPT_MULTI_PV,
  OPT_SKILL_LEVEL,
  OPT_MOVE_OVERHEAD,
  OPT_MIN_THINK_TIME,
  OPT_SLOW_MOVER,
  OPT_NODES_TIME,
  OPT_ANALYSE_MODE,
  OPT_CHESS960,
  OPT_SYZ_PATH,
  OPT_SYZ_PROBE_DEPTH,
  OPT_SYZ_50_MOVE,
  OPT_SYZ_PROBE_LIMIT,
  OPT_SYZ_USE_DTM,
  OPT_BOOK_FILE,
  OPT_BOOK_BEST_MOVE,
  OPT_BOOK_DEPTH,
  OPT_LARGE_PAGES,
  OPT_NUMA
};

struct Option {
  char *name;
  int type;
  int def, minVal, maxVal;
  char *defString;
  OnChange onChange;
  int value;
  char *valString;
};

void options_init(void);
void options_free(void);
void print_options(void);
int option_value(int opt);
const char *option_string_value(int opt);
void option_set_value(int opt, int value);
int option_set_by_name(char *name, char *value);

void setoption(char *str);
void position(Pos *pos, char *str);

void uci_loop(int argc, char* argv[]);
char *uci_value(char *str, Value v);
char *uci_square(char *str, Square s);
char *uci_move(char *str, Move m, int chess960);
void print_pv(Pos *pos, Depth depth, Value alpha, Value beta);
Move uci_to_move(const Pos *pos, char *str);

#endif
