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
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "evaluate.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "timeman.h"
#include "uci.h"

extern void benchmark(Pos *pos, char *str);

// FEN string of the initial position, normal chess
const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// A circular buffer to keep track of the position states for the moves
// received with the position command. These are used for draw by repetition
// detection. It is OK to wrap them around, since the 50-move rule ensures
// we don't need to look back more than 100 ply.

static State state[128];
static int state_idx;

// position() is called when the engine receives the "position" UCI
// command. The function sets up the position described in the given FEN
// string ("fen") or the starting position ("startpos") and then makes
// the moves given in the following move list ("moves").

void position(Pos *pos, char *str)
{
  char fen[128];
  char *moves;

  moves = strstr(str, "moves");
  if (moves) {
    if (moves > str) moves[-1] = 0;
    moves += 5;
  }

  if (strncmp(str, "fen ", 4) == 0) {
    strncpy(fen, str + 4, 127);
    fen[127] = 0;
  } else if (strncmp(str, "startpos", 8) == 0)
    strcpy(fen, StartFEN);
  else
    return;

  pos_set(pos, fen, option_value(OPT_CHESS960), &state[0], threads_main());
  state_idx = 1;

  // Parse move list (if any)
  if (moves)
    for (moves = strtok(moves, " \t"); moves; moves = strtok(NULL, " \t")) {
      Move m = uci_to_move(pos, moves);
      if (m == MOVE_NONE) break;
      CheckInfo ci;
      checkinfo_init(&ci, pos);
      do_move(pos, m, &state[state_idx], gives_check(pos, m, &ci));
      state_idx = (state_idx + 1) & 127;
    }
}


// setoption() is called when the engine receives the "setoption" UCI
// command. The function updates the UCI option ("name") to the given
// value ("value").

void setoption(char *str)
{
  char *name, *value;

  name = strstr(str, "name ");
  if (!name) {
    name = "";
    goto error;
  }

  name += 5;
  while (isspace(*name))
    name++;

  value = strstr(name, " value ");
  if (value) {
    char *p = value - 1;
    while (isspace(*p))
      p--;
    p[1] = 0;
    value += 7;
    while (isspace(*value))
      value++;
  }
  if (!value || strlen(value) == 0)
    value = "<empty>";

  if (option_set_by_name(name, value))
    return;

error:
  fprintf(stderr, "No such option: %s\n", name);
}


// go() is called when engine receives the "go" UCI command. The function sets
// the thinking time and other parameters from the input string, then starts
// the search.


void go(Pos *pos, char *str)
{
  LimitsType limits;
  char *token;

  limits.startTime = now(); // As early as possible!

  limits.time[0] = limits.time[1] = limits.inc[0] = limits.inc[1] = 0;
  limits.npmsec = limits.movestogo = limits.depth = limits.movetime = 0;
  limits.mate = limits.infinite = limits.ponder = limits.num_searchmoves = 0;
  limits.nodes = 0;

  for (token = strtok(str, " \t"); token; token = strtok(NULL, " \t")) {
    if (strcmp(token, "searchmoves") == 0)
      while ((token = strtok(NULL, " \t")))
        limits.searchmoves[limits.num_searchmoves++] = uci_to_move(pos, token);
    else if (strcmp(token, "wtime") == 0)
      limits.time[WHITE] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "btime") == 0)
      limits.time[BLACK] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "winc") == 0)
      limits.inc[WHITE] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "binc") == 0)
      limits.inc[BLACK] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "movestogo") == 0)
      limits.movestogo = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "depth") == 0)
      limits.depth = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "nodes") == 0)
      limits.nodes = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "movetime") == 0)
      limits.movetime = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "mate") == 0)
      limits.mate = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "infinite") == 0)
      limits.infinite = 1;
    else if (strcmp(token, "ponder") == 0)
      limits.ponder = 1;
  }

  threads_start_thinking(pos, &state[state_idx], &limits);
}


// uci_loop() waits for a command from stdin, parses it and calls the
// appropriate function. Also intercepts EOF from stdin to ensure
// gracefully exiting if the GUI dies unexpectedly. When called with some
// command line arguments, e.g. to run 'bench', once the command is
// executed the function returns immediately. In addition to the UCI ones,
// also some additional debug commands are supported.

void uci_loop(int argc, char **argv)
{
  Pos pos;
  char fen[strlen(StartFEN) + 1];
  char *token;

  setbuf(stdout, NULL);

  size_t buf_size = 1;
  for (int i = 1; i < argc; i++)
    buf_size += strlen(argv[i]) + 1;

  if (buf_size < 80) buf_size = 80;

  char *cmd = malloc(buf_size);

  cmd[0] = 0;
  for (int i = 1; i < argc; i++) {
    strcat(cmd, argv[i]);
    strcat(cmd, " ");
  }

  strcpy(fen, StartFEN);
  state_idx = 0;
  pos_set(&pos, fen, 0, &state[state_idx++], threads_main());

  do {
    if (argc == 1 && !getline(&cmd, &buf_size, stdin))
//    if (argc == 1 && !getdelim(&cmd, &buf_size, 0, stdin))
      strcpy(cmd, "quit");

    if (cmd[strlen(cmd) - 1] == '\n')
      cmd[strlen(cmd) - 1] = 0;

    token = cmd;
    while (isblank(*token))
      token++;

    char *str = token;
    while (*str && !isblank(*str))
      str++;

    if (*str) {
      *str++ = 0;
      while (isblank(*str))
        str++;
    }

    // The GUI sends 'ponderhit' to tell us to ponder on the same move the
    // opponent has played. In case Signals.stopOnPonderhit is set we are
    // waiting for 'ponderhit' to stop the search (for instance because we
    // already ran out of time), otherwise we should continue searching but
    // switching from pondering to normal search.
    if (    strcmp(token, "quit") == 0
        ||  strcmp(token, "stop") == 0
        || (strcmp(token, "ponderhit") == 0 && Signals.stopOnPonderhit)) {

      Signals.stop = 1;
      thread_start_searching(threads_main(), 1); // Could be sleeping
    }
    else if (strcmp(token, "ponderhit") == 0)
      Limits.ponder = 0; // Switch to normal search
    else if (strcmp(token, "uci") == 0) {
      printf("id name ");
      print_engine_info(1);
      printf("\n");
      print_options();
      printf("uciok\n");
    }
    else if (strcmp(token, "ucinewgame") == 0) {
      search_clear();
      Time.availableNodes = 0;
    }
    else if (strcmp(token, "isready") == 0)   printf("readyok\n");
    else if (strcmp(token, "go") == 0)        go(&pos, str);
    else if (strcmp(token, "position") == 0)  position(&pos, str);
    else if (strcmp(token, "setoption") == 0) setoption(str);

    // Additional custom non-UCI commands, useful for debugging
//    else if (strcmp(token, "flip") == 0)      pos_flip(&pos);
    else if (strcmp(token, "bench") == 0)     benchmark(&pos, str);
    else if (strcmp(token, "d") == 0)         print_pos(&pos);
//    else if (strcmp(token, "eval") == 0)      eval_trace(stdout, &pos);
    else if (strcmp(token, "perft") == 0) {
      char str2[64];
      sprintf(str2, "%d %d %d current perft", option_value(OPT_HASH),
                    option_value(OPT_THREADS), atoi(str));
      benchmark(&pos, str2);
    }
    else
      printf("Unknown command: %s %s\n", token, str);
  } while (argc == 1 && strcmp(token, "quit") != 0);

  free(cmd);

  thread_wait_for_search_finished(threads_main());
}


// uci_value() converts a Value to a string suitable for use with the UCI
// protocol specification:
//
// cp <x>    The score from the engine's point of view in centipawns.
// mate <y>  Mate in y moves, not plies. If the engine is getting mated
//           use negative values for y.

char *uci_value(char *str, Value v)
{
  if (abs(v) < VALUE_MATE - MAX_PLY)
    sprintf(str, "cp %d", v * 100 / PawnValueEg);
  else
    sprintf(str, "mate %d",
                 (v > 0 ? VALUE_MATE - v + 1 : -VALUE_MATE - v) / 2);

  return str;
}


// uci_square() converts a Square to a string in algebraic notation
// (g1, a7, etc.)

char *uci_square(char *str, Square s)
{
  str[0] = 'a' + file_of(s);
  str[1] = '1' + rank_of(s);
  str[2] = 0;

  return str;
}


// uci_move() converts a Move to a string in coordinate notation (g1f3,
// a7a8q). The only special case is castling, where we print in the e1g1
// notation in normal chess mode, and in e1h1 notation in chess960 mode.
// Internally all castling moves are always encoded as 'king captures rook'.

char *uci_move(char *str, Move m, int chess960)
{
  char buf1[8], buf2[8];
  Square from = from_sq(m);
  Square to = to_sq(m);

  if (m == 0)
    return "(none)";

  if (m == MOVE_NULL)
    return "0000";

  if (type_of_m(m) == CASTLING && !chess960)
    to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

  strcat(strcpy(str, uci_square(buf1, from)), uci_square(buf2, to));

  if (type_of_m(m) == PROMOTION) {
    str[strlen(str) + 1] = 0;
    str[strlen(str)] = " pnbrqk"[promotion_type(m)];
  }

  return str;
}


// uci_to_move() converts a string representing a move in coordinate
// notation (g1f3, a7a8q) to the corresponding legal Move, if any.

Move uci_to_move(Pos *pos, char *str)
{
  if (strlen(str) == 5) // Junior could send promotion piece in uppercase
    str[4] = tolower(str[4]);

  ExtMove list[MAX_MOVES];
  ExtMove *last = generate_legal(pos, list);

  char buf[16];

  for (ExtMove *m = list; m < last; m++)
    if (strcmp(str, uci_move(buf, m->move, pos->chess960)) == 0)
      return m->move;

  return MOVE_NONE;
}

