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

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "settings.h"
#include "thread.h"
#include "timeman.h"
#include "uci.h"

extern void benchmark(Pos *pos, char *str);

// FEN string of the initial position, normal chess
static const char StartFEN[] =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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

  if (strncmp(str, "fen", 3) == 0) {
    strncpy(fen, str + 4, 127);
    fen[127] = 0;
  } else if (strncmp(str, "startpos", 8) == 0)
    strcpy(fen, StartFEN);
  else
    return;

  pos->st = pos->stack + 100; // Start of circular buffer of 100 slots.
  pos_set(pos, fen, option_value(OPT_CHESS960));

  // Parse move list (if any).
  if (moves) {
    int ply = 0;

    for (moves = strtok(moves, " \t"); moves; moves = strtok(NULL, " \t")) {
      Move m = uci_to_move(pos, moves);
      if (!m) break;
      do_move(pos, m, gives_check(pos, pos->st, m));
      pos->gamePly++;
      // Roll over if we reach 100 plies.
      if (++ply == 100) {
        memcpy(pos->st - 100, pos->st, StateSize);
        pos->st -= 100;
        pos_set_check_info(pos);
        ply -= 100;
      }
    }

    // Make sure that is_draw() never tries to look back more than 99 ply.
    // This is enough, since 100 ply history means draw by 50-move rule.
    if (pos->st->pliesFromNull > 99)
      pos->st->pliesFromNull = 99;

    // Now move some of the game history at the end of the circular buffer
    // in front of that buffer.
    int k = (pos->st - (pos->stack + 100)) - max(5, pos->st->pliesFromNull);
    for (; k < 0; k++)
      memcpy(pos->stack + 100 + k, pos->stack + 200 + k, StateSize);
  }

  pos->rootKeyFlip = pos->st->key;
  (pos->st-1)->endMoves = pos->moveList;

  // Clear history position keys that have not yet repeated. This ensures
  // that is_draw() does not flag as a draw the first repetition of a
  // position coming before the root position. In addition, we set
  // pos->hasRepeated to indicate whether a position has repeated since
  // the last irreversible move.
  for (int k = 0; k <= pos->st->pliesFromNull; k++) {
    int l;
    for (l = k + 4; l <= pos->st->pliesFromNull; l += 2)
      if ((pos->st - k)->key == (pos->st - l)->key)
        break;
    if (l <= pos->st->pliesFromNull)
      pos->hasRepeated = 1;
    else
      (pos->st - k)->key = 0;
  }
  pos->rootKeyFlip ^= pos->st->key;
  pos->st->key ^= pos->rootKeyFlip;
}


// setoption() is called when the engine receives the "setoption" UCI
// command. The function updates the UCI option ("name") to the given
// value ("value").

void setoption(char *str)
{
  char *name, *value;

  name = strstr(str, "name");
  if (!name) {
    name = "";
    goto error;
  }

  name += 4;
  while (isblank(*name))
    name++;

  value = strstr(name, "value");
  if (value) {
    char *p = value - 1;
    while (isblank(*p))
      p--;
    p[1] = 0;
    value += 5;
    while (isblank(*value))
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

static void go(Pos *pos, char *str)
{
  char *token;

  process_delayed_settings();

  Limits = (struct LimitsType){ 0 };
  Limits.startTime = now(); // As early as possible!

  for (token = strtok(str, " \t"); token; token = strtok(NULL, " \t")) {
    if (strcmp(token, "searchmoves") == 0)
      while ((token = strtok(NULL, " \t")))
        Limits.searchmoves[Limits.numSearchmoves++] = uci_to_move(pos, token);
    else if (strcmp(token, "wtime") == 0)
      Limits.time[WHITE] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "btime") == 0)
      Limits.time[BLACK] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "winc") == 0)
      Limits.inc[WHITE] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "binc") == 0)
      Limits.inc[BLACK] = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "movestogo") == 0)
      Limits.movestogo = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "depth") == 0)
      Limits.depth = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "nodes") == 0)
      Limits.nodes = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "movetime") == 0)
      Limits.movetime = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "mate") == 0)
      Limits.mate = atoi(strtok(NULL, " \t"));
    else if (strcmp(token, "infinite") == 0)
      Limits.infinite = 1;
    else if (strcmp(token, "ponder") == 0)
      Limits.ponder = 1;
    else if (strcmp(token, "perft") == 0) {
      char str_buf[64];
      sprintf(str_buf, "%d %d %d current perft", option_value(OPT_HASH),
                    option_value(OPT_THREADS), atoi(strtok(NULL, " \t")));
      benchmark(pos, str_buf);
      return;
    }
  }

  start_thinking(pos);
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
  char str_buf[64];
  char *token;

  LOCK_INIT(Signals.lock);

  // Signals.searching is only read and set by the UI thread.
  // The UI thread uses it to know whether it must still call
  // thread_wait_until_sleeping() on the main search thread.
  // (This is important for our native Windows threading implementation.)
  Signals.searching = 0;

  // Signals.sleeping is set by the main search thread if it has run
  // out of work but must wait for a "stop" or "ponderhit" command from
  // the GUI to arrive before being allowed to output "bestmove". The main
  // thread will then go to sleep and has to be waken up by the UI thread.
  // This variable must be accessed only after acquiring Signals.lock.
  Signals.sleeping = 0;

  // Allocate 215 Stack slots.
  // Slots 100-200 form a circular buffer to be filled with game moves.
  // Slots 0-99 make room for prepending the part of game history relevant
  // for repetition detection.
  // Slots 201-214 may be used by TB root probing.
  pos.stack = malloc(215 * sizeof(Stack));
  pos.moveList = malloc(1000 * sizeof(ExtMove));
  pos.st = pos.stack + 100;
  pos.st[-1].endMoves = pos.moveList;

  size_t buf_size = 1;
  for (int i = 1; i < argc; i++)
    buf_size += strlen(argv[i]) + 1;

  if (buf_size < 1024) buf_size = 1024;

  char *cmd = malloc(buf_size);

  cmd[0] = 0;
  for (int i = 1; i < argc; i++) {
    strcat(cmd, argv[i]);
    strcat(cmd, " ");
  }

  strcpy(fen, StartFEN);
  pos_set(&pos, fen, 0);
  pos.rootKeyFlip = pos.st->key;

  do {
    if (argc == 1 && !getline(&cmd, &buf_size, stdin))
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
    if (strcmp(token, "quit") == 0 || strcmp(token, "stop") == 0) {
      if (Signals.searching) {
        Signals.stop = 1;
        LOCK(Signals.lock);
        if (Signals.sleeping)
          thread_wake_up(threads_main(), THREAD_RESUME);
        Signals.sleeping = 0;
        UNLOCK(Signals.lock);
      }
    }
    else if (strcmp(token, "ponderhit") == 0) {
      Limits.ponder = 0; // Switch to normal search
      if (Signals.stopOnPonderhit)
        Signals.stop = 1;
      LOCK(Signals.lock);
      if (Signals.sleeping) {
        Signals.stop = 1;
        thread_wake_up(threads_main(), THREAD_RESUME);
        Signals.sleeping = 0;
      }
      UNLOCK(Signals.lock);
    }
    else if (strcmp(token, "uci") == 0) {
      flockfile(stdout);
      printf("id name ");
      print_engine_info(1);
      printf("\n");
      print_options();
      printf("uciok\n");
      fflush(stdout);
      funlockfile(stdout);
    }
    else if (strcmp(token, "ucinewgame") == 0) {
      process_delayed_settings();
      search_clear();
    } else if (strcmp(token, "isready") == 0) {
      process_delayed_settings();
      printf("readyok\n");
      fflush(stdout);
    }
    else if (strcmp(token, "go") == 0)        go(&pos, str);
    else if (strcmp(token, "position") == 0)  position(&pos, str);
    else if (strcmp(token, "setoption") == 0) setoption(str);

    // Additional custom non-UCI commands, useful for debugging
    else if (strcmp(token, "bench") == 0)     benchmark(&pos, str);
    else if (strcmp(token, "d") == 0)         print_pos(&pos);
    else if (strcmp(token, "perft") == 0) {
      sprintf(str_buf, "%d %d %d current perft", option_value(OPT_HASH),
                    option_value(OPT_THREADS), atoi(str));
      benchmark(&pos, str_buf);
    }
    else {
      printf("Unknown command: %s %s\n", token, str);
      fflush(stdout);
    }
  } while (argc == 1 && strcmp(token, "quit") != 0);

  if (Signals.searching)
    thread_wait_until_sleeping(threads_main());

  free(cmd);
  free(pos.stack);
  free(pos.moveList);

  LOCK_DESTROY(Signals.lock);
}


// uci_value() converts a Value to a string suitable for use with the UCI
// protocol specification:
//
// cp <x>    The score from the engine's point of view in centipawns.
// mate <y>  Mate in y moves, not plies. If the engine is getting mated
//           use negative values for y.

char *uci_value(char *str, Value v)
{
  if (abs(v) < VALUE_MATE - MAX_MATE_PLY)
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

Move uci_to_move(const Pos *pos, char *str)
{
  if (strlen(str) == 5) // Junior could send promotion piece in uppercase
    str[4] = tolower(str[4]);

  ExtMove list[MAX_MOVES];
  ExtMove *last = generate_legal(pos, list);

  char buf[16];

  for (ExtMove *m = list; m < last; m++)
    if (strcmp(str, uci_move(buf, m->move, pos->chess960)) == 0)
      return m->move;

  return 0;
}
