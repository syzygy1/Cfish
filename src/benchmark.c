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

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "misc.h"
#include "position.h"
#include "search.h"
#include "settings.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

static char *Defaults[] = {
  "setoption name UCI_Chess960 value false",
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
  "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
  "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves d4e6",
  "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves g2g4",
  "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
  "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
  "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
  "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
  "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
  "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
  "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
  "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
  "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
  "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
  "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
  "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
  "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
  "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
  "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
  "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
  "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
  "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
  "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
  "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
  "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
  "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
  "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
  "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",

  // 5-man positions
  "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",     // Kc2 - mate
  "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",      // Na2 - mate
  "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",    // draw

  // 6-man positions
  "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",   // Re5 - mate
  "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",    // Ka2 - mate
  "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",  // Nd2 - draw

  // 7-man positions
  "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124", // Draw

  // Mate and stalemate positions
  "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
  "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
  "8/8/8/8/8/6k1/6p1/6K1 w - -",
  "7k/7P/6K1/8/3B4/8/8/8 b - -",

  // Chess 960
  "setoption name UCI_Chess960 value true",
  "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1 moves g2g3 d7d5 d2d4 c8h3 c1g5 e8d6 g5e7 f7f6",
  "setoption name UCI_Chess960 value false"
};

// benchmark() runs a simple benchmark by letting Stockfish analyze a set
// of positions for a given limit each. There are five parameters: the
// transposition table size, the number of search threads that should
// be used, the limit value spent for each position (optional, default is
// depth 13), an optional file name where to look for positions in FEN
// format (defaults are the positions defined above) and the type of the
// limit value: depth (default), time in millisecs or number of nodes.

void benchmark(Pos *current, char *str)
{
  char *token;
  char **fens;
  int numFens;

  Limits = (struct LimitsType){ 0 };

  int ttSize      = (token = strtok(str , " ")) ? atoi(token)  : 16;
  int threads     = (token = strtok(NULL, " ")) ? atoi(token)  : 1;
  int64_t limit   = (token = strtok(NULL, " ")) ? atoll(token) : 13;
  char *fenFile   = (token = strtok(NULL, " ")) ? token        : NULL;
  char *limitType = (token = strtok(NULL, " ")) ? token        : "";

  delayedSettings.ttSize = ttSize;
  delayedSettings.numThreads = threads;
  process_delayed_settings();
  search_clear();

  if (strcmp(limitType, "time") == 0)
    Limits.movetime = limit; // movetime is in millisecs
  else if (strcmp(limitType, "nodes") == 0)
    Limits.nodes = limit;
  else if (strcmp(limitType, "mate") == 0)
    Limits.mate = limit;
  else
    Limits.depth = limit;

  if (!fenFile || strcmp(fenFile, "default") == 0) {
    fens = Defaults;
    numFens = sizeof(Defaults) / sizeof(char *);
  }
  else if (strcmp(fenFile, "current") == 0) {
    fens = malloc(sizeof(*fens));
    fens[0] = malloc(128);
    pos_fen(current, fens[0]);
    numFens = 1;
  }
  else {
    int maxFens = 100;
    numFens = 0;
    FILE *F = fopen(fenFile, "r");
    if (!F) {
      fprintf(stderr, "Unable to open file %s\n", fenFile);
      return;
    }
    fens = malloc(maxFens * sizeof(*fens));
    fens[0] = NULL;
    size_t length = 0;
    while (getline(&fens[numFens], &length, F) > 0) {
      numFens++;
      if (numFens == maxFens) {
        maxFens += 100;
        fens = realloc(fens, maxFens * sizeof(*fens));
      }
      fens[numFens] = NULL;
      length = 0;
    }
    fclose(F);
  }

  uint64_t nodes = 0;
  Pos pos;
  pos.stack = malloc(215 * sizeof(*pos.stack));
  pos.st = pos.stack + 5;
  pos.moveList = malloc(10000 * sizeof(*pos.moveList));
  TimePoint elapsed = now();

  int numOpts = 0;
  for (int i = 0; i < numFens; i++)
    if (strncmp(fens[i], "setoption ", 9) == 0)
      numOpts++;

  for (int i = 0, j = 0; i < numFens; i++) {
    char buf[128];

    if (strncmp(fens[i], "setoption ", 9) == 0) {
      strncpy(buf, fens[i] + 10, 127 - 10);
      buf[127] = 0;
      setoption(buf);
      continue;
    }

    strcpy(buf, "fen ");
    strncat(buf, fens[i], 127 - 4);
    buf[127] = 0;

    position(&pos, buf);

    fprintf(stderr, "\nPosition: %d/%d\n", ++j, numFens - numOpts);

    if (strcmp(limitType, "perft") == 0)
      nodes += perft(&pos, Limits.depth * ONE_PLY);
    else {
      Limits.startTime = now();
      start_thinking(&pos);
      thread_wait_until_sleeping(threads_main());
      nodes += threads_nodes_searched();
    }
  }

  elapsed = now() - elapsed + 1; // Ensure positivity to avoid a 'divide by zero'

  fprintf(stderr, "\n==========================="
                  "\nTotal time (ms) : %" PRIu64
                  "\nNodes searched  : %" PRIu64
                  "\nNodes/second    : %" PRIu64 "\n",
                  elapsed, nodes, 1000 * nodes / elapsed);

  if (fens != Defaults) {
    for (int i = 0; i < numFens; i++)
      free(fens[i]);
    free(fens);
  }
  free(pos.stack);
  free(pos.moveList);
}
