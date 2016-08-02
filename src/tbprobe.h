#ifndef TBPROBE_H
#define TBPROBE_H

#include "search.h"

extern int TB_MaxCardinality;

void TB_init(char *path);
int TB_probe_wdl(Pos *pos, int *success);
int TB_probe_dtz(Pos *pos, int *success);
int TB_root_probe(Pos *pos, RootMoves *rootMoves, Value *score);
int TB_root_probe_wdl(Pos *pos, RootMoves *rootMoves, Value *score);
void TB_filter_root_moves(Pos *pos, RootMoves *rootMoves);

#endif
