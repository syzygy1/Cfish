#ifndef TBPROBE_H
#define TBPROBE_H

#include "movegen.h"

extern int TB_MaxCardinality;
extern int TB_MaxCardinalityDTM;

void TB_init(char *path);
void TB_free(void);
int TB_probe_wdl(Pos *pos, int *success);
int TB_probe_dtz(Pos *pos, int *success);
Value TB_probe_dtm(Pos *pos, int wdl, int *success);
int TB_root_probe_wdl(Pos *pos, RootMoves *rm);
int TB_root_probe_dtz(Pos *pos, RootMoves *rm);
int TB_root_probe_dtm(Pos *pos, RootMoves *rm);
void TB_expand_mate(Pos *pos, RootMove *move);

#endif
