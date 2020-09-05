#ifndef TBPROBE_H
#define TBPROBE_H

#include "movegen.h"

extern int TB_MaxCardinality;
extern int TB_MaxCardinalityDTM;

void TB_init(char *path);
void TB_free(void);
void TB_release(void);
int TB_probe_wdl(Position *pos, int *success);
int TB_probe_dtz(Position *pos, int *success);
Value TB_probe_dtm(Position *pos, int wdl, int *success);
bool TB_root_probe_wdl(Position *pos, RootMoves *rm);
bool TB_root_probe_dtz(Position *pos, RootMoves *rm);
bool TB_root_probe_dtm(Position *pos, RootMoves *rm);
void TB_expand_mate(Position *pos, RootMove *move);

#endif
