#ifndef TBPROBE_H
#define TBPROBE_H

#include "movegen.h"

extern int TB_MaxCardinality;

void TB_init(char *path);
void TB_free(void);
FAST int TB_probe_wdl(Pos *pos, int *success);
FAST int TB_probe_dtz(Pos *pos, int *success);
FAST int TB_root_probe(Pos *pos, ExtMove *rm, size_t *num_moves, Value *score);
FAST int TB_root_probe_wdl(Pos *pos, ExtMove *rm, size_t *num_moves, Value *score);
FAST ExtMove *TB_filter_root_moves(Pos *pos, ExtMove *begin, ExtMove *last);

#endif
