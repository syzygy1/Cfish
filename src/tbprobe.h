#ifndef TBPROBE_H
#define TBPROBE_H

#include "movegen.h"

extern int TB_MaxCardinality;

void TB_init(char *path);
void TB_free(void);
int TB_probe_wdl(Pos *pos, int *success);
int TB_probe_dtz(Pos *pos, int *success);
int TB_root_probe(Pos *pos, ExtMove *rm, size_t *num_moves, Value *score);
int TB_root_probe_wdl(Pos *pos, ExtMove *rm, size_t *num_moves, Value *score);
ExtMove *TB_filter_root_moves(Pos *pos, ExtMove *begin, ExtMove *last);

#endif
