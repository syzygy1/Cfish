#ifndef TBPROBE_H
#define TBPROBE_H

#include "movegen.h"

extern int TB_MaxCardinality;

void TB_init(char *path);
void TB_free(void);
int TB_probe_wdl(Pos *pos, Stack *st, int *success);
int TB_probe_dtz(Pos *pos, Stack *st, int *success);
int TB_root_probe(Pos *pos, Stack *st, ExtMove *rm, size_t num_moves);
int TB_root_probe_wdl(Pos *pos, Stack *st, ExtMove *rm, size_t num_moves);

#endif

