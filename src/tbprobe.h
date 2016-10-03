#ifndef TBPROBE_H
#define TBPROBE_H

#include "movegen.h"

extern int TB_MaxCardinality;

void TB_init(char *path);
void TB_free(void);
int TB_probe_wdl(Pos *pos, int *success);
int TB_probe_dtz(Pos *pos, int *success);
int TB_root_probe(Pos *pos, Move *rm, size_t *num_moves, Value *score);
int TB_root_probe_wdl(Pos *pos, Move *rm, size_t *num_moves, Value *score);
Move *TB_filter_root_moves(Pos *pos, Move *begin, Move *last);

#endif
