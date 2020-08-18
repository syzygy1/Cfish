#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

enum { Tempo = 28 };

#ifdef NNUE
enum { EVAL_HYBRID, EVAL_PURE, EVAL_CLASSICAL };
extern int useNNUE;
#endif

Value evaluate(const Position *pos);

#endif
