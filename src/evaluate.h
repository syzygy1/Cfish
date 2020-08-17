#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

enum { Tempo = 28 };

#ifdef NNUE
extern bool pureNNUE;
#endif

Value evaluate(const Position *pos);

#endif
