#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

typedef struct Pos Pos;

enum { Tempo = 20 };

extern _Atomic Score Contempt;

Value evaluate(const Pos *pos);

#endif
