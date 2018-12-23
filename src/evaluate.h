#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

typedef struct Pos Pos;

enum { Tempo = 28 };

Value evaluate(const Pos *pos);

#endif
