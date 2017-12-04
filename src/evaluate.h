#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

typedef struct Pos Pos;

#define Tempo ((Value)20)

extern Score Contempt;

Value evaluate(const Pos *pos);

#endif

