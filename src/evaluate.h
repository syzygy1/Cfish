#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

typedef struct Pos Pos;

#define Tempo ((Value)20)

void trace(Pos *pos);

//template<bool DoTrace = false>
Value evaluate(Pos *pos);

#endif

