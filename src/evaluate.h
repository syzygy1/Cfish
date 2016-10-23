#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"

typedef struct Pos Pos;
typedef struct Stack Stack;

#define Tempo ((Value)20)

Value evaluate(const Pos *pos, const Stack *st);

#endif

