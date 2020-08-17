#ifndef NNUE_H
#define NNUE_H

#include <stdalign.h>
#include <stdint.h>

#include "types.h"

typedef struct {
  alignas(64) int16_t accumulation[2][256];
  Value score;
  bool computedAccumulation;
} Accumulator;

void nnue_init(void);
Value nnue_evaluate(const Position *pos);

#endif
