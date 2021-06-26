#ifndef NNUE_H
#define NNUE_H

#include <stdalign.h>
#include <stdint.h>

#include "types.h"

enum { ACC_EMPTY, ACC_COMPUTED, ACC_INIT };

typedef struct {
  alignas(64) int16_t accumulation[2][256];
  uint8_t state[2];
} Accumulator;

void nnue_init(void);
void nnue_free(void);
Value nnue_evaluate(const Position *pos);
void nnue_export_net(void);

#endif
