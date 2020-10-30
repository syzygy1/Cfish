#include <assert.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#if defined(USE_AVX2)
#include <immintrin.h>

#elif defined(USE_SSE41)
#include <smmintrin.h>

#elif defined(USE_SSSE3)
#include <tmmintrin.h>

#elif defined(USE_SSE2)
#include <emmintrin.h>

#elif defined(USE_SSE)
#include <xmmintrin.h>

#elif defined(USE_MMX)
#include <mmintrin.h>

#elif defined(USE_NEON)
#include <arm_neon.h>
#endif

#include "evaluate.h"
#include "misc.h"
#include "nnue.h"
#include "position.h"
#include "settings.h"
#include "uci.h"

#ifndef NNUE_SPARSE
#define NNUE_REGULAR
#endif

#ifdef NNUE_EMBEDDED
#include "incbin.h"
INCBIN(Network, DefaultEvalFile);
#endif

// Old gcc on Windows is unable to provide a 32-byte aligned stack.
// We need to hack around this when using AVX2 and AVX512.
#if     defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32) \
    && !defined(__clang__) && !defined(__INTEL_COMPILER) \
    &&  defined(USE_AVX2)
#define ALIGNMENT_HACK
#endif

enum {
  PS_W_PAWN   =  1,
  PS_B_PAWN   =  1 * 64 + 1,
  PS_W_KNIGHT =  2 * 64 + 1,
  PS_B_KNIGHT =  3 * 64 + 1,
  PS_W_BISHOP =  4 * 64 + 1,
  PS_B_BISHOP =  5 * 64 + 1,
  PS_W_ROOK   =  6 * 64 + 1,
  PS_B_ROOK   =  7 * 64 + 1,
  PS_W_QUEEN  =  8 * 64 + 1,
  PS_B_QUEEN  =  9 * 64 + 1,
  PS_END      = 10 * 64 + 1
};

uint32_t PieceToIndex[2][16] = {
  { 0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, 0, 0,
    0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, 0, 0 },
  { 0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, 0, 0,
    0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, 0, 0 }
};

// Version of the evaluation file
static const uint32_t NnueVersion = 0x7AF32F16u;

// Constants used in evaluation value calculation
enum {
  FV_SCALE = 16,
  SHIFT = 6
};

enum {
  kHalfDimensions = 256,
  FtInDims = 64 * PS_END, // 64 * 641
};

// USE_MMX generates _mm_empty() instructions, so undefine if not needed
#if defined(USE_SSE2)
#undef USE_MMX
#endif

static_assert(kHalfDimensions % 256 == 0, "kHalfDimensions should be a multiple of 256");

#define VECTOR

#ifdef USE_AVX512
#define SIMD_WIDTH 512
typedef __m512i vec8_t, vec16_t;
typedef __mmask64 mask_t;
#define vec_add_16(a,b) _mm512_add_epi16(a,b)
#define vec_sub_16(a,b) _mm512_sub_epi16(a,b)
#define vec_packs(a,b) _mm512_packs_epi16(a,b)
#define vec_mask_pos(a) _mm512_cmpgt_epi8_mask(a,_mm512_setzero_si512())
#define NUM_REGS 8 // only 8 are needed

#elif USE_AVX2
#define SIMD_WIDTH 256
typedef __m256i vec8_t, vec16_t;
typedef uint32_t mask_t;
#define vec_add_16(a,b) _mm256_add_epi16(a,b)
#define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
#define vec_packs(a,b) _mm256_packs_epi16(a,b)
#define vec_mask_pos(a) _mm256_movemask_epi8(_mm256_cmpgt_epi8(a,_mm256_setzero_si256()))
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif

#elif USE_SSE2
#define SIMD_WIDTH 128
typedef __m128i vec8_t, vec16_t;
typedef uint16_t mask_t;
#define vec_add_16(a,b) _mm_add_epi16(a,b)
#define vec_sub_16(a,b) _mm_sub_epi16(a,b)
#define vec_packs(a,b) _mm_packs_epi16(a,b)
#define vec_mask_pos(a) _mm_movemask_epi8(_mm_cmpgt_epi8(a,_mm_setzero_si128()))
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif

#elif USE_MMX
#define SIMD_WIDTH 64
typedef __m64 vec8_t, vec16_t;
typedef uint8_t mask_t;
#define vec_add_16(a,b) _mm_add_pi16(a,b)
#define vec_sub_16(a,b) _mm_sub_pi16(a,b)
#define vec_packs(a,b) _mm_packs_pi16(a,b)
#define vec_mask_pos(a) _mm_movemask_pi8(_mm_cmpgt_pi8(a,_mm_setzero_si64()))
#define NUM_REGS 8

#elif USE_NEON
#define SIMD_WIDTH 128
typedef int8x16_t vec8_t;
typedef int16x8_t vec16_t;
typedef uint16_t mask_t;
#define vec_add_16(a,b) vaddq_s16(a,b)
#define vec_sub_16(a,b) vsubq_s16(a,b)
#define vec_packs(a,b) vcombine_s8(vqmovn_s16(a),vqmovn_s16(b))
#define vec_mask_pos(a) neon_movemask(vcgtq_s8(a,vdupq_n_s8(0)))
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif

#else
#undef VECTOR
#define SIMD_WIDTH 16 // dummy
typedef uint8_t mask_t; // dummy

#endif

typedef struct {
  unsigned size;
  unsigned values[30];
} IndexList;

INLINE Square orient(Color c, Square s)
{
  return s ^ (c == WHITE ? 0x00 : 0x3f);
}

INLINE unsigned make_index(Color c, Square s, Piece pc, Square ksq)
{
  return orient(c, s) + PieceToIndex[c][pc] + PS_END * ksq;
}

static void append_active_indices(const Position *pos, const Color c,
    IndexList *active)
{
  Square ksq = orient(c, square_of(c, KING));
  Bitboard bb = pieces() & ~pieces_p(KING);
  while (bb) {
    Square s = pop_lsb(&bb);
    active->values[active->size++] = make_index(c, s, piece_on(s), ksq);
  }
}

static void append_changed_indices(const Position *pos, const Color c,
    const DirtyPiece *dp, IndexList *removed, IndexList *added)
{
  Square ksq = orient(c, square_of(c, KING));
  for (int i = 0; i < dp->dirtyNum; i++) {
    Piece pc = dp->pc[i];
    if (type_of_p(pc) == KING) continue;
    if (dp->from[i] != SQ_NONE)
      removed->values[removed->size++] = make_index(c, dp->from[i], pc, ksq);
    if (dp->to[i] != SQ_NONE)
      added->values[added->size++] = make_index(c, dp->to[i], pc, ksq);
  }
}

// Input feature converter
static int16_t *ft_biases; // [kHalfDimensions]
static int16_t *ft_weights; // [kHalfDimenions * FtInDims]
static alloc_t ft_alloc;

#ifdef VECTOR
#define TILE_HEIGHT (NUM_REGS * SIMD_WIDTH / 16)
#endif

// Calculate cumulative value using difference calculation if possible
INLINE void update_accumulator(const Position *pos, const Color c)
{
#ifdef VECTOR
  vec16_t acc[NUM_REGS];
#endif

  Stack *st = pos->st;
  int gain = popcount(pieces()) - 2;
  while (st->accumulator.state[c] == ACC_EMPTY) {
    DirtyPiece *dp = &st->dirtyPiece;
    if (   dp->pc[0] == make_piece(c, KING)
        || (gain -= dp->dirtyNum + 1) < 0)
      break;
    st--;
  }

  if (st->accumulator.state[c] == ACC_COMPUTED) {
    if (st == pos->st)
      return;

    IndexList added[2], removed[2];
    added[0].size = added[1].size = removed[0].size = removed[1].size = 0;
    append_changed_indices(pos, c, &(st+1)->dirtyPiece, &removed[0], &added[0]);
    for (Stack *st2 = st + 2; st2 <= pos->st; st2++)
      append_changed_indices(pos, c, &st2->dirtyPiece, &removed[1], &added[1]);

    (st+1)->accumulator.state[c] = ACC_COMPUTED;
    pos->st->accumulator.state[c] = ACC_COMPUTED;

    Stack *stack[3] = { st + 1, st + 1 == pos->st ? NULL : pos->st, NULL };
#ifdef VECTOR
    for (unsigned i = 0; i < kHalfDimensions / TILE_HEIGHT; i++) {
      vec16_t *accTile = (vec16_t *)&st->accumulator.accumulation[c][i * TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_REGS; j++)
        acc[j] = accTile[j];
      for (unsigned l = 0; stack[l]; l++) {
        // Difference calculation for the deactivated features
        for (unsigned k = 0; k < removed[l].size; k++) {
          unsigned index = removed[l].values[k];
          const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
          vec16_t *column = (vec16_t *)&ft_weights[offset];
          for (unsigned j = 0; j < NUM_REGS; j++)
            acc[j] = vec_sub_16(acc[j], column[j]);
        }

        // Difference calculation for the activated features
        for (unsigned k = 0; k < added[l].size; k++) {
          unsigned index = added[l].values[k];
          const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
          vec16_t *column = (vec16_t *)&ft_weights[offset];
          for (unsigned j = 0; j < NUM_REGS; j++)
            acc[j] = vec_add_16(acc[j], column[j]);
        }

        accTile = (vec16_t *)&stack[l]->accumulator.accumulation[c][i * TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_REGS; j++)
          accTile[j] = acc[j];
      }
    }
#else
    for (unsigned l = 0; stack[l]; l++) {
      memcpy(&stack[l]->accumulator.accumulation[c],
          &st->accumulator.accumulation[c], kHalfDimensions * sizeof(int16_t));
      st = stack[l];

      // Difference calculation for the deactivated features
      for (unsigned k = 0; k < removed[l].size; k++) {
        unsigned index = removed[l].values[k];
        const unsigned offset = kHalfDimensions * index;

        for (unsigned j = 0; j < kHalfDimensions; j++)
          st->accumulator.accumulation[c][j] -= ft_weights[offset + j];
      }

      // Difference calculation for the activated features
      for (unsigned k = 0; k < added[l].size; k++) {
        unsigned index = added[l].values[k];
        const unsigned offset = kHalfDimensions * index;

        for (unsigned j = 0; j < kHalfDimensions; j++)
          st->accumulator.accumulation[c][j] += ft_weights[offset + j];
      }
    }
#endif
  } else {
    Accumulator *accumulator = &pos->st->accumulator;
    accumulator->state[c] = ACC_COMPUTED;
    IndexList active;
    active.size = 0;
    append_active_indices(pos, c, &active);
#ifdef VECTOR
    for (unsigned i = 0; i < kHalfDimensions / TILE_HEIGHT; i++) {
      vec16_t *ft_biases_tile = (vec16_t *)&ft_biases[i * TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_REGS; j++)
        acc[j] = ft_biases_tile[j];

      for (unsigned k = 0; k < active.size; k++) {
        unsigned index = active.values[k];
        unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
        vec16_t *column = (vec16_t *)&ft_weights[offset];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = vec_add_16(acc[j], column[j]);
      }

      vec16_t *accTile = (vec16_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_REGS; j++)
        accTile[j] = acc[j];
    }
#else
    memcpy(accumulator->accumulation[c], ft_biases,
        kHalfDimensions * sizeof(int16_t));

    for (unsigned k = 0; k < active.size; k++) {
      unsigned index = active.values[k];
      unsigned offset = kHalfDimensions * index;

      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];
    }
#endif
  }
}

#if 0
// Convert input features
INLINE void transform(const Position *pos, int8_t *output, mask_t *outMask)
{
  update_accumulator(pos, WHITE);
  update_accumulator(pos, BLACK);

  int16_t (*accumulation)[2][256] = &pos->st->accumulator.accumulation;

  const Color perspectives[2] = { stm(), !stm() };
  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#ifdef VECTOR
    const unsigned numChunks = (16 * kHalfDimensions) / SIMD_WIDTH;
    vec8_t *out = (vec8_t *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      vec16_t s0 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2];
      vec16_t s1 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      out[i] = vec_packs(s0, s1);
      *outMask++ = vec_mask_pos(out[i]);
    }

#else
    (void)outMask; // avoid compiler warning
    for (unsigned i = 0; i < kHalfDimensions; i++) {
      int16_t sum = (*accumulation)[perspectives[p]][i];
      output[offset + i] = clamp(sum, 0, 127);
    }

#endif

  }
}
#endif

#if 0
// Evaluation function
Value nnue_evaluate(const Position *pos)
{
  int32_t out_value;
  alignas(8) mask_t hidden1_mask[512 / (8 * sizeof(mask_t))];
  alignas(8) mask_t hidden2_mask[8 / sizeof(mask_t)] = { 0 };
#ifdef ALIGNMENT_HACK // work around a bug in old gcc on Windows
  uint8_t buf[sizeof(struct NetData) + 63];
  struct NetData *b = (struct NetData *)(buf + ((((uintptr_t)buf-1) ^ 0x3f) & 0x3f));
#define B(x) (b->x)
#else
  struct NetData buf;
#define B(x) (buf.x)
#endif

  transform(pos, B(input), hidden1_mask);

  hidden_layer(B(input), B(hidden1_out), 512, hidden1_biases,
      hidden1_weights, hidden1_mask, hidden2_mask, true);

  hidden_layer(B(hidden1_out), B(hidden2_out), 32, hidden2_biases,
      hidden2_weights, hidden2_mask, NULL, false);

  out_value = output_layer(B(hidden2_out), output_biases, output_weights);

#if defined(USE_MMX)
  _mm_empty();
#endif

  return out_value / FV_SCALE;
}
#endif

enum {
  TransformerStart = 3 * 4 + 177,
  NetworkStart = TransformerStart + 4 + 2 * 256 + 2 * 256 * 64 * 641
};

#include "nnue-regular.c"
#include "nnue-sparse.c"

static bool verify_net(const void *evalData, size_t size)
{
  if (size != 21022697) return false;

  const char *d = evalData;
  if (readu_le_u32(d) != NnueVersion) return false;
  if (readu_le_u32(d + 4) != 0x3e5aa6eeU) return false;
  if (readu_le_u32(d + 8) != 177) return false;
  if (readu_le_u32(d + TransformerStart) != 0x5d69d7b8) return false;
  if (readu_le_u32(d + NetworkStart) != 0x63337156) return false;

  return true;
}

static bool load_eval_file(const char *evalFile)
{
  const void *evalData;
  map_t mapping;
  size_t size;

#ifdef NNUE_EMBEDDED
  if (strcmp(evalFile, DefaultEvalFile) == 0) {
    evalData = gNetworkData;
    mapping = 0;
    size = gNetworkSize;
  } else
#endif
  {
    FD fd = open_file(evalFile);
    if (fd == FD_ERR) return false;
    evalData = map_file(fd, &mapping);
    size = file_size(fd);
    close_file(fd);
  }

  bool success = verify_net(evalData, size);
  if (success)
    init_weights(evalData);
  if (mapping) unmap_file(evalData, mapping);
  return success;
}

static char *loadedFile = NULL;

void nnue_init(void)
{
#ifndef NNUE_PURE
  const char *s = option_string_value(OPT_USE_NNUE);
  useNNUE =  strcmp(s, "classical") == 0 ? EVAL_CLASSICAL
           : strcmp(s, "pure"     ) == 0 ? EVAL_PURE : EVAL_HYBRID;
#endif

  const char *evalFile = option_string_value(OPT_EVAL_FILE);
  if (loadedFile && strcmp(evalFile, loadedFile) == 0)
    return;

  if (loadedFile)
    free(loadedFile);

  if (load_eval_file(evalFile)) {
    loadedFile = strdup(evalFile);
    return;
  }

  printf("info string ERROR: The network file %s was not loaded successfully.\n"
#ifdef NNUE_EMBEDDED
         , evalFile
#else
         "info string ERROR: The default net can be downloaded from:\n"
         "info string ERROR: https://tests.stockfishchess.org/api/nn/%s\n",
         evalFile, option_default_string_value(OPT_EVAL_FILE)
#endif
         );
  exit(EXIT_FAILURE);
}

void nnue_free(void)
{
  if (ft_biases)
    free_memory(&ft_alloc);
}
