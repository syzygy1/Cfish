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
#define vec_clip_8(a,b) _mm512_max_epi8(vec_packs(a,b),_mm512_setzero_si512())
#define NUM_REGS 8 // only 8 are needed

#elif USE_AVX2
#define SIMD_WIDTH 256
typedef __m256i vec8_t, vec16_t;
typedef uint32_t mask_t;
#define vec_add_16(a,b) _mm256_add_epi16(a,b)
#define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
#define vec_packs(a,b) _mm256_packs_epi16(a,b)
#define vec_mask_pos(a) _mm256_movemask_epi8(_mm256_cmpgt_epi8(a,_mm256_setzero_si256()))
#define vec_clip_8(a,b) _mm256_max_epi8(vec_packs(a,b),_mm256_setzero_si256())
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
#ifdef USE_SSE41
#define vec_clip_8(a,b) _mm_max_epi8(vec_packs(a,b),_mm_setzero_si128())
#elif USE_SSSE3
#define vec_clip_8(a,b) vec_packs(_mm_max_epi16(a,_mm_setzero_si128()),_mm_max_epi16(b,_mm_setzero_si128()))
#else
#define vec_clip_16(a) _mm_min_epi16(_mm_max_epi16(a,_mm_setzero_si128()),_mm_set1_epi16(127))
#endif
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
#ifdef USE_SSE
#define vec_clip_16(a) _mm_min_pi16(_mm_max_pi16(a,_mm_setzero_si64()),_mm_set1_pi16(127))
#else
#define vec_clip_16(a) _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(a, _mm_set1_pi16(0x7f80)), _mm_set1_pi16(0x0080)), _mm_set1_pi16(-0x8000))
#endif
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
#define vec_clip_8(a,b) vmaxq_s8(vec_packs(a,b),vdupq_n_s8(0))
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

#ifdef NNUE_SPARSE
typedef int8_t clipped_t;
#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_AVX2))
typedef int16_t weight_t, out_t;
#else
typedef int8_t weight_t, out_t;
#endif
#else
#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_SSSE3))
typedef int16_t weight_t, out_t, clipped_t;
#else
typedef int8_t weight_t, out_t, clipped_t;
#endif
#endif

#if defined(USE_MMX) && !defined(USE_SSE)
INLINE int _mm_movemask_pi8(__m64 v)
{
  const __m64 powers = _mm_set_pi8(-128, 64, 32, 16, 8, 4, 2, 1);
  __m64 m = _mm_and_si64(v, powers);
  m = _mm_or_si64(m, _mm_srli_si64(m, 32));
  m = _mm_or_si64(m, _mm_srli_pi32(m, 16));
  m = _mm_or_si64(m, _mm_srli_pi16(m, 8));
  return _mm_cvtsi64_si32(m) & 0xff;
}
#elif defined(USE_NEON)
INLINE int neon_movemask(uint8x16_t v)
{
  const uint8_t __attribute__((aligned(16))) powers[16] =
    { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
  const uint8x16_t kPowers = vld1q_u8(powers);

  uint64x2_t mask = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vandq_u8(v, kPowers))));
  return   vgetq_lane_u8((uint8x16_t)mask, 0)
        | (vgetq_lane_u8((uint8x16_t)mask, 8) << 8);
}
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

INLINE int32_t output_layer(const out_t *input, const int32_t *biases,
    const out_t *weights)
{
#if defined(USE_AVX2)
  __m256i *iv = (__m256i *)input;
  __m256i *row = (__m256i *)weights;
#if defined(USE_VNNI)
  __m256i prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), iv[0], row[0]);
#else
  __m256i prod = _mm256_maddubs_epi16(iv[0], row[0]);
  prod = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
#endif
  __m128i sum = _mm_add_epi32(
      _mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1b));
  return _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1) + biases[0];

#elif defined(USE_SSE2)
  __m128i *iv = (__m128i *)input;
  __m128i *row = (__m128i *)weights;
#if defined(USE_SSSE3) && !defined(NNUE_SPARSE)
  const __m128i kOnes = _mm_set1_epi16(1);
  __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(iv[0], row[0]), kOnes);
  __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(iv[1], row[1]), kOnes);
  __m128i sum = _mm_add_epi32(p0, p1);
#else
  __m128i p0 = _mm_madd_epi16(iv[0], row[0]);
  __m128i p1 = _mm_madd_epi16(iv[1], row[1]);
  __m128i p2 = _mm_madd_epi16(iv[2], row[2]);
  __m128i p3 = _mm_madd_epi16(iv[3], row[3]);
  __m128i sum = _mm_add_epi32(_mm_add_epi32(p0, p1), _mm_add_epi32(p2, p3));
#endif
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb));
#if defined(USE_SSE41)
  return _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1) + biases[0];
#else
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1));
  return _mm_cvtsi128_si32(sum) + biases[0];
#endif

#elif defined(USE_MMX)
  __m64 *iv = (__m64 *)input;
  __m64 *row = (__m64 *)weights;
  __m64 s0 = _mm_setzero_si64(), s1 = s0;
  for (unsigned j = 0; j < 4; j++) {
    s0 = _mm_add_pi32(s0, _mm_madd_pi16(row[2 * j], iv[2 * j]));
    s1 = _mm_add_pi32(s1, _mm_madd_pi16(row[2 * j + 1], iv[2 * j + 1]));
  }
  __m64 sum = _mm_add_pi32(s0, s1);
  sum = _mm_add_pi32(sum, _mm_unpackhi_pi32(sum, sum));
  return _mm_cvtsi64_si32(sum) + biases[0];

#elif defined(USE_NEON)
  int8x8_t *iv = (int8x8_t *)input;
  int8x8_t *row = (int8x8_t *)weights;
  int32x4_t sum = {biases[0]};
  for (unsigned j = 0; j < 2; j++) {
    int16x8_t prod = vmull_s8(iv[2 * j], row[2 * j]);
    prod = vmlal_s8(prod, iv[2 * j + 1], row[2 * j + 1]);
    sum = vpadalq_s16(sum, prod);
  }
  return sum[0] + sum[1] + sum[2] + sum[3];

#else
  int32_t sum = biases[0];
  for (unsigned j = 0; j < 32; j++)
    sum += weights[j] * input[j];
  return sum;

#endif
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

// Convert input features
INLINE void transform(const Position *pos, clipped_t *output, mask_t *outMask)
{
  (void)outMask;
  update_accumulator(pos, WHITE);
  update_accumulator(pos, BLACK);

  int16_t (*accumulation)[2][256] = &pos->st->accumulator.accumulation;

  const Color perspectives[2] = { stm(), !stm() };
  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#ifdef VECTOR
    const unsigned numChunks = (16 * kHalfDimensions) / SIMD_WIDTH;

#if defined(NNUE_SPARSE) || defined(USE_SSSE3) || defined(USE_NEON)
    vec8_t *out = (vec8_t *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      vec16_t s0 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2];
      vec16_t s1 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2 + 1];
#ifdef NNUE_SPARSE
      out[i] = vec_packs(s0, s1);
      *outMask++ = vec_mask_pos(out[i]);
#else
      out[i] = vec_clip_8(s0, s1);
#endif
    }

#else
    vec16_t *out = (vec16_t *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      vec16_t sum = ((vec16_t *)(*accumulation)[perspectives[p]])[i];
      out[i] = vec_clip_16(sum);
    }

#endif

#else
    for (unsigned i = 0; i < kHalfDimensions; i++) {
      int16_t sum = (*accumulation)[perspectives[p]][i];
      output[offset + i] = clamp(sum, 0, 127);
    }

#endif

  }
}

#ifndef USE_NEON
INLINE unsigned bit_shuffle(unsigned v, int left, int right, unsigned mask)
{
  unsigned w = v & mask;
  w = (w << left) | (w >> right);
  return (v & ~mask) | (w & mask);
}
#endif

enum {
  TransformerStart = 3 * 4 + 177,
  NetworkStart = TransformerStart + 4 + 2 * 256 + 2 * 256 * 64 * 641
};

#include "nnue-regular.c"
#include "nnue-sparse.c"

static const char *read_hidden_weights(weight_t *w, unsigned dims,
    const char *d)
{
  for (unsigned r = 0; r < 32; r++)
    for (unsigned c = 0; c < dims; c++)
      w[wt_idx(r, c, dims)] = *d++;

  return d;
}

static void init_weights(const void *evalData)
{
  if (!ft_biases) {
    if (settings.largePages)
      ft_biases = allocate_memory(2 * kHalfDimensions * (FtInDims + 1), true,
          &ft_alloc);
    if (!ft_biases)
      ft_biases = allocate_memory(2 * kHalfDimensions * (FtInDims + 1), false,
          &ft_alloc);
    if (!ft_biases) {
      fprintf(stdout, "Could not allocate enough memory.\n");
      exit(EXIT_FAILURE);
    }
    ft_weights = ft_biases + kHalfDimensions;
  }

  const char *d = (const char *)evalData + TransformerStart + 4;

  // Read transformer
  for (unsigned i = 0; i < kHalfDimensions; i++, d += 2)
    ft_biases[i] = readu_le_u16(d);
  for (unsigned i = 0; i < kHalfDimensions * FtInDims; i++, d += 2)
    ft_weights[i] = readu_le_u16(d);

  // Read network
  d += 4;
  for (unsigned i = 0; i < 32; i++, d += 4)
    hidden1_biases[i] = readu_le_u32(d);
  d = read_hidden_weights(hidden1_weights, 512, d);
  for (unsigned i = 0; i < 32; i++, d += 4)
    hidden2_biases[i] = readu_le_u32(d);
  d = read_hidden_weights(hidden2_weights, 32, d);
  for (unsigned i = 0; i < 1; i++, d += 4)
    output_biases[i] = readu_le_u32(d);
  read_output_weights(output_weights, d);

#if defined(NNUE_SPARSE) && defined(USE_AVX2)
  permute_biases(hidden1_biases);
  permute_biases(hidden2_biases);
#endif
}

void nnue_export_net(void) {
#ifdef NNUE_EMBEDDED
  FILE *F = fopen(DefaultEvalFile, "wb");
  if (F) {
    fwrite(gNetworkData, gNetworkSize, 1, F);
    fclose(F);
  }
#else
  printf("No embedded network fie.\n");
#endif
}

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
