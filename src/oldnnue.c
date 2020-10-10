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
#include "uci.h"

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

#if defined(USE_NEON) && !defined(IS_64BIT)
INLINE int16x8_t vmovl_high_s16(int8x16_t v)
{
  return vmovl_s16(vget_high_s16(v));
}
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
  FtOutDims = kHalfDimensions * 2
};

// USE_MMX generates _mm_empty() instructions, so undefine if not needed
#if defined(USE_SSE2)
#undef USE_MMX
#endif

static_assert(kHalfDimensions % 256 == 0, "kHalfDimensions should be a multiple of 256");

#define VECTOR

#ifdef USE_AVX512
#define SIMD_WIDTH 512
typedef __m512i vec_t;
#define vec_add_16(a,b) _mm512_add_epi16(a,b)
#define vec_sub_16(a,b) _mm512_sub_epi16(a,b)
#define NUM_REGS 8 // only 8 are needed

#elif USE_AVX2
#define SIMD_WIDTH 256
typedef __m256i vec_t;
#define vec_add_16(a,b) _mm256_add_epi16(a,b)
#define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
#define NUM_REGS 16

#elif USE_SSE2
#define SIMD_WIDTH 128
typedef __m128i vec_t;
#define vec_add_16(a,b) _mm_add_epi16(a,b)
#define vec_sub_16(a,b) _mm_sub_epi16(a,b)
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif

#elif USE_MMX
#define SIMD_WIDTH 64
typedef __m64 vec_t;
#define vec_add_16(a,b) _mm_add_pi16(a,b)
#define vec_sub_16(a,b) _mm_sub_pi16(a,b)
#define NUM_REGS 8

#elif USE_NEON
#define SIMD_WIDTH 128
typedef int16x8_t vec_t;
#define vec_add_16(a,b) vaddq_s16(a,b)
#define vec_sub_16(a,b) vsubq_s16(a,b)
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif

#else
#undef VECTOR
#define SIMD_WIDTH 16 // dummy

#endif

#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_SSSE3))
typedef int16_t clipped_t; // SSE2 and MMX have no int8 multiply.
typedef int16_t weight_t;
#else
typedef int8_t clipped_t;
typedef int8_t weight_t;
#endif

typedef uint8_t mask_t; // dummy

typedef struct {
  size_t size;
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

static void half_kp_append_active_indices(const Position *pos, const Color c,
    IndexList *active)
{
  Square ksq = orient(c, square_of(c, KING));
  Bitboard bb = pieces() & ~pieces_p(KING);
  while (bb) {
    Square s = pop_lsb(&bb);
    active->values[active->size++] = make_index(c, s, piece_on(s), ksq);
  }
}

static void half_kp_append_changed_indices(const Position *pos, const Color c,
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

static void append_active_indices(const Position *pos, IndexList active[2])
{
  for (unsigned c = 0; c < 2; c++)
    half_kp_append_active_indices(pos, c, &active[c]);
}

static void append_changed_indices(const Position *pos, IndexList removed[2],
    IndexList added[2], bool reset[2])
{
  const DirtyPiece *dp = &(pos->st->dirtyPiece);
  assert(dp->dirtyNum != 0);

  if ((pos->st-1)->accumulator.computedAccumulation) {
    for (unsigned c = 0; c < 2; c++) {
      reset[c] = dp->pc[0] == make_piece(c, KING);
      if (reset[c])
        half_kp_append_active_indices(pos, c, &added[c]);
      else
        half_kp_append_changed_indices(pos, c, dp, &removed[c], &added[c]);
    }
  } else {
    const DirtyPiece *dp2 = &((pos->st-1)->dirtyPiece);
    for (unsigned c = 0; c < 2; c++) {
      reset[c] =   dp->pc[0] == make_piece(c, KING)
                || dp2->pc[0] == make_piece(c, KING);
      if (reset[c])
        half_kp_append_active_indices(pos, c, &added[c]);
      else {
        half_kp_append_changed_indices(pos, c, dp, &removed[c], &added[c]);
        half_kp_append_changed_indices(pos, c, dp2, &removed[c], &added[c]);
      }
    }
  }
}

// InputLayer = InputSlice<256 * 2>
// out: 512 x clipped_t

// Hidden1Layer = ClippedReLu<AffineTransform<InputLayer, 32>>
// 512 x clipped_t -> 32 x int32_t -> 32 x clipped_t

// Hidden2Layer = ClippedReLu<AffineTransform<hidden1, 32>>
// 32 x clipped_t -> 32 x int32_t -> 32 x clipped_t

// OutputLayer = AffineTransform<HiddenLayer2, 1>
// 32 x clipped_t -> 1 x int32_t

static alignas(64) weight_t hidden1_weights[32 * 512];
static alignas(64) weight_t hidden2_weights[32 * 32];
static alignas(64) weight_t output_weights [1 * 32];

static alignas(64) int32_t hidden1_biases[32];
static alignas(64) int32_t hidden2_biases[32];
static int32_t output_biases[1];

INLINE void affine_propagate(clipped_t *input, int32_t *output, unsigned inDims,
    unsigned outDims, int32_t *biases, weight_t *weights)
{
  assert(inDims % 32 == 0);

#if defined(USE_AVX512)
  const unsigned numChunks = (inDims * 8) / SIMD_WIDTH;
  __m512i *inVec = (__m512i *)input;
#if !defined(USE_VNNI)
  const __m512i kOnes = _mm512_set1_epi16(1);
#endif

#elif defined(USE_AVX2)
  const unsigned numChunks = (inDims * 8) / SIMD_WIDTH;
  __m256i *inVec = (__m256i *)input;
#if !defined(USE_VNNI)
  const __m256i kOnes = _mm256_set1_epi16(1);
#endif

#elif defined(USE_SSSE3)
  const unsigned numChunks = (inDims * 8) / SIMD_WIDTH;
  __m128i *inVec = (__m128i *)input;
  const __m128i kOnes = _mm_set1_epi16(1);

#elif defined(USE_SSE2)
  const unsigned numChunks = (inDims * 16) / SIMD_WIDTH;
  __m128i *inVec = (__m128i *)input;

#elif defined(USE_MMX)
  const unsigned numChunks = (inDims * 16) / SIMD_WIDTH;
  __m64 *inVec = (__m64 *)input;

#elif defined(USE_NEON)
  const unsigned numChunks = (inDims * 8) / SIMD_WIDTH;
  int8x8_t *inVec = (int8x8_t *)input;

#endif

  for (unsigned i = 0; i < outDims; ++i) {
    unsigned offset = i * inDims;

#if defined(USE_AVX512)
    __m512i sum = _mm512_setzero_si512();
    __m512i *row = (__m512i *)&weights[offset];
    for (unsigned j = 0; j < numChunks; j++) {
#if defined(USE_VNNI)
      sum = _mm512_dpbusd_epi32(sum, inVec[j], row[j]);
#else
      __m512i product = _mm512_maddubs_epi16(inVec[j], row[j]);
      product = _mm512_madd_epi16(product, kOnes);
      sum = _mm512_add_epi32(sum, product);
#endif
    }

    if (inDims != numChunks * 64) {
      __m256i *iv256 = (__m256i *)(&inVec[numChunks]);
      __m256i *row256 = (__m256i *)(&row[numChunks]);
#if defined(USE_VNNI)
      __m256i product256 = _mm256_dpbusd_epi32(_mm512_castsi512_si256(sum),
          iv256[0], row256[0]);
      sum = _mm512_inserti32x8(sum, product256, 0);
#else
      __m256i product256 = _mm256_maddubs_epi16(iv256[0], row256[0]);
      sum = _mm512_add_epi32(sum, _mm512_cvtepi16_epi32(product256));
#endif
    }
    output[i] = _mm512_reduce_add_epi32(sum) + biases[i];

#elif defined(USE_AVX2)
    __m256i sum = _mm256_setzero_si256();
    __m256i *row = (__m256i *)&weights[offset];
    for (unsigned j = 0; j < numChunks; j++) {
#if defined(USE_VNNI)
      sum = _mm256_dpbusd_epi32(sum, inVec[j], row[j]);
#else
      __m256i product = _mm256_maddubs_epi16(inVec[j], row[j]);
      product = _mm256_madd_epi16(product, kOnes);
      sum = _mm256_add_epi32(sum, product);
#endif
    }
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    output[i] = _mm_cvtsi128_si32(sum128) + biases[i];

#elif defined(USE_SSSE3)
    __m128i sum = _mm_setzero_si128();
    __m128i *row = (__m128i *)&weights[offset];
    for (unsigned j = 0; j < numChunks / 2; j++) {
      __m128i product0 = _mm_maddubs_epi16(inVec[2 * j], row[2 * j]);
      product0 = _mm_madd_epi16(product0, kOnes);
      sum = _mm_add_epi32(sum, product0);
      __m128i product1 = _mm_maddubs_epi16(inVec[2 * j + 1], row[2 * j + 1]);
      product1 = _mm_madd_epi16(product1, kOnes);
      sum = _mm_add_epi32(sum, product1);
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E)); //_MM_PERM_BADC
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1)); //_MM_PERM_CDAB
    output[i] = _mm_cvtsi128_si32(sum) + biases[i];

#elif defined(USE_SSE2)
    __m128i sum = _mm_setzero_si128(), sum1 = sum;
    __m128i *row = (__m128i *)&weights[offset];
    for (unsigned j = 0; j < numChunks / 2; j++) {
      __m128i product0 = _mm_madd_epi16(inVec[2 * j], row[2 * j]);
      sum = _mm_add_epi32(sum, product0);
      __m128i product1 = _mm_madd_epi16(inVec[2 * j + 1], row[2 * j + 1]);
      sum1 = _mm_add_epi32(sum1, product1);
    }
    sum = _mm_add_epi32(sum, sum1);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xE));
    sum = _mm_add_epi32(sum, _mm_shufflelo_epi16(sum, 0xE));
    output[i] = _mm_cvtsi128_si32(sum) + biases[i];

#elif defined(USE_MMX)
    // adding 1 or 4 numbers per loop is slower, 2 seems optimal
    __m64 s0 = _mm_setzero_si64(), s1 = s0;
    __m64 *row = (__m64 *)&weights[offset];
    for (unsigned j = 0; j < numChunks / 2; j++) {
      s0 = _mm_add_pi32(s0, _mm_madd_pi16(row[2 * j], inVec[2 * j]));
      s1 = _mm_add_pi32(s1, _mm_madd_pi16(row[2 * j + 1], inVec[2 * j + 1]));
    }
    __m64 sum = _mm_add_pi32(s0, s1);
    sum = _mm_add_pi32(sum, _mm_unpackhi_pi32(sum, sum));
    output[i] = _mm_cvtsi64_si32(sum) + biases[i];

#elif defined(USE_NEON)
    int32x4_t sum = {biases[i]};
    int8x8_t *row = (int8x8_t *)&weights[offset];
    for (unsigned j = 0; j < numChunks; j++) {
      int16x8_t product = vmull_s8(inVec[2 * j], row[2 * j]);
      product = vmlal_s8(product, inVec[2 * j + 1], row[2 * j + 1]);
      sum = vpadalq_s16(sum, product);
    }
    output[i] = sum[0] + sum[1] + sum[2] + sum[3];

#else
    int32_t sum = biases[i];
    for (unsigned j = 0; j < inDims; j++)
      sum += weights[offset + j] * input[j];
    output[i] = sum;

#endif
  }
}

INLINE void clip_propagate(int32_t *input, clipped_t *output, unsigned numDims)
{
  assert(numDims == 32);

#if defined(USE_AVX512)
  (void)numDims;
  __m512i *in = (__m512i *)input;
  __m256i *out = (__m256i *)output;
  __m512i words = _mm512_srai_epi16(_mm512_packs_epi32(in[0], in[1]), SHIFT);
  __m256i packed = _mm256_packs_epi16(
      _mm512_castsi512_si256(words),_mm512_extracti64x4_epi64(words, 1));
  out[0] = _mm256_max_epi8(packed, _mm256_setzero_si256());

#elif defined(USE_AVX2)
  const unsigned numChunks = numDims / 32;
  const __m256i kZero = _mm256_setzero_si256();
  __m256i *in = (__m256i *)input;
  __m256i *out = (__m256i *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m256i words0 = _mm256_srai_epi16(_mm256_packs_epi32(
          in[i * 4 + 0], in[i * 4 + 1]), SHIFT);
    __m256i words1 = _mm256_srai_epi16(_mm256_packs_epi32(
          in[i * 4 + 2], in[i * 4 + 3]), SHIFT);
    out[i] = _mm256_max_epi8(_mm256_packs_epi16(words0, words1), kZero);
  }

#elif defined(USE_SSSE3)
  const unsigned numChunks = numDims / 16;
#ifdef USE_SSE41
  const __m128i kZero = _mm_setzero_si128();
#else
  const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

  __m128i *in = (__m128i *)input;
  __m128i *out = (__m128i *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m128i words0 = _mm_srai_epi16(
        _mm_packs_epi32(in[i * 4 + 0], in[i * 4 + 1]), SHIFT);
    __m128i words1 = _mm_srai_epi16(
        _mm_packs_epi32(in[i * 4 + 2], in[i * 4 + 3]), SHIFT);
    __m128i packed = _mm_packs_epi16(words0, words1);
#ifdef USE_SSE41
    out[i] = _mm_max_epi8(packed, kZero);
#else
    out[i] = _mm_subs_epi8(_mm_adds_epi8(packed, k0x80s), k0x80s);
#endif
  }

#elif defined(USE_SSE2)
  const unsigned numChunks = numDims / 8;
  const __m128i kZero = _mm_setzero_si128();
  const __m128i k0x7f = _mm_set1_epi16(0x7f);
  __m128i *in = (__m128i *)input;
  __m128i *out = (__m128i *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m128i words = _mm_srai_epi16(_mm_packs_epi32(in[i * 2], in[i * 2 + 1]),
        SHIFT);
    out[i] = _mm_min_epi16(_mm_max_epi16(words, kZero), k0x7f);
  }

#elif defined(USE_MMX)
  const unsigned numChunks = numDims / 4;
#ifdef USE_SSE
  const __m64 kZero = _mm_setzero_si64();
  const __m64 k0x7f = _mm_set1_pi16(0x7f);
#else
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  const __m64 k0x0080 = _mm_set1_pi16(0x0080);
  const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
#endif
  __m64 *in = (__m64 *)input;
  __m64 *out = (__m64 *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m64 words = _mm_srai_pi16(_mm_packs_pi32(in[i * 2], in[i * 2 + 1]),
        SHIFT);
#ifdef USE_SSE
    out[i] = _mm_min_epi16(_mm_max_epi16(words, kZero), k0x7f);
#else
    out[i] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(words, k0x7f80), k0x0080), k0x8000);
#endif
  }

#elif defined(USE_NEON)
  const unsigned numChunks = numDims / 8;
  const int8x8_t kZero = {0};
  int32x4_t *in = (int32x4_t *)input;
  int8x8_t *out = (int8x8_t *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    int16x8_t shifted = vcombine_s16(
        vqshrn_n_s32(in[i * 2], SHIFT), vqshrn_n_s32(in[i * 2 + 1], SHIFT));
    out[i] = vmax_s8(vqmovn_s16(shifted), kZero);
  }

#else
  for (unsigned i = 0; i < numDims; i++)
    output[i] = clamp(input[i] >> SHIFT, 0, 127);

#endif
}

// Input feature converter
static alignas(64) int16_t ft_biases[kHalfDimensions];
static alignas(64) int16_t ft_weights[kHalfDimensions * FtInDims];

#ifdef VECTOR
#define TILE_HEIGHT (NUM_REGS * SIMD_WIDTH / 16)
#endif

// Calculate cumulative value without using difference calculation
INLINE void refresh_accumulator(const Position *pos)
{
  Accumulator *accumulator = &(pos->st->accumulator);

  IndexList activeIndices[2];
  activeIndices[0].size = activeIndices[1].size = 0;
  append_active_indices(pos, activeIndices);

  for (unsigned c = 0; c < 2; c++) {
#ifdef VECTOR
    for (unsigned i = 0; i < kHalfDimensions / TILE_HEIGHT; i++) {
      vec_t *ft_biases_tile = (vec_t *)&ft_biases[i * TILE_HEIGHT];
      vec_t *accTile = (vec_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
      vec_t acc[NUM_REGS];

      for (unsigned j = 0; j < NUM_REGS; j++)
        acc[j] = ft_biases_tile[j];

      for (size_t k = 0; k < activeIndices[c].size; k++) {
        unsigned index = activeIndices[c].values[k];
        unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
        vec_t *column = (vec_t *)&ft_weights[offset];

        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = vec_add_16(acc[j], column[j]);
      }

      for (unsigned j = 0; j < NUM_REGS; j++)
        accTile[j] = acc[j];
    }
#else
    memcpy(accumulator->accumulation[c], ft_biases,
        kHalfDimensions * sizeof(int16_t));

    for (size_t k = 0; k < activeIndices[c].size; k++) {
      unsigned index = activeIndices[c].values[k];
      unsigned offset = kHalfDimensions * index;

      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];
    }
#endif
  }

  accumulator->computedAccumulation = true;
}

// Calculate cumulative value using difference calculation if possible
INLINE bool update_accumulator(const Position *pos)
{
  Accumulator *accumulator = &(pos->st->accumulator);
  if (accumulator->computedAccumulation)
    return true;

  Accumulator *prevAcc;
  if (   !(prevAcc = &(pos->st-1)->accumulator)->computedAccumulation
      && !(prevAcc = &(pos->st-2)->accumulator)->computedAccumulation)
    return false;

  IndexList removed_indices[2], added_indices[2];
  removed_indices[0].size = removed_indices[1].size = 0;
  added_indices[0].size = added_indices[1].size = 0;
  bool reset[2];
  append_changed_indices(pos, removed_indices, added_indices, reset);

#ifdef VECTOR
  for (unsigned i = 0; i< kHalfDimensions / TILE_HEIGHT; i++) {
    for (unsigned c = 0; c < 2; c++) {
      vec_t *accTile = (vec_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
      vec_t acc[NUM_REGS];

      if (reset[c]) {
        vec_t *ft_b_tile = (vec_t *)&ft_biases[i * TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = ft_b_tile[j];
      } else {
        vec_t *prevAccTile = (vec_t *)&prevAcc->accumulation[c][i * TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = prevAccTile[j];

        // Difference calculation for the deactivated features
        for (unsigned k = 0; k < removed_indices[c].size; k++) {
          unsigned index = removed_indices[c].values[k];
          const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;

          vec_t *column = (vec_t *)&ft_weights[offset];
          for (unsigned j = 0; j < NUM_REGS; j++)
            acc[j] = vec_sub_16(acc[j], column[j]);
        }
      }

      // Difference calculation for the activated features
      for (unsigned k = 0; k < added_indices[c].size; k++) {
        unsigned index = added_indices[c].values[k];
        const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;

        vec_t *column = (vec_t *)&ft_weights[offset];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = vec_add_16(acc[j], column[j]);
      }

      for (unsigned j = 0; j < NUM_REGS; j++)
        accTile[j] = acc[j];
    }
  }
#else
  for (unsigned c = 0; c < 2; c++) {
    if (reset[c]) {
      memcpy(accumulator->accumulation[c], ft_biases,
          kHalfDimensions * sizeof(int16_t));
    } else {
      memcpy(accumulator->accumulation[c], prevAcc->accumulation[c],
          kHalfDimensions * sizeof(int16_t));
      // Difference calculation for the deactivated features
      for (unsigned k = 0; k < removed_indices[c].size; k++) {
        unsigned index = removed_indices[c].values[k];
        const unsigned offset = kHalfDimensions * index;

        for (unsigned j = 0; j < kHalfDimensions; j++)
          accumulator->accumulation[c][j] -= ft_weights[offset + j];
      }
    }

    // Difference calculation for the activated features
    for (unsigned k = 0; k < added_indices[c].size; k++) {
      unsigned index = added_indices[c].values[k];
      const unsigned offset = kHalfDimensions * index;

      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];
    }
  }
#endif

  accumulator->computedAccumulation = true;
  return true;
}

// Convert input features
INLINE void transform(const Position *pos, clipped_t *output, mask_t *outMask)
{
  if (!update_accumulator(pos))
    refresh_accumulator(pos);

  int16_t (*accumulation)[2][256] = &pos->st->accumulator.accumulation;
  (void)outMask; // avoid compiler warning

  // Number of vectors to read
  const unsigned numChunks = (16 * kHalfDimensions) / SIMD_WIDTH;
#if defined(USE_AVX512)
  const __m512i kZero = _mm512_setzero_si512();

#elif defined(USE_AVX2)
  const __m256i kZero = _mm256_setzero_si256();

#elif defined(USE_SSE2)
#if defined(USE_SSE41)
  const __m128i kZero = _mm_setzero_si128();
#else
#if !defined(USE_SSSE3)
  const __m128i kZero = _mm_setzero_si128();
  const __m128i k0x7f = _mm_set1_epi16(127);
#else
  const __m128i k0x80s = _mm_set1_epi8(-128);
#endif
#endif

#elif defined(USE_MMX)
#ifdef USE_SSE
  const __m64 k0x7f = _mm_set1_pi16(127);
  const __m64 kZero = _mm_setzero_si64();
#else
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  const __m64 k0x0080 = _mm_set1_pi16(0x0080);
  const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
#endif

#elif defined(USE_NEON)
  const int8x16_t kZero = {0};

#endif

  const Color perspectives[2] = { stm(), !stm() };
  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#if defined(USE_AVX512)
    __m512i *out = (__m512i *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m512i sum0 = ((__m512i *)(*accumulation)[perspectives[p]])[i * 2];
      __m512i sum1 = ((__m512i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m512i packed = _mm512_packs_epi16(sum0, sum1);
      out[i] = _mm512_max_epi8(packed, kZero);
    }

#elif defined(USE_AVX2)
    __m256i *out = (__m256i *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m256i sum0 = ((__m256i *)(*accumulation)[perspectives[p]])[i * 2];
      __m256i sum1 = ((__m256i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m256i packed = _mm256_packs_epi16(sum0, sum1);
      out[i] = _mm256_max_epi8(packed, kZero);
    }

#elif defined(USE_SSE2)
    __m128i *out = (__m128i *)&output[offset];
#if defined(USE_SSSE3)
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m128i sum0 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2];
      __m128i sum1 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m128i packed = _mm_packs_epi16(sum0, sum1);
#if defined(USE_SSE41)
      out[i] = _mm_max_epi8(packed, kZero);
#else
      out[i] = _mm_subs_epi8(_mm_adds_epi8(packed, k0x80s), k0x80s);
#endif
    }
#else
    for (unsigned i = 0; i < numChunks; i++) {
      __m128i sum = ((__m128i *)(*accumulation)[perspectives[p]])[i];
      out[i] = _mm_min_epi16(_mm_max_epi16(sum, kZero), k0x7f);
    }
#endif

#elif defined(USE_MMX)
    __m64 *out = (__m64 *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m64 sum = ((__m64 *)(*accumulation)[perspectives[p]])[i];
#ifdef USE_SSE
      out[i] = _mm_min_pi16(_mm_max_pi16(sum, kZero), k0x7f);
#else
      out[i] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(sum, k0x7f80), k0x0080), k0x8000);
#endif
    }

#elif defined(USE_NEON)
    int8x16_t *out = (int8x16_t *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      int16x8_t sum = ((int16x8_t *)(*accumulation)[perspectives[p]])[2 * i];
      int16x8_t sum1 = ((int16x8_t *)(*accumulation)[perspectives[p]])[2 * i + 1];
      out[i] = vmaxq_s8(vcombine_s8(vqmovn_s16(sum), vqmovn_s16(sum1)), kZero);
    }

#else
    (void)numChunks;
    for (unsigned i = 0; i < kHalfDimensions; i++) {
      int16_t sum = (*accumulation)[perspectives[p]][i];
      output[offset + i] = clamp(sum, 0, 127);
    }

#endif

  }
}

struct NetData {
  alignas(64) clipped_t input[FtOutDims];
  int32_t hidden1_values[32];
  int32_t hidden2_values[32];
  clipped_t hidden1_clipped[32];
  clipped_t hidden2_clipped[32];
};

// Evaluation function
Value nnue_evaluate(const Position *pos)
{
  int32_t out_value;
#ifdef ALIGNMENT_HACK // work around a bug in old gcc on Windows
  uint8_t buf[sizeof(struct NetData) + 63];
  struct NetData *b = (struct NetData *)(buf + ((((uintptr_t)buf-1) ^ 0x3f) & 0x3f));
#define B(x) (b->x)
#else
  struct NetData buf;
#define B(x) (buf.x)
#endif

  transform(pos, B(input), NULL);

  affine_propagate(B(input), B(hidden1_values), FtOutDims, 32,
      hidden1_biases, hidden1_weights);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 32);

  affine_propagate(B(hidden1_clipped), B(hidden2_values), 32, 32,
      hidden2_biases, hidden2_weights);
  clip_propagate(B(hidden2_values), B(hidden2_clipped), 32);

  affine_propagate(B(hidden2_clipped), &out_value, 32, 1, output_biases,
      output_weights);

#if defined(USE_MMX)
  _mm_empty();
#endif

  return out_value / FV_SCALE;
}

static void read_output_weights(weight_t *w, const char *d)
{
  for (unsigned i = 0; i < 32; i++) {
    unsigned c = i;
#if defined(USE_AVX512)
    unsigned b = c & 0x14;
    b = (b << 2) | (b >> 2);
    c = (c & ~0x14) | (b & 0x14);
#elif defined(USE_AVX2)
    unsigned b = c & 0x1c;
    b = (b << 2) | (b >> 1);
    c = (c & ~0x1c) | (b & 0x1c);
#endif
    w[c] = *d++;
  }
}

INLINE unsigned wt_idx(unsigned r, unsigned c, unsigned dims)
{
  (void)dims;

#if defined(USE_AVX512)
  if (dims > 32) {
    unsigned b = c & 0x38;
    b = (b << 1) | (b >> 2);
    c = (c & ~0x38) | (b & 0x38);
  }
  else if (dims == 32) {
    unsigned b = c & 0x14;
    b = (b << 2) | (b >> 2);
    c = (c & ~0x14) | (b & 0x14);
  }

#elif defined(USE_AVX2)
  if (dims > 32) {
    unsigned b = c & 0x18;
    b = (b << 1) | (b >> 1);
    c = (c & ~0x18) | (b & 0x18);
  }
  else if (dims == 32) {
    unsigned b = c & 0x1c;
    b = (b << 2) | (b >> 1);
    c = (c & ~0x1c) | (b & 0x1c);
  }

#endif

  return r * dims + c;
}

static const char *read_hidden_weights(weight_t *w, unsigned dims,
    const char *d)
{
  for (unsigned r = 0; r < 32; r++)
    for (unsigned c = 0; c < dims; c++)
      w[wt_idx(r, c, dims)] = *d++;

  return d;
}

enum {
  TransformerStart = 3 * 4 + 177,
  NetworkStart = TransformerStart + 4 + 2 * 256 + 2 * 256 * 64 * 641
};

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

static void init_weights(const void *evalData)
{
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
