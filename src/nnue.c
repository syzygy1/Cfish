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

#if defined(USE_AVX512)
//#define TRANSPOSE

#elif defined(USE_AVX2)
#define TRANSPOSE

#elif defined(USE_SSE2)
#define TRANSPOSE

#elif defined(USE_MMX)
//#define TRANSPOSE

#elif defined(USE_NEON)
//#define TRANSPOSE

#else /* fallback code */
#define TRANSPOSE
#endif

#ifdef TRANSPOSE
#if defined(USE_SSE)
#define USE_MASK
#endif
#endif

static_assert(kHalfDimensions % 256 == 0, "kHalfDimensions should be a multiple of 256");

#ifdef USE_AVX512
#define SIMD_WIDTH 512
typedef __m512i vec_t;
#define vec_add_16(a,b) _mm512_add_epi16(a,b)
#define vec_sub_16(a,b) _mm512_sub_epi16(a,b)

#elif USE_AVX2
#define SIMD_WIDTH 256
typedef __m256i vec_t;
#define vec_add_16(a,b) _mm256_add_epi16(a,b)
#define vec_sub_16(a,b) _mm256_sub_epi16(a,b)

#elif USE_SSE2
#define SIMD_WIDTH 128
typedef __m128i vec_t;
#define vec_add_16(a,b) _mm_add_epi16(a,b)
#define vec_sub_16(a,b) _mm_sub_epi16(a,b)

#elif USE_MMX
#define SIMD_WIDTH 64
typedef __m64 vec_t;
#define vec_add_16(a,b) _mm_add_pi16(a,b)
#define vec_sub_16(a,b) _mm_sub_pi16(a,b)

#elif USE_NEON
#define SIMD_WIDTH 128
typedef int8x8_t vec_t; // unused

#else
#define SIMD_WIDTH 8 // dummy

#endif

// NUM_REGS is used only in refresh/update_accumulator()
#if defined(USE_AVX512)
#define NUM_REGS 8 // only 8 are needed

#elif defined(USE_SSE2) && defined(IS_64BIT)
#define NUM_REGS 16

#elif defined(USE_SSE2)
#define NUM_REGS 8

#elif USE_MMX
#define NUM_REGS 8

#endif

#ifndef TRANSPOSE

#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_SSSE3))
typedef int16_t clipped_t; // SSE2 and MMX have no int8 multiply.
typedef int16_t weight_t;
#else
typedef int8_t clipped_t;
typedef int8_t weight_t;
#endif

typedef uint8_t mask_t; // dummy

#else /* TRANSPOSE */

typedef int8_t clipped_t;
#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_AVX2))
typedef int16_t weight_t;
#else
typedef int8_t weight_t;
#endif

#if defined(USE_AVX512)
typedef __mmask64 mask_t;
#elif defined(USE_AVX2)
typedef uint32_t mask_t;
#elif defined(USE_SSE2)
typedef uint16_t mask_t;
#elif defined(USE_MMX)
typedef uint8_t mask_t;
#else
typedef uint8_t mask_t; // dummy
#endif

#endif

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

static void half_kp_append_active_indices(const Position *pos, Color c,
    IndexList *active)
{
  Square ksq = orient(c, square_of(c, KING));
  Bitboard bb = pieces() & ~pieces_p(KING);
  while (bb) {
    Square s = pop_lsb(&bb);
    active->values[active->size++] = make_index(c, s, piece_on(s), ksq);
  }
}

static void half_kp_append_changed_indices(const Position *pos, Color c,
    IndexList *removed, IndexList *added)
{
  Square ksq = orient(c, square_of(c, KING));
  DirtyPiece *dp = &(pos->st->dirtyPiece);
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
  if (dp->dirtyNum == 0) return;

  for (unsigned c = 0; c < 2; c++) {
    reset[c] = dp->pc[0] == make_piece(c, KING);
    if (reset[c])
      half_kp_append_active_indices(pos, c, &added[c]);
    else
      half_kp_append_changed_indices(pos, c, &removed[c], &added[c]);
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

#elif defined(USE_SSSE3) && !defined(TRANSPOSE)
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

#elif defined(USE_SSSE3) && !defined(TRANSPOSE)
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

#ifndef TRANSPOSE

INLINE void clip_propagate(int32_t *input, clipped_t *output, unsigned numDims)
{
  assert(numDims == 32);

#if defined(USE_AVX512)
  (void)numDims;
  const __m512i kZero = _mm512_setzero_si512();
  const __m512i kOffsets = _mm512_set_epi32(0,0,0,0,0,0,0,0,13,9,5,1,12,8,4,0);
  __m512i *in = (__m512i *)input;
  __m256i *out = (__m256i *)output;
  __m512i words = _mm512_srai_epi16(_mm512_packs_epi32(in[0], in[1]), SHIFT);
  out[0] = _mm256_max_epi8(_mm512_castsi512_si256(
        _mm512_permutexvar_epi32(kOffsets, _mm512_packs_epi16(words, kZero))),
        _mm256_setzero_si256());

#elif defined(USE_AVX2)
  const unsigned numChunks = numDims / 32;
  const __m256i kZero = _mm256_setzero_si256();
  const __m256i kOffsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  __m256i *in = (__m256i *)input;
  __m256i *out = (__m256i *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m256i words0 = _mm256_srai_epi16(_mm256_packs_epi32(
          in[i * 4 + 0], in[i * 4 + 1]), SHIFT);
    __m256i words1 = _mm256_srai_epi16(_mm256_packs_epi32(
          in[i * 4 + 2], in[i * 4 + 3]), SHIFT);
    out[i] = _mm256_permutevar8x32_epi32(_mm256_max_epi8(
          _mm256_packs_epi16(words0, words1), kZero), kOffsets);
  }

#elif defined(USE_SSSE3) || (defined(TRANSPOSE) && defined(USE_SSE2))
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
    __m128i packedbytes = _mm_packs_epi16(words0, words1);
#ifdef USE_SSE41
    out[i] = _mm_max_epi8(packedbytes, kZero);
#else
    out[i] = _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s);
#endif
  }

#elif defined(USE_SSE2)
  const unsigned numChunks = numDims / 8;
  const __m128i k0x7f80 = _mm_set1_epi16(0x7f80);
  const __m128i k0x0080 = _mm_set1_epi16(0x0080);
  const __m128i k0x8000 = _mm_set1_epi16(-0x8000);
  __m128i *in = (__m128i *)input;
  __m128i *out = (__m128i *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m128i words = _mm_srai_epi16(_mm_packs_epi32(in[i * 2], in[i * 2 + 1]),
        SHIFT);
    out[i] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(words, k0x7f80), k0x0080), k0x8000);
  }

#elif defined(USE_MMX)
  const unsigned numChunks = numDims / 4;
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  const __m64 k0x0080 = _mm_set1_pi16(0x0080);
  const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
  __m64 *in = (__m64 *)input;
  __m64 *out = (__m64 *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m64 words = _mm_srai_pi16(_mm_packs_pi32(in[i * 2], in[i * 2 + 1]),
        SHIFT);
    out[i] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(words, k0x7f80), k0x0080), k0x8000);
  }

#elif defined(USE_NEON)
  const unsigned numChunks = numDims / 8;
  const int8x8_t kZero = {0};
  int32x4_t *in = (int32x4_t *)input;
  int8x8_t *out = (int8x8_t *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    int16x8_t shifted;
    int16x4_t *pack = (int16x4_t *)&shifted;
    pack[0] = vqshrn_n_s32(in[i * 2 + 0], SHIFT);
    pack[1] = vqshrn_n_s32(in[i * 2 + 1], SHIFT);
    out[i] = vmax_s8(vqmovn_s16(shifted), kZero);
  }

#else
  for (unsigned i = 0; i < numDims; i++)
    output[i] = clamp(input[i] >> SHIFT, 0, 127);

#endif
}

#else /* TRANSPOSE */

static_assert(FtOutDims % 64 == 0, "FtOutDims not a multiple of 64");

INLINE bool next_idx(unsigned *idx, unsigned *offset, uint64_t *v,
    mask_t *mask, unsigned inDims)
{
  while (*v == 0) {
    *offset += 64;
    if (*offset >= inDims) return false;
    memcpy(v, (char *)mask + (*offset / 8), 8);
  }
  *idx = *offset + __builtin_ctzll(*v);
  *v &= *v - 1;
  return true;
}

#if defined(USE_AVX2)
INLINE void affine_txfm(int8_t *input, void *output, unsigned inDims,
    unsigned outDims, const int32_t *biases, weight_t *weights,
    mask_t *inMask, mask_t *outMask, const bool pack8_and_calc_mask)
{
  assert(outDims == 32);

  (void)outDims;
  const __m256i kZero = _mm256_setzero_si256();
  __m256i out_0 = ((__m256i *)biases)[0];
  __m256i out_1 = ((__m256i *)biases)[1];
  __m256i out_2 = ((__m256i *)biases)[2];
  __m256i out_3 = ((__m256i *)biases)[3];
  __m256i first, second;
  uint64_t v;
  unsigned idx;

  memcpy(&v, inMask, 8);
  for (unsigned offset = 0; offset < inDims;) {
    if (!next_idx(&idx, &offset, &v, inMask, inDims))
      break;
    first = ((__m256i *)weights)[idx];
    uint16_t factor = (uint8_t)input[idx];
    if (next_idx(&idx, &offset, &v, inMask, inDims)) {
      second = ((__m256i *)weights)[idx];
      factor |= input[idx] << 8;
    } else {
      second = kZero;
    }
    __m256i mul = _mm256_set1_epi16(factor), prod, signs;
    prod = _mm256_maddubs_epi16(mul, _mm256_unpacklo_epi8(first, second));
    signs = _mm256_cmpgt_epi16(kZero, prod);
    out_0 = _mm256_add_epi32(out_0, _mm256_unpacklo_epi16(prod, signs));
    out_1 = _mm256_add_epi32(out_1, _mm256_unpackhi_epi16(prod, signs));
    prod = _mm256_maddubs_epi16(mul, _mm256_unpackhi_epi8(first, second));
    signs = _mm256_cmpgt_epi16(kZero, prod);
    out_2 = _mm256_add_epi32(out_2, _mm256_unpacklo_epi16(prod, signs));
    out_3 = _mm256_add_epi32(out_3, _mm256_unpackhi_epi16(prod, signs));
  }

  __m256i out16_0 = _mm256_srai_epi16(_mm256_packs_epi32(out_0, out_1), SHIFT);
  __m256i out16_1 = _mm256_srai_epi16(_mm256_packs_epi32(out_2, out_3), SHIFT);

  __m256i *outVec = (__m256i *)output;
  if (pack8_and_calc_mask) {
    outVec[0] = _mm256_packs_epi16(out16_0, out16_1);
    outMask[0] = _mm256_movemask_epi8(_mm256_cmpgt_epi8(outVec[0], kZero));
  } else {
    outVec[0] = _mm256_max_epi8(_mm256_packs_epi16(out16_0, out16_1), kZero);
  }
}
#elif AVOID_USE_SSSE3
INLINE void affine_txfm(int8_t *input, void *output, unsigned inDims,
    unsigned outDims, const int32_t *biases, weight_t *weights,
    mask_t *inMask, mask_t *outMask, const bool pack8_and_calc_mask)
{
  assert(outDims == 32);

  const __m128i kZeros[2] = { 0 };
  __m128i out_0 = ((__m128i *)biases)[0];
  __m128i out_1 = ((__m128i *)biases)[1];
  __m128i out_2 = ((__m128i *)biases)[2];
  __m128i out_3 = ((__m128i *)biases)[3];
  __m128i out_4 = ((__m128i *)biases)[4];
  __m128i out_5 = ((__m128i *)biases)[5];
  __m128i out_6 = ((__m128i *)biases)[6];
  __m128i out_7 = ((__m128i *)biases)[7];
  const __m128i *first, *second;
  uint64_t v;
  unsigned idx;

  memcpy(&v, inMask, 8);
  for (unsigned offset = 0; offset < inDims;) {
    if (!next_idx(&idx, &offset, &v, inMask, inDims))
      break;
    first = (__m128i *)&weights[outDims * idx];
    uint16_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, inDims)) {
      second = (__m128i *)&weights[outDims * idx];
      factor |= input[idx] << 8;
    } else {
      second = kZeros;
    }
    __m128i mul = _mm_set1_epi16(factor), prod;
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[0], second[0]));
#if defined(USE_SSE41)
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[0], second[0]));
    out_0 = _mm_add_epi32(out_0, _mm_cvtepi16_epi32(prod));
    out_1 = _mm_add_epi32(out_1, _mm_cvtepi16_epi32(_mm_shuffle_epi32(prod, 0xE)));
    prod = _mm_maddubs_epi16(mul, _mm_unpackhi_epi8(first[0], second[0]));
    out_2 = _mm_add_epi32(out_2, _mm_cvtepi16_epi32(prod));
    out_3 = _mm_add_epi32(out_3, _mm_cvtepi16_epi32(_mm_shuffle_epi32(prod, 0xE)));
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[1], second[1]));
    out_4 = _mm_add_epi32(out_4, _mm_cvtepi16_epi32(prod));
    out_5 = _mm_add_epi32(out_5, _mm_cvtepi16_epi32(_mm_shuffle_epi32(prod, 0xE)));
    prod = _mm_maddubs_epi16(mul, _mm_unpackhi_epi8(first[1], second[1]));
    out_6 = _mm_add_epi32(out_6, _mm_cvtepi16_epi32(prod));
    out_7 = _mm_add_epi32(out_7, _mm_cvtepi16_epi32(_mm_shuffle_epi32(prod, 0xE)));
#else
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[0], second[0]));
    out_0 = _mm_add_epi32(out_0, _mm_srai_epi32(_mm_unpacklo_epi16(prod, prod), 16));
    out_1 = _mm_add_epi32(out_1, _mm_srai_epi32(_mm_unpackhi_epi16(prod, prod), 16));
    prod = _mm_maddubs_epi16(mul, _mm_unpackhi_epi8(first[0], second[0]));
    out_2 = _mm_add_epi32(out_2, _mm_srai_epi32(_mm_unpacklo_epi16(prod, prod), 16));
    out_3 = _mm_add_epi32(out_3, _mm_srai_epi32(_mm_unpackhi_epi16(prod, prod), 16));
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[1], second[1]));
    out_4 = _mm_add_epi32(out_4, _mm_srai_epi32(_mm_unpacklo_epi16(prod, prod), 16));
    out_5 = _mm_add_epi32(out_5, _mm_srai_epi32(_mm_unpackhi_epi16(prod, prod), 16));
    prod = _mm_maddubs_epi16(mul, _mm_unpackhi_epi8(first[1], second[1]));
    out_6 = _mm_add_epi32(out_6, _mm_srai_epi32(_mm_unpacklo_epi16(prod, prod), 16));
    out_7 = _mm_add_epi32(out_7, _mm_srai_epi32(_mm_unpackhi_epi16(prod, prod), 16));
#endif
  }

  __m128i out16_0 = _mm_srai_epi16(_mm_packs_epi32(out_0, out_1), SHIFT);
  __m128i out16_1 = _mm_srai_epi16(_mm_packs_epi32(out_2, out_3), SHIFT);
  __m128i out16_2 = _mm_srai_epi16(_mm_packs_epi32(out_4, out_5), SHIFT);
  __m128i out16_3 = _mm_srai_epi16(_mm_packs_epi32(out_6, out_7), SHIFT);

  __m128i *outVec = (__m128i *)output;
  if (pack8_and_calc_mask) {
    outVec[0] = _mm_packs_epi16(out16_0, out16_1);
    outMask[0] = _mm_movemask_epi8(_mm_cmpgt_epi8(outVec[0], kZeros[0]));
    outVec[1] = _mm_packs_epi16(out16_2, out16_3);
    outMask[1] = _mm_movemask_epi8(_mm_cmpgt_epi8(outVec[1], kZeros[0]));
  } else {
#if defined(USE_SSE41)
    outVec[0] = _mm_max_epi8(_mm_packs_epi16(out16_0, out16_1), kZeros[0]);
    outVec[1] = _mm_max_epi8(_mm_packs_epi16(out16_2, out16_3), kZeros[0]);
#else
    const __m128i k0x80s = _mm_set1_epi8(-128);
    outVec[0] = _mm_subs_epi8(_mm_adds_epi8(_mm_packs_epi16(out16_0, out16_1), k0x80s), k0x80s);
    outVec[1] = _mm_subs_epi8(_mm_adds_epi8(_mm_packs_epi16(out16_2, out16_3), k0x80s), k0x80s);
#endif
  }
}
#elif defined(USE_SSE2)
INLINE void affine_txfm(clipped_t *input, void *output, unsigned inDims,
    unsigned outDims, const int32_t *biases, weight_t *weights,
    mask_t *inMask, mask_t *outMask, const bool pack8_and_calc_mask)
{
  assert(outDims == 32);

  const __m128i kZeros[4] = { 0 };
  __m128i out_0 = ((__m128i *)biases)[0];
  __m128i out_1 = ((__m128i *)biases)[1];
  __m128i out_2 = ((__m128i *)biases)[2];
  __m128i out_3 = ((__m128i *)biases)[3];
  __m128i out_4 = ((__m128i *)biases)[4];
  __m128i out_5 = ((__m128i *)biases)[5];
  __m128i out_6 = ((__m128i *)biases)[6];
  __m128i out_7 = ((__m128i *)biases)[7];
  const __m128i *first, *second;
  uint64_t v;
  unsigned idx;

  memcpy(&v, inMask, 8);
  for (unsigned offset = 0; offset < inDims;) {
    if (!next_idx(&idx, &offset, &v, inMask, inDims))
      break;
    first = (__m128i *)&weights[outDims * idx];
    uint32_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, inDims)) {
      second = (__m128i *)&weights[outDims * idx];
      factor |= input[idx] << 16;
    } else {
      second = kZeros;
    }
    __m128i mul = _mm_set1_epi32(factor);
    out_0 = _mm_add_epi32(out_0, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[0],second[0])));
    out_1 = _mm_add_epi32(out_1, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[0],second[0])));
    out_2 = _mm_add_epi32(out_2, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[1],second[1])));
    out_3 = _mm_add_epi32(out_3, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[1],second[1])));
    out_4 = _mm_add_epi32(out_4, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[2],second[2])));
    out_5 = _mm_add_epi32(out_5, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[2],second[2])));
    out_6 = _mm_add_epi32(out_6, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[3],second[3])));
    out_7 = _mm_add_epi32(out_7, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[3],second[3])));
  }

  __m128i out16_0 = _mm_srai_epi16(_mm_packs_epi32(out_0, out_1), SHIFT);
  __m128i out16_1 = _mm_srai_epi16(_mm_packs_epi32(out_2, out_3), SHIFT);
  __m128i out16_2 = _mm_srai_epi16(_mm_packs_epi32(out_4, out_5), SHIFT);
  __m128i out16_3 = _mm_srai_epi16(_mm_packs_epi32(out_6, out_7), SHIFT);

  __m128i *outVec = (__m128i *)output;
  if (pack8_and_calc_mask) {
    outVec[0] = _mm_packs_epi16(out16_0, out16_1);
    outMask[0] = _mm_movemask_epi8(_mm_cmpgt_epi8(outVec[0], kZeros[0]));
    outVec[1] = _mm_packs_epi16(out16_2, out16_3);
    outMask[1] = _mm_movemask_epi8(_mm_cmpgt_epi8(outVec[1], kZeros[0]));
  } else {
#if defined(USE_SSE41)
    const __m128i kx07f = _mm_set1_epi16(127);
    outVec[0] = _mm_min_epi16(_mm_max_epi16(out16_0, kZeros[0]), kx07f);
    outVec[1] = _mm_min_epi16(_mm_max_epi16(out16_1, kZeros[0]), kx07f);
    outVec[2] = _mm_min_epi16(_mm_max_epi16(out16_2, kZeros[0]), kx07f);
    outVec[3] = _mm_min_epi16(_mm_max_epi16(out16_3, kZeros[0]), kx07f);
#else
    const __m128i k0x7f80 = _mm_set1_epi16(0x7f80);
    const __m128i k0x0080 = _mm_set1_epi16(0x0080);
    const __m128i k0x8000 = _mm_set1_epi16(-0x8000);
    outVec[0] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(out16_0, k0x7f80), k0x0080), k0x8000);
    outVec[1] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(out16_1, k0x7f80), k0x0080), k0x8000);
    outVec[2] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(out16_2, k0x7f80), k0x0080), k0x8000);
    outVec[3] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(out16_3, k0x7f80), k0x0080), k0x8000);
#endif
  }
}
#elif defined(USE_MMX) && defined(USE_SSE)
INLINE void affine_txfm(clipped_t *input, void *output, unsigned inDims,
    unsigned outDims, const int32_t *biases, weight_t *weights,
    mask_t *inMask, mask_t *outMask, const bool pack8_and_calc_mask)
{
  assert(outDims == 32);

  const __m64 kZeros[8] = { 0 };
  __m64 out_0 = ((__m64 *)biases)[0];
  __m64 out_1 = ((__m64 *)biases)[1];
  __m64 out_2 = ((__m64 *)biases)[2];
  __m64 out_3 = ((__m64 *)biases)[3];
  __m64 out_4 = ((__m64 *)biases)[4];
  __m64 out_5 = ((__m64 *)biases)[5];
  __m64 out_6 = ((__m64 *)biases)[6];
  __m64 out_7 = ((__m64 *)biases)[7];
  __m64 out_8 = ((__m64 *)biases)[8];
  __m64 out_9 = ((__m64 *)biases)[9];
  __m64 out_10 = ((__m64 *)biases)[10];
  __m64 out_11 = ((__m64 *)biases)[11];
  __m64 out_12 = ((__m64 *)biases)[12];
  __m64 out_13 = ((__m64 *)biases)[13];
  __m64 out_14 = ((__m64 *)biases)[14];
  __m64 out_15 = ((__m64 *)biases)[15];
  const __m64 *first, *second;
  uint64_t v;
  unsigned idx;

  memcpy(&v, inMask, 8);
  for (unsigned offset = 0; offset < inDims;) {
    if (!next_idx(&idx, &offset, &v, inMask, inDims))
      break;
    first = (__m64 *)&weights[outDims * idx];
    uint32_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, inDims)) {
      second = (__m64 *)&weights[outDims * idx];
      factor |= input[idx] << 16;
    } else {
      second = kZeros;
    }
    __m64 mul = _mm_set1_pi32(factor);
    out_0 = _mm_add_pi32(out_0, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[0],second[0])));
    out_1 = _mm_add_pi32(out_1, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[0],second[0])));
    out_2 = _mm_add_pi32(out_2, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[1],second[1])));
    out_3 = _mm_add_pi32(out_3, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[1],second[1])));
    out_4 = _mm_add_pi32(out_4, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[2],second[2])));
    out_5 = _mm_add_pi32(out_5, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[2],second[2])));
    out_6 = _mm_add_pi32(out_6, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[3],second[3])));
    out_7 = _mm_add_pi32(out_7, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[3],second[3])));
    out_8 = _mm_add_pi32(out_8, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[4],second[4])));
    out_9 = _mm_add_pi32(out_9, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[4],second[4])));
    out_10 = _mm_add_pi32(out_10, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[5],second[5])));
    out_11 = _mm_add_pi32(out_11, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[5],second[5])));
    out_12 = _mm_add_pi32(out_12, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[6],second[6])));
    out_13 = _mm_add_pi32(out_13, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[6],second[6])));
    out_14 = _mm_add_pi32(out_14, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[7],second[7])));
    out_15 = _mm_add_pi32(out_15, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[7],second[7])));
  }

  __m64 out16_0 = _mm_srai_pi16(_mm_packs_pi32(out_0, out_1), SHIFT);
  __m64 out16_1 = _mm_srai_pi16(_mm_packs_pi32(out_2, out_3), SHIFT);
  __m64 out16_2 = _mm_srai_pi16(_mm_packs_pi32(out_4, out_5), SHIFT);
  __m64 out16_3 = _mm_srai_pi16(_mm_packs_pi32(out_6, out_7), SHIFT);
  __m64 out16_4 = _mm_srai_pi16(_mm_packs_pi32(out_8, out_9), SHIFT);
  __m64 out16_5 = _mm_srai_pi16(_mm_packs_pi32(out_10, out_11), SHIFT);
  __m64 out16_6 = _mm_srai_pi16(_mm_packs_pi32(out_12, out_13), SHIFT);
  __m64 out16_7 = _mm_srai_pi16(_mm_packs_pi32(out_14, out_15), SHIFT);

  __m64 *outVec = (__m64 *)output;
  if (pack8_and_calc_mask) {
    outVec[0] = _mm_packs_pi16(out16_0, out16_1);
    outMask[0] = _mm_movemask_pi8(_mm_cmpgt_pi8(outVec[0], kZeros[0]));
    outVec[1] = _mm_packs_pi16(out16_2, out16_3);
    outMask[1] = _mm_movemask_pi8(_mm_cmpgt_pi8(outVec[1], kZeros[0]));
    outVec[2] = _mm_packs_pi16(out16_4, out16_5);
    outMask[2] = _mm_movemask_pi8(_mm_cmpgt_pi8(outVec[2], kZeros[0]));
    outVec[3] = _mm_packs_pi16(out16_6, out16_7);
    outMask[3] = _mm_movemask_pi8(_mm_cmpgt_pi8(outVec[3], kZeros[0]));
  } else {
    const __m64 kx07f = _mm_set1_pi16(127);
    outVec[0] = _mm_min_pi16(_mm_max_pi16(out16_0, kZeros[0]), kx07f);
    outVec[1] = _mm_min_pi16(_mm_max_pi16(out16_1, kZeros[0]), kx07f);
    outVec[2] = _mm_min_pi16(_mm_max_pi16(out16_2, kZeros[0]), kx07f);
    outVec[3] = _mm_min_pi16(_mm_max_pi16(out16_3, kZeros[0]), kx07f);
    outVec[4] = _mm_min_pi16(_mm_max_pi16(out16_4, kZeros[0]), kx07f);
    outVec[5] = _mm_min_pi16(_mm_max_pi16(out16_5, kZeros[0]), kx07f);
    outVec[6] = _mm_min_pi16(_mm_max_pi16(out16_6, kZeros[0]), kx07f);
    outVec[7] = _mm_min_pi16(_mm_max_pi16(out16_7, kZeros[0]), kx07f);
  }
}
#elif defined(USE_MMX)
INLINE void affine_txfm(clipped_t *input, void *output, unsigned inDims,
    unsigned outDims, const int32_t *biases, weight_t *weights,
    mask_t *inMask, mask_t *outMask, const bool pack8_and_calc_mask)
{
  assert(outDims == 32);

  (void)inMask; (void)outMask;
  const __m64 kZeros[8] = { 0 };
  __m64 out_0 = ((__m64 *)biases)[0];
  __m64 out_1 = ((__m64 *)biases)[1];
  __m64 out_2 = ((__m64 *)biases)[2];
  __m64 out_3 = ((__m64 *)biases)[3];
  __m64 out_4 = ((__m64 *)biases)[4];
  __m64 out_5 = ((__m64 *)biases)[5];
  __m64 out_6 = ((__m64 *)biases)[6];
  __m64 out_7 = ((__m64 *)biases)[7];
  __m64 out_8 = ((__m64 *)biases)[8];
  __m64 out_9 = ((__m64 *)biases)[9];
  __m64 out_10 = ((__m64 *)biases)[10];
  __m64 out_11 = ((__m64 *)biases)[11];
  __m64 out_12 = ((__m64 *)biases)[12];
  __m64 out_13 = ((__m64 *)biases)[13];
  __m64 out_14 = ((__m64 *)biases)[14];
  __m64 out_15 = ((__m64 *)biases)[15];
  const __m64 *first, *second;

  for (unsigned idx = 0; idx < inDims; idx++) {
    if (input[idx] <= 0)
      continue;
    uint32_t factor = input[idx];
    first = (__m64 *)&weights[outDims * idx];
    while (++idx < inDims && input[idx] <= 0);
    if (idx < inDims) {
      second = (__m64 *)&weights[outDims * idx];
      factor |= input[idx] << 16;
    } else
      second = kZeros;
    __m64 mul = _mm_set1_pi32(factor);
    out_0 = _mm_add_pi32(out_0, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[0],second[0])));
    out_1 = _mm_add_pi32(out_1, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[0],second[0])));
    out_2 = _mm_add_pi32(out_2, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[1],second[1])));
    out_3 = _mm_add_pi32(out_3, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[1],second[1])));
    out_4 = _mm_add_pi32(out_4, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[2],second[2])));
    out_5 = _mm_add_pi32(out_5, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[2],second[2])));
    out_6 = _mm_add_pi32(out_6, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[3],second[3])));
    out_7 = _mm_add_pi32(out_7, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[3],second[3])));
    out_8 = _mm_add_pi32(out_8, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[4],second[4])));
    out_9 = _mm_add_pi32(out_9, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[4],second[4])));
    out_10 = _mm_add_pi32(out_10, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[5],second[5])));
    out_11 = _mm_add_pi32(out_11, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[5],second[5])));
    out_12 = _mm_add_pi32(out_12, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[6],second[6])));
    out_13 = _mm_add_pi32(out_13, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[6],second[6])));
    out_14 = _mm_add_pi32(out_14, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[7],second[7])));
    out_15 = _mm_add_pi32(out_15, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[7],second[7])));
  }

  __m64 out16_0 = _mm_srai_pi16(_mm_packs_pi32(out_0, out_1), SHIFT);
  __m64 out16_1 = _mm_srai_pi16(_mm_packs_pi32(out_2, out_3), SHIFT);
  __m64 out16_2 = _mm_srai_pi16(_mm_packs_pi32(out_4, out_5), SHIFT);
  __m64 out16_3 = _mm_srai_pi16(_mm_packs_pi32(out_6, out_7), SHIFT);
  __m64 out16_4 = _mm_srai_pi16(_mm_packs_pi32(out_8, out_9), SHIFT);
  __m64 out16_5 = _mm_srai_pi16(_mm_packs_pi32(out_10, out_11), SHIFT);
  __m64 out16_6 = _mm_srai_pi16(_mm_packs_pi32(out_12, out_13), SHIFT);
  __m64 out16_7 = _mm_srai_pi16(_mm_packs_pi32(out_14, out_15), SHIFT);

  __m64 *outVec = (__m64 *)output;
  if (pack8_and_calc_mask) {
    outVec[0] = _mm_packs_pi16(out16_0, out16_1);
    outVec[1] = _mm_packs_pi16(out16_2, out16_3);
    outVec[2] = _mm_packs_pi16(out16_4, out16_5);
    outVec[3] = _mm_packs_pi16(out16_6, out16_7);
  } else {
    const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
    const __m64 k0x0080 = _mm_set1_pi16(0x0080);
    const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
    outVec[0] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_0, k0x7f80), k0x0080), k0x8000);
    outVec[1] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_1, k0x7f80), k0x0080), k0x8000);
    outVec[2] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_2, k0x7f80), k0x0080), k0x8000);
    outVec[3] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_3, k0x7f80), k0x0080), k0x8000);
    outVec[4] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_4, k0x7f80), k0x0080), k0x8000);
    outVec[5] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_5, k0x7f80), k0x0080), k0x8000);
    outVec[6] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_6, k0x7f80), k0x0080), k0x8000);
    outVec[7] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_7, k0x7f80), k0x0080), k0x8000);
  }
}
#else /* generic fallback */
INLINE void affine_txfm(clipped_t *input, void *output, unsigned inDims,
    unsigned outDims, int32_t *biases, weight_t *weights,
    mask_t *inMask, mask_t *outMask,
    const bool pack8_and_calc_mask)
{
  (void)inMask; (void)outMask; (void)pack8_and_calc_mask;

  int32_t tmp[outDims];

  for (unsigned i = 0; i < outDims; i++)
    tmp[i] = biases[i];

  for (unsigned idx = 0; idx < inDims; idx++)
    if (input[idx])
      for (unsigned i = 0; i < outDims; i++)
        tmp[i] += (int8_t)input[idx] * weights[outDims * idx + i];

  clipped_t *outVec = (clipped_t *)output;
  for (unsigned i = 0; i < outDims; i++)
    outVec[i] = clamp(tmp[i] >> SHIFT, 0, 127);
}
#endif

#endif

// Input feature converter
static alignas(64) int16_t ft_biases[kHalfDimensions];
static alignas(64) int16_t ft_weights[kHalfDimensions * FtInDims];

#if defined(USE_SSE2) || defined(USE_MMX)
#define TILE_HEIGHT (NUM_REGS * SIMD_WIDTH / 16)
#else
#define TILE_HEIGHT kHalfDimensions
#endif

// Calculate cumulative value without using difference calculation
INLINE void refresh_accumulator(const Position *pos)
{
  Accumulator *accumulator = &(pos->st->accumulator);

  IndexList activeIndices[2];
  activeIndices[0].size = activeIndices[1].size = 0;
  append_active_indices(pos, activeIndices);

  for (unsigned c = 0; c < 2; c++) {
    for (unsigned i = 0; i < kHalfDimensions / TILE_HEIGHT; i++) {
#if defined(USE_SSE2) || defined(USE_MMX)
      vec_t *ft_biases_tile = (vec_t *)&ft_biases[i * TILE_HEIGHT];
      vec_t *accTile = (vec_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
      vec_t acc[NUM_REGS];
      for (unsigned j = 0; j < NUM_REGS; j++)
        acc[j] = ft_biases_tile[j];

#else
      memcpy(&(accumulator->accumulation[c][i * TILE_HEIGHT]), 
          &ft_biases[i * TILE_HEIGHT], TILE_HEIGHT * sizeof(int16_t));

#endif
      for (size_t k = 0; k < activeIndices[c].size; k++) {
        unsigned index = activeIndices[c].values[k];
        unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;

#if defined(USE_SSE2) || defined(USE_MMX)
        vec_t *column = (vec_t *)&ft_weights[offset];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = vec_add_16(acc[j], column[j]);

#elif defined(USE_NEON)
        int16x8_t *accumulation = (int16x8_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
        int16x8_t *column = (int16x8_t *)&ft_weights[offset];
        const unsigned numChunks = kHalfDimensions / 8;
        for (unsigned j = 0; j < numChunks; j++)
          accumulation[j] = vaddq_s16(accumulation[j], column[j]);

#else
        for (unsigned j = 0; j < kHalfDimensions; j++)
          accumulator->accumulation[c][i * TILE_HEIGHT + j] += ft_weights[offset + j];

#endif
      }

#if defined(USE_SSE2) || defined(USE_MMX)
      for (unsigned j = 0; j < NUM_REGS; j++)
        accTile[j] = acc[j];

#endif
    }
  }

  accumulator->computedAccumulation = true;
}

// Calculate cumulative value using difference calculation if possible
INLINE bool update_accumulator_if_possible(const Position *pos)
{
  Accumulator *accumulator = &(pos->st->accumulator);
  if (accumulator->computedAccumulation)
    return true;

  Accumulator *prevAccumulator = &((pos->st-1)->accumulator);
  if (!prevAccumulator->computedAccumulation)
    return false;

  IndexList removed_indices[2], added_indices[2];
  removed_indices[0].size = removed_indices[1].size = 0;
  added_indices[0].size = added_indices[1].size = 0;
  bool reset[2];
  append_changed_indices(pos, removed_indices, added_indices, reset);
  for (unsigned i = 0; i< kHalfDimensions / TILE_HEIGHT; i++) {
    for (unsigned c = 0; c < 2; c++) {
#if defined(USE_SSE2) || defined(USE_MMX)
      vec_t *accTile = (vec_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
      vec_t acc[NUM_REGS];

#elif defined(USE_NEON)
      const unsigned numChunks = kHalfDimensions / 8;
      int16x8_t *accTile = (int16x8_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];

#endif

      if (reset[c]) {
#if defined(USE_SSE2) || defined(USE_MMX)
        vec_t *ft_b_tile = (vec_t *)&ft_biases[i * TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = ft_b_tile[j];
#else
        memcpy(&accumulator->accumulation[c][i * TILE_HEIGHT],
            &ft_biases[i * TILE_HEIGHT], TILE_HEIGHT * sizeof(int16_t));
#endif
      } else {
#if defined(USE_SSE2) || defined(USE_MMX)
        vec_t *prevAccTile = (vec_t *)&prevAccumulator->accumulation[c][i * TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = prevAccTile[j];
#else
        memcpy(&accumulator->accumulation[c][i * TILE_HEIGHT],
            &prevAccumulator->accumulation[c][i * TILE_HEIGHT],
            TILE_HEIGHT * sizeof(int16_t));
#endif
        // Difference calculation for the deactivated features
        for (unsigned k = 0; k < removed_indices[c].size; k++) {
          unsigned index = removed_indices[c].values[k];
          const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;

#if defined(USE_SSE2) || defined(USE_MMX)
          vec_t *column = (vec_t *)&ft_weights[offset];
          for (unsigned j = 0; j < NUM_REGS; j++)
            acc[j] = vec_sub_16(acc[j], column[j]);

#elif defined(USE_NEON)
          int16x8_t *column = (int16x8_t *)&ft_weights[offset];
          for (unsigned j = 0; j < numChunks; j++)
            accTile[j] = vsubq_s16(accTile[j], column[j]);

#else
          for (unsigned j = 0; j < kHalfDimensions; j++)
            accumulator->accumulation[c][i * TILE_HEIGHT + j] -= ft_weights[offset + j];

#endif
        }
      }

      // Difference calculation for the activated features
      for (unsigned k = 0; k < added_indices[c].size; k++) {
        unsigned index = added_indices[c].values[k];
        const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;

#if defined(USE_SSE2) || defined(USE_MMX)
        vec_t *column = (vec_t *)&ft_weights[offset];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = vec_add_16(acc[j], column[j]);

#elif defined(USE_NEON)
        int16x8_t *column = (int16x8_t *)&ft_weights[offset];
        for (unsigned j = 0; j < numChunks; j++)
          accTile[j] = vaddq_s16(accTile[j], column[j]);

#else
        for (unsigned j = 0; j < TILE_HEIGHT; j++)
          accumulator->accumulation[c][i * TILE_HEIGHT + j] += ft_weights[offset + j];

#endif
      }

#if defined(USE_SSE2) || defined(USE_MMX)
      for (unsigned j = 0; j < NUM_REGS; j++)
        accTile[j] = acc[j];

#endif
    }
  }

  accumulator->computedAccumulation = true;
  return true;
}

// Convert input features
INLINE void transform(const Position *pos, clipped_t *output,
    mask_t *outMask)
{
  if (!update_accumulator_if_possible(pos))
    refresh_accumulator(pos);

  int16_t (*accumulation)[2][256] = &pos->st->accumulator.accumulation;
  (void)outMask; // avoid compiler warning

  // Number of vectors to read
  const unsigned numChunks = (16 * kHalfDimensions) / SIMD_WIDTH;
#if defined(USE_AVX512)
  const __m512i kZero = _mm512_setzero_si512();
  const __m512i kOffsets = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

#elif defined(USE_AVX2)
  const __m256i kZero = _mm256_setzero_si256();

#elif defined(USE_SSE2)
#if defined(USE_SSE41) || defined(TRANSPOSE)
  const __m128i kZero = _mm_setzero_si128();
#else
#if !defined(USE_SSSE3)
  const __m128i k0x7f80 = _mm_set1_epi16(0x7f80);
  const __m128i k0x0080 = _mm_set1_epi16(0x0080);
  const __m128i k0x8000 = _mm_set1_epi16(-0x8000);
#else
  const __m128i k0x80s = _mm_set1_epi8(-128);
#endif
#endif

#elif defined(USE_MMX)
#ifndef TRANSPOSE
#ifdef USE_SSE
  const __m64 k0x7f = _mm_set1_pi16(127);
  const __m64 kZero = _mm_setzero_si64();
#else
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  const __m64 k0x0080 = _mm_set1_pi16(0x0080);
  const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
#endif
#elif USE_SSE
  const __m64 kZero = _mm_setzero_si64();
#endif

#elif defined(USE_NEON)
  const int8x8_t kZero = {0};

#endif

  const Color perspectives[2] = { stm(), !stm() };
  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#if defined(USE_AVX512)
    __m512i *out = (__m512i *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m512i sum0 = ((__m512i *)(*accumulation)[perspectives[p]])[i * 2 + 0];
      __m512i sum1 = ((__m512i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m512i packed = _mm512_packs_epi16(sum0, sum1);
#ifndef TRANSPOSE
      out[i] = _mm512_permutexvar_epi64(kOffsets, _mm512_max_epi8(packed, kZero));
#else
      out[i] = _mm512_permutexvar_epi64(kOffsets, packed);
      *outMask++ = _mm512_cmpgt_epi8_mask(out[i], kZero);
#endif
     }

#elif defined(USE_AVX2)
    __m256i *out = (__m256i *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m256i sum0 = ((__m256i *)(*accumulation)[perspectives[p]])[i * 2 + 0];
      __m256i sum1 = ((__m256i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m256i packed = _mm256_packs_epi16(sum0, sum1);
#ifndef TRANSPOSE
      out[i] = _mm256_permute4x64_epi64(_mm256_max_epi8(packed, kZero), 0xd8);
#else
      out[i] = _mm256_permute4x64_epi64(packed, 0xd8);
      *outMask++ = _mm256_movemask_epi8(_mm256_cmpgt_epi8(out[i], kZero));
#endif
    }

#elif defined(USE_SSE2)
    __m128i *out = (__m128i *)&output[offset];
#if defined(TRANSPOSE) || defined(USE_SSSE3)
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m128i sum0 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2];
      __m128i sum1 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m128i packed = _mm_packs_epi16(sum0, sum1);
#ifndef TRANSPOSE
#if defined(USE_SSE41)
      out[i] = _mm_max_epi8(packed, kZero);
#else
      out[i] = _mm_subs_epi8(_mm_adds_epi8(packed, k0x80s), k0x80s);
#endif
#else
      out[i] = packed;
      *outMask++ = _mm_movemask_epi8(_mm_cmpgt_epi8(packed, kZero));
#endif
    }
#else /* USE_SSE2 && !TRANSPOSE */
    for (unsigned i = 0; i < numChunks; i++) {
      __m128i sum = ((__m128i *)(*accumulation)[perspectives[p]])[i];
      out[i] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(sum, k0x7f80), k0x0080), k0x8000);
    }
#endif

#elif defined(USE_MMX)
    __m64 *out = (__m64 *)&output[offset];
#ifdef TRANSPOSE
    for (unsigned i = 0; i < numChunks / 2; i++) {
      __m64 sum0 = ((__m64 *)(*accumulation)[perspectives[p]])[i * 2];
      __m64 sum1 = ((__m64 *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      out[i] = _mm_packs_pi16(sum0, sum1);
#ifdef USE_MASK
      *outMask++ = _mm_movemask_pi8(_mm_cmpgt_pi8(out[i], kZero));
#endif
    }
#else /* !TRANSPOSE */
    for (unsigned i = 0; i < numChunks; i++) {
      __m64 sum = ((__m64 *)(*accumulation)[perspectives[p]])[i];
#ifdef USE_SSE
      out[i] = _mm_min_pi16(_mm_max_pi16(sum, kZero), k0x7f);
#else
      out[i] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(sum, k0x7f80), k0x0080), k0x8000);
#endif
    }
#endif

#elif defined(USE_NEON)
    int8x8_t *out = (int8x8_t *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      int16x8_t sum = ((int16x8_t *)(*accumulation)[perspectives[p]])[i];
      out[i] = vmax_s8(vqmovn_s16(sum), kZero);
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
#ifndef TRANSPOSE
  int32_t hidden1_values[32];
  int32_t hidden2_values[32];
  clipped_t hidden1_clipped[32];
  clipped_t hidden2_clipped[32];
#else
  clipped_t hidden1_out[32];
#if (defined(USE_SSE2) || defined(USE_MMX)) && !defined(USE_AVX2)
  int16_t hidden2_out[32];
#else
  int8_t hidden2_out[32];
#endif
#endif
};

// Evaluation function
Value nnue_evaluate(const Position *pos)
{
  int32_t out_value;
  alignas(8) mask_t input_mask[FtOutDims / (8 * sizeof(mask_t))];
#ifdef TRANSPOSE
  alignas(8) mask_t hidden1_mask[8 / sizeof(mask_t)] = { 0 };
#endif
#ifdef ALIGNMENT_HACK // work around a bug in old gcc on Windows
  uint8_t buf[sizeof(struct NetData) + 63];
  struct NetData *b = (struct NetData *)(buf + ((((uintptr_t)buf-1) ^ 0x3f) & 0x3f));
#define B(x) (b->x)
#else
  struct NetData buf;
#define B(x) (buf.x)
#endif

  transform(pos, B(input), input_mask);

#ifndef TRANSPOSE

  affine_propagate(B(input), B(hidden1_values), FtOutDims, 32,
      hidden1_biases, hidden1_weights);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 32);
  affine_propagate(B(hidden1_clipped), B(hidden2_values), 32, 32,
      hidden2_biases, hidden2_weights);
  clip_propagate(B(hidden2_values), B(hidden2_clipped), 32);
  affine_propagate(B(hidden2_clipped), &out_value, 32, 1, output_biases,
      output_weights);

#else

  affine_txfm(B(input), B(hidden1_out), FtOutDims, 32,
      hidden1_biases, hidden1_weights, input_mask, hidden1_mask, true);

  affine_txfm(B(hidden1_out), B(hidden2_out), 32, 32,
      hidden2_biases, hidden2_weights, hidden1_mask, NULL, false);

  affine_propagate((int8_t *)B(hidden2_out), &out_value, 32, 1, output_biases,
      output_weights);

#endif

#if defined(USE_MMX)
  _mm_empty();
#endif

  return out_value / FV_SCALE;
}

const char *read_weights(weight_t *w, unsigned width, unsigned height,
    const char *d)
{
  for (unsigned i = 0; i < height; i++)
    for (unsigned j = 0; j < width; j++) {
#ifndef TRANSPOSE
      w[i * width + j] = *d++;
#else
      w[j * height + i] = *d++;
#endif
    }

  return d;
}

#if defined(TRANSPOSE) && defined(USE_AVX2)
void permute_biases(int32_t *biases)
{
  __m128i *b = (__m128i *)biases;
  __m128i tmp[8];
  tmp[0] = b[0];
  tmp[1] = b[4];
  tmp[2] = b[1];
  tmp[3] = b[5];
  tmp[4] = b[2];
  tmp[5] = b[6];
  tmp[6] = b[3];
  tmp[7] = b[7];
  memcpy(b, tmp, 8 * sizeof(__m128i));
}
#endif

static const size_t TransformerStart = 3 * 4 + 177;
static const size_t NetworkStart =
    TransformerStart + 4 + 2 * 256 + 2 * 256 * 64 * 641;

bool verify_net(const void *evalData, size_t size)
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

void init_weights(const void *evalData)
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
  d = read_weights(hidden1_weights, 512, 32, d);
  for (unsigned i = 0; i < 32; i++, d += 4)
    hidden2_biases[i] = readu_le_u32(d);
  d = read_weights(hidden2_weights, 32, 32, d);
  for (unsigned i = 0; i < 1; i++, d += 4)
    output_biases[i] = readu_le_u32(d);
  read_weights(output_weights, 32, 1, d);

#if defined(TRANSPOSE) && defined(USE_AVX2)
  permute_biases(hidden1_biases);
  permute_biases(hidden2_biases);
#endif
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
  if (mapping) unmap_file((void *)evalData, mapping);
  return success;
}

static char *loadedFile = NULL;

void nnue_init(void)
{
  const char *s = option_string_value(OPT_USE_NNUE);
  useNNUE =  strcmp(s, "classical") == 0 ? EVAL_CLASSICAL
           : strcmp(s, "pure"     ) == 0 ? EVAL_PURE : EVAL_HYBRID;

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

// Incrementally update the accumulator if possible
void update_eval(const Position *pos)
{
  update_accumulator_if_possible(pos);

#ifdef USE_MMX
  _mm_empty();
#endif
}
