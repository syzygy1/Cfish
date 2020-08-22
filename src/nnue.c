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

ExtPieceSquare KppBoardIndex[] = {
  // convention: W - us, B - them
  // viewed from other side, W and B are reversed
  { PS_NONE,     PS_NONE     },
  { PS_W_PAWN,   PS_B_PAWN   },
  { PS_W_KNIGHT, PS_B_KNIGHT },
  { PS_W_BISHOP, PS_B_BISHOP },
  { PS_W_ROOK,   PS_B_ROOK   },
  { PS_W_QUEEN,  PS_B_QUEEN  },
  { PS_W_KING,   PS_B_KING   },
  { PS_NONE,     PS_NONE     },
  { PS_NONE,     PS_NONE     },
  { PS_B_PAWN,   PS_W_PAWN   },
  { PS_B_KNIGHT, PS_W_KNIGHT },
  { PS_B_BISHOP, PS_W_BISHOP },
  { PS_B_ROOK,   PS_W_ROOK   },
  { PS_B_QUEEN,  PS_W_QUEEN  },
  { PS_B_KING,   PS_W_KING   },
  { PS_NONE,     PS_NONE     }
};

#if defined(USE_SSE2)
#undef USE_MMX
#endif

#if !defined(USE_SSSE3) && (defined(USE_SSE41) || defined(USE_AVX2) || defined(USE_AVX512))
#error "USE_SSSE3 must be defined when USE_SSE41 or USE_AVX* is"
#endif

#if (defined(USE_SSE2) || defined(USE_MMX)) && !defined(USE_SSSE3)
typedef int16_t clipped_t; //SSE2 and MMX have no int8 multiply.
typedef int16_t weight_t;
#else
typedef uint8_t clipped_t;
typedef int8_t weight_t;
#endif

// Version of the evaluation file
static const uint32_t NnueVersion = 0x7AF32F16u;

// Constants used in evaluation value calculation
enum {
  FV_SCALE = 16,
  SHIFT = 6
};

enum {
  kTransformedFeatureDimensions = 256,
  kDimensions = 64 * PS_END, // HalfKP
  kMaxActiveDimensions = PIECE_ID_KING,
  kHalfDimensions = kTransformedFeatureDimensions,
  FtInDims = kDimensions,
  FtOutDims = kHalfDimensions * 2,
};

static uint32_t read_uint32_t(FILE *F)
{
  uint32_t v;
  fread(&v, 4, 1, F);
  return from_le_u32(v);
}

typedef struct {
  size_t size;
  unsigned values[kMaxActiveDimensions];
} IndexList;

INLINE unsigned make_index(Square sq, PieceSquare p)
{
  return PS_END * sq + p;
}

INLINE void get_pieces(const Position *pos, Color c, const PieceSquare **pcs,
    Square *sq)
{
  *pcs = c == WHITE ? pos->pieceListFw : pos->pieceListFb;
  PieceId target = PIECE_ID_KING + c;
  *sq = ((*pcs)[target] - PS_W_KING) & 0x3f;
}

static void half_kp_append_active_indices(const Position *pos, Color c,
    IndexList *active)
{
  const PieceSquare *pcs;
  Square sq;
  get_pieces(pos, c, &pcs, &sq);
  for (PieceId i = PIECE_ID_ZERO; i < PIECE_ID_KING; i++)
    if (pcs[i] != PS_NONE)
      active->values[active->size++] = make_index(sq, pcs[i]);
}

static void half_kp_append_changed_indices(const Position *pos, Color c,
    IndexList *removed, IndexList *added)
{
  const PieceSquare *pcs;
  Square sq;
  get_pieces(pos, c, &pcs, &sq);
  const DirtyPiece *dp = &(pos->st->dirtyPiece);
  for (int i = 0; i < dp->dirtyNum; i++) {
    if (dp->pieceId[i] >= PIECE_ID_KING) continue;
    PieceSquare old_p = dp->oldPiece[i][c];
    if (old_p != PS_NONE)
      removed->values[removed->size++] = make_index(sq, old_p);
    PieceSquare new_p = dp->newPiece[i][c];
    if (new_p != PS_NONE)
      added->values[added->size++] = make_index(sq, new_p);
  }
}

static void append_active_indices(const Position *pos, IndexList active[2])
{
  for (unsigned c = 0; c < 2; c++)
    half_kp_append_active_indices(pos, c, &(active[c]));
}

static void append_changed_indices(const Position *pos, IndexList removed[2],
    IndexList added[2], bool reset[2])
{
  const DirtyPiece *dp = &(pos->st->dirtyPiece);
  if (dp->dirtyNum == 0) return;

  for (unsigned c = 0; c < 2; c++) {
    reset[c] = dp->pieceId[0] == PIECE_ID_KING + c;
    if (reset[c])
      half_kp_append_active_indices(pos, c, &(added[c]));
    else
      half_kp_append_changed_indices(pos, c, &(removed[c]), &(added[c]));
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

static int32_t hidden1_biases[32];
static int32_t hidden2_biases[32];
static int32_t output_biases [1];

INLINE void affine_propagate(clipped_t *input, int32_t *output, unsigned inDims,
    unsigned outDims, int32_t *biases, weight_t *weights)
{
  assert(inDims % 32 == 0);

#if defined(USE_AVX512)
  const unsigned numChunks = inDims / 64;
#if !defined(USE_VNNI)
  const __m512i kOnes = _mm512_set1_epi16(1);
#endif
  __m512i *inVec = (__m512i *)input;

#elif defined(USE_AVX2)
  const unsigned numChunks = inDims / 32;
  __m256i *inVec = (__m256i *)input;
#if !defined(USE_VNNI)
  const __m256i kOnes = _mm256_set1_epi16(1);
#endif

#elif defined(USE_SSSE3)
  const unsigned numChunks = inDims / 16;
  const __m128i kOnes = _mm_set1_epi16(1);
  __m128i *inVec = (__m128i *)input;

#elif defined(USE_SSE2)
  const unsigned numChunks = inDims / 16;
  __m128i *inVec = (__m128i *)input;

#elif defined(USE_MMX)
  const unsigned numChunks = inDims / 8;
  __m64 *inVec = (__m64 *)input;

#elif defined(USE_NEON)
  const unsigned numChunks = inDims / 16;
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

    // Note: Changing kMaxSimdWidth from 32 to 64 breaks loading existing
    // networks. As a result kPaddedInputDimensions may not be an even
    // multiple of 64 bytes/512 bits and we have to do one more 256-bit chunk.
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
    for (unsigned j = 0; j < numChunks - 1; j += 2) {
      __m128i product0 = _mm_maddubs_epi16(inVec[j], row[j]);
      product0 = _mm_madd_epi16(product0, kOnes);
      sum = _mm_add_epi32(sum, product0);
      __m128i product1 = _mm_maddubs_epi16(inVec[j + 1], row[j + 1]);
      product1 = _mm_madd_epi16(product1, kOnes);
      sum = _mm_add_epi32(sum, product1);
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E)); //_MM_PERM_BADC
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1)); //_MM_PERM_CDAB
    output[i] = _mm_cvtsi128_si32(sum) + biases[i];

#elif defined(USE_SSE2)
    __m128i sum = _mm_setzero_si128(), sum1 = sum;
    __m128i *row = (__m128i *)&weights[offset];
    for (unsigned j = 0; j < numChunks; j++) {
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
    for (unsigned j = 0; j < numChunks; j++) {
      s0 = _mm_add_pi32(s0, _mm_madd_pi16(row[2 * j + 0], inVec[2 * j + 0]));
      s1 = _mm_add_pi32(s1, _mm_madd_pi16(row[2 * j + 1], inVec[2 * j + 1]));
    }
    __m64 sum = _mm_add_pi32(s0, s1);
    sum = _mm_add_pi32(sum, _mm_unpackhi_pi32(sum, sum));
    output[i] = _mm_cvtsi64_si32(sum) + biases[i];

#elif defined(USE_NEON)
    int32x4_t sum = {biases[i]};
    int8x8_t *row = (int8x8_t *)&weights[offset];
    for (unsigned j = 0; j < numChunks; j++) {
      int16x8_t product = vmull_s8(inVec[j * 2], row[j * 2]);
      product = vmlal_s8(product, inVec[j * 2 + 1], row[j * 2 + 1]);
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
  assert(numDims % 32 == 0);

#if defined(USE_AVX2)
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
  __m128i *in = (__m128i *)input;
  __m128i *out = (__m128i *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m128i words = _mm_srai_epi16(_mm_packs_epi32(in[i * 2], in[i * 2 + 1]),
        SHIFT);
    out[i] = _mm_subs_epu16(_mm_adds_epi16(words, k0x7f80), k0x7f80);
  }

#elif defined(USE_MMX)
  const unsigned numChunks = numDims / 4;
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  __m64 *in = (__m64 *)input;
  __m64 *out = (__m64 *)output;
  for (unsigned i = 0; i < numChunks; i++) {
    __m64 words = _mm_srai_pi16(_mm_packs_pi32(in[i * 2], in[i * 2 + 1]),
        SHIFT);
    out[i] = _mm_subs_pu16(_mm_adds_pi16(words, k0x7f80), k0x7f80);
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
    output[i] = max(0, min(127, input[i] >> SHIFT));

#endif
}

// Input feature converter
static alignas(64) int16_t ft_biases[kHalfDimensions];
static alignas(64) int16_t ft_weights[kHalfDimensions * FtInDims];

// Calculate cumulative value without using difference calculation
INLINE void refresh_accumulator(const Position *pos)
{
  Accumulator *accumulator = &(pos->st->accumulator);

  IndexList activeIndices[2];
  activeIndices[0].size = activeIndices[1].size = 0;
  append_active_indices(pos, activeIndices);

  for (unsigned c = 0; c < 2; c++) {
    memcpy(accumulator->accumulation[c], ft_biases, kHalfDimensions * sizeof(int16_t));

    for (size_t k = 0; k < activeIndices[c].size; k++) {
      unsigned index = activeIndices[c].values[k];
      unsigned offset = kHalfDimensions * index;

#if defined(USE_AVX512)
      __m512i *accumulation = (__m512i *)accumulator->accumulation[c];
      __m512i *column = (__m512i *)&ft_weights[offset];
      const unsigned numChunks = kHalfDimensions / 32;
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm512_add_epi16(accumulation[j], column[j]);

#elif defined(USE_AVX2)
      __m256i *accumulation = (__m256i *)accumulator->accumulation[c];
      __m256i *column = (__m256i *)&ft_weights[offset];
      const unsigned numChunks = kHalfDimensions / 16;
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);

#elif defined(USE_SSE2)
      __m128i *accumulation = (__m128i *)accumulator->accumulation[c];
      __m128i *column = (__m128i *)&ft_weights[offset];
      const unsigned numChunks = kHalfDimensions / 8;
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm_add_epi16(accumulation[j], column[j]);

#elif defined(USE_MMX)
      __m64 *accumulation = (__m64 *)accumulator->accumulation[c];
      __m64 *column = (__m64 *)&ft_weights[offset];
      const unsigned numChunks = kHalfDimensions / 4;
      for (unsigned j = 0; j < numChunks; ++j)
        accumulation[j] = _mm_add_pi16(accumulation[j], column[j]);

#elif defined(USE_NEON)
      int16x8_t *accumulation = (int16x8_t *)accumulator->accumulation[c];
      int16x8_t *column = (int16x8_t *)&ft_weights[offset];
      const unsigned numChunks = kHalfDimensions / 8;
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = vaddq_s16(accumulation[j], column[j]);

#else
      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];

#endif
    }
  }

  accumulator->computedAccumulation = true;
}

// Calculate cumulative value using difference calculation if possible
static bool update_accumulator_if_possible(const Position *pos)
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

  for (unsigned c = 0; c < 2; c++) {
#if defined(USE_AVX512)
    const unsigned numChunks = kHalfDimensions / 32;
    __m512i *accumulation = (__m512i *)accumulator->accumulation[c];

#elif defined(USE_AVX2)
    const unsigned numChunks = kHalfDimensions / 16;
    __m256i *accumulation = (__m256i *)accumulator->accumulation[c];

#elif defined(USE_SSE2)
    const unsigned numChunks = kHalfDimensions / 8;
    __m128i *accumulation = (__m128i *)accumulator->accumulation[c];

#elif defined(USE_MMX)
    const unsigned numChunks = kHalfDimensions / 4;
    __m64 *accumulation = (__m64 *)accumulator->accumulation[c];

#elif defined(USE_NEON)
    const unsigned numChunks = kHalfDimensions / 8;
    int16x8_t *accumulation = (int16x8_t *)accumulator->accumulation[c];
#endif

    if (reset[c]) {
      memcpy(&(accumulator->accumulation[c]), ft_biases,
          kHalfDimensions * sizeof(int16_t));
    } else {
      memcpy(&(accumulator->accumulation[c]),
          &(prevAccumulator->accumulation[c]),
          kHalfDimensions * sizeof(int16_t));

      // Difference calculation for the deactivated features
      for (unsigned k = 0; k < removed_indices[c].size; k++) {
        unsigned index = removed_indices[c].values[k];
        const unsigned offset = kHalfDimensions * index;

#if defined(USE_AVX512)
        __m512i *column = (__m512i *)&ft_weights[offset];
        for (unsigned j = 0; j < numChunks; j++)
          accumulation[j] = _mm512_sub_epi16(accumulation[j], column[j]);

#elif defined(USE_AVX2)
        __m256i *column = (__m256i *)&ft_weights[offset];
        for (unsigned j = 0; j < numChunks; j++)
          accumulation[j] = _mm256_sub_epi16(accumulation[j], column[j]);

#elif defined(USE_SSE2)
        __m128i *column = (__m128i *)&ft_weights[offset];
        for (unsigned j = 0; j < numChunks; j++)
          accumulation[j] = _mm_sub_epi16(accumulation[j], column[j]);

#elif defined(USE_MMX)
        __m64 *column = (__m64 *)&ft_weights[offset];
        for (unsigned j = 0; j < numChunks; j++)
          accumulation[j] = _mm_sub_pi16(accumulation[j], column[j]);

#elif defined(USE_NEON)
        int16x8_t *column = (int16x8_t *)&ft_weights[offset];
        for (unsigned j = 0; j < numChunks; j++)
          accumulation[j] = vsubq_s16(accumulation[j], column[j]);

#else
        for (unsigned j = 0; j < kHalfDimensions; j++)
          accumulator->accumulation[c][j] -= ft_weights[offset + j];
#endif
      }
    }

    // Difference calculation for the activated features
    for (unsigned k = 0; k < added_indices[c].size; k++) {
      unsigned index = added_indices[c].values[k];
      const unsigned offset = kHalfDimensions * index;

#if defined(USE_AVX512)
      __m512i *column = (__m512i *)&ft_weights[offset];
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm512_add_epi16(accumulation[j], column[j]);

#elif defined(USE_AVX2)
      __m256i *column = (__m256i *)&ft_weights[offset];
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);

#elif defined(USE_SSE2)
      __m128i *column = (__m128i *)&ft_weights[offset];
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm_add_epi16(accumulation[j], column[j]);

#elif defined(USE_MMX)
      __m64 *column = (__m64 *)&ft_weights[offset];
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = _mm_add_pi16(accumulation[j], column[j]);

#elif defined(USE_NEON)
      int16x8_t *column = (int16x8_t *)&ft_weights[offset];
      for (unsigned j = 0; j < numChunks; j++)
        accumulation[j] = vaddq_s16(accumulation[j], column[j]);

#else
      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];

#endif
    }
  }

  accumulator->computedAccumulation = true;
  return true;
}

// Convert input features
INLINE void transform(const Position *pos, clipped_t *output)
{
  if (!update_accumulator_if_possible(pos))
    refresh_accumulator(pos);

  int16_t (*accumulation)[2][256] = &pos->st->accumulator.accumulation;

#if defined(USE_AVX2)
  const unsigned numChunks = kHalfDimensions / 32;
  const __m256i kZero = _mm256_setzero_si256();

#elif defined(USE_SSE41)
  const unsigned numChunks = kHalfDimensions / 16;
  const __m128i kZero = _mm_setzero_si128();

#elif defined(USE_SSSE3)
  const unsigned numChunks = kHalfDimensions / 16;
  const __m128i k0x80s = _mm_set1_epi8(-128);

#elif defined(USE_SSE2)
  const unsigned numChunks = kHalfDimensions / 8;
  const __m128i k0x7f80 = _mm_set1_epi16(0x7f80);

#elif defined(USE_MMX)
  const unsigned numChunks = kHalfDimensions / 4;
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);

#elif defined(USE_NEON)
  const unsigned numChunks = kHalfDimensions / 8;
  const int8x8_t kZero = {0};

#endif

  const Color perspectives[2] = { stm(), !stm() };
  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#if defined(USE_AVX2)
    __m256i *out = (__m256i *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m256i sum0 = ((__m256i *)(*accumulation)[perspectives[p]])[i * 2 + 0];
      __m256i sum1 = ((__m256i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      out[i] = _mm256_permute4x64_epi64(_mm256_max_epi8(
          _mm256_packs_epi16(sum0, sum1), kZero), 0xd8);
    }

#elif defined(USE_SSSE3)
    __m128i *out = (__m128i *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m128i sum0 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2 + 0];
      __m128i sum1 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m128i packedbytes = _mm_packs_epi16(sum0, sum1);
#if defined(USE_SSE41)
      out[i] = _mm_max_epi8(packedbytes, kZero);
#else
      out[i] = _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s);
#endif
    }

#elif defined(USE_SSE2)
    __m128i *out = (__m128i *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m128i sum = ((__m128i *)(*accumulation)[perspectives[p]])[i];
      out[i] = _mm_subs_epu16(_mm_adds_epi16(sum, k0x7f80), k0x7f80);
    }

#elif defined(USE_MMX)
    __m64 *out = (__m64 *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m64 sum0 = ((__m64 *)(*accumulation)[perspectives[p]])[i];
      out[i] = _mm_subs_pu16(_mm_adds_pi16(sum0, k0x7f80), k0x7f80);
    }

#elif defined(USE_NEON)
    int8x8_t *out = (int8x8_t *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      int16x8_t sum = ((int16x8_t *)(*accumulation)[perspectives[p]])[i];
      out[i] = vmax_s8(vqmovn_s16(sum), kZero);
    }

#else
    for (unsigned i = 0; i < kHalfDimensions; i++) {
      int16_t sum = (*accumulation)[perspectives[p]][i];
      output[offset + i] = max(0, min(127, sum));
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
#if defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32)
  uint8_t buf[sizeof(struct NetData) + 63];
  struct NetData *b = (struct NetData *)(((uintptr_t)buf + 0x3f) & ~0x3f);
#define B(x) (b->x)
#else
  struct NetData buf;
#define B(x) (buf.x)
#endif

  transform(pos, B(input));
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

bool read_weights(weight_t *output_buf, unsigned count, FILE *F)
{
  for (unsigned i = 0; i < count; i++)
    output_buf[i] = (weight_t)(int8_t)fgetc(F);
  return true;
}

bool load_eval_file(const char *evalFile)
{
  FILE *F = fopen(evalFile, "rb");

  if (!F) return false;

  // Read network header
  uint32_t version = read_uint32_t(F);
  uint32_t hash = read_uint32_t(F);
  uint32_t len = read_uint32_t(F);
  for (unsigned i = 0; i < len; i++)
    fgetc(F);
  if (version != NnueVersion) return false;
  if (hash != 0x3e5aa6eeu) return false;

  // Read feature transformer
  hash = read_uint32_t(F);
  if (hash != 0x5d69d7b8) return false;
  fread(ft_biases, sizeof(int16_t), kHalfDimensions, F);
  fread(ft_weights, sizeof(int16_t), kHalfDimensions * FtInDims, F);

  // Read network
  hash = read_uint32_t(F);
  if (hash != 0x63337156) return false;
  fread(hidden1_biases, sizeof(int32_t), 32, F);
  read_weights(hidden1_weights, 32 * 512, F);
  fread(hidden2_biases, sizeof(int32_t), 32, F);
  read_weights(hidden2_weights, 32 * 32 , F);
  fread(output_biases, sizeof(int32_t), 1 , F);
  read_weights(output_weights, 1  * 32 , F);

  return true;
//  return feof(F);
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

  fprintf(stderr, "The network file %s was not loaded successfully.\n"
                  "The default net can be downloaded from: https://tests.stock"
                  "fishchess.org/api/nn/%s\n", evalFile,
                  option_default_string_value(OPT_EVAL_FILE));
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
