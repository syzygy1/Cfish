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

#elif defined(USE_NEON)
#include <arm_neon.h>
#endif

#include "evaluate.h"
#include "misc.h"
#include "nnue.h"
#include "position.h"
#include "uci.h"

// Version of the evaluation file
static const uint32_t NnueVersion = 0x7AF32F16u;

// Constant used in evaluation value calculation
static const int FV_SCALE = 16;
static const int kWeightScaleBits = 6;

// Size of cache line (in bytes)
enum { kCacheLineSize = 64 };

// SIMD width (in bytes)
#if defined(USE_AVX2)
static const size_t kSimdWidth = 32;

#elif defined(USE_SSE2)
static const size_t kSimdWidth = 16;

#elif defined(USE_NEON)
static const size_t kSimdWidth = 16;
#endif

//static const size_t kMaxSimdWidth = 32;

enum {
  kTransformedFeatureDimensions = 256,
  kDimensions = 64 * PS_END, // HalfKP
  kMaxActiveDimensions = PIECE_ID_KING,
  kHalfDimensions = kTransformedFeatureDimensions,
  FtInDims = kDimensions,
  FtOutDims = kHalfDimensions * 2,
  FtBufferSize = FtOutDims * sizeof(uint8_t)
};

static uint32_t read_uint32_t(FILE *F)
{
  uint32_t v;
  fread(&v, 4, 1, F);
  return from_le_u32(v);
}

// Round n up to be a multiple of base
#define ROUND_UP(n, base) (((n) + (base) - 1) / (base) * (base))

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

// from feature_set.h
static void append_active_indices(const Position *pos, //TriggerEvent trigger,
    IndexList active[2])
{
  for (unsigned c = 0; c < 2; c++) {
    //if (trigger == kFriendKingMoved)
      half_kp_append_active_indices(pos, c, &(active[c]));
  }
}

static void append_changed_indices(const Position *pos, //TriggerEvent trigger,
    IndexList removed[2], IndexList added[2], bool reset[2])
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
// uint8_t out, 512 dimensions
// kBufferSize = 0

// Hidden1Layer = ClippedReLu<AffineTransform<InputLayer, 32>>
// Affine: uint8_t in, int32_t out, 32 out dimensions
// kPaddedInputDimensions = ROUND_UP(512, kMaxSimdWidth) = 512
// kBufferSize = ROUND_UP(32 * sizeof(int32_t), kCacheLineSize) = 128
// Clipped: int32_t in, uint8_t out, 32 dimensions
// kBufferSize = ROUND_UP(32 * 1, kCachelineSize) = 32 -> 64

// Hidden2Layer = ClippedReLu<AffineTransform<hidden1, 32>>
// Affine: uint8_t in, int32_t out
// kPaddedInputDimensions = ROUND_UP(32, kMaxSimdWidth) = 32
// kOutputDimensions = 32
// kBufferSize = ROUND_UP(32 * sizeof(int32_t), kCacheLineSize) = 128
// Clipped: int32_t in, uint8_t out, 32 dimensions
// kBufferSize = ROUND_UP(32 * 1, kCacheLineSize) = 32 -> 64

// OutputLayer = AffineTransform<HiddenLayer2, 1>
// uint8_t in, int32_t out, 1 dimension
// kPaddedInputDimensions = ROUND_UP(32, kMaxSimdWidth) = 32
// kBufferSize = ROUND_UP(1 * sizeof(int32_t), kCacheLineSize) = 4 -> 64

enum { NetBufferSize = 128 + 64 + 128 + 64 + 64 };

static alignas(64) int8_t hidden1_weights[32 * 512];
static alignas(64) int8_t hidden2_weights[32 * 32];
static alignas(64) int8_t output_weights [1 * 32];

static int32_t hidden1_biases[32];
static int32_t hidden2_biases[32];
static int32_t output_biases [1];

void affine_propagate(uint8_t *input, int32_t *output, unsigned paddedInDims,
    unsigned outDims, int32_t biases[], int8_t weights[])
{
#if defined(USE_AVX512)
  const unsigned kNumChunks = paddedInDims / (kSimdWidth * 2);
  __m512i kOnes = _mm512_set1_epi16(1);
  __m512i *inVec = (__m512i *)input;

#elif defined(USE_AVX2)
  const unsigned kNumChunks = paddedInDims / kSimdWidth;
  __m256i kOnes = _mm256_set1_epi16(1);
  __m256i *inVec = (__m256i *)input;

#elif defined(USE_SSSE3)
  const unsigned kNumChunks = paddedInDims / kSimdWidth;
  __m128i kOnes = _mm_set1_epi16(1);
  __m128i *inVec = (__m128i *)input;

#elif defined(USE_NEON)
  const unsigned kNumChunks = paddedInDims / kSimdWidth;
  int8x8_t *inVec = (int8x8_t *)input;
#endif

  for (unsigned i = 0; i < outDims; ++i) {
    unsigned offset = i * paddedInDims;

#if defined(USE_AVX512)
    __m512i sum = _mm512_setzero_si512();
    __m512i *row = (__m512i *)&weights[offset];
    for (unsigned j = 0; j < kNumChunks; j++) {

#if defined(__MINGW32__) || defined(__MINGW64__)
      __m512i product = _mm512_maddubs_epi16(_mm512_loadu_si512(&inVec[j]), _mm512_load_si512(&row[j]));
#else
      __m512i product = _mm512_maddubs_epi16(_mm512_load_si512(&inVec[j]), _mm512_load_si512(&row[j]));
#endif

      product = _mm512_madd_epi16(product, kOnes);
      sum = _mm512_add_epi32(sum, product);
    }
    output[i] = _mm512_reduce_add_epi32(sum) + biases[i];

    // Note: Changing kMaxSimdWidth from 32 to 64 breaks loading existing networks.
    // As a result kPaddedInputDimensions may not be an even multiple of 64(512bit)
    // and we have to do one more 256bit chunk.
    if (paddedInDims != kNumChunks * kSimdWidth * 2)
    {
      __m256i *iv_256  = (__m256i *)input;
      __m256i *row_256 = (__m256i *)&weights[offset];
      int j = kNumChunks * 2;

#if defined(__MINGW32__) || defined(__MINGW64__)  // See HACK comment below in AVX2.
      __m256i sum256 = _mm256_maddubs_epi16(_mm256_loadu_si256(&iv_256[j]), _mm256_load_si256(&row_256[j]));
#else
      __m256i sum256 = _mm256_maddubs_epi16(_mm256_load_si256(&iv_256[j]), _mm256_load_si256(&row_256[j]));
#endif

      sum256 = _mm256_madd_epi16(sum256, _mm256_set1_epi16(1));
      sum256 = _mm256_hadd_epi32(sum256, sum256);
      sum256 = _mm256_hadd_epi32(sum256, sum256);
      const __m128i lo = _mm256_extracti128_si256(sum256, 0);
      const __m128i hi = _mm256_extracti128_si256(sum256, 1);
      output[i] += _mm_cvtsi128_si32(lo) + _mm_cvtsi128_si32(hi);
    }

#elif defined(USE_AVX2)
    __m256i sum = _mm256_setzero_si256();
    __m256i *row = (__m256i *)&weights[offset];
    for (unsigned j = 0; j < kNumChunks; j++) {
      __m256i product = _mm256_maddubs_epi16(

#if defined(__MINGW32__) || defined(__MINGW64__)
          // HACK: Use _mm256_loadu_si256() instead of _mm256_load_si256. Because the binary
          //       compiled with g++ in MSYS2 crashes here because the output memory is not aligned
          //       even though alignas is specified.
          _mm256_loadu_si256
#else
          _mm256_load_si256
#endif

          (&inVec[j]), _mm256_load_si256(&row[j]));
      product = _mm256_madd_epi16(product, kOnes);
      sum = _mm256_add_epi32(sum, product);
    }
    sum = _mm256_hadd_epi32(sum, sum);
    sum = _mm256_hadd_epi32(sum, sum);
    const __m128i lo = _mm256_extracti128_si256(sum, 0);
    const __m128i hi = _mm256_extracti128_si256(sum, 1);
    output[i] = _mm_cvtsi128_si32(lo) + _mm_cvtsi128_si32(hi) + biases[i];

#elif defined(USE_SSSE3)
    __m128i sum = _mm_cvtsi32_si128(biases[i]);
    __m128i *row = (__m128i *)&weights[offset];
    for (unsigned j = 0; j < kNumChunks; j++) {
      __m128i product = _mm_maddubs_epi16(
          _mm_load_si128(&inVec[j]), _mm_load_si128(&row[j]));
      product = _mm_madd_epi16(product, kOnes);
      sum = _mm_add_epi32(sum, product);
    }
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    output[i] = _mm_cvtsi128_si32(sum);

#elif defined(USE_NEON)
    int32x4_t sum = {biases[i]};
    int8x8_t *row = (int8x8_t *)&weights[offset];
    for (unsigned j = 0; j < kNumChunks; j++) {
      int16x8_t product = vmull_s8(inVec[j * 2], row[j * 2]);
      product = vmlal_s8(product, inVec[j * 2 + 1], row[j * 2 + 1]);
      sum = vpadalq_s16(sum, product);
    }
    output[i] = sum[0] + sum[1] + sum[2] + sum[3];

#else
    int32_t sum = biases[i];
    for (unsigned j = 0; j < paddedInDims; j++)
      sum += weights[offset + j] * input[j];
    output[i] = sum;
#endif

  }
}

void clip_propagate(int32_t *input, uint8_t *output, unsigned numDims)
{
#if defined(USE_AVX2)
  const unsigned kNumChunks = numDims / kSimdWidth;
  const __m256i kZero = _mm256_setzero_si256();
  const __m256i kOffsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  __m256i *in = (__m256i *)input;
  __m256i *out = (__m256i *)output;
  for (unsigned i = 0; i < kNumChunks; i++) {
    const __m256i words0 = _mm256_srai_epi16(_mm256_packs_epi32(

#if defined(__MINGW32__) || defined(__MINGW64__)
          // HACK: Use _mm256_loadu_si256() instead of _mm256_load_si256. Because the binary
          //       compiled with g++ in MSYS2 crashes here because the output memory is not aligned
          //       even though alignas is specified.
          _mm256_loadu_si256
#else
          _mm256_load_si256
#endif

          (&in[i * 4 + 0]),

#if defined(__MINGW32__) || defined(__MINGW64__)
          _mm256_loadu_si256
#else
          _mm256_load_si256
#endif

          (&in[i * 4 + 1])), kWeightScaleBits);
    const __m256i words1 = _mm256_srai_epi16(_mm256_packs_epi32(

#if defined(__MINGW32__) || defined(__MINGW64__)
          _mm256_loadu_si256
#else
          _mm256_load_si256
#endif

          (&in[i * 4 + 2]),

#if defined(__MINGW32__) || defined(__MINGW64__)
          _mm256_loadu_si256
#else
          _mm256_load_si256
#endif

          (&in[i * 4 + 3])), kWeightScaleBits);

#if defined(__MINGW32__) || defined(__MINGW64__)
      _mm256_storeu_si256
#else
      _mm256_store_si256
#endif

      (&out[i], _mm256_permutevar8x32_epi32(_mm256_max_epi8(
                                                            _mm256_packs_epi16(words0, words1), kZero), kOffsets));
  }
  const unsigned kStart = kNumChunks * kSimdWidth;

#elif defined(USE_SSSE3)
  const unsigned kNumChunks = numDims / kSimdWidth;

#ifdef USE_SSE41
  const __m128i kZero = _mm_setzero_si128();
#else
  const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

  __m128i *in = (__m128i *)input;
  __m128i *out = (__m128i *)output;
  for (unsigned i = 0; i < kNumChunks; i++) {
    const __m128i words0 = _mm_srai_epi16(_mm_packs_epi32(
          _mm_load_si128(&in[i * 4 + 0]),
          _mm_load_si128(&in[i * 4 + 1])), kWeightScaleBits);
    const __m128i words1 = _mm_srai_epi16(_mm_packs_epi32(
          _mm_load_si128(&in[i * 4 + 2]),
          _mm_load_si128(&in[i * 4 + 3])), kWeightScaleBits);
    const __m128i packedbytes = _mm_packs_epi16(words0, words1);
    _mm_store_si128(&out[i],

#ifdef USE_SSE41
        _mm_max_epi8(packedbytes, kZero)
#else
        _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
#endif

        );
  }
  const unsigned kStart = kNumChunks * kSimdWidth;

#elif defined(USE_NEON)
  const unsigned kNumChunks = numDims / (kSimdWidth / 2);
  const int8x8_t kZero = {0};
  int32x4_t *in = (int32x4_t *)input;
  int8x8_t *out = (int8x8_t *)output;
  for (unsigned i = 0; i < kNumChunks; i++) {
    int16x8_t shifted;
    int16x4_t *pack = (int16x4_t *)(&shifted);
    pack[0] = vqshrn_n_s32(in[i * 2 + 0], kWeightScaleBits);
    pack[1] = vqshrn_n_s32(in[i * 2 + 1], kWeightScaleBits);
    out[i] = vmax_s8(vqmovn_s16(shifted), kZero);
  }
  const unsigned kStart = kNumChunks * (kSimdWidth / 2);
#else
  const unsigned kStart = 0;
#endif

  for (unsigned i = kStart; i < numDims; i++)
    output[i] = max(0, min(127, input[i] >> kWeightScaleBits));
}

void propagate(uint8_t *tf, uint8_t *buffer)
{
  affine_propagate(tf, (int32_t *)buffer, 512, 32, hidden1_biases, hidden1_weights);
  clip_propagate((int32_t *)buffer, buffer + 128, 32);
  affine_propagate(buffer + 128, (int32_t *)(buffer + 192), 32, 32, hidden2_biases, hidden2_weights);
  clip_propagate((int32_t *)(buffer + 192), buffer + 320, 32);
  affine_propagate(buffer + 320, (int32_t *)(buffer + 384), 32, 1, output_biases, output_weights);
}

// Input feature converter
static alignas(64) int16_t ft_biases[kHalfDimensions];
static alignas(64) int16_t ft_weights[kHalfDimensions * FtInDims];

// Calculate cumulative value without using difference calculation
void refresh_accumulator(const Position *pos)
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

#if defined(USE_AVX2)
      __m256i *accumulation = (__m256i *)(&accumulator->accumulation[c][0]);
      __m256i *column = (__m256i *)(&ft_weights[offset]);
      const unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
      for (unsigned j = 0; j < kNumChunks; j++)
#if defined(__MINGW32__) || defined(__MINGW64__)
        _mm256_storeu_si256(&accumulation[j], _mm256_add_epi16(_mm256_loadu_si256(&accumulation[j]), column[j]));
#else
        accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);
#endif

#elif defined(USE_SSE2)
      __m128i *accumulation = (__m128i *)(&accumulator->accumulation[c][0]);
      __m128i *column = (__m128i *)(&ft_weights[offset]);
      const unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
      for (unsigned j = 0; j < kNumChunks; j++)
        accumulation[j] = _mm_add_epi16(accumulation[j], column[j]);

#elif defined(USE_NEON)
      int16x8_t *accumulation = (int16x8_t *)(
          &accumulator->accumulation[c][0]);
      int16x8_t *column = (int16x8_t *)(&ft_weights[offset]);
      constr unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
      for (unsigned j = 0; j < kNumChunks; j++)
        accumulation[j] = vaddq_s16(accumulation[j], column[j]);

#else
      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];
#endif

    }
  }

  accumulator->computedAccumulation = true;
}

bool update_accumulator_if_possible(const Position *pos);

// Convert input features
void transform(const Position *pos, uint8_t *output, bool refresh)
{
  if (refresh || !update_accumulator_if_possible(pos))
    refresh_accumulator(pos);

  int16_t (*accumulation)[2][256] = &(pos->st->accumulator.accumulation);

#if defined(USE_AVX2)
  const unsigned kNumChunks = kHalfDimensions / kSimdWidth;
  const int kControl = 0xd8; // 0b11011000
  const __m256i kZero = _mm256_setzero_si256();

#elif defined(USE_SSSE3)
  const unsigned kNumChunks = kHalfDimensions / kSimdWidth;

#ifdef USE_SSE41
  const __m128i kZero = _mm_setzero_si128();
#else
  const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

#elif defined(USE_NEON)
  const unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
  const int8x8_t kZero = {0};
#endif

  const Color perspectives[2] = { stm(), !stm() };
  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#if defined(USE_AVX2)
    __m256i *out = (__m256i *)(&output[offset]);
    for (unsigned i = 0; i < kNumChunks; i++) {
      __m256i sum0 =

#if defined(__MINGW32__) || defined(__MINGW64__)
        // HACK: Use _mm256_loadu_si256() instead of _mm256_load_si256. Because the binary
        //       compiled with g++ in MSYS2 crashes here because the output memory is not aligned
        //       even though alignas is specified.
        _mm256_loadu_si256
#else
        _mm256_load_si256
#endif

        (&((__m256i *)(&(*accumulation)[perspectives[p]]))[i * 2 + 0]);
      __m256i sum1 =

#if defined(__MINGW32__) || defined(__MINGW64__)
        _mm256_loadu_si256
#else
        _mm256_load_si256
#endif

        (&((__m256i *)(&(*accumulation)[perspectives[p]]))[i * 2 + 1]);

#if defined(__MINGW32__) || defined(__MINGW64__)
      _mm256_storeu_si256
#else
        _mm256_store_si256
#endif

        (&out[i], _mm256_permute4x64_epi64(_mm256_max_epi8(_mm256_packs_epi16(sum0, sum1), kZero), kControl));
    }

#elif defined(USE_SSSE3)
    __m128i *out = (__m128i *)(&output[offset]);
    for (unsigned i = 0; i < kNumChunks; i++) {
      __m128i sum0 = _mm_load_si128(&((__m128i *)(&
            (*accumulation)[perspectives[p]]))[i * 2 + 0]);
      __m128i sum1 = _mm_load_si128(&((__m128i *)(&
            (*accumulation)[perspectives[p]]))[i * 2 + 1]);
      __m128i packedbytes = _mm_packs_epi16(sum0, sum1);

      _mm_store_si128(&out[i],

#ifdef USE_SSE41
          _mm_max_epi8(packedbytes, kZero)
#else
          _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
#endif

      );
    }

#elif defined(USE_NEON)
    int8x8_t *out = (int8x8_t *)(&output[offset]);
    for (unsigned i = 0; i < kNumChunks; i++) {
      int16x8_t sum = ((int16x8_t *)((*accumulation)[perspectives[p]]))[i];
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

// Calculate cumulative value using difference calculation
void update_accumulator(const Position *pos)
{
  Accumulator *prev_accumulator = &((pos->st-1)->accumulator);
  Accumulator *accumulator = &(pos->st->accumulator);
  IndexList removed_indices[2], added_indices[2];
  removed_indices[0].size = removed_indices[1].size = 0;
  added_indices[0].size = added_indices[1].size = 0;
  bool reset[2];
  append_changed_indices(pos, removed_indices, added_indices, reset);

  for (unsigned perspective = 0; perspective < 2; perspective++) {

#if defined(USE_AVX2)
    const unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
    __m256i *accumulation = (__m256i *)(
        &accumulator->accumulation[perspective][0]);

#elif defined(USE_SSE2)
    const unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
    __m128i *accumulation = (__m128i *)(
        &accumulator->accumulation[perspective][0]);

#elif defined(USE_NEON)
    const unsigned kNumChunks = kHalfDimensions / (kSimdWidth / 2);
    int16x8t *accumulation = (int16x8_t *)(
        &accumulator->accumulation[perspective][0]);
#endif

    if (reset[perspective]) {
      memcpy(&(accumulator->accumulation[perspective]), ft_biases,
          kHalfDimensions * sizeof(int16_t));
    } else {
      memcpy(&(accumulator->accumulation[perspective]),
          &(prev_accumulator->accumulation[perspective]),
          kHalfDimensions * sizeof(int16_t));
      // Difference calculation for the deactivated features
      for (unsigned k = 0; k < removed_indices[perspective].size; k++) {
        unsigned index = removed_indices[perspective].values[k];
        const unsigned offset = kHalfDimensions * index;

#if defined(USE_AVX2)
        __m256i *column = (__m256i *)(&ft_weights[offset]);
        for (unsigned j = 0; j < kNumChunks; j++)
          accumulation[j] = _mm256_sub_epi16(accumulation[j], column[j]);

#elif defined(USE_SSE2)
        __m128i *column = (__m128i *)(&ft_weights[offset]);
        for (unsigned j = 0; j < kNumChunks; j++)
          accumulation[j] = _mm_sub_epi16(accumulation[j], column[j]);

#elif defined(USE_NEON)
        int16x8_t *column = (int16x8_t *)(&ft_weights[offset]);
        for (unsigned j = 0; j < kNumChunks; j++)
          accumulation[j] = vsubq_s16(accumulation[j], column[j]);

#else
        for (unsigned j = 0; j < kHalfDimensions; j++)
          accumulator->accumulation[perspective][j] -= ft_weights[offset + j];
#endif

      }
    }

    // Difference calculation for the activated features
    for (unsigned k = 0; k < added_indices[perspective].size; k++) {
      unsigned index = added_indices[perspective].values[k];
      const unsigned offset = kHalfDimensions * index;

#if defined(USE_AVX2)
      __m256i *column = (__m256i *)(&ft_weights[offset]);
      for (unsigned j = 0; j < kNumChunks; j++)
        accumulation[j] = _mm256_add_epi16(accumulation[j], column[j]);

#elif defined(USE_SSE2)
      __m128i *column = (__m128i *)(&ft_weights[offset]);
      for (unsigned j = 0; j < kNumChunks; j++)
        accumulation[j] = _mm_add_epi16(accumulation[j], column[j]);

#elif defined(USE_NEON)
      int16x8_t *column = (int16x8_t *)(&ft_weights[offset]);
      for (unsigned j = 0; j < kNumChunks; j++)
        accumulation[j] = vaddq_s16(accumulation[j], column[j]);

#else
      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[perspective][j] += ft_weights[offset + j];

#endif

    }
  }

  accumulator->computedAccumulation = true;
}

// Proceed with the difference calculation if possible
bool update_accumulator_if_possible(const Position *pos)
{
  Stack *st = pos->st;
  if (st->accumulator.computedAccumulation) {
    return true;
  }
  st--;
  if (st->accumulator.computedAccumulation) {
    update_accumulator(pos);
    return true;
  }
  return false;
}

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

// Calculate the evaluation value
static Value compute_score(const Position *pos, bool refresh)
{
  alignas(kCacheLineSize) uint8_t transformed_features[FtBufferSize];
  transform(pos, transformed_features, refresh);
  alignas(kCacheLineSize) uint8_t buffer[NetBufferSize];
  propagate(transformed_features, buffer);

  return *(int32_t *)(buffer + 384) / FV_SCALE;
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
  fread(hidden1_biases , sizeof(int32_t), 32      , F);
  fread(hidden1_weights, sizeof(uint8_t), 32 * 512, F);
  fread(hidden2_biases , sizeof(int32_t), 32      , F);
  fread(hidden2_weights, sizeof(uint8_t), 32 * 32 , F);
  fread(output_biases  , sizeof(int32_t), 1       , F);
  fread(output_weights , sizeof(uint8_t), 1  * 32 , F);

  return true;
//  return feof(F);
}

static char *loadedFile = NULL;

void nnue_init(void)
{
  pureNNUE = option_value(OPT_PURE_NNUE);

  const char *evalFile = option_string_value(OPT_EVAL_FILE);
  if (loadedFile && strcmp(evalFile, loadedFile) == 0)
    return;

  if (loadedFile)
    free(loadedFile);

  if (load_eval_file(evalFile)) {
    loadedFile = strdup(evalFile);
    return;
  }

  fprintf(stderr, "NNUE evaluation used, but the network file %s was not "
                  "loaded successfully. These network evaluation paramaters "
                  "must be available, and compatible with this version of the "
                  "code. The UCI option EvalFile might need to specify the "
                  "full path to the file, including the directory/folder name. "
                  "The default net can be downloaded from: https://tests.stock"
                  "fishchess.org/api/nn/%s\n", evalFile,
                  option_default_string_value(OPT_EVAL_FILE));
  exit(EXIT_FAILURE);
}

// Evaluation function. Perform differential calculation.
Value nnue_evaluate(const Position *pos)
{
  Value v = compute_score(pos, false);
  v = clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

  return v;
}

// Proceed with the difference calculation if possible
void update_eval(const Position *pos)
{
  update_accumulator_if_possible(pos);
}
