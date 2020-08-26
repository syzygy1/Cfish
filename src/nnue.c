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
#if defined(USE_SSE2)
#undef USE_MMX
#endif

#if (defined(USE_SSE2) || defined(USE_MMX)) && !defined(USE_SSSE3)
#ifdef NNUE_TRANSPOSE
typedef uint8_t clipped_t;
#else
typedef int16_t clipped_t; //SSE2 and MMX have no int8 multiply.
#endif
typedef int16_t weight_t;
#else
typedef uint8_t clipped_t;
typedef int8_t weight_t;
#endif

#if defined(USE_SSSE3) && !defined(USE_AVX2)
//#define NNUE_TRANSPOSE
#endif

#define LOOP_4(f) f(0);f(1);f(2);f(3)
#define LOOP_8(f) LOOP_4(f); f(4);f(5);f(6);f(7)
#define LOOP_16(f) LOOP_8(f); f(8);f(9);f(10);f(11);f(12);f(13);f(14);f(15)

// Version of the evaluation file
static const uint32_t NnueVersion = 0x7AF32F16u;

// Constants used in evaluation value calculation
enum {
  FV_SCALE = 16,
  SHIFT = 6
};

enum {
  kMaxActiveDimensions = 30,
  kHalfDimensions = 256,
  FtInDims = 64 * PS_END, // 64 * 641
  FtOutDims = kHalfDimensions * 2
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
#ifdef NNUE_TRANSPOSE
union mask_t {
    uint16_t asRegBitmask[4]; //change to 32 for AVX2
    uint64_t asInt[1]; //change according to max int size
};
static_assert(FtOutDims % 64 == 0, "FtOutDims not a multiple of 64"); // input_mask is read 64 bits per. If FtOutDims is changed to be not a multiple of 64, make sure to initialize end of input_mask.
#if defined(USE_SSE2) && defined(IS_64BIT)
INLINE void affine_txfm(const clipped_t *in, char *out, unsigned inDims,
    unsigned outDims, const int32_t *biases, weight_t *weights,
    const union mask_t* input_mask, union mask_t* output_mask, 
    bool pack8_and_calc_mask) {
  assert(outDims == 32);
  __m128i kZeros[4];
  memset(kZeros, 0, 64);
  __m128i *biases_cast = (__m128i *) biases;
#define TMP(j) __m128i out_##j = biases_cast[j]
  LOOP_8(TMP);
#undef TMP
  const uint64_t *inm_lbound = input_mask->asInt;
  const uint64_t *inm_cast = inm_lbound;
  const uint64_t *inm_ubound = inm_lbound + (inDims+63)/64;
  uint64_t pos;
  unsigned i = 0, i1; uint64_t inm_val = 0;
  __m128i *first, *second;
  do {
    while (inm_val == 0 && inm_cast < inm_ubound) {
      inm_val = *inm_cast;
      inm_cast++;
    }
    if (inm_val == 0) break; 
    pos = __builtin_ctzll(inm_val); // lsb position
    i = pos + ((inm_cast - inm_lbound) - 1)*64; // TODO: fix 32-bit
    inm_val &= inm_val - 1; //pop lsb
    first  = (__m128i*)&weights[outDims * i];
    while (inm_val == 0 && inm_cast < inm_ubound) {
      inm_val = *inm_cast;
      inm_cast++;
    }
    pos = __builtin_ctzll(inm_val); // lsb position
    i1 = pos + ((inm_cast - inm_lbound) - 1)*64; //TODO: fix 32-bit
    inm_val &= inm_val - 1; //pop lsb
    __m128i mul;
    if (i1 < inDims) {
        mul = _mm_set1_epi32(in[i1] << 16 | in[i] );
        second = (__m128i*)&weights[outDims * i1];
    } else {
        mul = _mm_set1_epi32(in[i]);
        second = kZeros;
    }
    out_0 = _mm_add_epi32(out_0, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[0],second[0])));
    out_1 = _mm_add_epi32(out_1, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[0],second[0])));
    out_2 = _mm_add_epi32(out_2, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[1],second[1])));
    out_3 = _mm_add_epi32(out_3, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[1],second[1])));
    out_4 = _mm_add_epi32(out_4, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[2],second[2])));
    out_5 = _mm_add_epi32(out_5, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[2],second[2])));
    out_6 = _mm_add_epi32(out_6, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[3],second[3])));
    out_7 = _mm_add_epi32(out_7, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[3],second[3])));
  } while (true);
  __m128i out_in16_0 = _mm_srai_epi16(_mm_packs_epi32(out_0, out_1), SHIFT);
  __m128i out_in16_1 = _mm_srai_epi16(_mm_packs_epi32(out_2, out_3), SHIFT);
  __m128i out_in16_2 = _mm_srai_epi16(_mm_packs_epi32(out_4, out_5), SHIFT);
  __m128i out_in16_3 = _mm_srai_epi16(_mm_packs_epi32(out_6, out_7), SHIFT);

  __m128i *out_cast = (__m128i*) out;
  if (pack8_and_calc_mask) {
    uint16_t *outmask_view = output_mask->asRegBitmask;
    const __m128i k0x80s = _mm_set1_epi8(-128);
    __m128i reg_tmp;
    reg_tmp = _mm_subs_epi8(_mm_adds_epi8(_mm_packs_epi16(out_in16_0, out_in16_1), k0x80s), k0x80s);
    out_cast[0] = reg_tmp;
    outmask_view[0] = _mm_movemask_epi8(_mm_cmpgt_epi8(reg_tmp, kZeros[0]));
    reg_tmp = _mm_subs_epi8(_mm_adds_epi8(_mm_packs_epi16(out_in16_2, out_in16_3), k0x80s), k0x80s);
    out_cast[1] = reg_tmp;
    outmask_view[1] = _mm_movemask_epi8(_mm_cmpgt_epi8(reg_tmp, kZeros[0]));
  } else {
    const __m128i k0x7f80s = _mm_set1_epi16(0x7f80);
#define TMP(j) out_cast[j] = _mm_subs_epu16(_mm_adds_epi16(out_in16_##j, k0x7f80s), k0x7f80s);
    LOOP_4(TMP);
#undef TMP
  }
}
INLINE void scalar_prod(int16_t *input, int32_t *output, unsigned inDims,
    unsigned outDims, int32_t *biases, weight_t *weights)
{
  assert(inDims % 32 == 0);
  assert(outDims == 1);
#if defined(USE_SSE2)
  const unsigned numChunks = inDims / 16;
  __m128i *inVec = (__m128i *)input;
#elif defined(USE_MMX)
  const unsigned numChunks = inDims / 8;
  __m64 *inVec = (__m64 *)input;

#endif
  
  for (unsigned i = 0; i < outDims; ++i) {
    unsigned offset = i * inDims;

#if defined(USE_SSE2)
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

#else
    int32_t sum = biases[i];
    for (unsigned j = 0; j < inDims; j++)
      sum += weights[offset + j] * input[j];
    output[i] = sum;

#endif
  }
}

#else
#error "NNUE_TRANSPOSE requires USE_SSE2 and IS_64BIT for now"
#endif
#endif
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

#elif defined(USE_SSSE3) || (defined(NNUE_TRANSPOSE) && defined(USE_SSE2))
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
    output[i] = max(0, min(127, input[i] >> SHIFT));

#endif
}

// Input feature converter
static alignas(64) int16_t ft_biases[kHalfDimensions];
static alignas(64) int16_t ft_weights[kHalfDimensions * FtInDims];

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

#endif

#if defined(USE_AVX512)
#define NUM_REGS 8 // only 8 are needed

#elif defined(USE_SSE2) && defined(IS_64BIT)
#define NUM_REGS 16

#elif defined(USE_SSE2)
#define NUM_REGS 8

#elif USE_MMX
#define NUM_REGS 8

#endif

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

  assert(kHalfDimensions % TILE_HEIGHT == 0);
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
INLINE void transform(const Position *pos, clipped_t *output, union mask_t* out_mask_buf)
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

#elif defined(USE_SSSE3) || (defined(USE_SSE2) && defined(NNUE_TRANSPOSE))
  const unsigned numChunks = kHalfDimensions / 16;
  const __m128i k0x80s = _mm_set1_epi8(-128);
  const __m128i kZero = _mm_setzero_si128();
#ifdef NNUE_TRANSPOSE
  uint16_t* out_mask = out_mask_buf->asRegBitmask;
#endif
#elif defined(USE_SSE2)
  const unsigned numChunks = kHalfDimensions / 8;
  const __m128i k0x7f80 = _mm_set1_epi16(0x7f80);
  const __m128i k0x0080 = _mm_set1_epi16(0x0080);
  const __m128i k0x8000 = _mm_set1_epi16(-0x8000);

#elif defined(USE_MMX)
  const unsigned numChunks = kHalfDimensions / 4;
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  const __m64 k0x0080 = _mm_set1_pi16(0x0080);
  const __m64 k0x8000 = _mm_set1_pi16(-0x8000);

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

#elif defined(USE_SSSE3) || (defined(NNUE_TRANSPOSE) && defined(USE_SSE2))
    __m128i *out = (__m128i *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m128i sum0 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2 + 0];
      __m128i sum1 = ((__m128i *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      __m128i packedbytes = _mm_packs_epi16(sum0, sum1);
      __m128i outres; 
#if defined(USE_SSE41)
      outres = _mm_max_epi8(packedbytes, kZero);
#else
      outres = _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s);
#endif
      out[i] = outres;
      *out_mask++ = _mm_movemask_epi8(_mm_cmpgt_epi8(outres, kZero));
    }

#elif defined(USE_SSE2)
    __m128i *out = (__m128i *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m128i sum = ((__m128i *)(*accumulation)[perspectives[p]])[i];
      out[i] = _mm_subs_epu16(_mm_add_epi16(_mm_adds_epi16(sum, k0x7f80), k0x0080), k0x8000);
    }

#elif defined(USE_MMX)
    __m64 *out = (__m64 *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      __m64 sum = ((__m64 *)(*accumulation)[perspectives[p]])[i];
      out[i] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(sum, k0x7f80), k0x0080), k0x8000);
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
  alignas(8) union mask_t input_mask[FtOutDims / 64 + 1];
  alignas(8) union mask_t hidden1_mask[1];
  //alignas(8) char hidden2_mask[16];
#if defined(__GNUC__ ) && (__GNUC__ < 9) && defined(_WIN32) && !defined(__clang__) && !defined(__INTEL_COMPILER)
  // work around a bug in old gcc on Windows
  uint8_t buf[sizeof(struct NetData) + 63];
  struct NetData *b = (struct NetData *)(buf + ((((uintptr_t)buf-1) ^ 0x3f) & 0x3f));
#define B(x) (b->x)
#else
  struct NetData buf;
#define B(x) (buf.x)
#endif

  transform(pos, B(input), input_mask);
#ifndef NNUE_TRANSPOSE
  affine_propagate(B(input), B(hidden1_values), FtOutDims, 32,
      hidden1_biases, hidden1_weights);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 32);
  affine_propagate(B(hidden1_clipped), B(hidden2_values), 32, 32,
      hidden2_biases, hidden2_weights);
  clip_propagate(B(hidden2_values), B(hidden2_clipped), 32);
  affine_propagate(B(hidden2_clipped), &out_value, 32, 1, output_biases,
      output_weights);
#else
  alignas(64) int16_t hidden2_out[32];
#ifdef IS_64BIT
  (*(union mask_t*)hidden1_mask).asInt[0] = 0ULL; // initialize, because final 32 bits won't be written but will be read
#endif
  affine_txfm(B(input), B(hidden1_clipped), FtOutDims, 32,
      hidden1_biases, hidden1_weights, input_mask, hidden1_mask, true);
  affine_txfm(B(hidden1_clipped), hidden2_out, 32, 32,
      hidden2_biases, hidden2_weights, hidden1_mask, NULL, false);
  scalar_prod(hidden2_out, &out_value, 32, 1, output_biases,
      output_weights);
#endif
#if defined(USE_MMX)
  _mm_empty();
#endif

  return out_value / FV_SCALE;
}

bool read_weights(weight_t *output_buf, unsigned width, unsigned height,
    FILE *F)
{
  for (unsigned i = 0; i < height; i++) {
    for (unsigned j = 0; j < width; j++) {
      int8_t tmp = (int8_t)fgetc(F);
#ifndef NNUE_TRANSPOSE
      output_buf[i * width + j] =
#else
      output_buf[j * height + i] =
#endif
        (weight_t)tmp;
    }
  }
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
  read_weights(hidden1_weights, 512, 32, F);
  fread(hidden2_biases, sizeof(int32_t), 32, F);
  read_weights(hidden2_weights, 32 , 32 , F);
  fread(output_biases, sizeof(int32_t), 1 , F);
  read_weights(output_weights, 32, 1 , F);
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

  printf("info string ERROR: The network file %s was not loaded successfully.\n"
         "info string ERROR: The default net can be downloaded from:\n"
         "info string ERROR: https://tests.stockfishchess.org/api/nn/%s\n",
         evalFile, option_default_string_value(OPT_EVAL_FILE));
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
