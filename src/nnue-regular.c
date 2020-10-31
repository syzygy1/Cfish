#ifdef NNUE_REGULAR

#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_SSSE3))
typedef int16_t clipped_t; // SSE2 and MMX have no int8 multiply.
typedef int16_t weight_t;
#else
typedef int8_t clipped_t;
typedef int8_t weight_t;
#endif

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

  for (unsigned i = 0; i < outDims; i++) {
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

#if defined(USE_SSSE3)
INLINE void affine_propagate2(clipped_t *input, int32_t *output,
    unsigned inDims, unsigned outDims, int32_t *biases, weight_t *weights)
{
  assert(inDims % 32 == 0);
  assert(outDims % 4 == 0);

  __m128i *outVec = (__m128i *)output;
  __m128i *biasVec = (__m128i *)biases;


  for (unsigned i = 0; i < outDims / 4; i++) {
    unsigned offset0 = (4 * i + 0) * inDims;
    unsigned offset1 = (4 * i + 1) * inDims;
    unsigned offset2 = (4 * i + 2) * inDims;
    unsigned offset3 = (4 * i + 3) * inDims;

#if defined(USE_AVX512)
    if (inDims >= 64) {
      __m512i *inVec = (__m512i *)input;
      __m512i sum0 = _mm512_setzero_si512();
      __m512i sum1 = _mm512_setzero_si512();
      __m512i sum2 = _mm512_setzero_si512();
      __m512i sum3 = _mm512_setzero_si512();
      __m512i *row0 = (__m512i *)&weights[offset0];
      __m512i *row1 = (__m512i *)&weights[offset1];
      __m512i *row2 = (__m512i *)&weights[offset2];
      __m512i *row3 = (__m512i *)&weights[offset3];
#if defined(USE_VNNI)
      for (unsigned j = 0; j < inDims / 64; j++) {
        sum0 = _mm512_dpbusd_epi32(sum0, inVec[j], row0[j]);
        sum1 = _mm512_dpbusd_epi32(sum1, inVec[j], row1[j]);
        sum2 = _mm512_dpbusd_epi32(sum2, inVec[j], row2[j]);
        sum3 = _mm512_dpbusd_epi32(sum3, inVec[j], row3[j]);
      }
#else
      const __m512i kOnes = _mm512_set1_epi16(1);
      for (unsigned j = 0; j < inDims / 64; j++) {
        __m512i prod = _mm512_maddubs_epi16(inVec[j], row0[j]);
        prod = _mm512_madd_epi16(prod, kOnes);
        sum0 = _mm512_add_epi32(sum0, prod);
        prod = _mm512_maddubs_epi16(inVec[j], row1[j]);
        prod = _mm512_madd_epi16(prod, kOnes);
        sum1 = _mm512_add_epi32(sum1, prod);
        prod = _mm512_maddubs_epi16(inVec[j], row2[j]);
        prod = _mm512_madd_epi16(prod, kOnes);
        sum2 = _mm512_add_epi32(sum2, prod);
        prod = _mm512_maddubs_epi16(inVec[j], row3[j]);
        prod = _mm512_madd_epi16(prod, kOnes);
        sum3 = _mm512_add_epi32(sum3, prod);
      }
#endif
#if 1
      __m512i sum01a = _mm512_unpacklo_epi32(sum0, sum1);
      __m512i sum01b = _mm512_unpackhi_epi32(sum0, sum1);

      __m512i sum23a = _mm512_unpacklo_epi32(sum2, sum3);
      __m512i sum23b = _mm512_unpackhi_epi32(sum2, sum3);

      __m512i sum01 = _mm512_add_epi32(sum01a, sum01b);
      __m512i sum23 = _mm512_add_epi32(sum23a, sum23b);

      __m512i sum0123a = _mm512_unpacklo_epi64(sum01, sum23);
      __m512i sum0123b = _mm512_unpackhi_epi64(sum01, sum23);

      __m512i sum = _mm512_add_epi32(sum0123a, sum0123b);

      __m256i sum256lo = _mm512_castsi512_si256(sum);
      __m256i sum256hi = _mm512_extracti64x4_epi64(sum, 1);

      sum256lo = _mm256_add_epi32(sum256lo, sum256hi);
#else
      __m256i sum256_0 = _mm256_add_epi32(_mm512_castsi512_si256(sum0),_mm512_extracti64x4_epi64(sum0, 1));
      __m256i sum256_1 = _mm256_add_epi32(_mm512_castsi512_si256(sum1),_mm512_extracti64x4_epi64(sum1, 1));
      __m256i sum256_2 = _mm256_add_epi32(_mm512_castsi512_si256(sum2),_mm512_extracti64x4_epi64(sum2, 1));
      __m256i sum256_3 = _mm256_add_epi32(_mm512_castsi512_si256(sum3),_mm512_extracti64x4_epi64(sum3, 1));
      sum256_0 = _mm256_hadd_epi32(sum256_0, sum256_1);
      sum256_2 = _mm256_hadd_epi32(sum256_2, sum256_3);
      sum256_0 = _mm256_hadd_epi32(sum256_0, sum256_2);
#endif
      __m128i sum128lo = _mm256_castsi256_si128(sum256lo);
      __m128i sum128hi = _mm256_extracti128_si256(sum256lo, 1);
      outVec[i] = _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), biasVec[i]);
    }
    else
#endif
#if defined(USE_AVX2)
    {
      __m256i *inVec = (__m256i *)input;
      __m256i sum0 = _mm256_setzero_si256();
      __m256i sum1 = _mm256_setzero_si256();
      __m256i sum2 = _mm256_setzero_si256();
      __m256i sum3 = _mm256_setzero_si256();
      __m256i *row0 = (__m256i *)&weights[offset0];
      __m256i *row1 = (__m256i *)&weights[offset1];
      __m256i *row2 = (__m256i *)&weights[offset2];
      __m256i *row3 = (__m256i *)&weights[offset3];
#if defined(USE_VNNI)
      for (unsigned j = 0; j < inDims / 32; j++) {
        sum0 = _mm256_dpbusd_epi32(sum0, inVec[j], row0[j]);
        sum1 = _mm256_dpbusd_epi32(sum1, inVec[j], row1[j]);
        sum2 = _mm256_dpbusd_epi32(sum2, inVec[j], row2[j]);
        sum3 = _mm256_dpbusd_epi32(sum3, inVec[j], row3[j]);
      }
#else
      const __m256i kOnes = _mm256_set1_epi16(1);
      for (unsigned j = 0; j < inDims / 32; j++) {
        __m256i prod = _mm256_maddubs_epi16(inVec[j], row0[j]);
        prod = _mm256_madd_epi16(prod, kOnes);
        sum0 = _mm256_add_epi32(sum0, prod);
        prod = _mm256_maddubs_epi16(inVec[j], row1[j]);
        prod = _mm256_madd_epi16(prod, kOnes);
        sum1 = _mm256_add_epi32(sum1, prod);
        prod = _mm256_maddubs_epi16(inVec[j], row2[j]);
        prod = _mm256_madd_epi16(prod, kOnes);
        sum2 = _mm256_add_epi32(sum2, prod);
        prod = _mm256_maddubs_epi16(inVec[j], row3[j]);
        prod = _mm256_madd_epi16(prod, kOnes);
        sum3 = _mm256_add_epi32(sum3, prod);
      }
#endif
      sum0 = _mm256_hadd_epi32(sum0, sum1);
      sum2 = _mm256_hadd_epi32(sum2, sum3);
      sum0 = _mm256_hadd_epi32(sum0, sum2);
      __m128i sum128lo = _mm256_castsi256_si128(sum0);
      __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);
      outVec[i] = _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), biasVec[i]);
    }

#elif defined(USE_SSSE3)
    __m128i *inVec = (__m128i *)input;
    __m128i sum0 = _mm_setzero_si128();
    __m128i sum1 = _mm_setzero_si128();
    __m128i sum2 = _mm_setzero_si128();
    __m128i sum3 = _mm_setzero_si128();
    __m128i *row0 = (__m128i *)&weights[offset0];
    __m128i *row1 = (__m128i *)&weights[offset1];
    __m128i *row2 = (__m128i *)&weights[offset2];
    __m128i *row3 = (__m128i *)&weights[offset3];
    const __m128i kOnes = _mm_set1_epi16(1);
    for (unsigned j = 0; j < inDims / 16; j++) {
      __m128i prod = _mm_maddubs_epi16(inVec[j], row0[j]);
      prod = _mm_madd_epi16(prod, kOnes);
      sum0 = _mm_add_epi32(sum0, prod);
      prod = _mm_maddubs_epi16(inVec[j], row1[j]);
      prod = _mm_madd_epi16(prod, kOnes);
      sum1 = _mm_add_epi32(sum1, prod);
      prod = _mm_maddubs_epi16(inVec[j], row2[j]);
      prod = _mm_madd_epi16(prod, kOnes);
      sum2 = _mm_add_epi32(sum2, prod);
      prod = _mm_maddubs_epi16(inVec[j], row3[j]);
      prod = _mm_madd_epi16(prod, kOnes);
      sum3 = _mm_add_epi32(sum3, prod);
    }
    sum0 = _mm_hadd_epi32(sum0, sum1);
    sum2 = _mm_hadd_epi32(sum2, sum3);
    sum0 = _mm_hadd_epi32(sum0, sum2);
    outVec[i] = _mm_add_epi32(sum0, biasVec[i]);

#endif
  }
}
#else
#define affine_propagate2 affine_propagate
#endif

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
    out[i] = _mm_min_pi16(_mm_max_pi16(words, kZero), k0x7f);
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

// Convert input features
INLINE void transform(const Position *pos, clipped_t *output, mask_t *outMask)
{
  update_accumulator(pos, WHITE);
  update_accumulator(pos, BLACK);

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
  alignas(64) clipped_t input[512];
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

  affine_propagate2(B(input), B(hidden1_values), 512, 32,
      hidden1_biases, hidden1_weights);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 32);

  affine_propagate2(B(hidden1_clipped), B(hidden2_values), 32, 32,
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
}

#endif
