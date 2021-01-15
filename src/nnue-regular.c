#ifdef NNUE_REGULAR

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

INLINE void affine_propagate(clipped_t *input, int32_t *output,
    unsigned inDims, unsigned outDims, int32_t *biases, weight_t *weights)
{
  assert(inDims == 32 || inDims % 128 == 0);
  assert(outDims % 8 == 0);

#if defined(USE_AVX512)
  if (inDims >= 64) {
    __m128i *outVec = (__m128i *)output;
    __m128i *biasVec = (__m128i *)biases;
    for (unsigned i = 0; i < outDims / 4; i++) {
      __m512i *inVec = (__m512i *)input;
      __m512i *w = (__m512i *)&weights[4 * i * inDims];
      __m512i s0, s1, s2, s3;
      s0 = s1 = s2 = s3 = _mm512_setzero_si512();
#if defined(USE_VNNI)
      for (unsigned j = 0; j < inDims / 64; j++) {
        s0 = _mm512_dpbusd_epi32(s0, inVec[j], w[0 * inDims / 64 + j]);
        s1 = _mm512_dpbusd_epi32(s1, inVec[j], w[1 * inDims / 64 + j]);
        s2 = _mm512_dpbusd_epi32(s2, inVec[j], w[2 * inDims / 64 + j]);
        s3 = _mm512_dpbusd_epi32(s3, inVec[j], w[3 * inDims / 64 + j]);
      }
#else
      const __m512i kOnes = _mm512_set1_epi16(1);
      __m512i p1, p2;
#if 0
      p = _mm512_maddubs_epi16(inVec[0], w[0 * inDims / 64]);
      __m512i s0 = _mm512_madd_epi16(p, kOnes);
      p = _mm512_maddubs_epi16(inVec[0], w[1 * inDims / 64]);
      __m512i s1 = _mm512_madd_epi16(p, kOnes);
      p = _mm512_maddubs_epi16(inVec[0], w[2 * inDims / 64]);
      __m512i s2 = _mm512_madd_epi16(p, kOnes);
      p = _mm512_maddubs_epi16(inVec[0], w[3 * inDims / 64]);
      __m512i s3 = _mm512_madd_epi16(p, kOnes);
#endif
      for (unsigned j = 0; j < inDims / 128; j++) {
        p1 = _mm512_maddubs_epi16(inVec[2 * j], w[0 * inDims / 64 + 2 * j]);
        p2 = _mm512_maddubs_epi16(inVec[2 * j + 1], w[0 * inDims / 64 + 2 * j + 1]);
        s0 = _mm512_add_epi32(s0, _mm512_madd_epi16(_mm512_add_epi16(p1, p2), kOnes));
        p1 = _mm512_maddubs_epi16(inVec[2 * j], w[1 * inDims / 64 + 2 * j]);
        p2 = _mm512_maddubs_epi16(inVec[2 * j + 1], w[1 * inDims / 64 + 2 * j + 1]);
        s1 = _mm512_add_epi32(s1, _mm512_madd_epi16(_mm512_add_epi16(p1, p2), kOnes));
        p1 = _mm512_maddubs_epi16(inVec[2 * j], w[2 * inDims / 64 + 2 * j]);
        p2 = _mm512_maddubs_epi16(inVec[2 * j + 1], w[2 * inDims / 64 + 2 * j + 1]);
        s2 = _mm512_add_epi32(s2, _mm512_madd_epi16(_mm512_add_epi16(p1, p2), kOnes));
        p1 = _mm512_maddubs_epi16(inVec[2 * j], w[3 * inDims / 64 + 2 * j]);
        p2 = _mm512_maddubs_epi16(inVec[2 * j + 1], w[3 * inDims / 64 + 2 * j + 1]);
        s3 = _mm512_add_epi32(s3, _mm512_madd_epi16(_mm512_add_epi16(p1, p2), kOnes));
      }
#endif
#if 1
      s0 = _mm512_add_epi32(_mm512_unpacklo_epi32(s0, s1),
          _mm512_unpackhi_epi32(s0, s1));
      s2 = _mm512_add_epi32(_mm512_unpacklo_epi32(s2, s3),
          _mm512_unpackhi_epi32(s2, s3));
      s0 = _mm512_add_epi32(_mm512_unpacklo_epi64(s0, s2),
          _mm512_unpackhi_epi64(s0, s2));
      __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(s0),
          _mm512_extracti64x4_epi64(s0, 1));
      __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
          _mm256_extracti128_si256(sum256, 1));
      outVec[i] = _mm_add_epi32(sum128, biasVec[i]);
#else
      __m256i sum256_0 = _mm256_add_epi32(_mm512_castsi512_si256(sum0),_mm512_extracti64x4_epi64(sum0, 1));
      __m256i sum256_1 = _mm256_add_epi32(_mm512_castsi512_si256(sum1),_mm512_extracti64x4_epi64(sum1, 1));
      __m256i sum256_2 = _mm256_add_epi32(_mm512_castsi512_si256(sum2),_mm512_extracti64x4_epi64(sum2, 1));
      __m256i sum256_3 = _mm256_add_epi32(_mm512_castsi512_si256(sum3),_mm512_extracti64x4_epi64(sum3, 1));
      sum256_0 = _mm256_hadd_epi32(sum256_0, sum256_1);
      sum256_2 = _mm256_hadd_epi32(sum256_2, sum256_3);
      sum256_0 = _mm256_hadd_epi32(sum256_0, sum256_2);
      __m128i sum128 = _mm128_add_epi32(_mm256_castsi256_si128(sum256_0),
          _mm256_extracti128_si256(sum256_0, 1));
      outVec[i] = _mm_add_epi32(sum128, biasVec[i]);
#endif
    }
  } else { // 32 x 32 multiplication
    __m512i *outVec = (__m512i *)output;
    __m512i *biasVec = (__m512i *)biases;
    __m128i *inVec = (__m128i *)input;
    __m512i in0 = _mm512_broadcast_i32x4(inVec[0]);
    __m512i in1 = _mm512_broadcast_i32x4(inVec[1]);
#if defined(USE_VNNI)
    const __m512i kZero = _mm512_setzero_si512();
    __m512i s0, s1, s2, s3;
    for (unsigned i = 0; i < outDims / 16; i++) {
      __m512i *w = (__m512i *)&weights[16 * i * 32];
      s0 = _mm512_dpbusd_epi32(kZero, in0, w[0]); // first half rows 0,4,8,12
      s0 = _mm512_dpbusd_epi32(s0, in1, w[1]); // second half rows 0,4,8,12
      s1 = _mm512_dpbusd_epi32(kZero, in0, w[2]); // first half rows 1,5,9,13
      s1 = _mm512_dpbusd_epi32(s1, in1, w[3]);
      s2 = _mm512_dpbusd_epi32(kZero, in0, w[4]);
      s2 = _mm512_dpbusd_epi32(s2, in1, w[5]);
      s3 = _mm512_dpbusd_epi32(kZero, in0, w[6]);
      s3 = _mm512_dpbusd_epi32(s3, in1, w[7]);
      s0 = _mm512_add_epi32(
          _mm512_unpacklo_epi32(s0, s1), _mm512_unpackhi_epi32(s0, s1));
      s2 = _mm512_add_epi32(
          _mm512_unpacklo_epi32(s2, s3), _mm512_unpackhi_epi32(s2, s3));
      s0 = _mm512_add_epi32(
          _mm512_unpacklo_epi64(s0, s2), _mm512_unpackhi_epi64(s0, s2));
      outVec[i] = _mm512_add_epi32(s0, biasVec[i]);
    }
#else
    const __m512i kOnes = _mm512_set1_epi16(1);
    __m512i s0, s1, s2, s3, p;
    for (unsigned i = 0; i < outDims / 16; i++) {
      __m512i *w = (__m512i *)&weights[16 * i * 32];
      s0 = _mm512_maddubs_epi16(in0, w[0]); // first half of rows 0,4,8,12
      s0 = _mm512_madd_epi16(s0, kOnes);
      p  = _mm512_maddubs_epi16(in1, w[1]); // second half of rows 0,4,8,12
      p  = _mm512_madd_epi16(p, kOnes);
      s0 = _mm512_add_epi32(s0, p);
      s1 = _mm512_maddubs_epi16(in0, w[2]); // first half of rows 1,5,9,13
      s1 = _mm512_madd_epi16(s1, kOnes);
      p  = _mm512_maddubs_epi16(in1, w[3]);
      p  = _mm512_madd_epi16(p, kOnes);
      s1 = _mm512_add_epi32(s1, p);
      s2 = _mm512_maddubs_epi16(in0, w[4]);
      s2 = _mm512_madd_epi16(s2, kOnes);
      p  = _mm512_maddubs_epi16(in1, w[5]);
      p  = _mm512_madd_epi16(p, kOnes);
      s2 = _mm512_add_epi32(s2, p);
      s3 = _mm512_maddubs_epi16(in0, w[6]);
      s3 = _mm512_madd_epi16(s3, kOnes);
      p  = _mm512_maddubs_epi16(in1, w[7]);
      p  = _mm512_madd_epi16(p, kOnes);
      s3 = _mm512_add_epi32(s3, p);
      s0 = _mm512_add_epi32(
          _mm512_unpacklo_epi32(s0, s1), _mm512_unpackhi_epi32(s0, s1));
      s2 = _mm512_add_epi32(
          _mm512_unpacklo_epi32(s2, s3), _mm512_unpackhi_epi32(s2, s3));
      s0 = _mm512_add_epi32(
          _mm512_unpacklo_epi64(s0, s2), _mm512_unpackhi_epi64(s0, s2));
      outVec[i] = _mm512_add_epi32(s0, biasVec[i]);
    }
#endif
  }

#elif defined(USE_AVX2)
#if 1
  if (inDims > 32) {
    __m128i *outVec = (__m128i *)output;
    __m128i *biasVec = (__m128i *)biases;
    __m256i *inVec = (__m256i *)input;
    for (unsigned i = 0; i < outDims / 4; i++) {
      __m256i *w = (__m256i *)&weights[4 * i * inDims];
      __m256i s0, s1, s2, s3;
      s0 = s1 = s2 = s3 = _mm256_setzero_si256();
#if defined(USE_VNNI)
      for (unsigned j = 0; j < inDims / 32; j++) {
        s0 = _mm256_dpbusd_epi32(s0, inVec[j], w[0 * inDims / 32 + j]);
        s1 = _mm256_dpbusd_epi32(s1, inVec[j], w[1 * inDims / 32 + j]);
        s2 = _mm256_dpbusd_epi32(s2, inVec[j], w[2 * inDims / 32 + j]);
        s3 = _mm256_dpbusd_epi32(s3, inVec[j], w[3 * inDims / 32 + j]);
      }
#else
      const __m256i kOnes = _mm256_set1_epi16(1);
      __m256i p1, p2;
      for (unsigned j = 0; j < inDims / 64; j++) {
        p1 = _mm256_maddubs_epi16(inVec[2 * j], w[0 * inDims / 32 + 2 * j]);
        p2 = _mm256_maddubs_epi16(inVec[2 * j + 1], w[0 * inDims / 32 + 2 * j + 1]);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_add_epi16(p1, p2), kOnes));
        p1 = _mm256_maddubs_epi16(inVec[2 * j], w[1 * inDims / 32 + 2 * j]);
        p2 = _mm256_maddubs_epi16(inVec[2 * j + 1], w[1 * inDims / 32 + 2 * j + 1]);
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_add_epi16(p1, p2), kOnes));
        p1 = _mm256_maddubs_epi16(inVec[2 * j], w[2 * inDims / 32 + 2 * j]);
        p2 = _mm256_maddubs_epi16(inVec[2 * j + 1], w[2 * inDims / 32 + 2 * j + 1]);
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_add_epi16(p1, p2), kOnes));
        p1 = _mm256_maddubs_epi16(inVec[2 * j], w[3 * inDims / 32 + 2 * j]);
        p2 = _mm256_maddubs_epi16(inVec[2 * j + 1], w[3 * inDims / 32 + 2 * j + 1]);
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_add_epi16(p1, p2), kOnes));
      }
#endif
      s0 = _mm256_hadd_epi32(s0, s1);
      s2 = _mm256_hadd_epi32(s2, s3);
      s0 = _mm256_hadd_epi32(s0, s2);
      __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s0),
          _mm256_extracti128_si256(s0, 1));
      outVec[i] = _mm_add_epi32(sum128, biasVec[i]);
    }
  } else { // 32x32 multiplication
    __m256i *outVec = (__m256i *)output;
    __m256i *biasVec = (__m256i *)biases;
    __m128i *inVec = (__m128i *)input;
    __m256i in0 = _mm256_broadcastsi128_si256(inVec[0]);
    __m256i in1 = _mm256_broadcastsi128_si256(inVec[1]);
#if defined(USE_VNNI)
    const __m256i kZero = _mm256_setzero_si256();
    __m256i s0, s1, s2, s3;
    for (unsigned i = 0; i < outDims / 8; i++) {
      __m256i *w = (__m256i *)&weights[8 * i * 32];
      s0 = _mm256_dpbusd_epi32(kZero, in0, w[0]);
      s0 = _mm256_dpbusd_epi32(s0, in1, w[1]);
      s1 = _mm256_dpbusd_epi32(kZero, in0, w[2]);
      s1 = _mm256_dpbusd_epi32(s1, in1, w[3]);
      s2 = _mm256_dpbusd_epi32(kZero, in0, w[4]);
      s2 = _mm256_dpbusd_epi32(s2, in1, w[5]);
      s3 = _mm256_dpbusd_epi32(kZero, in0, w[6]);
      s3 = _mm256_dpbusd_epi32(s3, in1, w[7]);
      s0 = _mm256_hadd_epi32(s0, s1);
      s2 = _mm256_hadd_epi32(s2, s3);
      s0 = _mm256_hadd_epi32(s0, s2);
      outVec[i] = _mm256_add_epi32(s0, biasVec[i]);
    }
#else
    const __m256i kOnes = _mm256_set1_epi16(1);
    __m256i s0, s1, s2, s3, p;
    for (unsigned i = 0; i < outDims / 8; i++) {
      __m256i *w = (__m256i *)&weights[8 * i * 32];
      s0 = _mm256_maddubs_epi16(in0, w[0]); // first half of rows 0,4
      s0 = _mm256_madd_epi16(s0, kOnes);
      p  = _mm256_maddubs_epi16(in1, w[1]); // second half of rows 0,4
      p  = _mm256_madd_epi16(p, kOnes);
      s0 = _mm256_add_epi32(s0, p);
      s1 = _mm256_maddubs_epi16(in0, w[2]); // first half of rows 1,5
      s1 = _mm256_madd_epi16(s1, kOnes);
      p  = _mm256_maddubs_epi16(in1, w[3]); // second half of rows 1,5
      p  = _mm256_madd_epi16(p, kOnes);
      s1 = _mm256_add_epi32(s1, p);
      s2 = _mm256_maddubs_epi16(in0, w[4]); // first half of rows 2,6
      s2 = _mm256_madd_epi16(s2, kOnes);
      p  = _mm256_maddubs_epi16(in1, w[5]); // second half of rows 2,6
      p  = _mm256_madd_epi16(p, kOnes);
      s2 = _mm256_add_epi32(s2, p);
      s3 = _mm256_maddubs_epi16(in0, w[6]); // first half of rows 3,7
      s3 = _mm256_madd_epi16(s3, kOnes);
      p  = _mm256_maddubs_epi16(in1, w[7]); // second half of rows 3,7
      p  = _mm256_madd_epi16(p, kOnes);
      s3 = _mm256_add_epi32(s3, p);
      s0 = _mm256_hadd_epi32(s0, s1);
      s2 = _mm256_hadd_epi32(s2, s3);
      s0 = _mm256_hadd_epi32(s0, s2);
      outVec[i] = _mm256_add_epi32(s0, biasVec[i]);
    }
#endif
  }
#else
  { // 8 at a time does not seem to be an improvement
    __m256i *outVec = (__m256i *)output;
    __m256i *biasVec = (__m256i *)biases;
    __m256i *inVec = (__m256i *)input;
    for (unsigned i = 0; i < outDims / 8; i++) {
      __m256i *w = (__m256i *)&weights[8 * i * inDims];
      __m256i sum0 = _mm256_setzero_si256();
      __m256i sum1 = _mm256_setzero_si256();
      __m256i sum2 = _mm256_setzero_si256();
      __m256i sum3 = _mm256_setzero_si256();
      __m256i sum4 = _mm256_setzero_si256();
      __m256i sum5 = _mm256_setzero_si256();
      __m256i sum6 = _mm256_setzero_si256();
      __m256i sum7 = _mm256_setzero_si256();
#if defined(USE_VNNI)
      for (unsigned j = 0; j < inDims / 32; j++) {
        sum0 = _mm256_dpbusd_epi32(sum0, inVec[j], w[0 * inDims / 32 + j]);
        sum1 = _mm256_dpbusd_epi32(sum1, inVec[j], w[1 * inDims / 32 + j]);
        sum2 = _mm256_dpbusd_epi32(sum2, inVec[j], w[2 * inDims / 32 + j]);
        sum3 = _mm256_dpbusd_epi32(sum3, inVec[j], w[3 * inDims / 32 + j]);
        sum4 = _mm256_dpbusd_epi32(sum4, inVec[j], w[4 * inDims / 32 + j]);
        sum5 = _mm256_dpbusd_epi32(sum5, inVec[j], w[5 * inDims / 32 + j]);
        sum6 = _mm256_dpbusd_epi32(sum6, inVec[j], w[6 * inDims / 32 + j]);
        sum7 = _mm256_dpbusd_epi32(sum7, inVec[j], w[7 * inDims / 32 + j]);
      }
#else
      const __m256i kOnes = _mm256_set1_epi16(1);
      __m256i prod;
      for (unsigned j = 0; j < inDims / 32; j++) {
        prod = _mm256_maddubs_epi16(inVec[j], w[0 * inDims / 32 + j]);
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[1 * inDims / 32 + j]);
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[2 * inDims / 32 + j]);
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[3 * inDims / 32 + j]);
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[4 * inDims / 32 + j]);
        sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[5 * inDims / 32 + j]);
        sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[6 * inDims / 32 + j]);
        sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(prod, kOnes));
        prod = _mm256_maddubs_epi16(inVec[j], w[7 * inDims / 32 + j]);
        sum7 = _mm256_add_epi32(sum7, _mm256_madd_epi16(prod, kOnes));
      }
#endif
      sum0 = _mm256_hadd_epi32(sum0, sum1);
      sum2 = _mm256_hadd_epi32(sum2, sum3);
      sum4 = _mm256_hadd_epi32(sum4, sum5);
      sum6 = _mm256_hadd_epi32(sum6, sum7);
      sum0 = _mm256_hadd_epi32(sum0, sum2);
      sum4 = _mm256_hadd_epi32(sum4, sum6);
      sum2 = _mm256_permute2x128_si256(sum0, sum4, 0x20);
      sum6 = _mm256_permute2x128_si256(sum0, sum4, 0x31);
      outVec[i] = _mm256_add_epi32(_mm256_add_epi32(sum2, sum6), biasVec[i]);
    }
  }
#endif

#elif defined(USE_SSSE3)
  __m128i *outVec = (__m128i *)output;
  __m128i *biasVec = (__m128i *)biases;
  __m128i *inVec = (__m128i *)input;
  const __m128i kOnes = _mm_set1_epi16(1);
  for (unsigned i = 0; i < outDims / 4; i++) {
    __m128i *w = (__m128i *)&weights[4 * i * inDims], p1, p2, s0, s1, s2, s3;
    s0 = s1 = s2 = s3 = _mm_setzero_si128();
    for (unsigned j = 0; j < inDims / 32; j++) {
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[0 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[0 * inDims / 16 + 2 * j + 1]);
      s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_add_epi16(p1, p2), kOnes));
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[1 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[1 * inDims / 16 + 2 * j + 1]);
      s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_add_epi16(p1, p2), kOnes));
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[2 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[2 * inDims / 16 + 2 * j + 1]);
      s2 = _mm_add_epi32(s2, _mm_madd_epi16(_mm_add_epi16(p1, p2), kOnes));
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[3 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[3 * inDims / 16 + 2 * j + 1]);
      s3 = _mm_add_epi32(s3, _mm_madd_epi16(_mm_add_epi16(p1, p2), kOnes));
    }
    s0 = _mm_hadd_epi32(s0, s1);
    s2 = _mm_hadd_epi32(s2, s3);
    s0 = _mm_hadd_epi32(s0, s2);
    outVec[i] = _mm_add_epi32(s0, biasVec[i]);
  }

#elif defined(USE_SSE2)
  __m128i *outVec = (__m128i *)output;
  __m128i *biasVec = (__m128i *)biases;
  __m128i *inVec = (__m128i *)input;
  for (unsigned i = 0; i < outDims / 4; i++) {
    __m128i *w = (__m128i *)&weights[4 * i * inDims], p, s0, s1, s2, s3;
    s0 = s1 = s2 = s3 = _mm_setzero_si128();
    for (unsigned j = 0; j < inDims / 8; j++) {
      p = _mm_madd_epi16(inVec[j], w[0 * inDims / 8 + j]);
      s0 = _mm_add_epi32(s0, p);
      p = _mm_madd_epi16(inVec[j], w[1 * inDims / 8 + j]);
      s1 = _mm_add_epi32(s1, p);
      p = _mm_madd_epi16(inVec[j], w[2 * inDims / 8 + j]);
      s2 = _mm_add_epi32(s2, p);
      p = _mm_madd_epi16(inVec[j], w[3 * inDims / 8 + j]);
      s3 = _mm_add_epi32(s3, p);
    }
    s0 = _mm_add_epi32( _mm_unpacklo_epi32(s0, s1), _mm_unpackhi_epi32(s0, s1));
    s2 = _mm_add_epi32( _mm_unpacklo_epi32(s2, s3), _mm_unpackhi_epi32(s2, s3));
    s0 = _mm_add_epi32( _mm_unpacklo_epi64(s0, s2), _mm_unpackhi_epi64(s0, s2));
    outVec[i] = _mm_add_epi32(s0, biasVec[i]);
  }

#elif defined(USE_MMX)
  __m64 *outVec = (__m64 *)output;
  __m64 *biasVec = (__m64 *)biases;
  __m64 *inVec = (__m64 *)input;
  for (unsigned i = 0; i < outDims / 2; i++) {
    __m64 *w = (__m64 *)&weights[2 * i * inDims], p, s0, s1, s2, s3;
    s0 = s1 = s2 = s3 = _mm_setzero_si64();
    for (unsigned j = 0; j < inDims / 8; j++) {
      p = _mm_madd_pi16(inVec[2 * j + 0], w[0 * inDims / 4 + 2 * j + 0]);
      s0 = _mm_add_pi32(s0, p);
      p = _mm_madd_pi16(inVec[2 * j + 0], w[1 * inDims / 4 + 2 * j + 0]);
      s1 = _mm_add_pi32(s1, p);
      p = _mm_madd_pi16(inVec[2 * j + 1], w[0 * inDims / 4 + 2 * j + 1]);
      s2 = _mm_add_pi32(s2, p);
      p = _mm_madd_pi16(inVec[2 * j + 1], w[1 * inDims / 4 + 2 * j + 1]);
      s3 = _mm_add_pi32(s3, p);
    }
    s0 = _mm_add_pi32(s0, s2);
    s1 = _mm_add_pi32(s1, s3);
    s0 = _mm_add_pi32(_mm_unpacklo_pi32(s0, s1), _mm_unpackhi_pi32(s0, s1));
    outVec[i] = _mm_add_pi32(s0, biasVec[i]);
  }

#elif defined(USE_NEON)
  int32x4_t *outVec = (int32x4_t *)output;
  int32x4_t *biasVec = (int32x4_t *)biases;
  int8x8_t *inVec = (int8x8_t *)input;
  int16x8_t p;
  for (unsigned i = 0; i < outDims / 4; i++) {
    int8x8_t *w = (int8x8_t *)&weights[4 * i * inDims];
    int32x4_t s0 = { 0 }, s1 = { 0 }, s2 = { 0 }, s3 = { 0 };
    for (unsigned j = 0; j < inDims / 16; j++) {
      p = vmull_s8(inVec[2 * j], w[0 * inDims / 8 + 2 * j]);
      p = vmlal_s8(p, inVec[2 * j + 1], w[0 * inDims / 8 + 2 * j + 1]);
      s0 = vpadalq_s16(s0, p);
      p = vmull_s8(inVec[2 * j], w[1 * inDims / 8 + 2 * j]);
      p = vmlal_s8(p, inVec[2 * j + 1], w[1 * inDims / 8 + 2 * j + 1]);
      s1 = vpadalq_s16(s1, p);
      p = vmull_s8(inVec[2 * j], w[2 * inDims / 8 + 2 * j]);
      p = vmlal_s8(p, inVec[2 * j + 1], w[2 * inDims / 8 + 2 * j + 1]);
      s2 = vpadalq_s16(s2, p);
      p = vmull_s8(inVec[2 * j], w[3 * inDims / 8 + 2 * j]);
      p = vmlal_s8(p, inVec[2 * j + 1], w[3 * inDims / 8 + 2 * j + 1]);
      s3 = vpadalq_s16(s3, p);
    }
    s0 = vpaddq_s32(s0, s1);
    s2 = vpaddq_s32(s2, s3);
    s0 = vpaddq_s32(s0, s2);
    outVec[i] = vaddq_s32(s0, biasVec[i]);
  }

#else
  for (unsigned i = 0; i < outDims; i++) {
    unsigned int offset = i * inDims;
    int32_t sum = biases[i];
    for (unsigned j = 0; j < inDims; j++)
      sum += weights[offset + j] * input[j];
    output[i] = sum;
  }

#endif
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

  affine_propagate(B(input), B(hidden1_values), 512, 32,
      hidden1_biases, hidden1_weights);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 32);

  affine_propagate(B(hidden1_clipped), B(hidden2_values), 32, 32,
      hidden2_biases, hidden2_weights);
  clip_propagate(B(hidden2_values), B(hidden2_clipped), 32);

  out_value = output_layer(B(hidden2_clipped), output_biases, output_weights);

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
    c = bit_shuffle(c, 2, 2, 0x14);
#elif defined(USE_AVX2)
    c = bit_shuffle(c, 2, 1, 0x1c);
#endif
    w[c] = *d++;
  }
}

INLINE unsigned wt_idx(unsigned r, unsigned c, unsigned dims)
{
  (void)dims;

  unsigned k = r * dims + c;

#if defined(USE_AVX512)
  if (dims > 32)
    k = bit_shuffle(k, 1, 2, 0x38);
  else if (dims == 32) {
    k = bit_shuffle(k, 2, 2, 0x14);
    k = bit_shuffle(k, 2, 3, 0x1f0);
  }

#elif defined(USE_AVX2)
  if (dims > 32)
    k = bit_shuffle(k, 1, 1, 0x18);
  else if (dims == 32) {
    k = bit_shuffle(k, 2, 1, 0x1c);
    k = bit_shuffle(k, 1, 3, 0xf0);
  }

#endif

  return k;
}

#endif
