#ifdef NNUE_SPARSE

#if defined(USE_NEON) && !defined(IS_64BIT)
INLINE int16x8_t vmovl_high_s16(int8x16_t v)
{
  return vmovl_s16(vget_high_s16(v));
}
#endif

#ifdef IS_64BIT
typedef uint64_t mask2_t;
#else
typedef uint32_t mask2_t;
#endif

// InputLayer = InputSlice<256 * 2>
// out: 512 x int8_t

// Hidden1Layer = ClippedReLu<AffineTransform<InputLayer, 32>>
// 512 x int8_t -> 32 x int32_t -> 32 x int8_t

// Hidden2Layer = ClippedReLu<AffineTransform<hidden1, 32>>
// 32 x int8_t -> 32 x int32_t -> 32 x out_t

// OutputLayer = AffineTransform<HiddenLayer2, 1>
// 32 x out_t -> 1 x int32_t

#if !defined(USE_AVX512)
static alignas(64) weight_t hidden1_weights[32 * 512];
static alignas(64) weight_t hidden2_weights[32 * 32];
#else
static alignas(64) weight_t hidden1_weights[64 * 512];
static alignas(64) weight_t hidden2_weights[64 * 32];
#endif
static alignas(64) out_t output_weights[1 * 32];

static alignas(64) int32_t hidden1_biases[32];
static alignas(64) int32_t hidden2_biases[32];
static int32_t output_biases[1];

#ifdef VECTOR
INLINE bool next_idx(unsigned *idx, unsigned *offset, mask2_t *v,
    mask_t *mask, unsigned dims)
{
  while (*v == 0) {
    *offset += 8 * sizeof(mask2_t);
    if (*offset >= dims) return false;
    memcpy(v, (char *)mask + (*offset / 8), sizeof(mask2_t));
  }
#ifdef IS_64BIT
  *idx = *offset + __builtin_ctzll(*v);
#else
  *idx = *offset + __builtin_ctzl(*v);
#endif
  *v &= *v - 1;
  return true;
}
#endif

INLINE void hidden_layer(const int8_t *input, void *output, unsigned dims,
    const int32_t *biases, const weight_t *weights, mask_t *inMask,
    mask_t *outMask, const bool pack8_and_calc_mask)
{
#if defined(USE_AVX512)
  const __m512i kZero = _mm512_setzero_si512();
  __m512i out_0 = ((__m512i *)biases)[0];
  __m512i out_1 = ((__m512i *)biases)[1];
  __m512i first, second;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = ((__m512i *)weights)[idx];
    uint16_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = ((__m512i *)weights)[idx];
      factor |= input[idx] << 8;
    } else {
      second = kZero;
    }
    __m512i mul = _mm512_set1_epi16(factor), prod, signs;
    prod = _mm512_maddubs_epi16(mul, _mm512_unpacklo_epi8(first, second));
    signs = _mm512_srai_epi16(prod, 15);
    out_0 = _mm512_add_epi32(out_0, _mm512_unpacklo_epi16(prod, signs));
    out_1 = _mm512_add_epi32(out_1, _mm512_unpackhi_epi16(prod, signs));
  }

  __m512i out16 = _mm512_srai_epi16(_mm512_packs_epi32(out_0, out_1), SHIFT);

  __m256i *outVec = (__m256i *)output;
  const __m256i kZero256 = _mm256_setzero_si256();
  outVec[0] = _mm256_packs_epi16(
      _mm512_castsi512_si256(out16),_mm512_extracti64x4_epi64(out16, 1));
  if (pack8_and_calc_mask)
    outMask[0] = (uint32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(outVec[0], kZero256));
  else
    outVec[0] = _mm256_max_epi8(outVec[0], kZero256);

#elif defined(USE_AVX2)
  const __m256i kZero = _mm256_setzero_si256();
  __m256i out_0 = ((__m256i *)biases)[0];
  __m256i out_1 = ((__m256i *)biases)[1];
  __m256i out_2 = ((__m256i *)biases)[2];
  __m256i out_3 = ((__m256i *)biases)[3];
  __m256i first, second;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = ((__m256i *)weights)[idx];
    uint16_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
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
  outVec[0] = _mm256_packs_epi16(out16_0, out16_1);
  if (pack8_and_calc_mask)
    outMask[0] = _mm256_movemask_epi8(_mm256_cmpgt_epi8(outVec[0], kZero));
  else
    outVec[0] = _mm256_max_epi8(outVec[0], kZero);

#elif AVOID_USE_SSSE3
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
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (__m128i *)&weights[32 * idx];
    uint16_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = (__m128i *)&weights[32 * idx];
      factor |= input[idx] << 8;
    } else {
      second = kZeros;
    }
    __m128i mul = _mm_set1_epi16(factor), prod, signs;
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[0], second[0]));
    signs = _mm_cmpgt_epi16(kZeros[0], prod);
    out_0 = _mm_add_epi32(out_0, _mm_unpacklo_epi16(prod, signs));
    out_1 = _mm_add_epi32(out_1, _mm_unpackhi_epi16(prod, signs));
    prod = _mm_maddubs_epi16(mul, _mm_unpackhi_epi8(first[0], second[0]));
    signs = _mm_cmpgt_epi16(kZeros[0], prod);
    out_2 = _mm_add_epi32(out_2, _mm_unpacklo_epi16(prod, signs));
    out_3 = _mm_add_epi32(out_3, _mm_unpackhi_epi16(prod, signs));
    prod = _mm_maddubs_epi16(mul, _mm_unpacklo_epi8(first[1], second[1]));
    signs = _mm_cmpgt_epi16(kZeros[0], prod);
    out_4 = _mm_add_epi32(out_4, _mm_unpacklo_epi16(prod, signs));
    out_5 = _mm_add_epi32(out_5, _mm_unpackhi_epi16(prod, signs));
    prod = _mm_maddubs_epi16(mul, _mm_unpackhi_epi8(first[1], second[1]));
    signs = _mm_cmpgt_epi16(kZeros[0], prod);
    out_6 = _mm_add_epi32(out_6, _mm_unpacklo_epi16(prod, signs));
    out_7 = _mm_add_epi32(out_7, _mm_unpackhi_epi16(prod, signs));
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
    outVec[0] = _mm_packs_epi16(
        _mm_max_epi16(out16_0, kZeros[0]), _mm_max_epi16(out16_1, kZeros[0]));
    outVec[1] = _mm_packs_epi16(
        _mm_max_epi16(out16_2, kZeros[0]), _mm_max_epi16(out16_3, kZeros[0]));
#endif
  }

#elif defined(USE_SSE2)
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
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (__m128i *)&weights[32 * idx];
    uint32_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = (__m128i *)&weights[32 * idx];
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
    const __m128i kx07f = _mm_set1_epi16(127);
    outVec[0] = _mm_min_epi16(_mm_max_epi16(out16_0, kZeros[0]), kx07f);
    outVec[1] = _mm_min_epi16(_mm_max_epi16(out16_1, kZeros[0]), kx07f);
    outVec[2] = _mm_min_epi16(_mm_max_epi16(out16_2, kZeros[0]), kx07f);
    outVec[3] = _mm_min_epi16(_mm_max_epi16(out16_3, kZeros[0]), kx07f);
  }

#elif defined(USE_MMX)

#if 0
  const __m64 kZeros[2] = { 0 };
  for (unsigned t = 0; t < 4; t++) {
    __m64 out_0 = ((__m64 *)biases)[4 * t + 0];
    __m64 out_1 = ((__m64 *)biases)[4 * t + 1];
    __m64 out_2 = ((__m64 *)biases)[4 * t + 2];
    __m64 out_3 = ((__m64 *)biases)[4 * t + 3];
    const __m64 *first, *second;
    mask2_t v;
    unsigned idx;

    memcpy(&v, inMask, sizeof(mask2_t));
    for (unsigned offset = 0; offset < dims;) {
      if (!next_idx(&idx, &offset, &v, inMask, dims))
        break;
      first = &((__m64 *)&weights[32 * idx])[2  * t];
      uint32_t factor = input[idx];
      if (next_idx(&idx, &offset, &v, inMask, dims)) {
        second = &((__m64 *)&weights[32 * idx])[2 * t];
        factor |= input[idx] << 16;
      } else {
        second = kZeros;
      }
      __m64 mul = _mm_set1_pi32(factor);
      out_0 = _mm_add_pi32(out_0, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[0],second[0])));
      out_1 = _mm_add_pi32(out_1, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[0],second[0])));
      out_2 = _mm_add_pi32(out_2, _mm_madd_pi16(mul, _mm_unpacklo_pi16(first[1],second[1])));
      out_3 = _mm_add_pi32(out_3, _mm_madd_pi16(mul, _mm_unpackhi_pi16(first[1],second[1])));
    }

    __m64 out16_0 = _mm_srai_pi16(_mm_packs_pi32(out_0, out_1), SHIFT);
    __m64 out16_1 = _mm_srai_pi16(_mm_packs_pi32(out_2, out_3), SHIFT);

    __m64 *outVec = (__m64 *)output;
    if (pack8_and_calc_mask) {
      outVec[t] = _mm_packs_pi16(out16_0, out16_1);
      outMask[t] = _mm_movemask_pi8(_mm_cmpgt_pi8(outVec[t], kZeros[0]));
    } else {
#ifdef USE_SSE
      const __m64 kx07f = _mm_set1_pi16(127);
      outVec[2 * t] = _mm_min_pi16(_mm_max_pi16(out16_0, kZeros[0]), kx07f);
      outVec[2 * t + 1] = _mm_min_pi16(_mm_max_pi16(out16_1, kZeros[0]), kx07f);
#else
      const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
      const __m64 k0x0080 = _mm_set1_pi16(0x0080);
      const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
      outVec[2 * t] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_0, k0x7f80), k0x0080), k0x8000);
      outVec[2 * t + 1] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_1, k0x7f80), k0x0080), k0x8000);
#endif
    }
  }
#else
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
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (__m64 *)&weights[32 * idx];
    uint32_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = (__m64 *)&weights[32 * idx];
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
#ifdef USE_SSE
    const __m64 kx07f = _mm_set1_pi16(127);
    outVec[0] = _mm_min_pi16(_mm_max_pi16(out16_0, kZeros[0]), kx07f);
    outVec[1] = _mm_min_pi16(_mm_max_pi16(out16_1, kZeros[0]), kx07f);
    outVec[2] = _mm_min_pi16(_mm_max_pi16(out16_2, kZeros[0]), kx07f);
    outVec[3] = _mm_min_pi16(_mm_max_pi16(out16_3, kZeros[0]), kx07f);
    outVec[4] = _mm_min_pi16(_mm_max_pi16(out16_4, kZeros[0]), kx07f);
    outVec[5] = _mm_min_pi16(_mm_max_pi16(out16_5, kZeros[0]), kx07f);
    outVec[6] = _mm_min_pi16(_mm_max_pi16(out16_6, kZeros[0]), kx07f);
    outVec[7] = _mm_min_pi16(_mm_max_pi16(out16_7, kZeros[0]), kx07f);
#else
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
#endif
  }
#endif

#elif defined(USE_NEON)
  int32x4_t out_0 = ((int32x4_t *)biases)[0];
  int32x4_t out_1 = ((int32x4_t *)biases)[1];
  int32x4_t out_2 = ((int32x4_t *)biases)[2];
  int32x4_t out_3 = ((int32x4_t *)biases)[3];
  int32x4_t out_4 = ((int32x4_t *)biases)[4];
  int32x4_t out_5 = ((int32x4_t *)biases)[5];
  int32x4_t out_6 = ((int32x4_t *)biases)[6];
  int32x4_t out_7 = ((int32x4_t *)biases)[7];
  const int8x8_t *first;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (int8x8_t *)&weights[32 * idx];
    int16_t factor = input[idx];

    int16x8_t prod;
    prod = vmulq_n_s16(vmovl_s8(first[0]), factor);
    out_0 = vaddq_s32(out_0, vmovl_s16(vget_low_s16(prod)));
    out_1 = vaddq_s32(out_1, vmovl_high_s16(prod));
    prod = vmulq_n_s16(vmovl_s8(first[1]), factor);
    out_2 = vaddq_s32(out_2, vmovl_s16(vget_low_s16(prod)));
    out_3 = vaddq_s32(out_3, vmovl_high_s16(prod));
    prod = vmulq_n_s16(vmovl_s8(first[2]), factor);
    out_4 = vaddq_s32(out_4, vmovl_s16(vget_low_s16(prod)));
    out_5 = vaddq_s32(out_5, vmovl_high_s16(prod));
    prod = vmulq_n_s16(vmovl_s8(first[3]), factor);
    out_6 = vaddq_s32(out_6, vmovl_s16(vget_low_s16(prod)));
    out_7 = vaddq_s32(out_7, vmovl_high_s16(prod));
  }

  int16x8_t out16_0 = vcombine_s16(vqshrn_n_s32(out_0, SHIFT), vqshrn_n_s32(out_1, SHIFT));
  int16x8_t out16_1 = vcombine_s16(vqshrn_n_s32(out_2, SHIFT), vqshrn_n_s32(out_3, SHIFT));
  int16x8_t out16_2 = vcombine_s16(vqshrn_n_s32(out_4, SHIFT), vqshrn_n_s32(out_5, SHIFT));
  int16x8_t out16_3 = vcombine_s16(vqshrn_n_s32(out_6, SHIFT), vqshrn_n_s32(out_7, SHIFT));

  if (pack8_and_calc_mask) {
    const int8x16_t kZero = { 0 };
    int8x16_t *outVec = (int8x16_t *)output;
    outVec[0] = vcombine_s8(vqmovn_s16(out16_0), vqmovn_s16(out16_1));
    outMask[0] = neon_movemask(vcgtq_s8(outVec[0], kZero));
    outVec[1] = vcombine_s8(vqmovn_s16(out16_2), vqmovn_s16(out16_3));
    outMask[1] = neon_movemask(vcgtq_s8(outVec[1], kZero));
  } else {
    // The next step takes int8x8_t as input, so store as int8x8_t
    const int8x8_t kZero = { 0 };
    int8x8_t *outVec = (int8x8_t *)output;
    outVec[0] = vmax_s8(vqmovn_s16(out16_0), kZero);
    outVec[1] = vmax_s8(vqmovn_s16(out16_1), kZero);
    outVec[2] = vmax_s8(vqmovn_s16(out16_2), kZero);
    outVec[3] = vmax_s8(vqmovn_s16(out16_3), kZero);
  }

#else /* generic fallback */
  (void)inMask; (void)outMask; (void)pack8_and_calc_mask;

  int32_t tmp[32];

  for (unsigned i = 0; i < 32; i++)
    tmp[i] = biases[i];

  for (unsigned idx = 0; idx < dims; idx++)
    if (input[idx])
      for (unsigned i = 0; i < 32; i++)
        tmp[i] += (int8_t)input[idx] * weights[32 * idx + i];

  int8_t *outVec = (int8_t *)output;
  for (unsigned i = 0; i < 32; i++)
    outVec[i] = clamp(tmp[i] >> SHIFT, 0, 127);

#endif
}

struct NetData {
  alignas(64) int8_t input[512];
  int8_t hidden1_out[32];
  out_t hidden2_out[32];
};

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

static void read_output_weights(out_t *w, const char *d)
{
  for (unsigned i = 0; i < 32; i++) {
    unsigned c = i;
#if defined(USE_AVX512)
    c = bit_shuffle(c, 1, 1, 0x18);
#endif
    w[c] = *d++;
  }
}

INLINE unsigned wt_idx(unsigned r, unsigned c, unsigned dims)
{
  (void)dims;

#if defined(USE_AVX512)
  if (dims > 32)
    c = bit_shuffle(c, 1, 2, 0x38);
  else if (dims == 32)
    c = bit_shuffle(c, 1, 1, 0x18);

#elif defined(USE_AVX2)
  if (dims > 32)
    c = bit_shuffle(c, 1, 1, 0x18);

#endif

#if defined(USE_AVX512)
  return c * 64 + r + (r & ~7);

#else
  return c * 32 + r;

#endif
}

#ifdef USE_AVX2
static void permute_biases(int32_t *biases)
{
  __m128i *b = (__m128i *)biases;
  __m128i tmp[8];
#ifdef USE_AVX512
  tmp[0] = b[0];
  tmp[1] = b[2];
  tmp[2] = b[4];
  tmp[3] = b[6];
  tmp[4] = b[1];
  tmp[5] = b[3];
  tmp[6] = b[5];
  tmp[7] = b[7];
#elif USE_AVX2
  tmp[0] = b[0];
  tmp[1] = b[4];
  tmp[2] = b[1];
  tmp[3] = b[5];
  tmp[4] = b[2];
  tmp[5] = b[6];
  tmp[6] = b[3];
  tmp[7] = b[7];
#else
#error
#endif
  memcpy(b, tmp, 8 * sizeof(__m128i));
}
#endif

#endif
