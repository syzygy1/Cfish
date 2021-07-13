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
// 32 x int8_t -> 32 x int32_t -> 32 x out_t_sparse

// OutputLayer = AffineTransform<HiddenLayer2, 1>
// 32 x out_t_sparse -> 1 x int32_t

static alignas(64) weight_t_sparse hidden1_weights[8][16 * 1024];
static alignas(64) weight_t hidden2_weights[8][32 * 32];
static alignas(64) weight_t output_weights[8][1 * 32];

static alignas(64) int32_t hidden1_biases[8][16];
static alignas(64) int32_t hidden2_biases[8][32];
static int32_t output_biases[8][1];

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
    const int32_t *biases, const weight_t *weights, mask_t *inMask)
{
#if defined(USE_SSSE3) && !AVOID_USE_SSSE3
  const __m128i kZeros[2] = { 0 };
  __m128i out_0 = ((__m128i *)biases)[0];
  __m128i out_1 = ((__m128i *)biases)[1];
  __m128i out_2 = ((__m128i *)biases)[2];
  __m128i out_3 = ((__m128i *)biases)[3];
  const __m128i *first, *second;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (__m128i *)&weights[16 * idx];
    uint16_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = (__m128i *)&weights[16 * idx];
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
  }

  __m128i out16_0 = _mm_srai_epi16(_mm_packs_epi32(out_0, out_1), SHIFT);
  __m128i out16_1 = _mm_srai_epi16(_mm_packs_epi32(out_2, out_3), SHIFT);

  __m128i *outVec = (__m128i *)output;
#if defined(USE_SSE41)
  outVec[0] = _mm_max_epi8(_mm_packs_epi16(out16_0, out16_1), kZeros[0]);
#else
  outVec[0] = _mm_packs_epi16(
      _mm_max_epi16(out16_0, kZeros[0]), _mm_max_epi16(out16_1, kZeros[0]));
#endif

#elif defined(USE_SSE2)
  const __m128i kZeros[4] = { 0 };
  __m128i out_0 = ((__m128i *)biases)[0];
  __m128i out_1 = ((__m128i *)biases)[1];
  __m128i out_2 = ((__m128i *)biases)[2];
  __m128i out_3 = ((__m128i *)biases)[3];
  const __m128i *first, *second;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (__m128i *)&weights[16 * idx];
    uint32_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = (__m128i *)&weights[16 * idx];
      factor |= input[idx] << 16;
    } else {
      second = kZeros;
    }
    __m128i mul = _mm_set1_epi32(factor);
    out_0 = _mm_add_epi32(out_0, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[0],second[0])));
    out_1 = _mm_add_epi32(out_1, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[0],second[0])));
    out_2 = _mm_add_epi32(out_2, _mm_madd_epi16(mul, _mm_unpacklo_epi16(first[1],second[1])));
    out_3 = _mm_add_epi32(out_3, _mm_madd_epi16(mul, _mm_unpackhi_epi16(first[1],second[1])));
  }

  __m128i out16_0 = _mm_srai_epi16(_mm_packs_epi32(out_0, out_1), SHIFT);
  __m128i out16_1 = _mm_srai_epi16(_mm_packs_epi32(out_2, out_3), SHIFT);

  __m128i *outVec = (__m128i *)output;
  const __m128i kx07f = _mm_set1_epi16(127);
  outVec[0] = _mm_min_epi16(_mm_max_epi16(out16_0, kZeros[0]), kx07f);
  outVec[1] = _mm_min_epi16(_mm_max_epi16(out16_1, kZeros[0]), kx07f);

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
      first = &((__m64 *)&weights[16 * idx])[2  * t];
      uint32_t factor = input[idx];
      if (next_idx(&idx, &offset, &v, inMask, dims)) {
        second = &((__m64 *)&weights[16 * idx])[2 * t];
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
  const __m64 *first, *second;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (__m64 *)&weights[16 * idx];
    uint32_t factor = input[idx];
    if (next_idx(&idx, &offset, &v, inMask, dims)) {
      second = (__m64 *)&weights[16 * idx];
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
  }

  __m64 out16_0 = _mm_srai_pi16(_mm_packs_pi32(out_0, out_1), SHIFT);
  __m64 out16_1 = _mm_srai_pi16(_mm_packs_pi32(out_2, out_3), SHIFT);
  __m64 out16_2 = _mm_srai_pi16(_mm_packs_pi32(out_4, out_5), SHIFT);
  __m64 out16_3 = _mm_srai_pi16(_mm_packs_pi32(out_6, out_7), SHIFT);

  __m64 *outVec = (__m64 *)output;
#ifdef USE_SSE
  const __m64 kx07f = _mm_set1_pi16(127);
  outVec[0] = _mm_min_pi16(_mm_max_pi16(out16_0, kZeros[0]), kx07f);
  outVec[1] = _mm_min_pi16(_mm_max_pi16(out16_1, kZeros[0]), kx07f);
  outVec[2] = _mm_min_pi16(_mm_max_pi16(out16_2, kZeros[0]), kx07f);
  outVec[3] = _mm_min_pi16(_mm_max_pi16(out16_3, kZeros[0]), kx07f);
#else
  const __m64 k0x7f80 = _mm_set1_pi16(0x7f80);
  const __m64 k0x0080 = _mm_set1_pi16(0x0080);
  const __m64 k0x8000 = _mm_set1_pi16(-0x8000);
  outVec[0] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_0, k0x7f80), k0x0080), k0x8000);
  outVec[1] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_1, k0x7f80), k0x0080), k0x8000);
  outVec[2] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_2, k0x7f80), k0x0080), k0x8000);
  outVec[3] = _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(out16_3, k0x7f80), k0x0080), k0x8000);
#endif
#endif

#elif defined(USE_NEON)
  int32x4_t out_0 = ((int32x4_t *)biases)[0];
  int32x4_t out_1 = ((int32x4_t *)biases)[1];
  int32x4_t out_2 = ((int32x4_t *)biases)[2];
  int32x4_t out_3 = ((int32x4_t *)biases)[3];
  const int8x8_t *first;
  mask2_t v;
  unsigned idx;

  memcpy(&v, inMask, sizeof(mask2_t));
  for (unsigned offset = 0; offset < dims;) {
    if (!next_idx(&idx, &offset, &v, inMask, dims))
      break;
    first = (int8x8_t *)&weights[16 * idx];
    int16_t factor = input[idx];

    int16x8_t prod;
    prod = vmulq_n_s16(vmovl_s8(first[0]), factor);
    out_0 = vaddq_s32(out_0, vmovl_s16(vget_low_s16(prod)));
    out_1 = vaddq_s32(out_1, vmovl_high_s16(prod));
    prod = vmulq_n_s16(vmovl_s8(first[1]), factor);
    out_2 = vaddq_s32(out_2, vmovl_s16(vget_low_s16(prod)));
    out_3 = vaddq_s32(out_3, vmovl_high_s16(prod));
  }

  int16x8_t out16_0 = vcombine_s16(vqshrn_n_s32(out_0, SHIFT), vqshrn_n_s32(out_1, SHIFT));
  int16x8_t out16_1 = vcombine_s16(vqshrn_n_s32(out_2, SHIFT), vqshrn_n_s32(out_3, SHIFT));

  // The next step takes int8x8_t as input, so store as int8x8_t
  const int8x8_t kZero = { 0 };
  int8x8_t *outVec = (int8x8_t *)output;
  outVec[0] = vmax_s8(vqmovn_s16(out16_0), kZero);
  outVec[1] = vmax_s8(vqmovn_s16(out16_1), kZero);

#else /* generic fallback */
  (void)inMask;

  int32_t tmp[16];

  for (unsigned i = 0; i < 16; i++)
    tmp[i] = biases[i];

  for (unsigned idx = 0; idx < dims; idx++)
    if (input[idx])
      for (unsigned i = 0; i < 16; i++)
        tmp[i] += (int8_t)input[idx] * weights[16 * idx + i];

  int8_t *outVec = (int8_t *)output;
  for (unsigned i = 0; i < 16; i++)
    outVec[i] = clamp(tmp[i] >> SHIFT, 0, 127);

#endif
}

struct NetData {
  alignas(64) int8_t input[1024];
  out_t_sparse hidden1_out[32];
  clipped_t hidden1_clipped[32];
  int32_t hidden2_values[32];
  clipped_t hidden2_clipped[32];
};

// Evaluation function
Value nnue_evaluate(const Position *pos)
{
  int32_t out_value;
  alignas(8) mask_t hidden1_mask[1024 / (8 * sizeof(mask_t))];
#ifdef ALIGNMENT_HACK // work around a bug in old gcc on Windows
  uint8_t buf[sizeof(struct NetData) + 63];
  struct NetData *b = (struct NetData *)(buf + ((((uintptr_t)buf-1) ^ 0x3f) & 0x3f));
#define B(x) (b->x)
#else
  struct NetData buf;
#define B(x) (buf.x)
#endif

  int32_t bucket = (popcount(pieces()) - 1) / 4;
  int32_t psqt_val;

  if (transform(pos, B(input), hidden1_mask, bucket, &psqt_val))
  {
#if defined(USE_MMX)
  _mm_empty();
#endif
    return psqt_val / FV_SCALE;
  }
  else
  {
    hidden_layer(B(input), B(hidden1_out), 1024, hidden1_biases[bucket],
        hidden1_weights[bucket], hidden1_mask);
    for (unsigned i = 0; i < 16; ++i)
      B(hidden1_clipped)[i] = B(hidden1_out)[i];
    for (unsigned i = 16; i < 32; ++i)
      B(hidden1_clipped)[i] = 0;

    affine_propagate(B(hidden1_clipped), B(hidden2_values), 32, 32,
        hidden2_biases[bucket], hidden2_weights[bucket]);
    clip_propagate(B(hidden2_values), B(hidden2_clipped), 32);

    out_value = output_layer(B(hidden2_clipped), output_biases[bucket], output_weights[bucket]);

  #if defined(USE_MMX)
    _mm_empty();
  #endif

    return (out_value + psqt_val) / FV_SCALE;
  }
}

INLINE unsigned wt_idx_sparse(unsigned r, unsigned c, unsigned dims)
{
  (void)dims;

  return c * 16 + r;

}

#endif
