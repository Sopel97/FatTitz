#ifdef NNUE_REGULAR
#error "Not supported"
// InputLayer = InputSlice<256 * 2>
// out: 512 x clipped_t

// Hidden1Layer = ClippedReLu<AffineTransform<InputLayer, 32>>
// 512 x clipped_t -> 32 x int32_t -> 32 x clipped_t

// Hidden2Layer = ClippedReLu<AffineTransform<hidden1, 32>>
// 32 x clipped_t -> 32 x int32_t -> 32 x clipped_t

// OutputLayer = AffineTransform<HiddenLayer2, 1>
// 32 x clipped_t -> 1 x int32_t

static alignas(64) weight_t hidden1_weights[8][16 * 1024];
static alignas(64) weight_t hidden2_weights[8][32 * 32];
static alignas(64) weight_t output_weights[8][1 * 32];

static alignas(64) int32_t hidden1_biases[8][16];
static alignas(64) int32_t hidden2_biases[8][32];
static int32_t output_biases[8][1];

#endif

INLINE void affine_propagate(clipped_t *input, int32_t *output,
    unsigned inDims, unsigned outDims, int32_t *biases, weight_t *weights)
{
  assert(inDims == 32 || inDims % 64 == 0);
  assert(outDims % 8 == 0);

#if defined(USE_SSSE3)
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
      s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_adds_epi16(p1, p2), kOnes));
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[1 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[1 * inDims / 16 + 2 * j + 1]);
      s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_adds_epi16(p1, p2), kOnes));
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[2 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[2 * inDims / 16 + 2 * j + 1]);
      s2 = _mm_add_epi32(s2, _mm_madd_epi16(_mm_adds_epi16(p1, p2), kOnes));
      p1 = _mm_maddubs_epi16(inVec[2 * j], w[3 * inDims / 16 + 2 * j]);
      p2 = _mm_maddubs_epi16(inVec[2 * j + 1], w[3 * inDims / 16 + 2 * j + 1]);
      s3 = _mm_add_epi32(s3, _mm_madd_epi16(_mm_adds_epi16(p1, p2), kOnes));
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
      s0 = _mm_adds_epi32(s0, p);
      p = _mm_madd_epi16(inVec[j], w[1 * inDims / 8 + j]);
      s1 = _mm_adds_epi32(s1, p);
      p = _mm_madd_epi16(inVec[j], w[2 * inDims / 8 + j]);
      s2 = _mm_adds_epi32(s2, p);
      p = _mm_madd_epi16(inVec[j], w[3 * inDims / 8 + j]);
      s3 = _mm_adds_epi32(s3, p);
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
      s0 = _mm_adds_pi32(s0, p);
      p = _mm_madd_pi16(inVec[2 * j + 0], w[1 * inDims / 4 + 2 * j + 0]);
      s1 = _mm_adds_pi32(s1, p);
      p = _mm_madd_pi16(inVec[2 * j + 1], w[0 * inDims / 4 + 2 * j + 1]);
      s2 = _mm_adds_pi32(s2, p);
      p = _mm_madd_pi16(inVec[2 * j + 1], w[1 * inDims / 4 + 2 * j + 1]);
      s3 = _mm_adds_pi32(s3, p);
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
  assert(numDims % 32 == 0);

#if defined(USE_SSSE3)
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

#ifdef NNUE_REGULAR
struct NetData {
  alignas(64) clipped_t input[1024];
  int32_t hidden1_values[32];
  int32_t hidden2_values[32];
  clipped_t hidden1_clipped[32];
  clipped_t hidden2_clipped[32];
};

// Evaluation function
Value nnue_evaluate(const Position *pos, bool adjusted)
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

  int32_t bucket = (popcount(pieces()) - 1) / 4;
  int32_t psqt_val;

  transform(pos, B(input), NULL, bucket, &psqt_val);

  affine_propagate(B(input), B(hidden1_values), 1024, 16,
      hidden1_biases[bucket], hidden1_weights[bucket]);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 32);
  memset(B(hidden1_clipped) + 16, 0, sizeof(clipped_t) * 16);

  affine_propagate(B(hidden1_clipped), B(hidden2_values), 32, 32,
      hidden2_biases[bucket], hidden2_weights[bucket]);
  clip_propagate(B(hidden2_values), B(hidden2_clipped), 32);

  out_value = output_layer(B(hidden2_clipped), output_biases[bucket], output_weights[bucket]);

#if defined(USE_MMX)
  _mm_empty();
#endif

  int materialist = psqt_val;
  int positional = out_value;

  int delta_npm = abs(non_pawn_material_c(WHITE) - non_pawn_material_c(BLACK));
  int entertainment = (adjusted && delta_npm <= BishopValueMg - KnightValueMg ? 7 : 0);

  int A = 128 - entertainment;
  int B = 128 + entertainment;

  int sum = (A * materialist + B * positional) / 128;

  return sum / FV_SCALE;
}
#endif

static const char* read_output_weights_dense(weight_t *w, const char *d)
{
  for (unsigned i = 0; i < 64; i++) {
    unsigned c = i;
    w[c] = *d++;
  }
  return d;
}

INLINE unsigned wt_idx_dense(unsigned r, unsigned c, unsigned dims)
{
  (void)dims;

  unsigned k = r * dims + c;

  return k;
}
