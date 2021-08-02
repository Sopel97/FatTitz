#ifdef NNUE_SPARSE

#if defined(USE_NEON) || !defined(IS_64BIT)
#error "Not supported"
#endif

#if !defined(USE_SSE2)
#error "Not supported"
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

static alignas(64) weight_t_sparse hidden1_weights[8][64 * 2049];
static alignas(64) weight_t hidden2_weights[8][64 * 64];
static alignas(64) weight_t output_weights[8][1 * 64];

static alignas(64) int32_t hidden1_biases[8][64];
static alignas(64) int32_t hidden2_biases[8][64];
static int32_t output_biases[8][1];

INLINE void hidden_layer(const clipped_t *input, int32_t *output, unsigned dims,
    const int32_t *biases, const weight_t_sparse *weights, uint16_t *nnz_indices, int num_nnz_indices)
{

  const int PaddedOutputDimensions = 64;

#if defined (USE_AVX2)

  const int ChunkSize = 8;
  const int NumChunks = 8;
  const int TileSize = NumChunks * ChunkSize;
  const int NumTiles = PaddedOutputDimensions / TileSize;

  const __m256i ones = _mm256_set1_epi16(1);

  while (num_nnz_indices % 4 != 0)
    nnz_indices[num_nnz_indices++] = dims;

  __m256i acc[NumChunks];

  for (int i = 0; i < NumTiles; ++i)
  {
    const __m256i* biasesTile = (&biases[i * TileSize]);
    __m256i* outputTile = (&output[i * TileSize]);

    for (int k = 0; k < NumChunks; ++k)
      acc[k] = _mm256_setzero_si256();

    for (int j = 0; j < num_nnz_indices; j += 4)
    {
      const __m256i mul0 = _mm256_set1_epi16(input[nnz_indices[j+0]] | (input[nnz_indices[j+1]] << 8));
      const __m256i mul2 = _mm256_set1_epi16(input[nnz_indices[j+2]] | (input[nnz_indices[j+3]] << 8));
      const __m256i* col0 = (&weights[nnz_indices[j+0] * PaddedOutputDimensions + i * TileSize]);
      const __m256i* col1 = (&weights[nnz_indices[j+1] * PaddedOutputDimensions + i * TileSize]);
      const __m256i* col2 = (&weights[nnz_indices[j+2] * PaddedOutputDimensions + i * TileSize]);
      const __m256i* col3 = (&weights[nnz_indices[j+3] * PaddedOutputDimensions + i * TileSize]);
      for (int k = 0; k < NumChunks / 4; ++k)
      {
        __m256i prod0 = _mm256_maddubs_epi16(mul0, _mm256_unpacklo_epi8(col0[k], col1[k]));
        __m256i prod1 = _mm256_maddubs_epi16(mul0, _mm256_unpackhi_epi8(col0[k], col1[k]));
        __m256i prod2 = _mm256_maddubs_epi16(mul2, _mm256_unpacklo_epi8(col2[k], col3[k]));
        __m256i prod3 = _mm256_maddubs_epi16(mul2, _mm256_unpackhi_epi8(col2[k], col3[k]));
        acc[k*4 + 0] = _mm256_add_epi32(acc[k*4 + 0], _mm256_madd_epi16(ones, _mm256_unpacklo_epi16(prod0, prod2)));
        acc[k*4 + 1] = _mm256_add_epi32(acc[k*4 + 1], _mm256_madd_epi16(ones, _mm256_unpackhi_epi16(prod0, prod2)));
        acc[k*4 + 2] = _mm256_add_epi32(acc[k*4 + 2], _mm256_madd_epi16(ones, _mm256_unpacklo_epi16(prod1, prod3)));
        acc[k*4 + 3] = _mm256_add_epi32(acc[k*4 + 3], _mm256_madd_epi16(ones, _mm256_unpackhi_epi16(prod1, prod3)));
      }
    }

    for (int k = 0; k < NumChunks / 4; ++k)
    {
      __m128i acc00 = _mm256_extracti128_si256(acc[k*4 + 0], 0);
      __m128i acc01 = _mm256_extracti128_si256(acc[k*4 + 0], 1);
      __m128i acc10 = _mm256_extracti128_si256(acc[k*4 + 1], 0);
      __m128i acc11 = _mm256_extracti128_si256(acc[k*4 + 1], 1);
      __m128i acc20 = _mm256_extracti128_si256(acc[k*4 + 2], 0);
      __m128i acc21 = _mm256_extracti128_si256(acc[k*4 + 2], 1);
      __m128i acc30 = _mm256_extracti128_si256(acc[k*4 + 3], 0);
      __m128i acc31 = _mm256_extracti128_si256(acc[k*4 + 3], 1);

      outputTile[k*4 + 0] = _mm256_add_epi32(_mm256_setr_m128i(acc00, acc10), biasesTile[k*4 + 0]);
      outputTile[k*4 + 1] = _mm256_add_epi32(_mm256_setr_m128i(acc20, acc30), biasesTile[k*4 + 1]);
      outputTile[k*4 + 2] = _mm256_add_epi32(_mm256_setr_m128i(acc01, acc11), biasesTile[k*4 + 2]);
      outputTile[k*4 + 3] = _mm256_add_epi32(_mm256_setr_m128i(acc21, acc31), biasesTile[k*4 + 3]);
    }
  }

#elif defined (USE_SSSE3)

  const int ChunkSize = 4;
  const int NumChunks = 8;
  const int TileSize = NumChunks * ChunkSize;
  const int NumTiles = PaddedOutputDimensions / TileSize;

  const __m128i ones = _mm_set1_epi16(1);

  while (num_nnz_indices % 4 != 0)
    nnz_indices[num_nnz_indices++] = dims;

  __m128i acc[NumChunks];

  for (int i = 0; i < NumTiles; ++i)
  {
    const __m128i* biasesTile = (&biases[i * TileSize]);
    __m128i* outputTile = (&output[i * TileSize]);

    for (int k = 0; k < NumChunks; ++k)
      acc[k] = biasesTile[k];

    for (int j = 0; j < num_nnz_indices; j += 4)
    {
      const __m128i mul0 = _mm_set1_epi16(input[nnz_indices[j+0]] | (input[nnz_indices[j+1]] << 8));
      const __m128i mul2 = _mm_set1_epi16(input[nnz_indices[j+2]] | (input[nnz_indices[j+3]] << 8));
      const __m128i* col0 = (&weights[nnz_indices[j+0] * PaddedOutputDimensions + i * TileSize]);
      const __m128i* col1 = (&weights[nnz_indices[j+1] * PaddedOutputDimensions + i * TileSize]);
      const __m128i* col2 = (&weights[nnz_indices[j+2] * PaddedOutputDimensions + i * TileSize]);
      const __m128i* col3 = (&weights[nnz_indices[j+3] * PaddedOutputDimensions + i * TileSize]);
      for (int k = 0; k < NumChunks / 4; ++k)
      {
        __m128i prod0 = _mm_maddubs_epi16(mul0, _mm_unpacklo_epi8(col0[k], col1[k]));
        __m128i prod1 = _mm_maddubs_epi16(mul0, _mm_unpackhi_epi8(col0[k], col1[k]));
        __m128i prod2 = _mm_maddubs_epi16(mul2, _mm_unpacklo_epi8(col2[k], col3[k]));
        __m128i prod3 = _mm_maddubs_epi16(mul2, _mm_unpackhi_epi8(col2[k], col3[k]));
        acc[k*4 + 0] = _mm_add_epi32(acc[k*4 + 0], _mm_madd_epi16(ones, _mm_unpacklo_epi16(prod0, prod2)));
        acc[k*4 + 1] = _mm_add_epi32(acc[k*4 + 1], _mm_madd_epi16(ones, _mm_unpackhi_epi16(prod0, prod2)));
        acc[k*4 + 2] = _mm_add_epi32(acc[k*4 + 2], _mm_madd_epi16(ones, _mm_unpacklo_epi16(prod1, prod3)));
        acc[k*4 + 3] = _mm_add_epi32(acc[k*4 + 3], _mm_madd_epi16(ones, _mm_unpackhi_epi16(prod1, prod3)));
      }
    }

    for (int k = 0; k < NumChunks; ++k)
      outputTile[k] = acc[k];
  }

#elif defined (USE_SSE2)

  const int ChunkSize = 4;
  const int NumChunks = 8;
  const int TileSize = NumChunks * ChunkSize;
  const int NumTiles = PaddedOutputDimensions / TileSize;

  while (num_nnz_indices % 2 != 0)
    nnz_indices[num_nnz_indices++] = dims;

  __m128i acc[NumChunks];

  for (int i = 0; i < NumTiles; ++i)
  {
    const __m128i* biasesTile = (&biases[i * TileSize]);
    __m128i* outputTile = (&output[i * TileSize]);

    for (int k = 0; k < NumChunks; ++k)
      acc[k] = biasesTile[k];

    for (int j = 0; j < num_nnz_indices; j += 2)
    {
      const __m128i mul0 = _mm_set1_epi32(input[nnz_indices[j+0]] | (input[nnz_indices[j+1]] << 16));
      const __m128i* col0 = (&weights[nnz_indices[j+0] * PaddedOutputDimensions + i * TileSize]);
      const __m128i* col1 = (&weights[nnz_indices[j+1] * PaddedOutputDimensions + i * TileSize]);
      for (int k = 0; k < NumChunks / 2; ++k)
      {
        acc[k*2 + 0] = _mm_add_epi32(acc[k*2 + 0], _mm_madd_epi16(mul0, _mm_unpacklo_epi16(col0[k], col1[k])));
        acc[k*2 + 1] = _mm_add_epi32(acc[k*2 + 1], _mm_madd_epi16(mul0, _mm_unpackhi_epi16(col0[k], col1[k])));
      }
    }

    for (int k = 0; k < NumChunks; ++k)
      outputTile[k] = acc[k];
  }

#else
#error "Not supported"
#endif
}

struct NetData {
  alignas(64) int8_t input[2048];
  int32_t hidden1_values[64];
  clipped_t hidden1_clipped[64];
  int32_t hidden2_values[64];
  clipped_t hidden2_clipped[64];
};

// Evaluation function
Value nnue_evaluate(const Position *pos, bool adjusted)
{
  int32_t out_value;
  alignas(8) uint16_t nnz_indices[2048 + 16];
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

  int num_nnz_indices = transform(pos, B(input), nnz_indices, bucket, &psqt_val);

  hidden_layer(B(input), B(hidden1_values), 2048, hidden1_biases[bucket],
      hidden1_weights[bucket], nnz_indices, num_nnz_indices);
  clip_propagate(B(hidden1_values), B(hidden1_clipped), 64);

  affine_propagate(B(hidden1_clipped), B(hidden2_values), 64, 64,
      hidden2_biases[bucket], hidden2_weights[bucket]);
  clip_propagate(B(hidden2_values), B(hidden2_clipped), 64);

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
