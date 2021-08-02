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
#include "settings.h"
#include "uci.h"

#ifndef NNUE_SPARSE
#define NNUE_REGULAR
#endif

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
  PS_W_PAWN   =  0,
  PS_B_PAWN   =  1 * 64,
  PS_W_KNIGHT =  2 * 64,
  PS_B_KNIGHT =  3 * 64,
  PS_W_BISHOP =  4 * 64,
  PS_B_BISHOP =  5 * 64,
  PS_W_ROOK   =  6 * 64,
  PS_B_ROOK   =  7 * 64,
  PS_W_QUEEN  =  8 * 64,
  PS_B_QUEEN  =  9 * 64,
  PS_KING     = 10 * 64,
  PS_END      = 11 * 64
};

uint32_t PieceToIndex[2][16] = {
  { 0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, 0,
    0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, 0 },
  { 0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, 0,
    0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, 0 }
};

// Version of the evaluation file
static const uint32_t NnueVersion = 0x7AF32F20u;

// Constants used in evaluation value calculation
enum {
  FV_SCALE = 16,
  SHIFT = 6
};

enum {
  kHalfDimensions = 1024,
  kPsqtBuckets = 8,
  kLayerStacks = 8,
  FtInDims = 32 * PS_END, // 32 * 704
};

// USE_MMX generates _mm_empty() instructions, so undefine if not needed
#if defined(USE_SSE2)
#undef USE_MMX
#endif

static_assert(kHalfDimensions % 1024 == 0, "kHalfDimensions should be a multiple of 512");

#define VECTOR

#ifdef USE_AVX512
#define SIMD_WIDTH 512
#define PSQT_SIMD_WIDTH 256
typedef __m512i vec8_t, vec16_t;
typedef __m256i vec32_psqt_t;
typedef __mmask64 mask_t;
#define vec_add_16(a,b) _mm512_add_epi16(a,b)
#define vec_sub_16(a,b) _mm512_sub_epi16(a,b)
#define vec_add_psqt_32(a,b) _mm256_add_epi32(a,b)
#define vec_sub_psqt_32(a,b) _mm256_sub_epi32(a,b)
#define vec_packs(a,b) _mm512_packs_epi16(a,b)
#define vec_mask_pos(a) _mm512_cmpgt_epi8_mask(a,_mm512_setzero_si512())
#define vec_clip_8(a,b) _mm512_max_epi8(vec_packs(a,b),_mm512_setzero_si512())
#define vec_zero_psqt() _mm256_setzero_si256()
#define NUM_REGS 32
#define NUM_PSQT_REGS 1

#elif USE_AVX2
#define SIMD_WIDTH 256
#define PSQT_SIMD_WIDTH 256
typedef __m256i vec8_t, vec16_t;
typedef __m256i vec32_psqt_t;
typedef uint32_t mask_t;
#define vec_add_16(a,b) _mm256_add_epi16(a,b)
#define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
#define vec_add_psqt_32(a,b) _mm256_add_epi32(a,b)
#define vec_sub_psqt_32(a,b) _mm256_sub_epi32(a,b)
#define vec_packs(a,b) _mm256_packs_epi16(a,b)
#define vec_mask_pos(a) _mm256_movemask_epi8(_mm256_cmpgt_epi8(a,_mm256_setzero_si256()))
#define vec_clip_8(a,b) _mm256_max_epi8(vec_packs(a,b),_mm256_setzero_si256())
#define vec_zero_psqt() _mm256_setzero_si256()
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif
#define NUM_PSQT_REGS 1

#elif USE_SSE2
#define SIMD_WIDTH 128
#define PSQT_SIMD_WIDTH 128
typedef __m128i vec8_t, vec16_t;
typedef __m128i vec32_psqt_t;
typedef uint16_t mask_t;
#define vec_add_16(a,b) _mm_add_epi16(a,b)
#define vec_sub_16(a,b) _mm_sub_epi16(a,b)
#define vec_add_psqt_32(a,b) _mm_add_epi32(a,b)
#define vec_sub_psqt_32(a,b) _mm_sub_epi32(a,b)
#define vec_packs(a,b) _mm_packs_epi16(a,b)
#define vec_mask_pos(a) _mm_movemask_epi8(_mm_cmpgt_epi8(a,_mm_setzero_si128()))
#ifdef USE_SSE41
#define vec_clip_8(a,b) _mm_max_epi8(vec_packs(a,b),_mm_setzero_si128())
#elif USE_SSSE3
#define vec_clip_8(a,b) vec_packs(_mm_max_epi16(a,_mm_setzero_si128()),_mm_max_epi16(b,_mm_setzero_si128()))
#else
#define vec_clip_16(a) _mm_min_epi16(_mm_max_epi16(a,_mm_setzero_si128()),_mm_set1_epi16(127))
#endif
#define vec_zero_psqt() _mm_setzero_si128()
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif
#define NUM_PSQT_REGS 2

#elif USE_MMX
#define SIMD_WIDTH 64
#define PSQT_SIMD_WIDTH 64
typedef __m64 vec8_t, vec16_t;
typedef __m64 vec32_psqt_t;
typedef uint8_t mask_t;
#define vec_add_16(a,b) _mm_add_pi16(a,b)
#define vec_sub_16(a,b) _mm_sub_pi16(a,b)
#define vec_add_psqt_32(a,b) _mm_add_pi32(a,b)
#define vec_sub_psqt_32(a,b) _mm_sub_pi32(a,b)
#define vec_packs(a,b) _mm_packs_pi16(a,b)
#define vec_mask_pos(a) _mm_movemask_pi8(_mm_cmpgt_pi8(a,_mm_setzero_si64()))
#ifdef USE_SSE
#define vec_clip_16(a) _mm_min_pi16(_mm_max_pi16(a,_mm_setzero_si64()),_mm_set1_pi16(127))
#else
#define vec_clip_16(a) _mm_subs_pu16(_mm_add_pi16(_mm_adds_pi16(a, _mm_set1_pi16(0x7f80)), _mm_set1_pi16(0x0080)), _mm_set1_pi16(-0x8000))
#endif
#define vec_zero_psqt() 0
#define NUM_REGS 8
#define NUM_PSQT_REGS 4

#elif USE_NEON
#define SIMD_WIDTH 128
#define PSQT_SIMD_WIDTH 128
typedef int8x16_t vec8_t;
typedef int16x8_t vec16_t;
typedef int32x4_t vec32_psqt_t;
typedef uint16_t mask_t;
#define vec_add_16(a,b) vaddq_s16(a,b)
#define vec_sub_16(a,b) vsubq_s16(a,b)
#define vec_add_psqt_32(a,b) vaddq_s32(a,b)
#define vec_sub_psqt_32(a,b) vsubq_s32(a,b)
#define vec_packs(a,b) vcombine_s8(vqmovn_s16(a),vqmovn_s16(b))
#define vec_mask_pos(a) neon_movemask(vcgtq_s8(a,vdupq_n_s8(0)))
#define vec_clip_8(a,b) vmaxq_s8(vec_packs(a,b),vdupq_n_s8(0))
#define vec_zero_psqt() vec32_psqt_t{0}
#ifdef IS_64BIT
#define NUM_REGS 16
#else
#define NUM_REGS 8
#endif
#define NUM_PSQT_REGS 2

#else
#undef VECTOR
#define SIMD_WIDTH 16 // dummy
typedef uint8_t mask_t; // dummy

#endif

#ifdef NNUE_SPARSE
#if (defined(USE_MMX) || (defined(USE_SSE2))) && !(defined(USE_SSSE3))
typedef int16_t weight_t_sparse, out_t_sparse;
#else
typedef int8_t weight_t_sparse, out_t_sparse;
#endif
#else
#error "Not supported"
#endif

#if defined(USE_MMX) || (defined(USE_SSE2) && !defined(USE_SSSE3))
typedef int16_t weight_t, out_t, clipped_t;
#else
typedef int8_t weight_t, out_t, clipped_t;
#endif

#if defined(USE_MMX) && !defined(USE_SSE)
INLINE int _mm_movemask_pi8(__m64 v)
{
  const __m64 powers = _mm_set_pi8(-128, 64, 32, 16, 8, 4, 2, 1);
  __m64 m = _mm_and_si64(v, powers);
  m = _mm_or_si64(m, _mm_srli_si64(m, 32));
  m = _mm_or_si64(m, _mm_srli_pi32(m, 16));
  m = _mm_or_si64(m, _mm_srli_pi16(m, 8));
  return _mm_cvtsi64_si32(m) & 0xff;
}
#elif defined(USE_NEON)
INLINE int neon_movemask(uint8x16_t v)
{
  const uint8_t __attribute__((aligned(16))) powers[16] =
    { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
  const uint8x16_t kPowers = vld1q_u8(powers);

  uint64x2_t mask = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vandq_u8(v, kPowers))));
  return   vgetq_lane_u8((uint8x16_t)mask, 0)
        | (vgetq_lane_u8((uint8x16_t)mask, 8) << 8);
}
#endif

static alignas(64) uint16_t LookupTableIndices[256][8];
static alignas(64) uint8_t LookupTableCounts[256];

static inline int lsb_constexpr(uint32_t v)
{
  int c = 0;
  if (!v) return 32;
  while (!(v & 1))
  {
    v >>= 1;
    ++c;
  }
  return c;
}

void init_lookup()
{
  for (int i = 0; i < 256; ++i)
  {
    int j = i;
    int k = 0;
    while(j)
    {
      const int lsbIndex = lsb_constexpr(j);
      j &= j - 1;
      LookupTableIndices[i][k] = lsbIndex;
      ++k;
    }
    LookupTableCounts[i] = k;
  }
}

typedef struct {
  unsigned size;
  unsigned values[32];
} IndexList;

INLINE Square orient(Color c, Square s, Square ksq)
{
  return s ^ (c * SQ_A8) ^ ((file_of(ksq) < FILE_E) * SQ_H1);
}

static const int KingBuckets[64] = {
  -1, -1, -1, -1, 31, 30, 29, 28,
  -1, -1, -1, -1, 27, 26, 25, 24,
  -1, -1, -1, -1, 23, 22, 21, 20,
  -1, -1, -1, -1, 19, 18, 17, 16,
  -1, -1, -1, -1, 15, 14, 13, 12,
  -1, -1, -1, -1, 11, 10, 9, 8,
  -1, -1, -1, -1, 7, 6, 5, 4,
  -1, -1, -1, -1, 3, 2, 1, 0
};

INLINE unsigned make_index(Color c, Square s, Piece pc, Square ksq)
{
  int o_ksq = orient(c, ksq, ksq);
  return orient(c, s, ksq) + PieceToIndex[c][pc] + PS_END * KingBuckets[o_ksq];
}

static void append_active_indices(const Position *pos, const Color c,
    IndexList *active)
{
  Square ksq = square_of(c, KING);
  Bitboard bb = pieces();
  while (bb) {
    Square s = pop_lsb(&bb);
    active->values[active->size++] = make_index(c, s, piece_on(s), ksq);
  }
}

static void append_changed_indices(const Position *pos, const Color c,
    const DirtyPiece *dp, IndexList *removed, IndexList *added)
{
  Square ksq = square_of(c, KING);
  for (int i = 0; i < dp->dirtyNum; i++) {
    Piece pc = dp->pc[i];
    if (dp->from[i] != SQ_NONE)
      removed->values[removed->size++] = make_index(c, dp->from[i], pc, ksq);
    if (dp->to[i] != SQ_NONE)
      added->values[added->size++] = make_index(c, dp->to[i], pc, ksq);
  }
}

INLINE int32_t output_layer(const clipped_t *input, const int32_t *biases,
    const weight_t *weights)
{
  int32_t sum = biases[0];
  for (unsigned j = 0; j < 64; j++)
    sum += weights[j] * input[j];
  return sum;
}

// Input feature converter
static int16_t *ft_biases; // [kHalfDimensions]
static int16_t *ft_weights; // [kHalfDimenions * FtInDims]
static int32_t *ft_weights_psqt; // [kHalfDimenions * FtInDims]
static alloc_t ft_alloc;

#ifdef VECTOR
#define TILE_HEIGHT (NUM_REGS * SIMD_WIDTH / 16)
#define PSQT_TILE_HEIGHT  (NUM_PSQT_REGS * PSQT_SIMD_WIDTH / 32)
#endif

// Calculate cumulative value using difference calculation if possible
INLINE void update_accumulator(const Position *pos, const Color c)
{
#ifdef VECTOR
  vec16_t acc[NUM_REGS];
  vec32_psqt_t acc_psqt[NUM_PSQT_REGS];
#endif

  Stack *st = pos->st;
  int gain = popcount(pieces());
  while (st->accumulator.state[c] == ACC_EMPTY) {
    DirtyPiece *dp = &st->dirtyPiece;
    if (   dp->pc[0] == make_piece(c, KING)
        || (gain -= dp->dirtyNum + 1) < 0)
      break;
    st--;
  }

  if (st->accumulator.state[c] == ACC_COMPUTED) {
    if (st == pos->st)
      return;

    IndexList added[2], removed[2];
    added[0].size = added[1].size = removed[0].size = removed[1].size = 0;
    append_changed_indices(pos, c, &(st+1)->dirtyPiece, &removed[0], &added[0]);
    for (Stack *st2 = st + 2; st2 <= pos->st; st2++)
      append_changed_indices(pos, c, &st2->dirtyPiece, &removed[1], &added[1]);

    (st+1)->accumulator.state[c] = ACC_COMPUTED;
    pos->st->accumulator.state[c] = ACC_COMPUTED;

    Stack *stack[3] = { st + 1, st + 1 == pos->st ? NULL : pos->st, NULL };
#ifdef VECTOR
    for (unsigned i = 0; i < kHalfDimensions / TILE_HEIGHT; i++) {
      vec16_t *accTile = (vec16_t *)&st->accumulator.accumulation[c][i * TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_REGS; j++)
        acc[j] = accTile[j];
      for (unsigned l = 0; stack[l]; l++) {
        // Difference calculation for the deactivated features
        for (unsigned k = 0; k < removed[l].size; k++) {
          unsigned index = removed[l].values[k];
          const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
          vec16_t *column = (vec16_t *)&ft_weights[offset];
          for (unsigned j = 0; j < NUM_REGS; j++)
            acc[j] = vec_sub_16(acc[j], column[j]);
        }

        // Difference calculation for the activated features
        for (unsigned k = 0; k < added[l].size; k++) {
          unsigned index = added[l].values[k];
          const unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
          vec16_t *column = (vec16_t *)&ft_weights[offset];
          for (unsigned j = 0; j < NUM_REGS; j++)
            acc[j] = vec_add_16(acc[j], column[j]);
        }

        accTile = (vec16_t *)&stack[l]->accumulator.accumulation[c][i * TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_REGS; j++)
          accTile[j] = acc[j];
      }
    }

    for (unsigned i = 0; i < kPsqtBuckets / PSQT_TILE_HEIGHT; i++) {
      vec32_psqt_t *accTile = (vec32_psqt_t *)&st->accumulator.accumulation_psqt[c][i * PSQT_TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
        acc_psqt[j] = accTile[j];
      for (unsigned l = 0; stack[l]; l++) {
        // Difference calculation for the deactivated features
        for (unsigned k = 0; k < removed[l].size; k++) {
          unsigned index = removed[l].values[k];
          const unsigned offset = kPsqtBuckets * index + i * PSQT_TILE_HEIGHT;
          vec32_psqt_t *column = (vec32_psqt_t *)&ft_weights_psqt[offset];
          for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
            acc_psqt[j] = vec_sub_psqt_32(acc_psqt[j], column[j]);
        }

        // Difference calculation for the activated features
        for (unsigned k = 0; k < added[l].size; k++) {
          unsigned index = added[l].values[k];
          const unsigned offset = kPsqtBuckets * index + i * PSQT_TILE_HEIGHT;
          vec32_psqt_t *column = (vec32_psqt_t *)&ft_weights_psqt[offset];
          for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
            acc_psqt[j] = vec_add_psqt_32(acc_psqt[j], column[j]);
        }

        accTile = (vec32_psqt_t *)&stack[l]->accumulator.accumulation_psqt[c][i * PSQT_TILE_HEIGHT];
        for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
          accTile[j] = acc_psqt[j];
      }
    }
#else
    for (unsigned l = 0; stack[l]; l++) {
      memcpy(&stack[l]->accumulator.accumulation[c],
          &st->accumulator.accumulation[c], kHalfDimensions * sizeof(int16_t));
      memcpy(&stack[l]->accumulator.accumulation_psqt[c],
          &st->accumulator.accumulation_psqt[c], kPsqtBuckets * sizeof(int32_t));
      st = stack[l];

      // Difference calculation for the deactivated features
      for (unsigned k = 0; k < removed[l].size; k++) {
        unsigned index = removed[l].values[k];
        const unsigned offset = kHalfDimensions * index;
        const unsigned offset_psqt = kPsqtBuckets * index;

        for (unsigned j = 0; j < kHalfDimensions; j++)
          st->accumulator.accumulation[c][j] -= ft_weights[offset + j];

        for (unsigned j = 0; j < kPsqtBuckets; j++)
          st->accumulator.accumulation_psqt[c][j] -= ft_weights_psqt[offset_psqt + j];
      }

      // Difference calculation for the activated features
      for (unsigned k = 0; k < added[l].size; k++) {
        unsigned index = added[l].values[k];
        const unsigned offset = kHalfDimensions * index;
        const unsigned offset_psqt = kPsqtBuckets * index;

        for (unsigned j = 0; j < kHalfDimensions; j++)
          st->accumulator.accumulation[c][j] += ft_weights[offset + j];

        for (unsigned j = 0; j < kPsqtBuckets; j++)
          st->accumulator.accumulation_psqt[c][j] += ft_weights_psqt[offset_psqt + j];
      }
    }
#endif
  } else {
    Accumulator *accumulator = &pos->st->accumulator;
    accumulator->state[c] = ACC_COMPUTED;
    IndexList active;
    active.size = 0;
    append_active_indices(pos, c, &active);
#ifdef VECTOR
    for (unsigned i = 0; i < kHalfDimensions / TILE_HEIGHT; i++) {
      vec16_t *ft_biases_tile = (vec16_t *)&ft_biases[i * TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_REGS; j++)
        acc[j] = ft_biases_tile[j];

      for (unsigned k = 0; k < active.size; k++) {
        unsigned index = active.values[k];
        unsigned offset = kHalfDimensions * index + i * TILE_HEIGHT;
        vec16_t *column = (vec16_t *)&ft_weights[offset];
        for (unsigned j = 0; j < NUM_REGS; j++)
          acc[j] = vec_add_16(acc[j], column[j]);
      }

      vec16_t *accTile = (vec16_t *)&accumulator->accumulation[c][i * TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_REGS; j++)
        accTile[j] = acc[j];
    }

    for (unsigned i = 0; i < kPsqtBuckets / PSQT_TILE_HEIGHT; i++) {
      for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
        acc_psqt[j] = vec_zero_psqt();

      for (unsigned k = 0; k < active.size; k++) {
        unsigned index = active.values[k];
        unsigned offset = kPsqtBuckets * index + i * PSQT_TILE_HEIGHT;
        vec32_psqt_t *column = (vec32_psqt_t *)&ft_weights_psqt[offset];
        for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
          acc_psqt[j] = vec_add_psqt_32(acc_psqt[j], column[j]);
      }

      vec32_psqt_t *accTile = (vec32_psqt_t *)&accumulator->accumulation_psqt[c][i * PSQT_TILE_HEIGHT];
      for (unsigned j = 0; j < NUM_PSQT_REGS; j++)
        accTile[j] = acc_psqt[j];
    }
#else
    memcpy(accumulator->accumulation[c], ft_biases,
        kHalfDimensions * sizeof(int16_t));
    memset(accumulator->accumulation_psqt[c], 0,
        kPsqtBuckets * sizeof(int32_t));

    for (unsigned k = 0; k < active.size; k++) {
      unsigned index = active.values[k];
      unsigned offset = kHalfDimensions * index;
      unsigned offset_psqt = kPsqtBuckets * index;

      for (unsigned j = 0; j < kHalfDimensions; j++)
        accumulator->accumulation[c][j] += ft_weights[offset + j];

      for (unsigned j = 0; j < kPsqtBuckets; j++)
        accumulator->accumulation_psqt[c][j] += ft_weights_psqt[offset_psqt + j];
    }
#endif
  }
}

// Convert input features
INLINE int transform(const Position *pos, clipped_t *output, uint16_t *nnz_indices, int32_t psqtBucket, int32_t* psqt_val)
{
  update_accumulator(pos, WHITE);
  update_accumulator(pos, BLACK);

  int16_t (*accumulation)[2][1024] = &pos->st->accumulator.accumulation;
  int32_t (*accumulation_psqt)[2][8] = &pos->st->accumulator.accumulation_psqt;

  const Color perspectives[2] = { stm(), !stm() };

  *psqt_val = (
      (*accumulation_psqt)[perspectives[0]][psqtBucket]
    - (*accumulation_psqt)[perspectives[1]][psqtBucket]
  ) / 2;

  int num_nnz_indices = 0;
  __m128i base = _mm_set1_epi16(0);
  __m128i increment = _mm_set1_epi16(8);

  for (unsigned p = 0; p < 2; p++) {
    const unsigned offset = kHalfDimensions * p;

#if defined (USE_AVX2)

    const unsigned numChunks = (16 * kHalfDimensions) / 256;
    vec8_t *out = (vec8_t *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      vec16_t s0 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2];
      vec16_t s1 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      out[i] = vec_packs(s0, s1);

      unsigned nnz = _mm256_movemask_epi8(_mm256_cmpgt_epi8(out[i], _mm256_setzero_si256()));
      unsigned b3 = (nnz >> 24) & 0xFF;
      unsigned b2 = (nnz >> 16) & 0xFF;
      unsigned b1 = (nnz >> 8) & 0xFF;
      unsigned b0 = (nnz) & 0xFF;
      unsigned c0 = LookupTableCounts[b0];
      unsigned c1 = LookupTableCounts[b1];
      unsigned c2 = LookupTableCounts[b2];
      unsigned c3 = LookupTableCounts[b3];
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b0]) + base);
      num_nnz_indices += c0;
      base += increment;
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b1]) + base);
      num_nnz_indices += c1;
      base += increment;
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b2]) + base);
      num_nnz_indices += c2;
      base += increment;
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b3]) + base);
      num_nnz_indices += c3;
      base += increment;
    }

#elif defined (USE_SSSE3)

    const unsigned numChunks = (16 * kHalfDimensions) / 128;
    vec8_t *out = (vec8_t *)&output[offset];
    for (unsigned i = 0; i < numChunks / 2; i++) {
      vec16_t s0 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2];
      vec16_t s1 = ((vec16_t *)(*accumulation)[perspectives[p]])[i * 2 + 1];
      out[i] = vec_packs(s0, s1);

      unsigned nnz = _mm_movemask_epi8(_mm_cmpgt_epi8(out[i], _mm_setzero_si128()));
      unsigned b1 = (nnz >> 8) & 0xFF;
      unsigned b0 = (nnz) & 0xFF;
      unsigned c0 = LookupTableCounts[b0];
      unsigned c1 = LookupTableCounts[b1];
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b0]) + base);
      num_nnz_indices += c0;
      base += increment;
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b1]) + base);
      num_nnz_indices += c1;
      base += increment;
    }

#elif defined (USE_SSE2)

    const unsigned numChunks = (16 * kHalfDimensions) / 128;
    vec16_t *out = (vec16_t *)&output[offset];
    for (unsigned i = 0; i < numChunks; i++) {
      vec16_t sum = ((vec16_t *)(*accumulation)[perspectives[p]])[i];
      out[i] = vec_clip_16(sum);
#error "error because no _mm_movemask_epi16 and I don't care"
      unsigned nnz = _mm_movemask_epi16(_mm_cmpgt_epi16(out[i], _mm_setzero_si128()));
      unsigned b0 = (nnz) & 0xFF;
      unsigned c0 = LookupTableCounts[b0];
      _mm_storeu_si128(nnz_indices + num_nnz_indices, _mm_loadu_si128(&LookupTableIndices[b0]) + base);
      num_nnz_indices += c0;
      base += increment;
    }

#else
#error "Not supported"
#endif

  }

  return num_nnz_indices;
}

#ifndef USE_NEON
INLINE unsigned bit_shuffle(unsigned v, int left, int right, unsigned mask)
{
  unsigned w = v & mask;
  w = (w << left) | (w >> right);
  return (v & ~mask) | (w & mask);
}
#endif

#include "nnue-regular.c"
#include "nnue-sparse.c"

static const char *read_hidden_weights_dense(weight_t *w, unsigned outDims, unsigned dims,
    const char *d)
{
  for (unsigned r = 0; r < outDims; r++)
  {
    for (unsigned c = 0; c < dims; c++)
      w[wt_idx_dense(r, c, dims)] = *d++;
  }

  return d;
}

#if defined (NNUE_SPARSE)
static const char *read_hidden_weights_sparse(weight_t_sparse *w, unsigned outDims, unsigned dims,
    const char *d)
{
  for (unsigned i = 0; i < outDims; i++)
  {
    for (unsigned j = 0; j < dims; j++)
      w[j*outDims + i] = *d++;

     w[dims * outDims + i] = 0;
  }

  return d;
}
#endif

static bool init_weights(const void *evalData, unsigned size)
{
  if (!ft_biases) {
    if (settings.largePages)
      ft_biases = allocate_memory(2 * kHalfDimensions * (FtInDims + 2) + (4 * kPsqtBuckets * FtInDims), true,
          &ft_alloc);
    if (!ft_biases)
      ft_biases = allocate_memory(2 * kHalfDimensions * (FtInDims + 2) + (4 * kPsqtBuckets * FtInDims), false,
          &ft_alloc);
    if (!ft_biases) {
      fprintf(stdout, "Could not allocate enough memory.\n");
      exit(EXIT_FAILURE);
    }
    ft_weights = ft_biases + kHalfDimensions;
    ft_weights_psqt = (int32_t*)(ft_weights + kHalfDimensions * (FtInDims + 1));
  }

  const char *d = (const char *)evalData;
  unsigned s = readu_le_u32(d+8);
  d += 4 + 4 + 4 + s + 4;

  // Read transformer
  memcpy(ft_biases, d, kHalfDimensions * 2);
  d += kHalfDimensions * 2;
  memcpy(ft_weights, d, kHalfDimensions * FtInDims * 2);
  d += kHalfDimensions * FtInDims * 2;
  memcpy(ft_weights_psqt, d, kPsqtBuckets * FtInDims * 4);
  d += kPsqtBuckets * FtInDims * 4;

  // Read network
  for (unsigned k = 0; k < kLayerStacks; ++k) {
    d += 4;
    for (unsigned i = 0; i < 64; i++, d += 4)
      hidden1_biases[k][i] = readu_le_u32(d);
#if defined (NNUE_SPARSE)
    d = read_hidden_weights_sparse(hidden1_weights[k], 64, 2048, d);
#else
    d = read_hidden_weights_dense(hidden1_weights[k], 64, 2048, d);
#endif

    for (unsigned i = 0; i < 64; i++, d += 4)
      hidden2_biases[k][i] = readu_le_u32(d);
    d = read_hidden_weights_dense(hidden2_weights[k], 64, 64, d);

    for (unsigned i = 0; i < 1; i++, d += 4)
      output_biases[k][i] = readu_le_u32(d);

    d = read_output_weights_dense(output_weights[k], d);
  }

  init_lookup();

  return d == ((const char*)evalData) + size;
}

void nnue_export_net(void) {
#ifdef NNUE_EMBEDDED
  FILE *F = fopen(DefaultEvalFile, "wb");
  if (F) {
    fwrite(gNetworkData, gNetworkSize, 1, F);
    fclose(F);
  }
#else
  printf("No embedded network fie.\n");
#endif
}

static bool verify_net(const void *evalData, size_t size)
{
  const char *d = evalData;
  if (readu_le_u32(d) != NnueVersion) return false;

  return true;
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
    success = init_weights(evalData, size);
  if (mapping) unmap_file(evalData, mapping);
  return success;
}

static char *loadedFile = NULL;

void nnue_init(void)
{
#ifndef NNUE_PURE
  const char *s = option_string_value(OPT_USE_NNUE);
  useNNUE =  strcmp(s, "classical") == 0 ? EVAL_CLASSICAL
           : strcmp(s, "pure"     ) == 0 ? EVAL_PURE : EVAL_HYBRID;
#endif

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

void nnue_free(void)
{
  if (ft_biases)
    free_memory(&ft_alloc);
}
