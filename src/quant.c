/*
 * quant.c - quantised-domain dot products (see quant.h for the rule).
 *
 * Every kernel here has the same shape:
 *
 *   for each block of the row:
 *        w8  = expand this block of weights to int8 (type specific, vector)
 *        s   = sum_i xq_i * w8_i            (int32, no floats involved)
 *        acc += (d_w * d_x) * s             (one float FMA per block)
 *
 * The block scale multiplies the accumulator, never an individual weight, and
 * the weights are never converted to float. Types that carry a per-block
 * minimum (Q4_K, Q5_K) subtract it as a second accumulator term using the sum
 * of the activations, which is why nita_aq8_t keeps that sum.
 */
#include "quant.h"
#include "nita.h"
#include "iq_tables.h"

#include <math.h>
#include <string.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

/* ------------------------------------------------------------------ */
/* Activation quantisation                                             */
/* ------------------------------------------------------------------ */

void nita_aq8_quantise(const float *x, int k, nita_aq8_t *out)
{
    int b;

    for (b = 0; b < k / 32; ++b) {
        const float *xb = x + b * 32;
        nita_aq8_t  *o  = &out[b];
        int          i;
        float        amax = 0.0f, d;
        int32_t      sum  = 0;

#if defined(__AVX2__)
        {
            __m256 vmax = _mm256_setzero_ps();
            int    s;

            for (s = 0; s < 32; s += 8) {
                __m256 v = _mm256_loadu_ps(xb + s);
                vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(
                           _mm256_set1_ps(-0.0f), v));
            }
            {
                __m128 lo = _mm256_castps256_ps128(vmax);
                __m128 hi = _mm256_extractf128_ps(vmax, 1);
                __m128 m  = _mm_max_ps(lo, hi);
                m = _mm_max_ps(m, _mm_movehl_ps(m, m));
                m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
                amax = _mm_cvtss_f32(m);
            }
        }
#else
        for (i = 0; i < 32; ++i) {
            float a = xb[i] < 0.0f ? -xb[i] : xb[i];

            if (a > amax) amax = a;
        }
#endif

        /* A zero block keeps d = 0 and all-zero quants: 0/0 must not produce
         * NaNs that would poison the accumulator. */
        d = amax > 0.0f ? amax / 127.0f : 0.0f;
        {
            float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

#if defined(__AVX2__)
            int s;

            for (s = 0; s < 32; s += 8) {
                __m256  v   = _mm256_mul_ps(_mm256_loadu_ps(xb + s),
                                            _mm256_set1_ps(inv));
                __m256i q32 = _mm256_cvtps_epi32(v);       /* nearest even */
                __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q32),
                                              _mm256_extracti128_si256(q32, 1));
                __m128i q8  = _mm_packs_epi16(q16, q16);   /* saturating     */

                _mm_storel_epi64((__m128i *)(o->qs + s), q8);
            }
#else
            for (i = 0; i < 32; ++i) {
                long q = lrintf(xb[i] * inv);               /* same rounding */

                if (q > 127) q = 127;
                if (q < -127) q = -127;
                o->qs[i] = (int8_t)q;
            }
#endif
        }
        for (i = 0; i < 32; ++i) sum += o->qs[i];
        {
            int g;

            for (g = 0; g < 4; ++g) {
                int32_t gsum = (int32_t)o->qs[g * 8 + 0] + o->qs[g * 8 + 1]
                             + o->qs[g * 8 + 2] + o->qs[g * 8 + 3]
                             + o->qs[g * 8 + 4] + o->qs[g * 8 + 5]
                             + o->qs[g * 8 + 6] + o->qs[g * 8 + 7];
                o->sums[g] = gsum;
            }
        }
        o->d   = d;
        o->sum = sum;
    }
}

/* ------------------------------------------------------------------ */
/* int8 x int8 dot, one 32-element block                               */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)
static int32_t hsum_i32(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));

    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static float hsum256_ps(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* The vector forms below return the eight int32 lane partial sums. They exist
 * because reducing to a scalar per block is the whole performance question:
 * one horizontal sum per block costs more than the multiply it is summing,
 * while a vertical accumulator costs one convert and one FMA and is reduced
 * once per row. */
static __m256 dot_i8_vec(const int8_t *x, __m256i w)
{
    const __m256i bx = _mm256_loadu_si256((const __m256i *)x);
    const __m256i ax = _mm256_sign_epi8(bx, bx);

    return _mm256_cvtepi32_ps(_mm256_madd_epi16(
               _mm256_maddubs_epi16(ax, _mm256_sign_epi8(w, bx)),
               _mm256_set1_epi16(1)));
}

static __m256 dot_u8_vec(const int8_t *x, __m256i w)
{
    const __m256i bx = _mm256_loadu_si256((const __m256i *)x);

    return _mm256_cvtepi32_ps(_mm256_madd_epi16(
               _mm256_maddubs_epi16(w, bx), _mm256_set1_epi16(1)));
}

/* sum of x[i] * w[i] over 16 signed bytes */
static int32_t dot_i8_16(const int8_t *x, __m128i w)
{
    const __m128i bx = _mm_loadu_si128((const __m128i *)x);
    const __m128i ax = _mm_sign_epi8(bx, bx);
    const __m128i sw = _mm_sign_epi8(w, bx);
    __m128i       p  = _mm_madd_epi16(_mm_maddubs_epi16(ax, sw),
                                      _mm_set1_epi16(1));

    p = _mm_hadd_epi32(p, p);
    p = _mm_hadd_epi32(p, p);
    return _mm_cvtsi128_si32(p);
}

/* sum of x[i] * w[i] for unsigned w in 0..31, 32 elements */
static int32_t dot_u8_32(const int8_t *x, __m256i w)
{
    const __m256i bx = _mm256_loadu_si256((const __m256i *)x);

    return hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(w, bx),
                                      _mm256_set1_epi16(1)));
}

/* Exact for every int8 pair, including -128, at the cost of widening to
 * int16: the maddubs trick above needs |w| <= 127 in its signed operand,
 * because -(-128) saturates, and it saturates the int16 intermediate as soon
 * as one side is biased into 0..255. Every int8 x int8 product fits an int16,
 * so this path has no such corner. */
static __m256 dot_i8_vec_exact(const int8_t *x, __m256i w)
{
    const __m256i bx = _mm256_loadu_si256((const __m256i *)x);
    const __m256i xl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bx));
    const __m256i xh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bx, 1));
    const __m256i wl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
    const __m256i wh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xl, wl),
                                               _mm256_madd_epi16(xh, wh)));
}

static float hsum128_ps(__m128 v)
{
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

/* 16 stored bytes -> 32 bytes, low nibbles first then high, which is the
 * element order every 4-bit block uses. */
static __m256i nibbles_32(const uint8_t *qs)
{
    const __m128i q = _mm_loadu_si128((const __m128i *)qs);

    return _mm256_and_si256(_mm256_set1_epi8(0x0F),
                            _mm256_set_m128i(_mm_srli_epi16(q, 4), q));
}

/* one 32-bit bitfield -> 32 bytes, 0xFF where the bit is set. Byte j is bit
 * j, so the vector lines up with nibbles_32 element for element. */
static __m256i bits_32(const uint8_t *qh)
{
    uint32_t      w;
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL,
                                           0x0202020202020202LL,
                                           0x0101010101010101LL,
                                           0x0000000000000000LL);
    const __m256i mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    __m256i       b;

    memcpy(&w, qh, sizeof w);
    /* each shuffled byte holds the byte of w containing bit j, so OR-ing in
     * every lower bit and testing the sign isolates that one bit */
    b = _mm256_shuffle_epi8(_mm256_set1_epi32((int)w), shuf);
    b = _mm256_or_si256(b, mask);
    return _mm256_cmpeq_epi8(b, _mm256_set1_epi8((char)-1));
}
#endif

/* ------------------------------------------------------------------ */
/* Q8_0: 32 weights per block, d at +0, qs at +2, 34 bytes            */
/* ------------------------------------------------------------------ */
static float dot_q8_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 34;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            __m256         p   = dot_i8_vec(a[b].qs,
                                    _mm256_loadu_si256((const __m256i *)(blk + 2)));

            vacc = _mm256_fmadd_ps(p, _mm256_set1_ps(dw * a[b].d), vacc);
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 34;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        int32_t        s   = 0;
        int            i;

        for (i = 0; i < 32; ++i) s += (int32_t)a[b].qs[i] * (int32_t)(int8_t)blk[2 + i];
        acc += dw * a[b].d * (float)s;
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q4_K: 256 weights, 144 bytes                                       */
/*   d +0, dmin +2, 12 scale bytes +4, qs +16                         */
/*   chunk j (0..3), 64 elements: low nibbles of 32 bytes qs[j*32..]   */
/*              -> elements j*64 .. j*64+32, scale sc[2j], min mn[2j]  */
/*   high nibbles of the same bytes -> j*64+32 .. j*64+64,             */
/*              scale sc[2j+1], min mn[2j+1]                           */
/*   value = d*sc*w - dmin*mn                                          */
/* ------------------------------------------------------------------ */
static void q4k_scales(const uint8_t *s, int8_t sc[8], int8_t mn[8])
{
    int j;

    for (j = 0; j < 4; ++j) {
        sc[j] = (int8_t)(s[j] & 63);
        mn[j] = (int8_t)(s[j + 4] & 63);
    }
    for (j = 4; j < 8; ++j) {
        sc[j] = (int8_t)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        mn[j] = (int8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}
static float dot_q4_K(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f, msum = 0.0f;
    int   sb;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();

        for (sb = 0; sb < nb / 8; ++sb) {
            const uint8_t *blk  = row + (size_t)sb * 144;
            const uint8_t *qs   = blk + 16;
            float          d    = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            float          dmin = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
            int8_t         sc[8], mn[8];
            int            j;

            q4k_scales(blk + 4, sc, mn);

            for (j = 0; j < 4; ++j) {
                const nita_aq8_t *ab0 = &a[sb * 8 + 2 * j];
                const nita_aq8_t *ab1 = &a[sb * 8 + 2 * j + 1];
                const __m256i q256 = _mm256_loadu_si256((const __m256i *)(qs + j * 32));
                const __m256i m0f  = _mm256_set1_epi8(0x0F);
                const __m256i wlo  = _mm256_and_si256(q256, m0f);
                const __m256i whi  = _mm256_and_si256(_mm256_srli_epi16(q256, 4), m0f);

                vacc = _mm256_fmadd_ps(dot_u8_vec(ab0->qs, wlo),
                                       _mm256_set1_ps(d * (float)sc[2 * j] * ab0->d), vacc);
                vacc = _mm256_fmadd_ps(dot_u8_vec(ab1->qs, whi),
                                       _mm256_set1_ps(d * (float)sc[2 * j + 1] * ab1->d), vacc);
                msum += dmin * (float)mn[2 * j] * ab0->d * (float)ab0->sum;
                msum += dmin * (float)mn[2 * j + 1] * ab1->d * (float)ab1->sum;
            }
        }
        acc = hsum256_ps(vacc) - msum;
    }
#else
    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk  = row + (size_t)sb * 144;
        const uint8_t *qs   = blk + 16;
        float          d    = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        float          dmin = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
        int8_t         sc[8], mn[8];
        int            j;

        q4k_scales(blk + 4, sc, mn);
        for (j = 0; j < 4; ++j) {
            const nita_aq8_t *ab0 = &a[sb * 8 + 2 * j];
            const nita_aq8_t *ab1 = &a[sb * 8 + 2 * j + 1];
            int8_t            wlo[32], whi[32];
            int32_t           s0 = 0, s1 = 0;
            int               l;

            for (l = 0; l < 32; ++l) {
                wlo[l] = (int8_t)(qs[j * 32 + l] & 0x0F);
                whi[l] = (int8_t)(qs[j * 32 + l] >> 4);
            }
            for (l = 0; l < 32; ++l) s0 += (int32_t)ab0->qs[l] * wlo[l];
            for (l = 0; l < 32; ++l) s1 += (int32_t)ab1->qs[l] * whi[l];
            acc  += d * (float)sc[2 * j] * ab0->d * (float)s0;
            acc  += d * (float)sc[2 * j + 1] * ab1->d * (float)s1;
            acc  -= dmin * (float)mn[2 * j] * ab0->d * (float)ab0->sum;
            acc  -= dmin * (float)mn[2 * j + 1] * ab1->d * (float)ab1->sum;
        }
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q5_K: 176 bytes; qh +16, ql +48; five bits per weight              */
/*   sub-block 2j+{0,1} of 64 elements from ql[j*32 + l]              */
/* ------------------------------------------------------------------ */
static float dot_q5_K(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk  = row + (size_t)sb * 176;
        const uint8_t *qh   = blk + 16;
        const uint8_t *ql   = blk + 48;
        float          d    = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        float          dmin = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
        int8_t         sc[8], mn[8];
        int            j;

        q4k_scales(blk + 4, sc, mn);

        for (j = 0; j < 4; ++j) {
            const nita_aq8_t *a0 = &a[sb * 8 + 2 * j + 0];
            const nita_aq8_t *a1 = &a[sb * 8 + 2 * j + 1];
            float             d0 = d * (float)sc[2 * j + 0];
            float             m0 = dmin * (float)mn[2 * j + 0];
            float             d1 = d * (float)sc[2 * j + 1];
            float             m1 = dmin * (float)mn[2 * j + 1];
            int32_t           s0, s1;

#if defined(__AVX2__)
            {
                const __m256i qlv = _mm256_loadu_si256(
                                        (const __m256i *)(ql + j * 32));
                const __m256i qhv = _mm256_loadu_si256(
                                        (const __m256i *)(qh));
                const __m256i m0f = _mm256_set1_epi8(0x0F);
                const __m256i bit0 = _mm256_set1_epi8((char)(1 << (2 * j)));
                const __m256i bit1 = _mm256_set1_epi8((char)(2 << (2 * j)));
                const __m256i h16  = _mm256_set1_epi8(16);
                const __m256i zero = _mm256_setzero_si256();

                /* bit clear -> 16, set -> 0: cmpeq gives 0xFF where the bit
                 * is clear, so andnot turns those lanes into 16 and the rest
                 * into 0, which is the +16 the fifth bit contributes. */
                __m256i lo0 = _mm256_cmpeq_epi8(_mm256_and_si256(qhv, bit0), zero);
                __m256i lo1 = _mm256_cmpeq_epi8(_mm256_and_si256(qhv, bit1), zero);
                __m256i h0  = _mm256_andnot_si256(lo0, h16);
                __m256i h1  = _mm256_andnot_si256(lo1, h16);
                __m256i w0  = _mm256_or_si256(_mm256_and_si256(qlv, m0f), h0);
                __m256i w1  = _mm256_or_si256(
                                  _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m0f), h1);

                s0 = dot_u8_32(a0->qs, w0);
                s1 = dot_u8_32(a1->qs, w1);
            }
#else
            {
                int8_t w8[32];
                int    l;

                for (l = 0; l < 32; ++l)
                    w8[l] = (int8_t)((ql[j * 32 + l] & 0x0F) +
                                     ((qh[l] & (1 << (2 * j))) ? 16 : 0));
                s0 = 0;
                for (l = 0; l < 32; ++l) s0 += (int32_t)a0->qs[l] * w8[l];

                for (l = 0; l < 32; ++l)
                    w8[l] = (int8_t)((ql[j * 32 + l] >> 4) +
                                     ((qh[l] & (2 << (2 * j))) ? 16 : 0));
                s1 = 0;
                for (l = 0; l < 32; ++l) s1 += (int32_t)a1->qs[l] * w8[l];
            }
#endif
            acc += d0 * a0->d * (float)s0;
            acc -= m0 * a0->d * (float)a0->sum;
            acc += d1 * a1->d * (float)s1;
            acc -= m1 * a1->d * (float)a1->sum;
        }
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q6_K: 210 bytes; ql +0, qh +128, 16 int8 scales +192, d +208       */
/*   scale changes every 16 elements, so the int32 sum is per 16      */
/* ------------------------------------------------------------------ */
static float dot_q6_K(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 210;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + 128;
        const int8_t  *sc  = (const int8_t *)(blk + 192);
        float          d   = nita_f16_to_f32((uint16_t)(blk[208] | (blk[209] << 8)));
        int            half;

        /* Each half of the super-block holds 128 weights in four 32-element
         * groups, and the scale changes every 16 elements inside those
         * groups, so the integer sum is taken per 16. */
        for (half = 0; half < 2; ++half) {
            const uint8_t    *lql = ql + half * 64;
            const uint8_t    *lqh = qh + half * 32;
            const int8_t     *lsc = sc + half * 8;
            const nita_aq8_t *g0  = &a[sb * 8 + half * 4 + 0];
            const nita_aq8_t *g1  = &a[sb * 8 + half * 4 + 1];
            const nita_aq8_t *g2  = &a[sb * 8 + half * 4 + 2];
            const nita_aq8_t *g3  = &a[sb * 8 + half * 4 + 3];
            int               is;

            for (is = 0; is < 2; ++is) {
                int32_t s1, s2, s3, s4;

#if defined(__AVX2__)
                {
                    const __m128i m0f = _mm_set1_epi8(0x0F);
                    const __m128i m3  = _mm_set1_epi8(0x03);
                    const __m128i c32 = _mm_set1_epi8(32);
                    const __m128i qla = _mm_loadu_si128((const __m128i *)(lql + is * 16));
                    const __m128i qlb = _mm_loadu_si128((const __m128i *)(lql + 32 + is * 16));
                    const __m128i qha = _mm_loadu_si128((const __m128i *)(lqh + is * 16));

                    /* q = low nibble | (high two bits << 4), then -32 */
                    __m128i w1 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(qla, m0f),
                                       _mm_slli_epi16(_mm_and_si128(qha, m3), 4)), c32);
                    __m128i w2 = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(qlb, m0f),
                                       _mm_slli_epi16(_mm_and_si128(
                                           _mm_srli_epi16(qha, 2), m3), 4)), c32);
                    __m128i w3 = _mm_sub_epi8(_mm_or_si128(
                                       _mm_and_si128(_mm_srli_epi16(qla, 4), m0f),
                                       _mm_slli_epi16(_mm_and_si128(
                                           _mm_srli_epi16(qha, 4), m3), 4)), c32);
                    __m128i w4 = _mm_sub_epi8(_mm_or_si128(
                                       _mm_and_si128(_mm_srli_epi16(qlb, 4), m0f),
                                       _mm_slli_epi16(_mm_and_si128(
                                           _mm_srli_epi16(qha, 6), m3), 4)), c32);

                    s1 = dot_i8_16(g0->qs + is * 16, w1);
                    s2 = dot_i8_16(g1->qs + is * 16, w2);
                    s3 = dot_i8_16(g2->qs + is * 16, w3);
                    s4 = dot_i8_16(g3->qs + is * 16, w4);
                }
#else
                {
                    int l;

                    s1 = s2 = s3 = s4 = 0;
                    for (l = is * 16; l < is * 16 + 16; ++l) {
                        int q1 = (int)((lql[l + 0] & 0x0F) | (((lqh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = (int)((lql[l + 32] & 0x0F) | (((lqh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((lql[l + 0] >> 4) | (((lqh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((lql[l + 32] >> 4) | (((lqh[l] >> 6) & 3) << 4)) - 32;

                        s1 += (int32_t)g0->qs[l] * q1;
                        s2 += (int32_t)g1->qs[l] * q2;
                        s3 += (int32_t)g2->qs[l] * q3;
                        s4 += (int32_t)g3->qs[l] * q4;
                    }
                }
#endif
                acc += d * (float)lsc[is + 0] * g0->d * (float)s1;
                acc += d * (float)lsc[is + 2] * g1->d * (float)s2;
                acc += d * (float)lsc[is + 4] * g2->d * (float)s3;
                acc += d * (float)lsc[is + 6] * g3->d * (float)s4;
            }
        }
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q4_0: 18 bytes / 32 weights                                        */
/*   d +0, qs[16] +2; low nibble -> element l, high -> element 16+l    */
/*   value = d * (nibble - 8), and -8 fits in the int8 weight, so the  */
/*   whole block is one signed dot with no second term.                */
/* ------------------------------------------------------------------ */
static float dot_q4_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 18;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            __m256i        w   = _mm256_sub_epi8(nibbles_32(blk + 2),
                                                 _mm256_set1_epi8(8));

            vacc = _mm256_fmadd_ps(dot_i8_vec(a[b].qs, w),
                                   _mm256_set1_ps(dw * a[b].d), vacc);
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 18;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        int32_t        s   = 0;
        int            l;

        for (l = 0; l < 16; ++l) {
            s += (int32_t)a[b].qs[l]      * ((int32_t)(blk[2 + l] & 0x0F) - 8);
            s += (int32_t)a[b].qs[16 + l] * ((int32_t)(blk[2 + l] >> 4) - 8);
        }
        acc += dw * a[b].d * (float)s;
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q5_0: 22 bytes / 32 weights                                        */
/*   d +0, qh +2 (32 bits), qs[16] +6; value = d * (5-bit - 16)        */
/*   bit j of qh is the 5th bit of element j, bit j+16 of element      */
/*   16+j, and -16 likewise folds into the int8 weight.                */
/* ------------------------------------------------------------------ */
static float dot_q5_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 22;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            /* 0xF0 where the 5th bit is clear: nib | 0xF0 is then nib - 16 */
            __m256i        hi  = _mm256_andnot_si256(bits_32(blk + 2),
                                        _mm256_set1_epi8((char)0xF0));
            __m256i        w   = _mm256_or_si256(nibbles_32(blk + 6), hi);

            vacc = _mm256_fmadd_ps(dot_i8_vec(a[b].qs, w),
                                   _mm256_set1_ps(dw * a[b].d), vacc);
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 22;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        uint32_t       qh  = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                           | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
        int32_t        s   = 0;
        int            l;

        for (l = 0; l < 16; ++l) {
            int lo = (blk[6 + l] & 0x0F) | (int)((qh >> l) & 1) << 4;
            int hi = (blk[6 + l] >> 4)    | (int)((qh >> (l + 16)) & 1) << 4;

            s += (int32_t)a[b].qs[l]      * (lo - 16);
            s += (int32_t)a[b].qs[16 + l] * (hi - 16);
        }
        acc += dw * a[b].d * (float)s;
    }
#endif
    return acc;
}

/* The 16 sub-block scales of Q3_K are 6 bits each, packed across 12 bytes in
 * a way that no shift alone undoes: the 64 kB-side bits live in the top two
 * bits of the third word. Identical to the reference in dequant.c. */
static void q3k_scales(const uint8_t *s, int8_t sc[16])
{
    static const uint32_t kmask1 = 0x03030303u;
    static const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t              aux[4];
    int                   i;

    for (i = 0; i < 3; ++i)
        aux[i] = (uint32_t)s[i * 4 + 0] | ((uint32_t)s[i * 4 + 1] << 8)
               | ((uint32_t)s[i * 4 + 2] << 16) | ((uint32_t)s[i * 4 + 3] << 24);
    {
        uint32_t tmp = aux[2];

        /* the four 2-bit highs travel in the top two bits of the third word,
         * one pair per scale word: word 0 takes bits 0-1, word 1 bits 2-3,
         * word 2 bits 4-5 and word 3 bits 6-7 */
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2)         | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2)         | (((tmp >> 2) & kmask1) << 4);
    }
    for (i = 0; i < 16; ++i)
        sc[i] = (int8_t)((aux[i / 4] >> (8 * (i % 4))) & 0xFF);
}

/* ------------------------------------------------------------------ */
/* Q8_1: 36 bytes / 32 weights                                        */
/*   d +0, s +2 (sum of the quants, unused by a dot), qs[32] int8 +4   */
/*                                                                     */
/* Q8_1 is the one type whose quantiser uses a -128 scale (-128/max,   */
/* unlike Q8_0's -127/max), so a byte may legitimately be -128 - and   */
/* both fast int8 dots lose that value: the sign trick because          */
/* -(-128) saturates, the unsigned one because 255*127 overflows the    */
/* int16 intermediate. This kernel takes the exact widening path.       */
/* ------------------------------------------------------------------ */
static float dot_q8_1(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 36;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            __m256i        w   = _mm256_loadu_si256((const __m256i *)(blk + 4));

            vacc = _mm256_fmadd_ps(dot_i8_vec_exact(a[b].qs, w),
                                   _mm256_set1_ps(dw * a[b].d), vacc);
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 36;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        int32_t        s   = 0;
        int            l;

        for (l = 0; l < 32; ++l)
            s += (int32_t)a[b].qs[l] * (int32_t)(int8_t)blk[4 + l];
        acc += dw * a[b].d * (float)s;
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q4_1: 20 bytes / 32 weights                                        */
/*   d +0, m +2, qs[16] +4; value = d*nib + m, nib in 0..15            */
/*   The min is per block and a float, so it rides a second term fed   */
/*   by the sum of the activations rather than the int8 weight.        */
/* ------------------------------------------------------------------ */
static float dot_q4_1(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();
        float  corr = 0.0f;

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 20;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            float          mw  = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
            float          dx  = a[b].d;

            vacc = _mm256_fmadd_ps(dot_u8_vec(a[b].qs, nibbles_32(blk + 4)),
                                   _mm256_set1_ps(dw * dx), vacc);
            corr += mw * dx * (float)a[b].sum;
        }
        acc = hsum256_ps(vacc) + corr;
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 20;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        float          mw  = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
        int32_t        s   = 0;
        int            l;

        for (l = 0; l < 16; ++l) {
            s += (int32_t)a[b].qs[l]      * (blk[4 + l] & 0x0F);
            s += (int32_t)a[b].qs[16 + l] * (blk[4 + l] >> 4);
        }
        acc += a[b].d * (dw * (float)s + mw * (float)a[b].sum);
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q5_1: 24 bytes / 32 weights                                        */
/*   d +0, m +2, qh +4, qs[16] +8; value = d*(5-bit) + m, no -16       */
/* ------------------------------------------------------------------ */
static float dot_q5_1(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        __m256 vacc = _mm256_setzero_ps();
        float  corr = 0.0f;

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 24;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            float          mw  = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
            float          dx  = a[b].d;
            /* 0x10 where the 5th bit is set: nib | 0x10 is the 0..31 code */
            __m256i        hi  = _mm256_and_si256(bits_32(blk + 4),
                                        _mm256_set1_epi8(0x10));
            __m256i        w   = _mm256_or_si256(nibbles_32(blk + 8), hi);

            vacc = _mm256_fmadd_ps(dot_u8_vec(a[b].qs, w),
                                   _mm256_set1_ps(dw * dx), vacc);
            corr += mw * dx * (float)a[b].sum;
        }
        acc = hsum256_ps(vacc) + corr;
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 24;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        float          mw  = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
        uint32_t       qh  = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8)
                           | ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
        int32_t        s   = 0;
        int            l;

        for (l = 0; l < 16; ++l) {
            int lo = (blk[8 + l] & 0x0F) | (int)((qh >> l) & 1) << 4;
            int hi = (blk[8 + l] >> 4)    | (int)((qh >> (l + 16)) & 1) << 4;

            s += (int32_t)a[b].qs[l]      * lo;
            s += (int32_t)a[b].qs[16 + l] * hi;
        }
        acc += a[b].d * (dw * (float)s + mw * (float)a[b].sum);
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q8_K: 292 bytes / 256 weights                                      */
/*   d f32 +0, qs[256] int8 +4, bsums[16] +260 (unused for a dot)      */
/*   The weights are already int8, so this is the plain signed dot    */
/*   with one scale for the whole 256.                                */
/* ------------------------------------------------------------------ */
static float dot_q8_K(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 292;
        float          dw;
        int            j;

        memcpy(&dw, blk, sizeof dw);

#if defined(__AVX2__)
        {
            __m256 vacc = _mm256_setzero_ps();

            for (j = 0; j < 8; ++j) {
                const nita_aq8_t *ab = &a[sb * 8 + j];

                vacc = _mm256_fmadd_ps(
                           dot_i8_vec(ab->qs,
                                      _mm256_loadu_si256(
                                          (const __m256i *)(blk + 4 + j * 32))),
                           _mm256_set1_ps(dw * ab->d), vacc);
            }
            acc += hsum256_ps(vacc);
        }
#else
        for (j = 0; j < 8; ++j) {
            const nita_aq8_t *ab = &a[sb * 8 + j];
            int32_t           s  = 0;
            int               l;

            for (l = 0; l < 32; ++l)
                s += (int32_t)ab->qs[l]
                   * (int32_t)(int8_t)blk[4 + j * 32 + l];
            acc += dw * ab->d * (float)s;
        }
#endif
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* PrismML ternary packings                                           */
/*                                                                    */
/* All four store one scale per block and weights that are already    */
/* small integers, so once the packing is expanded the dot is a plain */
/* signed int8 dot with no second term - the whole problem is turning */
/* bits, 2-bit slots and base-3 trits into bytes.                     */
/*                                                                    */
/* The tables hold the weights packed into a machine word (eight or   */
/* four int8 values per entry) so the kernels never re-type a byte    */
/* array as int32/int64: doing that is alignment and aliasing UB and  */
/* gcc -O2 duly mangled it, while -O1 was correct. Packed words load  */
/* into an integer register and go straight to the vector.            */
/* ------------------------------------------------------------------ */

/* Q1_0: 8 sign bytes per stored byte, 1 for a set bit, -1 for clear.
 * Only the scalar path calls this; the AVX2 path expands in registers. */
#if !defined(__AVX2__)
static const int64_t *q1_table(void)
{
    static int64_t t[256];
    static int     built = 0;

    if (!built) {
        int b, k;

        for (b = 0; b < 256; ++b) {
            uint64_t v = 0;

            for (k = 0; k < 8; ++k)
                v |= (uint64_t)(uint8_t)(((b >> k) & 1) ? 1 : 0xFF) << (8 * k);
            t[b] = (int64_t)v;
        }
        built = 1;
    }
    return t;
}
#endif

/* Q2_0 / PQ2_0: 4 two-bit slots per stored byte, weight is slot - 1.
 * Only the scalar path needs this; the AVX2 path expands by shifting. */
#if !defined(__AVX2__)
static const int32_t *q2_table(void)
{
    static int32_t t[256];
    static int     built = 0;

    if (!built) {
        int b, k;

        for (b = 0; b < 256; ++b) {
            uint32_t v = 0;

            for (k = 0; k < 4; ++k)
                v |= (uint32_t)(uint8_t)(int8_t)(((b >> (2 * k)) & 3) - 1)
                     << (8 * k);
            t[b] = (int32_t)v;
        }
        built = 1;
    }
    return t;
}

#endif

/* PTQ1_0: 5 base-3 trits per stored byte. The trit is (byte * 3^n) ceiling-
 * divided by 243, which the reference decode does as truncate-to-8-bits
 * followed by (q * 3) >> 8. Entry [byte*5 + n] is the weight, trit - 1. */
static const int8_t *ptq_table(void)
{
    static int8_t        t[256 * 5];
    static int           built = 0;
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };

    if (!built) {
        int b, n;

        for (b = 0; b < 256; ++b)
            for (n = 0; n < 5; ++n) {
                uint8_t q  = (uint8_t)(b * pow3[n]);
                int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);

                t[b * 5 + n] = (int8_t)(xi - 1);
            }
        built = 1;
    }
    return t;
}

#if !defined(__AVX2__)
static float dot_i8_scalar(const int8_t *x, const int8_t *w)
{
    int32_t s = 0;
    int     l;

    for (l = 0; l < 32; ++l) s += (int32_t)x[l] * (int32_t)w[l];
    return (float)s;
}
#endif

#if defined(__AVX2__)
/* eight stored slot bytes become one 32-byte weight vector.
 *
 * No table here: the eight table loads this started as were data-dependent
 * addresses, and gcc -O2 fused them into a gather, which faults whenever the
 * table lands near an unmapped page (it segfaulted roughly one run in five and
 * never under gdb, which disables ASLR). Replicating each stored byte four
 * times and shifting puts the slots in place with no data-dependent address
 * at all, and is one op cheaper. */
static __m256i q2_expand(const uint8_t *s)
{
    static const int8_t rep_i[32] = {
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
    };
    static const int8_t m1_i[32] = {
        0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
        0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0
    };
    static const int8_t m2_i[32] = {
        0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0,
        0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0
    };
    static const int8_t m3_i[32] = {
        0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1,
        0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1
    };
    const __m128i b   = _mm_loadl_epi64((const __m128i *)s);
    const __m256i rep = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(b),
                                            _mm256_loadu_si256(
                                                (const __m256i *)rep_i));
    const __m256i m3v = _mm256_set1_epi8(0x03);
    __m256i       q;

    q = _mm256_and_si256(rep, m3v);
    q = _mm256_blendv_epi8(q, _mm256_and_si256(_mm256_srli_epi16(rep, 2), m3v),
                           _mm256_loadu_si256((const __m256i *)m1_i));
    q = _mm256_blendv_epi8(q, _mm256_and_si256(_mm256_srli_epi16(rep, 4), m3v),
                           _mm256_loadu_si256((const __m256i *)m2_i));
    q = _mm256_blendv_epi8(q, _mm256_and_si256(_mm256_srli_epi16(rep, 6), m3v),
                           _mm256_loadu_si256((const __m256i *)m3_i));
    return _mm256_sub_epi8(q, _mm256_set1_epi8(1));
}
#endif

#if defined(__AVX2__)
/* 32 packed +-1 weights against 32 int8 activations, as eight int32 lanes.
 *
 * The weights are +-1, so the multiply is a sign flip: xor by an all-ones
 * byte where the sign is negative, then subtract the same mask (that is the
 * two's-complement negation; it is exact, and the pair sums that follow stay
 * inside int16 for any int8 activation). The sign bytes come from a byte
 * shuffle of the 32-bit bitfield, so this loop has no data-dependent address
 * and no dependence on the previous block. */
static inline __m256 q1_dot32(const uint8_t *s, const int8_t *x,
                              __m256i ones8, __m256i ones16)
{
    const __m256i q  = _mm256_loadu_si256((const __m256i *)x);
    const __m256i m  = _mm256_xor_si256(bits_32(s),
                                         _mm256_set1_epi8((char)-1));
    const __m256i sq = _mm256_sub_epi8(_mm256_xor_si256(q, m), m);

    return _mm256_cvtepi32_ps(_mm256_madd_epi16(
               _mm256_maddubs_epi16(ones8, sq), ones16));
}
#endif

/* Q1_0: 18 bytes / 128 weights: d f16 + 16 sign bytes; element j is bit
 * j%8 of byte j/8, so 32 consecutive elements come from 4 stored bytes.
 *
 * The four sub-blocks of a stored block are independent, so they get one
 * float accumulator each: a single chain serialised every block behind the
 * four-cycle FMA latency, which is what held this kernel at six weights per
 * cycle no matter how fast the memory was. */
static float dot_q1_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    __m256 v2 = _mm256_setzero_ps(), v3 = _mm256_setzero_ps();
#endif

    for (sb = 0; sb < nb / 4; ++sb) {
        const uint8_t    *blk = row + (size_t)sb * 18;
        const uint8_t    *qs  = blk + 2;
        float             dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const nita_aq8_t *ab  = &a[sb * 4];

#if defined(__AVX2__)
        v0 = _mm256_fmadd_ps(q1_dot32(qs + 0,  ab[0].qs, ones8, ones16),
                             _mm256_set1_ps(dw * ab[0].d), v0);
        v1 = _mm256_fmadd_ps(q1_dot32(qs + 4,  ab[1].qs, ones8, ones16),
                             _mm256_set1_ps(dw * ab[1].d), v1);
        v2 = _mm256_fmadd_ps(q1_dot32(qs + 8,  ab[2].qs, ones8, ones16),
                             _mm256_set1_ps(dw * ab[2].d), v2);
        v3 = _mm256_fmadd_ps(q1_dot32(qs + 12, ab[3].qs, ones8, ones16),
                             _mm256_set1_ps(dw * ab[3].d), v3);
#else
        {
            const int64_t *T = q1_table();
            int            j;

            for (j = 0; j < 4; ++j) {
                const nita_aq8_t *aq = &ab[j];
                int8_t            w[32];
                int               k, m;

                for (k = 0; k < 4; ++k) {
                    int64_t v = T[qs[j * 4 + k]];

                    for (m = 0; m < 8; ++m)
                        w[k * 8 + m] = (int8_t)(uint8_t)((uint64_t)v >> (8 * m));
                }
                acc += dw * aq->d * dot_i8_scalar(aq->qs, w);
            }
        }
#endif
    }
#if defined(__AVX2__)
    acc = hsum256_ps(_mm256_add_ps(_mm256_add_ps(v0, v1),
                                   _mm256_add_ps(v2, v3)));
#endif
    return acc;
}

/* Q2_0: 18 bytes / 64 weights: d f16 + 16 slot bytes, 2 bits each      */
static float dot_q2_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif

    for (sb = 0; sb < nb / 2; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 18;
        const uint8_t *qs  = blk + 2;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        int            j;

        for (j = 0; j < 2; ++j) {
            const nita_aq8_t *ab = &a[sb * 2 + j];
            int8_t            w[32];
            int               k;

#if defined(__AVX2__)
            vacc = _mm256_fmadd_ps(dot_i8_vec(ab->qs, q2_expand(qs + j * 8)),
                                   _mm256_set1_ps(dw * ab->d), vacc);
            (void)w;
            (void)k;
#else
            const int32_t *T = q2_table();
            for (k = 0; k < 8; ++k) {
                int32_t v = T[qs[j * 8 + k]];
                int     m;

                for (m = 0; m < 4; ++m)
                    w[k * 4 + m] = (int8_t)(uint8_t)((uint32_t)v >> (8 * m));
            }
            acc += dw * ab->d * dot_i8_scalar(ab->qs, w);
#endif
        }
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* PQ2_0: 34 bytes / 128 weights: the same packing over a longer group  */
static float dot_pq2_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif

    for (sb = 0; sb < nb / 4; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 34;
        const uint8_t *qs  = blk + 2;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        int            j;

        for (j = 0; j < 4; ++j) {
            const nita_aq8_t *ab = &a[sb * 4 + j];
            int8_t            w[32];
            int               k;

#if defined(__AVX2__)
            vacc = _mm256_fmadd_ps(dot_i8_vec(ab->qs, q2_expand(qs + j * 8)),
                                   _mm256_set1_ps(dw * ab->d), vacc);
            (void)w;
            (void)k;
#else
            const int32_t *T = q2_table();
            for (k = 0; k < 8; ++k) {
                int32_t v = T[qs[j * 8 + k]];
                int     m;

                for (m = 0; m < 4; ++m)
                    w[k * 4 + m] = (int8_t)(uint8_t)((uint32_t)v >> (8 * m));
            }
            acc += dw * ab->d * dot_i8_scalar(ab->qs, w);
#endif
        }
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* PTQ1_0: 28 bytes / 128 weights: 24 bytes of 5 trits each, 2 bytes of 4
 * trits each, then the fp16 scale last. Element order is trit-major: five
 * 16-wide planes from the first 16 stored bytes, five 8-wide planes from the
 * next 8, then the two tail bytes.
 *
 * Same expansion as TQ1_0: the trit of byte b at plane n is floor(b * 3^(n+1)
 * / 256) mod 3, so t = 3*r, trit = t >> 8, r = t & 0xFF. That is three ops per
 * plane for the whole lane batch, where the table version paid a load, an
 * address computation and a store per weight - and because the pieces are
 * assembled in registers the block no longer round-trips through a 128-byte
 * buffer. At this width the planes are 16 and 8 values wide, so the four
 * 32-element chunks are built by pairing them up. */
static float dot_ptq1_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif

    for (sb = 0; sb < nb / 4; ++sb) {
        const uint8_t    *blk = row + (size_t)sb * 28;
        const uint8_t    *qs  = blk;
        const uint8_t    *qh  = blk + 24;
        float             dw  = nita_f16_to_f32((uint16_t)(blk[26] | (blk[27] << 8)));
        const nita_aq8_t *ab  = &a[sb * 4];

#if defined(__AVX2__)
        const __m128i three = _mm_set1_epi16(3);
        const __m128i ff    = _mm_set1_epi16(0xFF);
        const __m128i one   = _mm_set1_epi16(1);
        __m128i       p16[5];
        __m128i       p8[5];
        __m128i       tail;
        int           n;

        /* bytes 0..15 -> five planes of 16 weights */
        {
            const __m256i three2 = _mm256_set1_epi16(3);
            const __m256i ff2    = _mm256_set1_epi16(0xFF);
            const __m256i one2   = _mm256_set1_epi16(1);
            __m256i       r      = _mm256_cvtepu8_epi16(
                                       _mm_loadu_si128((const __m128i *)qs));

            for (n = 0; n < 5; ++n) {
                const __m256i t = _mm256_mullo_epi16(r, three2);
                const __m256i d = _mm256_sub_epi16(_mm256_srli_epi16(t, 8), one2);

                p16[n] = _mm_packs_epi16(_mm256_castsi256_si128(d),
                                         _mm256_extracti128_si256(d, 1));
                r = _mm256_and_si256(t, ff2);
            }
        }

        /* bytes 16..23 -> five planes of 8 weights */
        {
            __m128i r = _mm_cvtepu8_epi16(
                            _mm_loadl_epi64((const __m128i *)(qs + 16)));

            for (n = 0; n < 5; ++n) {
                const __m128i t = _mm_mullo_epi16(r, three);
                const __m128i d = _mm_sub_epi16(_mm_srli_epi16(t, 8), one);

                p8[n] = _mm_packs_epi16(d, d);   /* low 8 bytes are the plane */
                r = _mm_and_si128(t, ff);
            }
        }

        /* the two qh bytes hold four trits each: the last eight weights */
        {
            const int8_t *T = ptq_table();
            int8_t        t8[8];
            int           k, m;

            for (k = 0; k < 4; ++k)
                for (m = 0; m < 2; ++m)
                    t8[k * 2 + m] = T[(size_t)qh[m] * 5 + k];
            tail = _mm_loadl_epi64((const __m128i *)t8);
        }

        vacc = _mm256_fmadd_ps(
                   dot_i8_vec(ab[0].qs, _mm256_set_m128i(p16[1], p16[0])),
                   _mm256_set1_ps(dw * ab[0].d), vacc);
        vacc = _mm256_fmadd_ps(
                   dot_i8_vec(ab[1].qs, _mm256_set_m128i(p16[3], p16[2])),
                   _mm256_set1_ps(dw * ab[1].d), vacc);
        vacc = _mm256_fmadd_ps(
                   dot_i8_vec(ab[2].qs,
                              _mm256_set_m128i(
                                  _mm_unpacklo_epi64(p8[0], p8[1]), p16[4])),
                   _mm256_set1_ps(dw * ab[2].d), vacc);
        vacc = _mm256_fmadd_ps(
                   dot_i8_vec(ab[3].qs,
                              _mm256_set_m128i(
                                  _mm_unpacklo_epi64(p8[4], tail),
                                  _mm_unpacklo_epi64(p8[2], p8[3]))),
                   _mm256_set1_ps(dw * ab[3].d), vacc);
#else
        static const size_t  stages[3] = { 32, 16, 8 };
        const int8_t        *T = ptq_table();
        int8_t               w[128];
        size_t               j = 0, s, m;
        int                  idx = 0, b, k, n;

        for (s = 0; s < 3; ++s) {
            size_t c = stages[s];

            for (; j + c <= 24; j += c)
                for (n = 0; n < 5; ++n)
                    for (m = 0; m < c; ++m)
                        w[idx++] = T[(size_t)qs[j + m] * 5 + n];
        }
        for (k = 0; k < 4; ++k)
            for (m = 0; m < 2; ++m)
                w[idx++] = T[(size_t)qh[m] * 5 + k];

        for (b = 0; b < 4; ++b)
            acc += dw * ab[b].d * dot_i8_scalar(ab[b].qs, w + b * 32);
#endif
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* TQ2_0: 66 bytes / 256 weights, 2 bits per weight, group 256         */
/*   qs[64] +0, d +64 (scale last, unlike Q2_0)                        */
/*   slot l is a bit plane across 32 stored bytes, so it covers 32     */
/*   consecutive elements rather than interleaving like Q2_0/PQ2_0     */
/* ------------------------------------------------------------------ */
static float dot_tq2_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 66;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[64] | (blk[65] << 8)));
        int            g, l;

        for (g = 0; g < 2; ++g) {
#if defined(__AVX2__)
            const __m256i b = _mm256_loadu_si256((const __m256i *)(blk + g * 32));
#endif

            for (l = 0; l < 4; ++l) {
                const nita_aq8_t *ab = &a[sb * 8 + g * 4 + l];

#if defined(__AVX2__)
                /* one shift per plane: no slot interleave, so no blends */
                __m256i w = _mm256_srli_epi16(b, 2 * l);

                w = _mm256_and_si256(w, _mm256_set1_epi8(0x03));
                w = _mm256_sub_epi8(w, _mm256_set1_epi8(1));
                vacc = _mm256_fmadd_ps(dot_i8_vec(ab->qs, w),
                                       _mm256_set1_ps(dw * ab->d), vacc);
#else
                int8_t w[32];
                int    m;

                for (m = 0; m < 32; ++m)
                    w[m] = (int8_t)(((blk[g * 32 + m] >> (2 * l)) & 3) - 1);
                acc += dw * ab->d * dot_i8_scalar(ab->qs, w);
#endif
            }
        }
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* TQ1_0: 54 bytes / 256 weights, dense base-3 trits, group 256        */
/*   qs[48] +0 (5 trits each), qh[4] +48 (4 trits each), d +52        */
/*                                                                    */
/* The trit of byte b at plane n is floor(b * 3^(n+1) / 256) mod 3,    */
/* which PTQ1_0 decodes through a table. Here it is a remainder chain:  */
/* t = 3*r, trit = t >> 8, r = t & 0xFF, three ops per plane per lane  */
/* batch, and no table, no staging buffer and no per-weight address.    */
/*                                                                    */
/* Upstream's element order at this block size makes the first five     */
/* planes exactly the first five 32-element chunks, so those dots run   */
/* straight from registers. The next 16 stored bytes give five planes   */
/* of 16, which are two chunks of two planes each, and the last chunk   */
/* is the fifth 16-plane followed by the four qh bytes.                 */
/* ------------------------------------------------------------------ */
static float dot_tq1_0(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t    *blk = row + (size_t)sb * 54;
        const uint8_t    *qs  = blk;
        const uint8_t    *qh  = blk + 48;
        float             dw  = nita_f16_to_f32((uint16_t)(blk[52] | (blk[53] << 8)));
        const nita_aq8_t *ab  = &a[sb * 8];

#if defined(__AVX2__)
        const __m256i three = _mm256_set1_epi16(3);
        const __m256i ff    = _mm256_set1_epi16(0xFF);
        const __m256i one8  = _mm256_set1_epi8(1);
        const __m256i one16 = _mm256_set1_epi16(1);
        __m128i       p[5];
        __m256i       lo, hi;
        int           n, k, m;

        /* bytes 0..31 -> five planes of 32, each its own activation chunk */
        lo = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs));
        hi = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(qs + 16)));

        for (n = 0; n < 5; ++n) {
            const __m256i tl = _mm256_mullo_epi16(lo, three);
            const __m256i th = _mm256_mullo_epi16(hi, three);
            __m256i       d;

            d = _mm256_packs_epi16(_mm256_srli_epi16(tl, 8),
                                   _mm256_srli_epi16(th, 8));
            d = _mm256_permute4x64_epi64(d, 0xD8);   /* undo packs lane order */
            d = _mm256_sub_epi8(d, one8);            /* trit 0..2 -> -1..1 */
            vacc = _mm256_fmadd_ps(dot_i8_vec(ab[n].qs, d),
                                   _mm256_set1_ps(dw * ab[n].d), vacc);
            lo = _mm256_and_si256(tl, ff);
            hi = _mm256_and_si256(th, ff);
        }

        /* bytes 32..47 -> five planes of 16 */
        {
            __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(qs + 32)));

            for (n = 0; n < 5; ++n) {
                const __m256i t = _mm256_mullo_epi16(r, three);
                const __m256i d = _mm256_sub_epi16(_mm256_srli_epi16(t, 8), one16);
                __m128i       packed;

                packed = _mm_packs_epi16(_mm256_castsi256_si128(d),
                                         _mm256_extracti128_si256(d, 1));
                if (n == 4) p[4] = packed;
                else        p[n] = packed;
                r = _mm256_and_si256(t, ff);
            }
        }

        /* chunks 5 and 6 take two 16-planes each */
        vacc = _mm256_fmadd_ps(
                   dot_i8_vec(ab[5].qs, _mm256_set_m128i(p[1], p[0])),
                   _mm256_set1_ps(dw * ab[5].d), vacc);
        vacc = _mm256_fmadd_ps(
                   dot_i8_vec(ab[6].qs, _mm256_set_m128i(p[3], p[2])),
                   _mm256_set1_ps(dw * ab[6].d), vacc);

        /* chunk 7: fifth 16-plane, then the four qh bytes at four trits each */
        {
            const int8_t *T = ptq_table();
            int8_t        tail[16];

            for (k = 0; k < 4; ++k)
                for (m = 0; m < 4; ++m)
                    tail[k * 4 + m] = T[(size_t)qh[m] * 5 + k];

            vacc = _mm256_fmadd_ps(
                       dot_i8_vec(ab[7].qs,
                                  _mm256_set_m128i(
                                      _mm_loadu_si128((const __m128i *)tail), p[4])),
                       _mm256_set1_ps(dw * ab[7].d), vacc);
        }
#else
        static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
        int8_t               w[256];
        int                  b, n, m;

        for (n = 0; n < 5; ++n) {
            for (m = 0; m < 32; ++m)
                w[n * 32 + m] = (int8_t)((int16_t)(((uint16_t)(uint8_t)
                                    (qs[m] * pow3[n]) * 3) >> 8) - 1);
            for (m = 0; m < 16; ++m)
                w[160 + n * 16 + m] = (int8_t)((int16_t)(((uint16_t)(uint8_t)
                                    (qs[32 + m] * pow3[n]) * 3) >> 8) - 1);
        }
        for (n = 0; n < 4; ++n)
            for (m = 0; m < 4; ++m)
                w[240 + n * 4 + m] = ptq_table()[(size_t)qh[m] * 5 + n];

        for (b = 0; b < 8; ++b)
            acc += dw * ab[b].d * dot_i8_scalar(ab[b].qs, w + b * 32);
#endif
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q2_K: 84 bytes / 256 weights, 2 bits per weight                     */
/*   scales[16] +0 (low nibble sc, high nibble min, per 16 elements)   */
/*   qs[64] +16, d +80, dmin +82                                       */
/*   value = d*sc*q - dmin*m                                           */
/* ------------------------------------------------------------------ */
static float dot_q2_K(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    {
        const __m256i m3    = _mm256_set1_epi8(0x03);
        const __m128i ones  = _mm_set1_epi8(1);
        const __m128i onesw = _mm_set1_epi16(1);
        __m128        vacc  = _mm_setzero_ps();

        for (sb = 0; sb < nb / 8; ++sb) {
            const uint8_t *blk  = row + (size_t)sb * 84;
            const uint8_t *qs   = blk + 16;
            float d    = nita_f16_to_f32((uint16_t)(blk[80] | (blk[81] << 8)));
            float dmin = nita_f16_to_f32((uint16_t)(blk[82] | (blk[83] << 8)));
            int   is   = 0;
            int   chunk;

            for (chunk = 0; chunk < 2; ++chunk) {
                const __m256i q2v = _mm256_loadu_si256((const __m256i *)qs);
                int shift = 0;
                int j;

                for (j = 0; j < 4; ++j) {
                    const __m256i bits = _mm256_and_si256(
                                             _mm256_srli_epi16(q2v, shift), m3);
                    const __m128i lo = _mm256_castsi256_si128(bits);
                    const __m128i hi = _mm256_extracti128_si256(bits, 1);
                    const nita_aq8_t *ab = &a[sb * 8 + chunk * 4 + j];
                    int h;

                    /* the two 16-element halves carry their own scale and min,
                     * so each keeps its own four lanes of the accumulator */
                    for (h = 0; h < 2; ++h) {
                        const __m128i x16 = _mm_loadu_si128(
                                               (const __m128i *)(ab->qs + h * 16));
                        int     sc = blk[is++];
                        float   dl = d    * (float)(sc & 0x0F);
                        float   ml = dmin * (float)(sc >> 4);
                        __m128i p4, c4;

                        p4 = _mm_madd_epi16(
                                 _mm_maddubs_epi16(h ? hi : lo, x16), onesw);
                        c4 = _mm_madd_epi16(_mm_maddubs_epi16(ones, x16), onesw);
                        vacc = _mm_fmadd_ps(_mm_cvtepi32_ps(p4),
                                            _mm_set1_ps(dl * ab->d), vacc);
                        vacc = _mm_fmadd_ps(_mm_cvtepi32_ps(c4),
                                            _mm_set1_ps(-ml * ab->d), vacc);
                    }
                    shift += 2;
                }
                qs += 32;
            }
        }
        acc = hsum128_ps(vacc);
    }
#else
    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk  = row + (size_t)sb * 84;
        const uint8_t *qs   = blk + 16;
        float d    = nita_f16_to_f32((uint16_t)(blk[80] | (blk[81] << 8)));
        float dmin = nita_f16_to_f32((uint16_t)(blk[82] | (blk[83] << 8)));
        int   is   = 0;
        int   chunk;

        for (chunk = 0; chunk < 2; ++chunk) {
            int shift = 0;
            int j;

            for (j = 0; j < 4; ++j) {
                const nita_aq8_t *ab = &a[sb * 8 + chunk * 4 + j];
                int h;

                for (h = 0; h < 2; ++h) {
                    int     sc = blk[is++];
                    float   dl = d    * (float)(sc & 0x0F);
                    float   ml = dmin * (float)(sc >> 4);
                    int32_t s  = 0, xs = 0;
                    int     l;

                    for (l = 0; l < 16; ++l) {
                        int q = (qs[h * 16 + l] >> shift) & 3;

                        s  += (int32_t)ab->qs[h * 16 + l] * q;
                        xs += ab->qs[h * 16 + l];
                    }
                    acc += ab->d * (dl * (float)s - ml * (float)xs);
                }
                shift += 2;
            }
            qs += 32;
        }
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Q3_K: 110 bytes / 256 weights, 3 bits per weight                    */
/*   hmask[32] +0, qs[64] +32, scales[12] +96, d +108                  */
/*   value = d*(sc-32) * q, q = 2-bit - 4 when the hmask bit is clear  */
/*   and q = 2-bit when it is set, i.e. q in -4..3: a signed byte, so  */
/*   the scale is the only per-half factor left.                       */
/* ------------------------------------------------------------------ */
static float dot_q3_K(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 110;
        float          d   = nita_f16_to_f32((uint16_t)(blk[108] | (blk[109] << 8)));
        int8_t         sc[16];
        int            is = 0;
        int            chunk;
        uint8_t        m  = 1;

        q3k_scales(blk + 96, sc);

#if defined(__AVX2__)
        {
            const __m256i m3    = _mm256_set1_epi8(0x03);
            const __m128i onesw = _mm_set1_epi16(1);
            const __m256i hm    = _mm256_loadu_si256((const __m256i *)blk);
            __m128        vacc  = _mm_setzero_ps();

            for (chunk = 0; chunk < 2; ++chunk) {
                const __m256i q3v = _mm256_loadu_si256(
                                        (const __m256i *)(blk + 32 + chunk * 32));
                int shift = 0;
                int j;

                for (j = 0; j < 4; ++j) {
                    const __m256i bits = _mm256_and_si256(
                                             _mm256_srli_epi16(q3v, shift), m3);
                    const __m256i bit  = _mm256_set1_epi8((char)m);
                    const __m256i clr  = _mm256_cmpeq_epi8(
                                             _mm256_and_si256(hm, bit),
                                             _mm256_setzero_si256());
                    const __m256i w    = _mm256_sub_epi8(bits,
                                             _mm256_and_si256(clr,
                                                 _mm256_set1_epi8(4)));
                    const __m128i lo = _mm256_castsi256_si128(w);
                    const __m128i hi = _mm256_extracti128_si256(w, 1);
                    const nita_aq8_t *ab = &a[sb * 8 + chunk * 4 + j];
                    int h;

                    for (h = 0; h < 2; ++h) {
                        const __m128i x16 = _mm_loadu_si128(
                                               (const __m128i *)(ab->qs + h * 16));
                        const __m128i ax  = _mm_sign_epi8(x16, x16);
                        const __m128i sw  = _mm_sign_epi8(h ? hi : lo, x16);
                        const __m128i p4  = _mm_madd_epi16(
                                                _mm_maddubs_epi16(ax, sw), onesw);
                        float dl = d * (float)(sc[is++] - 32) * ab->d;

                        vacc = _mm_fmadd_ps(_mm_cvtepi32_ps(p4),
                                            _mm_set1_ps(dl), vacc);
                    }
                    shift += 2;
                    m = (uint8_t)(m << 1);
                }
            }
            acc += hsum128_ps(vacc);
        }
#else
        for (chunk = 0; chunk < 2; ++chunk) {
            int shift = 0;
            int j;

            for (j = 0; j < 4; ++j) {
                const nita_aq8_t *ab = &a[sb * 8 + chunk * 4 + j];
                int h;

                for (h = 0; h < 2; ++h) {
                    float   dl = d * (float)(sc[is++] - 32) * ab->d;
                    int32_t s  = 0;
                    int     l;

                    for (l = 0; l < 16; ++l) {
                        int q = ((blk[32 + chunk * 32 + h * 16 + l] >> shift) & 3)
                              - ((blk[h * 16 + l] & m) ? 0 : 4);

                        s += (int32_t)ab->qs[h * 16 + l] * q;
                    }
                    acc += dl * (float)s;
                }
                shift += 2;
                m = (uint8_t)(m << 1);
            }
        }
#endif
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* IQ4_XS: 136 bytes / 256 weights, non-linear 4-bit codebook         */
/*   d +0, scales_h +2, scales_l +4, qs +8                            */
/*   sub-block ib: 16 bytes -> low nibbles 0..15, high nibbles 16..31  */
/* ------------------------------------------------------------------ */
static const int8_t g_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static const int8_t g_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

/* 2^(e-128): the E8M0 exponent, halved because the codebook is doubled.
 * Built once; a concurrent duplicate fill writes identical values, and a
 * float store is a single aligned write, so the race is benign. */
static float e8m0_half(uint8_t e)
{
    static float  tab[256];
    static int    ready = 0;

    if (!ready) {
        int i;

        for (i = 0; i < 256; ++i) tab[i] = exp2f((float)i - 128.0f);
        ready = 1;
    }
    return tab[e];
}

static float dot_iq4_xs(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    {
        const __m256i tbl = _mm256_broadcastsi128_si256(
                                _mm_loadu_si128((const __m128i *)g_iq4nl));
        const __m128i m0f = _mm_set1_epi8(0x0F);
        __m256        vacc = _mm256_setzero_ps();

        for (sb = 0; sb < nb / 8; ++sb) {
            const uint8_t *blk = row + (size_t)sb * 136;
            float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            uint16_t       h   = (uint16_t)(blk[2] | (blk[3] << 8));
            const uint8_t *sl  = blk + 4;
            const uint8_t *qs  = blk + 8;
            int            ib;

            for (ib = 0; ib < 8; ++ib) {
                const nita_aq8_t *ab = &a[sb * 8 + ib];
                int   ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                         | (((h >> (2 * ib)) & 3) << 4);
                float dl = d * (float)(ls - 32);
                const __m128i b = _mm_loadu_si128(
                                      (const __m128i *)(qs + ib * 16));
                const __m256i nib = _mm256_set_m128i(
                                        _mm_and_si128(_mm_srli_epi16(b, 4), m0f),
                                        _mm_and_si128(b, m0f));
                const __m256i w8 = _mm256_shuffle_epi8(tbl, nib);

                vacc = _mm256_fmadd_ps(dot_i8_vec(ab->qs, w8),
                                       _mm256_set1_ps(dl * ab->d), vacc);
            }
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 136;
        float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        uint16_t       h   = (uint16_t)(blk[2] | (blk[3] << 8));
        const uint8_t *sl  = blk + 4;
        const uint8_t *qs  = blk + 8;
        int            ib;

        for (ib = 0; ib < 8; ++ib) {
            const nita_aq8_t *ab = &a[sb * 8 + ib];
            int   ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                     | (((h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            int32_t s = 0;
            int   l;

            for (l = 0; l < 16; ++l) {
                s += (int32_t)ab->qs[l]      * (int32_t)g_iq4nl[qs[ib * 16 + l] & 0x0F];
                s += (int32_t)ab->qs[16 + l] * (int32_t)g_iq4nl[qs[ib * 16 + l] >> 4];
            }
            acc += dl * ab->d * (float)s;
        }
    }
#endif
    return acc;
}

/* IQ4_NL: 18 bytes / 32 weights, fp16 d, NIBBLE is a codebook index.
 * Same layout as IQ4_XS without the per-sub-block scales: low nibbles hold
 * elements 0-15, high nibbles elements 16-31. */
static float dot_iq4_nl(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        const __m256i tbl = _mm256_broadcastsi128_si256(
                                _mm_loadu_si128((const __m128i *)g_iq4nl));
        const __m128i m0f = _mm_set1_epi8(0x0F);
        __m256        vacc = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t  *blk = row + (size_t)b * 18;
            const nita_aq8_t *ab = &a[b];
            float           d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const __m128i   q   = _mm_loadu_si128((const __m128i *)(blk + 2));
            const __m256i   nib = _mm256_set_m128i(
                                      _mm_and_si128(_mm_srli_epi16(q, 4), m0f),
                                      _mm_and_si128(q, m0f));
            const __m256i   w8  = _mm256_shuffle_epi8(tbl, nib);

            vacc = _mm256_fmadd_ps(dot_i8_vec(ab->qs, w8),
                                   _mm256_set1_ps(d * ab->d), vacc);
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t  *blk = row + (size_t)b * 18;
        const nita_aq8_t *ab = &a[b];
        float           d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t  *qs  = blk + 2;
        int32_t         s   = 0;
        int             l;

        for (l = 0; l < 16; ++l) {
            s += (int32_t)ab->qs[l]      * (int32_t)g_iq4nl[qs[l] & 0x0F];
            s += (int32_t)ab->qs[16 + l] * (int32_t)g_iq4nl[qs[l] >> 4];
        }
        acc += d * ab->d * (float)s;
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* IQ-quant (i-quant) integer kernels.                                */
/*                                                                     */
/* Grid/sign tables are vendored verbatim in iq_tables.h (upstream     */
/* ggml-common.h, MIT). Every grid value fits an int8, so each 32-     */
/* element block decodes into a plain int8 vector and rides the same   */
/* maddubs/madd path as the K-quants. Float scale multiplies happen    */
/* once per sub-block, never per weight.                               */
/*                                                                     */
/* Sign conventions differ per type, and mixing them up is silent      */
/* corruption, so they are spelled out at each site:                   */
/*   IQ2_XXS/IQ2_XS/IQ3_XXS : 7-bit index -> ksigns64[i] (0x00/0xff)   */
/*   IQ2_S/IQ3_S            : sign byte stored raw -> signmask64[b]    */
/*   IQ1_S/IQ1_M            : no sign byte; the grid holds int8 incl.  */
/*                            sign, and the delta term needs the sum   */
/*                            of the activations per block/8 group.    */
/* ------------------------------------------------------------------ */
static int8_t iq_sneg(uint8_t g, uint8_t m)
{
    /* m is 0x00 or 0xff. (g ^ 0xff) - 0xff == ~g + 1 == -g (byte wrap). */
    return (int8_t)(((g ^ m) - m) & 0xFFu);
}

static void iq_grid8(const uint64_t *grid, int idx, uint64_t m64, int8_t *w)
{
    uint64_t  g;
    const uint8_t *gb = (const uint8_t *)&g;
    int       j;

    memcpy(&g, grid + idx, sizeof g);
    for (j = 0; j < 8; ++j)
        w[j] = iq_sneg(gb[j], (uint8_t)(m64 >> (8 * j)));
}

static void iq_grid4x2(const uint32_t *grid, int idx1, int idx2, uint64_t m64,
                       int8_t *w)
{
    uint32_t  g1, g2;
    const uint8_t *a = (const uint8_t *)&g1, *b = (const uint8_t *)&g2;
    int       j;

    memcpy(&g1, grid + idx1, sizeof g1);
    memcpy(&g2, grid + idx2, sizeof g2);
    for (j = 0; j < 4; ++j) w[j]     = iq_sneg(a[j], (uint8_t)(m64 >> (8 * j)));
    for (j = 0; j < 4; ++j) w[4 + j] = iq_sneg(b[j], (uint8_t)(m64 >> (8 * (4 + j))));
}

#if defined(__AVX2__)
static __m256 iq_dot32(const int8_t *x, const int8_t *w8)
{
    return dot_i8_vec(x, _mm256_loadu_si256((const __m256i *)w8));
}
static int iq_dot16(const int8_t *x, const int8_t *w8)
{
    return dot_i8_16(x, _mm_loadu_si128((const __m128i *)w8));
}
#else
static int iq_dot32_scalar(const int8_t *x, const int8_t *w8)
{
    int32_t s = 0;
    int     j;

    for (j = 0; j < 32; ++j) s += (int32_t)x[j] * (int32_t)w8[j];
    return s;
}
#endif

/* ggml type 16: IQ2_XXS. 66 bytes / 256 weights.
 *   d fp16; qs i.e. 8 sub-blocks of (4 grid idx bytes + uint32 scales/signs) */
static float dot_iq2_xxs(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif
    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 66;
        float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qsb = blk + 2;
        int            ib32;

        for (ib32 = 0; ib32 < 8; ++ib32) {
            const nita_aq8_t *ab = &a[sb * 8 + ib32];
            uint32_t        aux0, aux1;
            float           db;
            const uint8_t  *gi;
            int8_t          w8[32];
            int             l;

            memcpy(&aux0, qsb + 8 * ib32, sizeof aux0);
            memcpy(&aux1, qsb + 8 * ib32 + 4, sizeof aux1);
            db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            gi = (const uint8_t *)&aux0;

            for (l = 0; l < 4; ++l)
                iq_grid8(nita_iq2xxs_grid, gi[l],
                         nita_ksigns64[(aux1 >> 7 * l) & 127], w8 + 8 * l);
#if defined(__AVX2__)
            vacc = _mm256_fmadd_ps(iq_dot32(ab->qs, w8),
                                   _mm256_set1_ps(db * ab->d), vacc);
#else
            acc += db * ab->d * (float)iq_dot32_scalar(ab->qs, w8);
#endif
        }
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* ggml type 17: IQ2_XS. 74 bytes / 256 weights.
 *   d fp16; qs uint16[32] (9 grid bits + 7 sign bits); scales uint8[8]. */
static float dot_iq2_xs(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 74;
        float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint16_t *qs = (const uint16_t *)(blk + 2);
        const uint8_t *sc  = blk + 66;
        int            ib32;

        for (ib32 = 0; ib32 < 8; ++ib32) {
            const nita_aq8_t *ab = &a[sb * 8 + ib32];
            const uint8_t   s    = sc[ib32];
            const float     db0  = d * (0.5f + (float)(s & 0x0F)) * 0.25f;
            const float     db1  = d * (0.5f + (float)(s >> 4)) * 0.25f;
            int8_t          w8[32];
            int             l;

            for (l = 0; l < 4; ++l) {
                const uint16_t q = qs[4 * ib32 + l];

                iq_grid8(nita_iq2xs_grid, (int)(q & 511),
                         nita_ksigns64[q >> 9], w8 + 8 * l);
            }
#if defined(__AVX2__)
            acc += ab->d * (db0 * (float)iq_dot16(ab->qs, w8)
                          + db1 * (float)iq_dot16(ab->qs + 16, w8 + 16));
#else
            {
                int32_t s0 = 0, s1 = 0, j;

                for (j = 0; j < 16; ++j) s0 += (int32_t)ab->qs[j] * w8[j];
                for (j = 0; j < 16; ++j) s1 += (int32_t)ab->qs[16 + j] * w8[16 + j];
                acc += ab->d * (db0 * (float)s0 + db1 * (float)s1);
            }
#endif
        }
    }
    return acc;
}

/* ggml type 22: IQ2_S. 82 bytes / 256 weights.
 *   d fp16; qs uint8[64] (first 32 grid idx, next 32 raw sign bits);
 *   qh uint8[8] (grid idx bits 8-9); scales uint8[8]. */
static float dot_iq2_s(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 82;
        float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs  = blk + 2;
        const uint8_t *qh  = blk + 66;
        const uint8_t *signs = qs + 32;
        const uint8_t *sc  = blk + 74;
        int            ib32;

        for (ib32 = 0; ib32 < 8; ++ib32) {
            const nita_aq8_t *ab = &a[sb * 8 + ib32];
            const uint8_t   s    = sc[ib32];
            const float     db0  = d * (0.5f + (float)(s & 0x0F)) * 0.25f;
            const float     db1  = d * (0.5f + (float)(s >> 4)) * 0.25f;
            int8_t          w8[32];
            int             l;

            for (l = 0; l < 4; ++l) {
                const int idx = qs[4 * ib32 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);

                iq_grid8(nita_iq2s_grid, idx,
                         nita_signmask64[signs[4 * ib32 + l]], w8 + 8 * l);
            }
#if defined(__AVX2__)
            acc += ab->d * (db0 * (float)iq_dot16(ab->qs, w8)
                          + db1 * (float)iq_dot16(ab->qs + 16, w8 + 16));
#else
            {
                int32_t s0 = 0, s1 = 0, j;

                for (j = 0; j < 16; ++j) s0 += (int32_t)ab->qs[j] * w8[j];
                for (j = 0; j < 16; ++j) s1 += (int32_t)ab->qs[16 + j] * w8[16 + j];
                acc += ab->d * (db0 * (float)s0 + db1 * (float)s1);
            }
#endif
        }
    }
    return acc;
}

/* ggml type 18: IQ3_XXS. 98 bytes / 256 weights.
 *   d fp16; qs uint8[96] (first 64 grid idx, next 32 scales+signs). */
static float dot_iq3_xxs(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif
    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 98;
        float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs  = blk + 2;
        const uint8_t *scales_and_signs = qs + 64;
        int            ib32;

        for (ib32 = 0; ib32 < 8; ++ib32) {
            const nita_aq8_t *ab = &a[sb * 8 + ib32];
            uint32_t        aux32;
            float           db;
            const uint8_t  *q   = qs + 8 * ib32;
            int8_t          w8[32];
            int             l;

            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof aux32);
            db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (l = 0; l < 4; ++l)
                iq_grid4x2(nita_iq3xxs_grid, q[2 * l + 0], q[2 * l + 1],
                           nita_ksigns64[(aux32 >> 7 * l) & 127], w8 + 8 * l);
#if defined(__AVX2__)
            vacc = _mm256_fmadd_ps(iq_dot32(ab->qs, w8),
                                   _mm256_set1_ps(db * ab->d), vacc);
#else
            acc += db * ab->d * (float)iq_dot32_scalar(ab->qs, w8);
#endif
        }
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* ggml type 21: IQ3_S. 110 bytes / 256 weights.
 *   d fp16; qs uint8[64]; qh uint8[8]; signs uint8[32]; scales uint8[4]
 *   (2 nibbles per 32-block, one pair of blocks per byte). */
static float dot_iq3_s(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif
    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk  = row + (size_t)sb * 110;
        float          d    = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs   = blk + 2;
        const uint8_t *qh   = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *sc   = blk + 106;
        int            ib32;

        for (ib32 = 0; ib32 < 8; ib32 += 2) {
            const nita_aq8_t *ab1 = &a[sb * 8 + ib32];
            const nita_aq8_t *ab2 = &a[sb * 8 + ib32 + 1];
            const float      db1  = d * (1.0f + 2.0f * (float)(sc[ib32 / 2] & 0x0F));
            const float      db2  = d * (1.0f + 2.0f * (float)(sc[ib32 / 2] >> 4));
            const uint8_t   *q1   = qs + 8 * ib32;
            const uint8_t   *sn1  = signs + 4 * ib32;
            int8_t           w8[32];
            int              l;

            for (l = 0; l < 4; ++l)
                iq_grid4x2(nita_iq3s_grid,
                           (int)(q1[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 0x100)),
                           (int)(q1[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 0x100)),
                           nita_signmask64[sn1[l]], w8 + 8 * l);
#if defined(__AVX2__)
            vacc = _mm256_fmadd_ps(iq_dot32(ab1->qs, w8),
                                   _mm256_set1_ps(db1 * ab1->d), vacc);
#else
            acc += db1 * ab1->d * (float)iq_dot32_scalar(ab1->qs, w8);
#endif
            {
                const uint8_t *q2  = q1 + 8;
                const uint8_t *sn2 = sn1 + 4;

                for (l = 0; l < 4; ++l)
                    iq_grid4x2(nita_iq3s_grid,
                               (int)(q2[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 0x100)),
                               (int)(q2[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 0x100)),
                               nita_signmask64[sn2[l]], w8 + 8 * l);
#if defined(__AVX2__)
                vacc = _mm256_fmadd_ps(iq_dot32(ab2->qs, w8),
                                       _mm256_set1_ps(db2 * ab2->d), vacc);
#else
                acc += db2 * ab2->d * (float)iq_dot32_scalar(ab2->qs, w8);
#endif
            }
            qh    += 2;
        }
    }
#if defined(__AVX2__)
    acc = hsum256_ps(vacc);
#endif
    return acc;
}

/* ggml type 19: IQ1_S. 50 bytes / 256 weights.
 *   d fp16; qs uint8[32] grid idx bytes; qh uint16[8] (idx bits 8-10,
 *   per-block scale and delta sign). value = dl*(grid + delta). */
static float dot_iq1_s(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row + (size_t)sb * 50;
        float          d   = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs  = blk + 2;
        const uint16_t *qh = (const uint16_t *)(blk + 34);
        int            ib;

        for (ib = 0; ib < 8; ++ib) {
            const nita_aq8_t *ab = &a[sb * 8 + ib];
            const uint16_t  qhv  = qh[ib];
            const float     dl   = d * (2.0f * (float)((qhv >> 12) & 7) + 1.0f);
            const float     delta = (qhv & 0x8000u) ? -0.125f : 0.125f;
            int8_t          w8[32];
            int             l;

            for (l = 0; l < 4; ++l) {
                const uint64_t *g = nita_iq1s_grid + (qs[4 * ib + l] | (((qhv >> (3 * l)) & 7) << 8));
                memcpy(w8 + 8 * l, g, 8);   /* grid bytes are int8 already */
            }
#if defined(__AVX2__)
            acc += ab->d * dl * (hsum256_ps(iq_dot32(ab->qs, w8)) + delta * (float)ab->sum);
#else
            acc += ab->d * dl * ((float)iq_dot32_scalar(ab->qs, w8) + delta * (float)ab->sum);
#endif
        }
    }
    return acc;
}

/* ggml type 29: IQ1_M. 56 bytes / 256 weights, no fp16 d in the block:
 * the scale is synthesised from 12 bits spread over the four scale u16s.
 *   qs uint8[32] grid idx bytes; qh uint8[16] (idx bits 8-10, delta
 *   signs); scales uint8[8] (3-bit scales + 12 bits of d). */
static float dot_iq1_m(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   sb;

    for (sb = 0; sb < nb / 8; ++sb) {
        const uint8_t *blk = row;
        uint16_t       sc[4];
        uint16_t       su;
        float          d;

        row += 56;
        memcpy(sc, blk + 48, sizeof sc);
        su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                      | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        d  = nita_f16_to_f32(su);
        {
            const uint8_t *qs = blk + 0;
            const uint8_t *qh = blk + 32;
            int            ib;

            for (ib = 0; ib < 8; ++ib) {
                const nita_aq8_t *ab = &a[sb * 8 + ib];
                const uint16_t   s   = sc[ib / 2];
                const float      dl1 = d * (2.0f * (float)((s >> (6 * (ib % 2) + 0)) & 7) + 1.0f);
                const float      dl2 = d * (2.0f * (float)((s >> (6 * (ib % 2) + 3)) & 7) + 1.0f);
                const uint8_t   q_0  = qs[4 * ib + 0], q_1 = qs[4 * ib + 1];
                const uint8_t   q_2  = qs[4 * ib + 2], q_3 = qs[4 * ib + 3];
                const uint8_t   h_0  = qh[2 * ib + 0], h_1 = qh[2 * ib + 1];
                const int       idx0 = q_0 | ((h_0 << 8) & 0x700);
                const int       idx1 = q_1 | ((h_0 << 4) & 0x700);
                const int       idx2 = q_2 | ((h_1 << 8) & 0x700);
                const int       idx3 = q_3 | ((h_1 << 4) & 0x700);
                const float     d0 = (h_0 & 0x08) ? -0.125f : 0.125f;
                const float     d1 = (h_0 & 0x80) ? -0.125f : 0.125f;
                const float     d2 = (h_1 & 0x08) ? -0.125f : 0.125f;
                const float     d3 = (h_1 & 0x80) ? -0.125f : 0.125f;
                int8_t          w8[32];
                uint64_t        g;

                memcpy(&g, nita_iq1s_grid + idx0, 8); memcpy(w8 +  0, (const uint8_t *)&g, 8);
                memcpy(&g, nita_iq1s_grid + idx1, 8); memcpy(w8 +  8, (const uint8_t *)&g, 8);
                memcpy(&g, nita_iq1s_grid + idx2, 8); memcpy(w8 + 16, (const uint8_t *)&g, 8);
                memcpy(&g, nita_iq1s_grid + idx3, 8); memcpy(w8 + 24, (const uint8_t *)&g, 8);
#if defined(__AVX2__)
                {
                    const int lo = iq_dot16(ab->qs, w8);
                    const int hi = iq_dot16(ab->qs + 16, w8 + 16);

                    acc += ab->d * (dl1 * ((float)lo + d0 * (float)ab->sums[0]
                                                      + d1 * (float)ab->sums[1])
                                  + dl2 * ((float)hi + d2 * (float)ab->sums[2]
                                                      + d3 * (float)ab->sums[3]));
                }
#else
                {
                    int32_t s0 = 0, s1 = 0, s2_ = 0, s3 = 0, j;

                    for (j = 0; j < 8; ++j) s0 += (int32_t)ab->qs[j] * w8[j];
                    for (j = 0; j < 8; ++j) s1 += (int32_t)ab->qs[8 + j] * w8[8 + j];
                    for (j = 0; j < 8; ++j) s2_ += (int32_t)ab->qs[16 + j] * w8[16 + j];
                    for (j = 0; j < 8; ++j) s3 += (int32_t)ab->qs[24 + j] * w8[24 + j];
                    acc += ab->d * (dl1 * ((float)(s0 + s1) + d0 * (float)ab->sums[0]
                                                              + d1 * (float)ab->sums[1])
                                  + dl2 * ((float)(s2_ + s3) + d2 * (float)ab->sums[2]
                                                              + d3 * (float)ab->sums[3]));
                }
#endif
            }
        }
    }
    return acc;
}

/* MXFP4: 17 bytes / 32 weights, E8M0 block scale, doubled e2m1 codes */
static float dot_mxfp4(const uint8_t *row, const nita_aq8_t *a, int nb)
{
    float acc = 0.0f;
    int   b;

#if defined(__AVX2__)
    {
        const __m256i tbl = _mm256_broadcastsi128_si256(
                                _mm_loadu_si128((const __m128i *)g_fp4));
        const __m128i m0f = _mm_set1_epi8(0x0F);
        __m256        vacc = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 17;
            float          d   = e8m0_half(blk[0]);
            const __m128i  q   = _mm_loadu_si128((const __m128i *)(blk + 1));
            const __m256i nib  = _mm256_set_m128i(
                                     _mm_and_si128(_mm_srli_epi16(q, 4), m0f),
                                     _mm_and_si128(q, m0f));
            const __m256i w8   = _mm256_shuffle_epi8(tbl, nib);

            vacc = _mm256_fmadd_ps(dot_i8_vec(a[b].qs, w8),
                                   _mm256_set1_ps(d * a[b].d), vacc);
        }
        acc = hsum256_ps(vacc);
    }
#else
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 17;
        float          d   = e8m0_half(blk[0]);
        int32_t        s   = 0;
        int            l;

        for (l = 0; l < 16; ++l) {
            s += (int32_t)a[b].qs[l]      * (int32_t)g_fp4[blk[1 + l] & 0x0F];
            s += (int32_t)a[b].qs[16 + l] * (int32_t)g_fp4[blk[1 + l] >> 4];
        }
        acc += d * a[b].d * (float)s;
    }
#endif
    return acc;
}

/* ------------------------------------------------------------------ */
/* Tile form: expand each weight block once, dot it with every token  */
/* ------------------------------------------------------------------ */

static void dot_tile_q8_0(const uint8_t *row, const nita_aq8_t *a, int stride,
                          int t, int nb, float *out, int out_stride)
{
    int i, b, i0;

#if defined(__AVX2__)
    /* Tokens are taken eight at a time: one accumulator per token has to live
     * in a register, and there are only sixteen of those. Wider tiles simply
     * make another pass over the weights, exactly as the float kernel does
     * with four. */
    for (i0 = 0; i0 < t; i0 += 8) {
        const int tc = (t - i0) < 8 ? (t - i0) : 8;
        __m256    vacc[8];

        for (i = 0; i < tc; ++i) vacc[i] = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 34;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const __m256i  w   = _mm256_loadu_si256((const __m256i *)(blk + 2));

            for (i = 0; i < tc; ++i) {
                const nita_aq8_t *ai = &a[(size_t)(i0 + i) * stride + b];

                vacc[i] = _mm256_fmadd_ps(dot_i8_vec(ai->qs, w),
                                          _mm256_set1_ps(dw * ai->d), vacc[i]);
            }
        }
        for (i = 0; i < tc; ++i)
            out[(size_t)(i0 + i) * out_stride] = hsum256_ps(vacc[i]);
    }
    return;
#endif
    for (i = 0; i < t; ++i) out[(size_t)i * out_stride] = 0.0f;
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 34;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));

        for (i = 0; i < t; ++i) {
            const nita_aq8_t *ai = &a[(size_t)i * stride + b];
            int32_t           sum = 0;
            int               l;

            for (l = 0; l < 32; ++l)
                sum += (int32_t)ai->qs[l] * (int32_t)(int8_t)blk[2 + l];
            out[(size_t)i * out_stride] += dw * ai->d * (float)sum;
        }
    }
}

static void dot_tile_q4_K(const uint8_t *row, const nita_aq8_t *a, int stride,
                          int t, int nb, float *out, int out_stride)
{
    int i, i0;

#if defined(__AVX2__)
    for (i0 = 0; i0 < t; i0 += 8) {
        const int tc = (t - i0) < 8 ? (t - i0) : 8;
        __m256    vacc[8];
        float     msum[8];
        int       sb;

        for (i = 0; i < tc; ++i) {
            vacc[i] = _mm256_setzero_ps();
            msum[i] = 0.0f;
        }
        for (sb = 0; sb < nb / 8; ++sb) {
            const uint8_t *blk  = row + (size_t)sb * 144;
            const uint8_t *qs   = blk + 16;
            float          d    = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            float          dmin = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
            int8_t         sc[8], mn[8];
            int            j;

            q4k_scales(blk + 4, sc, mn);
            for (j = 0; j < 4; ++j) {
                /* Mirror the single-vector kernel's layout: 32 bytes of qs
                 * cover two 32-element groups — low nibbles then high
                 * nibbles — with scales/mins interleaved as sc[2j], sc[2j+1].
                 * An earlier version sliced qs linearly in 16-byte chunks,
                 * which scrambled both the byte order and the scale pairing. */
                const __m256i q256 = _mm256_loadu_si256((const __m256i *)(qs + j * 32));
                const __m256i m0f256 = _mm256_set1_epi8(0x0F);
                const __m256i wlo  = _mm256_and_si256(q256, m0f256);
                const __m256i whi  = _mm256_and_si256(_mm256_srli_epi16(q256, 4), m0f256);

                for (i = 0; i < tc; ++i) {
                    const nita_aq8_t *ai0 = &a[(size_t)(i0 + i) * stride + sb * 8 + 2 * j];
                    const nita_aq8_t *ai1 = &a[(size_t)(i0 + i) * stride + sb * 8 + 2 * j + 1];

                    vacc[i] = _mm256_fmadd_ps(
                                  dot_u8_vec(ai0->qs, wlo),
                                  _mm256_set1_ps(d * (float)sc[2 * j] * ai0->d), vacc[i]);
                    vacc[i] = _mm256_fmadd_ps(
                                  dot_u8_vec(ai1->qs, whi),
                                  _mm256_set1_ps(d * (float)sc[2 * j + 1] * ai1->d), vacc[i]);
                    msum[i] += dmin * (float)mn[2 * j]     * ai0->d * (float)ai0->sum;
                    msum[i] += dmin * (float)mn[2 * j + 1] * ai1->d * (float)ai1->sum;
                }
            }
        }
        for (i = 0; i < tc; ++i)
            out[(size_t)(i0 + i) * out_stride] = hsum256_ps(vacc[i]) - msum[i];
    }
    return;
#endif
    for (i = 0; i < t; ++i) out[(size_t)i * out_stride] = 0.0f;
    {
        int sb;

        for (sb = 0; sb < nb / 8; ++sb) {
            const uint8_t *blk  = row + (size_t)sb * 144;
            const uint8_t *qs   = blk + 16;
            float          d    = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            float          dmin = nita_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
            int8_t         sc[8], mn[8];
            int            j;

            q4k_scales(blk + 4, sc, mn);
            for (j = 0; j < 4; ++j) {
                int8_t wlo[32], whi[32];
                int    l;

                for (l = 0; l < 32; ++l) {
                    wlo[l] = (int8_t)(qs[j * 32 + l] & 0x0F);
                    whi[l] = (int8_t)(qs[j * 32 + l] >> 4);
                }
                for (i = 0; i < t; ++i) {
                    const nita_aq8_t *ai0 = &a[(size_t)i * stride + sb * 8 + 2 * j];
                    const nita_aq8_t *ai1 = &a[(size_t)i * stride + sb * 8 + 2 * j + 1];
                    int32_t           s0 = 0, s1 = 0;

                    for (l = 0; l < 32; ++l) s0 += (int32_t)ai0->qs[l] * wlo[l];
                    for (l = 0; l < 32; ++l) s1 += (int32_t)ai1->qs[l] * whi[l];
                    out[(size_t)i * out_stride] += d * (float)sc[2 * j] * ai0->d * (float)s0
                                                 + d * (float)sc[2 * j + 1] * ai1->d * (float)s1
                                                 - dmin * (float)mn[2 * j] * ai0->d * (float)ai0->sum
                                                 - dmin * (float)mn[2 * j + 1] * ai1->d * (float)ai1->sum;
                }
            }
        }
    }
}

/* IQ4_NL tile form: expand each weight block once, use it for every token in
 * the tile. Mirrors dot_tile_q8_0. */
static void dot_tile_iq4_nl(const uint8_t *row, const nita_aq8_t *a, int stride,
                            int t, int nb, float *out, int out_stride)
{
    int i, b, i0;

#if defined(__AVX2__)
    for (i0 = 0; i0 < t; i0 += 8) {
        const int tc = (t - i0) < 8 ? (t - i0) : 8;
        const __m256i tbl = _mm256_broadcastsi128_si256(
                                _mm_loadu_si128((const __m128i *)g_iq4nl));
        const __m128i m0f = _mm_set1_epi8(0x0F);
        __m256    vacc[8];

        for (i = 0; i < tc; ++i) vacc[i] = _mm256_setzero_ps();

        for (b = 0; b < nb; ++b) {
            const uint8_t *blk = row + (size_t)b * 18;
            float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const __m128i  q   = _mm_loadu_si128((const __m128i *)(blk + 2));
            const __m256i  nib = _mm256_set_m128i(
                                     _mm_and_si128(_mm_srli_epi16(q, 4), m0f),
                                     _mm_and_si128(q, m0f));
            const __m256i  w   = _mm256_shuffle_epi8(tbl, nib);

            for (i = 0; i < tc; ++i) {
                const nita_aq8_t *ai = &a[(size_t)(i0 + i) * stride + b];

                vacc[i] = _mm256_fmadd_ps(dot_i8_vec(ai->qs, w),
                                          _mm256_set1_ps(dw * ai->d), vacc[i]);
            }
        }
        for (i = 0; i < tc; ++i)
            out[(size_t)(i0 + i) * out_stride] = hsum256_ps(vacc[i]);
    }
    return;
#endif
    for (i = 0; i < t; ++i) out[(size_t)i * out_stride] = 0.0f;
    for (b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * 18;
        float          dw  = nita_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs  = blk + 2;

        for (i = 0; i < t; ++i) {
            const nita_aq8_t *ai = &a[(size_t)i * stride + b];
            int32_t           sum = 0;
            int               l;

            for (l = 0; l < 16; ++l) {
                sum += (int32_t)ai->qs[l]      * (int32_t)g_iq4nl[qs[l] & 0x0F];
                sum += (int32_t)ai->qs[16 + l] * (int32_t)g_iq4nl[qs[l] >> 4];
            }
            out[(size_t)i * out_stride] += dw * ai->d * (float)sum;
        }
    }
}

void nita_quant_dot_tile(uint32_t type, const uint8_t *row,
                         const nita_aq8_t *a, int stride, int t, int nb,
                         float *out, int out_stride)
{
    if (type == 8) { dot_tile_q8_0(row, a, stride, t, nb, out, out_stride); return; }
    if (type == 12) { dot_tile_q4_K(row, a, stride, t, nb, out, out_stride); return; }
    if (type == 20) { dot_tile_iq4_nl(row, a, stride, t, nb, out, out_stride); return; }

    /* Types without a fused tile form still stay in the integer domain, they
     * just expand the weights once per token instead of once per tile. */
    {
        int i;

        for (i = 0; i < t; ++i)
            out[(size_t)i * out_stride] = nita_quant_dot(type, row,
                                                        &a[(size_t)i * stride], nb);
    }
}

/* ------------------------------------------------------------------ */

int nita_quant_has_int_dot(uint32_t type)
{
    switch (type) {
    case 2:  /* Q4_0   */
    case 3:  /* Q4_1   */
    case 6:  /* Q5_0   */
    case 7:  /* Q5_1   */
    case 8:  /* Q8_0   */
    case 9:  /* Q8_1   */
    case 10: /* Q2_K   */
    case 11: /* Q3_K   */
    case 12: /* Q4_K   */
    case 13: /* Q5_K   */
    case 14: /* Q6_K   */
    case 15: /* Q8_K   */
    case 16: /* IQ2_XXS */
    case 17: /* IQ2_XS  */
    case 18: /* IQ3_XXS */
    case 19: /* IQ1_S   */
    case 20: /* IQ4_NL  */
    case 21: /* IQ3_S   */
    case 22: /* IQ2_S   */
    case 23: /* IQ4_XS  */
    case 29: /* IQ1_M   */
    case 34: /* TQ1_0  */
    case 35: /* TQ2_0  */
    case 39: /* MXFP4  */
    case 41: /* Q1_0   */
    case 42: /* Q2_0   */
    case 142:/* PQ2_0  */
    case 143:/* PTQ1_0 */
        return 1;
    default:
        return 0;
    }
}

float nita_quant_dot(uint32_t type, const uint8_t *row, const nita_aq8_t *a,
                     int nb)
{
    switch (type) {
    case 2:  return dot_q4_0(row, a, nb);
    case 3:  return dot_q4_1(row, a, nb);
    case 6:  return dot_q5_0(row, a, nb);
    case 7:  return dot_q5_1(row, a, nb);
    case 8:  return dot_q8_0(row, a, nb);
    case 9:  return dot_q8_1(row, a, nb);
    case 10: return dot_q2_K(row, a, nb);
    case 11: return dot_q3_K(row, a, nb);
    case 15: return dot_q8_K(row, a, nb);
    case 12: return dot_q4_K(row, a, nb);
    case 13: return dot_q5_K(row, a, nb);
    case 14: return dot_q6_K(row, a, nb);
    case 16: return dot_iq2_xxs(row, a, nb);
    case 17: return dot_iq2_xs(row, a, nb);
    case 18: return dot_iq3_xxs(row, a, nb);
    case 19: return dot_iq1_s(row, a, nb);
    case 20: return dot_iq4_nl(row, a, nb);
    case 21: return dot_iq3_s(row, a, nb);
    case 22: return dot_iq2_s(row, a, nb);
    case 23: return dot_iq4_xs(row, a, nb);
    case 29: return dot_iq1_m(row, a, nb);
    case 34: return dot_tq1_0(row, a, nb);
    case 35: return dot_tq2_0(row, a, nb);
    case 39: return dot_mxfp4(row, a, nb);
    case 41: return dot_q1_0(row, a, nb);
    case 42: return dot_q2_0(row, a, nb);
    case 142:return dot_pq2_0(row, a, nb);
    case 143:return dot_ptq1_0(row, a, nb);
    default: return 0.0f;
    }
}
