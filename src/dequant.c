/*
 * dequant.c – GGUF block dequantisation.
 *
 * The engine never materialises a whole dequantised weight matrix. Rows are
 * dequantised on demand into a small reusable buffer, which honours the
 * project rule of only expanding the tile that is actually being consumed.
 *
 * Block layouts follow the ggml definitions. Little-endian throughout.
 */

#include "nita.h"
#include "iq_tables.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Half precision                                                      */
/* ------------------------------------------------------------------ */
float nita_f16_to_f32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    {
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
}

static uint16_t rd_u16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static float rd_f16(const uint8_t *p)
{
    return nita_f16_to_f32(rd_u16(p));
}

static float rd_f32(const uint8_t *p)
{
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
                  | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ------------------------------------------------------------------ */
/* Byte size of one row                                                */
/* ------------------------------------------------------------------ */
size_t nita_row_bytes(uint32_t type, uint64_t ne0)
{
    uint64_t bs = ggml_blck_size(type);
    uint64_t sz = ggml_type_size(type);
    uint64_t blocks;

    if (bs == 0) bs = 1;
    blocks = (ne0 + bs - 1) / bs;
    return (size_t)(blocks * sz);
}

/* ------------------------------------------------------------------ */
/* K-quant scale unpacking (shared by Q4_K / Q5_K)                     */
/* ------------------------------------------------------------------ */
/* Upstream get_scale_min_k4, byte for byte: the ggml AVX2 unpack used to be
 * transcribed here and decoded the wrong lanes, which silently produced
 * garbage for every Q4_K / Q5_K tensor. Keep it scalar and obviously equal
 * to get_scale_min_k4. */
static void unpack_q4k_scales(const uint8_t *scales_in, int8_t scales[8], int8_t mins[8])
{
    int j;

    for (j = 0; j < 4; ++j) {
        scales[j] = (int8_t)(scales_in[j] & 63);
        mins[j]   = (int8_t)(scales_in[j + 4] & 63);
    }
    for (j = 4; j < 8; ++j) {
        scales[j] = (int8_t)((scales_in[j + 4] & 0x0F) | ((scales_in[j - 4] >> 6) << 4));
        mins[j]   = (int8_t)((scales_in[j + 4] >> 4) | ((scales_in[j] >> 6) << 4));
    }
}

/* ------------------------------------------------------------------ */
/* Row dequantisation                                                  */
/* ------------------------------------------------------------------ */
int nita_dequant_row(const uint8_t *src, uint32_t type, uint64_t ne0, float *dst)
{
    uint64_t i;

    switch (type) {
    /* ---------------- unquantised ---------------- */
    case 0: /* F32 */
        for (i = 0; i < ne0; ++i) dst[i] = rd_f32(src + i * 4);
        return 0;

    case 1: /* F16 */
        for (i = 0; i < ne0; ++i) dst[i] = rd_f16(src + i * 2);
        return 0;

    case 30: /* BF16: high 16 bits of an f32 */
        for (i = 0; i < ne0; ++i) {
            uint32_t bits = ((uint32_t)rd_u16(src + i * 2)) << 16;
            float f;
            memcpy(&f, &bits, 4);
            dst[i] = f;
        }
        return 0;

    case 28: /* F64 */
        for (i = 0; i < ne0; ++i) {
            uint64_t bits = 0;
            double d;
            int k;
            for (k = 7; k >= 0; --k) bits = (bits << 8) | src[i * 8 + (uint64_t)k];
            memcpy(&d, &bits, 8);
            dst[i] = (float)d;
        }
        return 0;

    case 24: /* I8 */
        for (i = 0; i < ne0; ++i) dst[i] = (float)(int8_t)src[i];
        return 0;

    case 25: /* I16 */
        for (i = 0; i < ne0; ++i) dst[i] = (float)(int16_t)rd_u16(src + i * 2);
        return 0;

    case 26: /* I32 */
        for (i = 0; i < ne0; ++i) {
            uint32_t v = (uint32_t)src[i * 4] | ((uint32_t)src[i * 4 + 1] << 8)
                       | ((uint32_t)src[i * 4 + 2] << 16) | ((uint32_t)src[i * 4 + 3] << 24);
            dst[i] = (float)(int32_t)v;
        }
        return 0;

    case 27: { /* I64: low 64 bits, cast through double; values above 2^53
                  lose integer precision in fp32, which is inherent */
        for (i = 0; i < ne0; ++i) {
            uint64_t v = 0;
            int      b;

            for (b = 7; b >= 0; --b) v = (v << 8) | src[i * 8 + b];
            dst[i] = (float)(int64_t)v;
        }
        return 0;
    }

    /* ---------------- legacy quants ---------------- */
    case 2: { /* Q4_0: 32 elements, 18 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float d = rd_f16(blk);
            const uint8_t *qs = blk + 2;
            int j;
            for (j = 0; j < 16; ++j) {
                int lo = (qs[j] & 0x0F) - 8;
                int hi = (qs[j] >> 4) - 8;
                dst[i * 32 + j]      = d * (float)lo;
                dst[i * 32 + j + 16] = d * (float)hi;
            }
        }
        return 0;
    }

    case 3: { /* Q4_1: 32 elements, 20 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 20;
            float d = rd_f16(blk);
            float m = rd_f16(blk + 2);
            const uint8_t *qs = blk + 4;
            int j;
            for (j = 0; j < 16; ++j) {
                int lo = (qs[j] & 0x0F);
                int hi = (qs[j] >> 4);
                dst[i * 32 + j]      = d * (float)lo + m;
                dst[i * 32 + j + 16] = d * (float)hi + m;
            }
        }
        return 0;
    }

    case 6: { /* Q5_0: 32 elements, 22 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 22;
            float d = rd_f16(blk);
            uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                        | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
            const uint8_t *qs = blk + 6;
            int j;
            for (j = 0; j < 16; ++j) {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int lo = (qs[j] & 0x0F) | xh0;
                int hi = (qs[j] >> 4) | xh1;
                dst[i * 32 + j]      = d * (float)(lo - 16);
                dst[i * 32 + j + 16] = d * (float)(hi - 16);
            }
        }
        return 0;
    }

    case 7: { /* Q5_1: 32 elements, 24 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 24;
            float d = rd_f16(blk);
            float m = rd_f16(blk + 2);
            uint32_t qh = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8)
                        | ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
            const uint8_t *qs = blk + 8;
            int j;
            for (j = 0; j < 16; ++j) {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int lo = (qs[j] & 0x0F) | xh0;
                int hi = (qs[j] >> 4) | xh1;
                dst[i * 32 + j]      = d * (float)lo + m;
                dst[i * 32 + j + 16] = d * (float)hi + m;
            }
        }
        return 0;
    }

    case 8: { /* Q8_0: 32 elements, 34 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 34;
            float d = rd_f16(blk);
            const int8_t *qs = (const int8_t *)(blk + 2);
            int j;
            for (j = 0; j < 32; ++j)
                dst[i * 32 + j] = d * (float)qs[j];
        }
        return 0;
    }

    case 9: { /* Q8_1: 32 elements, 36 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 36;
            float d = rd_f16(blk);
            const int8_t *qs = (const int8_t *)(blk + 4);
            int j;
            for (j = 0; j < 32; ++j)
                dst[i * 32 + j] = d * (float)qs[j];
        }
        return 0;
    }

    /* ---------------- k-quants (super-block = 256) ---------------- */
    case 10: { /* Q2_K: 256 elements, 84 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 84;
            const uint8_t *scales = blk;
            const uint8_t *qs = blk + 16;
            float d    = rd_f16(blk + 80);
            float dmin = rd_f16(blk + 82);
            int is = 0;
            int n;
            float *y = dst + i * 256;

            for (n = 0; n < 256; n += 128) {
                int shift = 0;
                int j;
                for (j = 0; j < 4; ++j) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0x0F);
                    float ml = dmin * (float)(sc >> 4);
                    int l;
                    for (l = 0; l < 16; ++l)
                        y[n + j * 32 + l] = dl * (float)((qs[l] >> shift) & 3) - ml;

                    sc = scales[is++];
                    dl = d * (float)(sc & 0x0F);
                    ml = dmin * (float)(sc >> 4);
                    for (l = 0; l < 16; ++l)
                        y[n + j * 32 + 16 + l] = dl * (float)((qs[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                qs += 32;
            }
        }
        return 0;
    }

    case 11: { /* Q3_K: 256 elements, 110 bytes */
        uint64_t nb = ne0 / 256;
        static const uint32_t kmask1 = 0x03030303u;
        static const uint32_t kmask2 = 0x0f0f0f0fu;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 110;
            const uint8_t *hmask = blk;
            const uint8_t *qs = blk + 32;
            const uint8_t *scale_bytes = blk + 96;
            float d_all = rd_f16(blk + 108);
            uint32_t aux[4];
            int8_t scales[16];
            int is = 0;
            int n;
            float *y = dst + i * 256;
            uint8_t m = 1;

            for (is = 0; is < 3; ++is) {
                aux[is] = (uint32_t)scale_bytes[is * 4 + 0]
                        | ((uint32_t)scale_bytes[is * 4 + 1] << 8)
                        | ((uint32_t)scale_bytes[is * 4 + 2] << 16)
                        | ((uint32_t)scale_bytes[is * 4 + 3] << 24);
            }
            {
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = ((aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4));
                aux[1] = ((aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4));
            }
            for (is = 0; is < 16; ++is) {
                int v = (int)((aux[is / 4] >> (8 * (is % 4))) & 0xFF);
                scales[is] = (int8_t)v;
            }

            is = 0;
            for (n = 0; n < 256; n += 128) {
                int shift = 0;
                int j;
                for (j = 0; j < 4; ++j) {
                    float dl = d_all * (float)(scales[is++] - 32);
                    int l;
                    for (l = 0; l < 16; ++l) {
                        int q = ((qs[l] >> shift) & 3)
                              - ((hmask[l] & m) ? 0 : 4);
                        y[n + j * 32 + l] = dl * (float)q;
                    }
                    dl = d_all * (float)(scales[is++] - 32);
                    for (l = 0; l < 16; ++l) {
                        int q = ((qs[l + 16] >> shift) & 3)
                              - ((hmask[l + 16] & m) ? 0 : 4);
                        y[n + j * 32 + 16 + l] = dl * (float)q;
                    }
                    shift += 2;
                    m <<= 1;
                }
                qs += 32;
            }
            (void)hmask;
        }
        return 0;
    }

    case 12: { /* Q4_K: 256 elements, 144 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 144;
            float d    = rd_f16(blk);
            float dmin = rd_f16(blk + 2);
            int8_t scales[8], mins[8];
            const uint8_t *qs = blk + 16;
            float *y = dst + i * 256;
            int j;

            unpack_q4k_scales(blk + 4, scales, mins);

            for (j = 0; j < 4; ++j) {
                float d1 = d * (float)scales[2 * j];
                float m1 = dmin * (float)mins[2 * j];
                float d2 = d * (float)scales[2 * j + 1];
                float m2 = dmin * (float)mins[2 * j + 1];
                int l;
                for (l = 0; l < 32; ++l) y[j * 64 + l] = d1 * (float)(qs[l] & 0x0F) - m1;
                for (l = 0; l < 32; ++l) y[j * 64 + 32 + l] = d2 * (float)(qs[l] >> 4) - m2;
                qs += 32;
            }
        }
        return 0;
    }

    case 13: { /* Q5_K: 256 elements, 176 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 176;
            float d    = rd_f16(blk);
            float dmin = rd_f16(blk + 2);
            int8_t scales[8], mins[8];
            const uint8_t *qh = blk + 16;
            const uint8_t *ql = blk + 48;
            float *y = dst + i * 256;
            uint8_t u1 = 1, u2 = 2;
            int j;

            unpack_q4k_scales(blk + 4, scales, mins);

            for (j = 0; j < 4; ++j) {
                float d1 = d * (float)scales[2 * j + 0];
                float m1 = dmin * (float)mins[2 * j + 0];
                float d2 = d * (float)scales[2 * j + 1];
                float m2 = dmin * (float)mins[2 * j + 1];
                int l;
                for (l = 0; l < 32; ++l) {
                    int hi = (qh[l] & u1) ? 16 : 0;
                    y[j * 64 + l] = d1 * (float)((ql[l] & 0x0F) + hi) - m1;
                }
                for (l = 0; l < 32; ++l) {
                    int hi = (qh[l] & u2) ? 16 : 0;
                    y[j * 64 + 32 + l] = d2 * (float)((ql[l] >> 4) + hi) - m2;
                }
                ql += 32;
                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
        }
        return 0;
    }

    case 14: { /* Q6_K: 256 elements, 210 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 210;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const int8_t  *sc = (const int8_t *)(blk + 192);
            float d = rd_f16(blk + 208);
            float *y = dst + i * 256;
            int n;

            for (n = 0; n < 256; n += 128) {
                int l;
                for (l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    y[n + l + 0]  = d * (float)sc[is + 0] * (float)q1;
                    y[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
                    y[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
                    y[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                /* y must not advance here: n already carries the 128-element
                 * half, and doing both walked 256 floats past the row. */
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return 0;
    }

    case 20: { /* IQ4_NL: 32 elements, 18 bytes: fp16 d + 16 code bytes.
                 * The nibble is an index into the ggml non-linear codebook,
                 * not a linear offset. Low nibbles hold elements 0-15, high
                 * nibbles elements 16-31 (same layout as Q4_0). */
        static const int8_t kv[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
               1,   13,  25,  38,  53,  69,  89, 113
        };
        uint64_t nb = ne0 / 32;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float          d   = rd_f16(blk);
            const uint8_t *qs  = blk + 2;
            int            l;

            for (l = 0; l < 16; ++l) {
                dst[i * 32 + l]      = d * (float)kv[qs[l] & 0x0F];
                dst[i * 32 + 16 + l] = d * (float)kv[qs[l] >> 4];
            }
        }
        return 0;
    }

    case 23: { /* IQ4_XS: 256 elements, 136 bytes */
        /* Non-linear 4-bit codebook; the values are the ggml table verbatim.
         * Indexing it with the nibble is the whole of IQ4_XS decoding: unlike
         * Q4_K the nibble is not a linear offset. */
        static const int8_t kv[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
               1,   13,  25,  38,  53,  69,  89, 113
        };
        uint64_t nb = ne0 / 256;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 136;
            float          d   = rd_f16(blk);
            uint16_t       h   = (uint16_t)(blk[2] | (blk[3] << 8));
            const uint8_t *sl  = blk + 4;
            const uint8_t *qs  = blk + 8;
            float         *y   = dst + i * 256;
            int            ib;

            for (ib = 0; ib < 8; ++ib) {
                int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                       | (((h >> (2 * ib)) & 3) << 4);
                float dl = d * (float)(ls - 32);
                int   l;

                for (l = 0; l < 16; ++l) {
                    y[ib * 32 + l]      = dl * (float)kv[qs[ib * 16 + l] & 0x0F];
                    y[ib * 32 + 16 + l] = dl * (float)kv[qs[ib * 16 + l] >> 4];
                }
            }
        }
        return 0;
    }

    case 39: { /* MXFP4: 32 elements, 17 bytes: E8M0 exponent + 16 code bytes */
        /* e2m1 values, doubled, so the scale carries the missing half. */
        static const int8_t kv[16] = {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
        };
        uint64_t nb = ne0 / 32;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 17;
            float          d   = exp2f((float)blk[0] - 128.0f);
            int            l;

            for (l = 0; l < 16; ++l) {
                dst[i * 32 + l]      = d * (float)kv[blk[1 + l] & 0x0F];
                dst[i * 32 + 16 + l] = d * (float)kv[blk[1 + l] >> 4];
            }
        }
        return 0;
    }

    case 16: { /* IQ2_XXS: 256 elements, 66 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t  *blk = src + i * 66;
            float           d   = rd_f16(blk);
            const uint16_t *qs  = (const uint16_t *)(blk + 2);
            float          *y   = dst + i * 256;
            int             ib32;

            for (ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t aux[2];
                const uint8_t *aux8 = (const uint8_t *)aux;
                float db;
                int l, j;

                memcpy(aux, qs + 4 * ib32, sizeof aux);
                db = d * (0.5f + (float)(aux[1] >> 28)) * 0.25f;
                for (l = 0; l < 4; ++l) {
                    const uint8_t *grid = (const uint8_t *)(nita_iq2xxs_grid + aux8[l]);
                    const uint8_t  sb   = nita_ksigns_iq2xs[(aux[1] >> 7 * l) & 127];

                    for (j = 0; j < 8; ++j)
                        y[ib32 * 32 + l * 8 + j] =
                            db * (float)(int8_t)grid[j] * ((sb & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        }
        return 0;
    }

    case 17: { /* IQ2_XS: 256 elements, 74 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t  *blk = src + i * 74;
            float           d   = rd_f16(blk);
            const uint16_t *qs  = (const uint16_t *)(blk + 2);
            const uint8_t  *sl  = blk + 66;
            float          *y   = dst + i * 256;
            int             ib32;

            for (ib32 = 0; ib32 < 8; ++ib32) {
                const float db[2] = { d * (0.5f + (float)(sl[ib32] & 0x0F)) * 0.25f,
                                      d * (0.5f + (float)(sl[ib32] >> 4)) * 0.25f };
                int         l, j;

                for (l = 0; l < 4; ++l) {
                    const uint16_t q    = qs[4 * ib32 + l];
                    const uint8_t *grid = (const uint8_t *)(nita_iq2xs_grid + (q & 511));
                    const uint8_t  sb   = nita_ksigns_iq2xs[q >> 9];
                    const float    dl   = db[l / 2];

                    for (j = 0; j < 8; ++j)
                        y[ib32 * 32 + l * 8 + j] =
                            dl * (float)(int8_t)grid[j] * ((sb & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        }
        return 0;
    }

    case 22: { /* IQ2_S: 256 elements, 82 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk   = src + i * 82;
            float          d     = rd_f16(blk);
            const uint8_t *qs    = blk + 2;
            const uint8_t *qh    = blk + 66;
            const uint8_t *signs = qs + 32;
            const uint8_t *sl    = blk + 74;
            float         *y     = dst + i * 256;
            int            ib32;

            for (ib32 = 0; ib32 < 8; ++ib32) {
                const float db[2] = { d * (0.5f + (float)(sl[ib32] & 0x0F)) * 0.25f,
                                      d * (0.5f + (float)(sl[ib32] >> 4)) * 0.25f };
                int         l, j;

                for (l = 0; l < 4; ++l) {
                    const uint8_t *grid = (const uint8_t *)(
                        nita_iq2s_grid + (qs[4 * ib32 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                    const uint8_t  sb = signs[4 * ib32 + l];
                    const float    dl = db[l / 2];

                    for (j = 0; j < 8; ++j)
                        y[ib32 * 32 + l * 8 + j] =
                            dl * (float)(int8_t)grid[j] * ((sb & (1u << j)) ? -1.0f : 1.0f);
                }
            }
        }
        return 0;
    }

    case 18: { /* IQ3_XXS: 256 elements, 98 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 98;
            float          d   = rd_f16(blk);
            const uint8_t *qs  = blk + 2;
            const uint8_t *scales_and_signs = qs + 64;
            float         *y   = dst + i * 256;
            int            ib32;

            for (ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t        aux32;
                const uint8_t  *q = qs + 8 * ib32;
                float           db;
                int             l, j;

                memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof aux32);
                db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
                for (l = 0; l < 4; ++l) {
                    const uint8_t *grid1 = (const uint8_t *)(nita_iq3xxs_grid + q[2 * l + 0]);
                    const uint8_t *grid2 = (const uint8_t *)(nita_iq3xxs_grid + q[2 * l + 1]);
                    const uint8_t  sb    = nita_ksigns_iq2xs[(aux32 >> 7 * l) & 127];

                    for (j = 0; j < 4; ++j) {
                        y[ib32 * 32 + l * 8 + j]     = db * (float)(int8_t)grid1[j]
                                                     * ((sb & (1u << (j + 0))) ? -1.0f : 1.0f);
                        y[ib32 * 32 + l * 8 + 4 + j] = db * (float)(int8_t)grid2[j]
                                                     * ((sb & (1u << (j + 4))) ? -1.0f : 1.0f);
                    }
                }
            }
        }
        return 0;
    }

    case 21: { /* IQ3_S: 256 elements, 110 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk   = src + i * 110;
            float          d     = rd_f16(blk);
            const uint8_t *qs    = blk + 2;
            const uint8_t *qh    = blk + 66;
            const uint8_t *signs = blk + 74;
            const uint8_t *sc    = blk + 106;
            float         *y     = dst + i * 256;
            int            ib32;

            for (ib32 = 0; ib32 < 8; ib32 += 2) {
                const float db1 = d * (1.0f + 2.0f * (float)(sc[ib32 / 2] & 0x0F));
                const float db2 = d * (1.0f + 2.0f * (float)(sc[ib32 / 2] >> 4));
                const uint8_t *q1 = qs + 8 * ib32, *sn1 = signs + 4 * ib32;
                const uint8_t *q2 = q1 + 8, *sn2 = sn1 + 4;
                int            l, j;

                for (l = 0; l < 4; ++l) {
                    const uint8_t *grid1 = (const uint8_t *)(
                        nita_iq3s_grid + (q1[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 0x100)));
                    const uint8_t *grid2 = (const uint8_t *)(
                        nita_iq3s_grid + (q1[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 0x100)));
                    const uint8_t  sb = sn1[l];

                    for (j = 0; j < 4; ++j) {
                        y[ib32 * 32 + l * 8 + j]     = db1 * (float)(int8_t)grid1[j]
                                                     * ((sb & (1u << (j + 0))) ? -1.0f : 1.0f);
                        y[ib32 * 32 + l * 8 + 4 + j] = db1 * (float)(int8_t)grid2[j]
                                                     * ((sb & (1u << (j + 4))) ? -1.0f : 1.0f);
                    }
                }
                for (l = 0; l < 4; ++l) {
                    const uint8_t *grid1 = (const uint8_t *)(
                        nita_iq3s_grid + (q2[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 0x100)));
                    const uint8_t *grid2 = (const uint8_t *)(
                        nita_iq3s_grid + (q2[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 0x100)));
                    const uint8_t  sb = sn2[l];

                    for (j = 0; j < 4; ++j) {
                        y[(ib32 + 1) * 32 + l * 8 + j]     = db2 * (float)(int8_t)grid1[j]
                                                           * ((sb & (1u << (j + 0))) ? -1.0f : 1.0f);
                        y[(ib32 + 1) * 32 + l * 8 + 4 + j] = db2 * (float)(int8_t)grid2[j]
                                                           * ((sb & (1u << (j + 4))) ? -1.0f : 1.0f);
                    }
                }
                qh += 2;
            }
        }
        return 0;
    }

    case 19: { /* IQ1_S: 256 elements, 50 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 50;
            float          d   = rd_f16(blk);
            const uint8_t *qs  = blk + 2;
            const uint16_t *qh = (const uint16_t *)(blk + 34);
            float         *y   = dst + i * 256;
            int            ib;

            for (ib = 0; ib < 8; ++ib) {
                const uint16_t qhv   = qh[ib];
                const float    dl    = d * (2.0f * (float)((qhv >> 12) & 7) + 1.0f);
                const float    delta = (qhv & 0x8000u) ? -0.125f : 0.125f;
                int            l, j;

                for (l = 0; l < 4; ++l) {
                    const int8_t *grid = (const int8_t *)(
                        nita_iq1s_grid + (qs[4 * ib + l] | (((qhv >> (3 * l)) & 7) << 8)));

                    for (j = 0; j < 8; ++j)
                        y[ib * 32 + l * 8 + j] = dl * ((float)grid[j] + delta);
                }
            }
        }
        return 0;
    }

    case 29: { /* IQ1_M: 256 elements, 56 bytes; d synthesised from scales */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 56;
            const uint8_t *qs  = blk + 0;
            const uint8_t *qh  = blk + 32;
            const uint16_t *sc = (const uint16_t *)(blk + 48);
            uint16_t       su  = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                                          | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
            float          d   = rd_f16((const uint8_t *)&su);
            float         *y   = dst + i * 256;
            int            ib;

            for (ib = 0; ib < 8; ++ib) {
                const uint16_t s   = sc[ib / 2];
                const float    dl1 = d * (2.0f * (float)((s >> (6 * (ib % 2) + 0)) & 7) + 1.0f);
                const float    dl2 = d * (2.0f * (float)((s >> (6 * (ib % 2) + 3)) & 7) + 1.0f);
                const uint8_t  q_0 = qs[4 * ib + 0], q_1 = qs[4 * ib + 1];
                const uint8_t  q_2 = qs[4 * ib + 2], q_3 = qs[4 * ib + 3];
                const uint8_t  h_0 = qh[2 * ib + 0], h_1 = qh[2 * ib + 1];
                const int      idx[4] = { q_0 | ((h_0 << 8) & 0x700), q_1 | ((h_0 << 4) & 0x700),
                                          q_2 | ((h_1 << 8) & 0x700), q_3 | ((h_1 << 4) & 0x700) };
                const float    del[4] = { (h_0 & 0x08) ? -0.125f : 0.125f,
                                          (h_0 & 0x80) ? -0.125f : 0.125f,
                                          (h_1 & 0x08) ? -0.125f : 0.125f,
                                          (h_1 & 0x80) ? -0.125f : 0.125f };
                const float    dl[2]  = { dl1, dl2 };
                int            l, j;

                for (l = 0; l < 4; ++l) {
                    const int8_t *grid = (const int8_t *)(nita_iq1s_grid + idx[l]);
                    const float   dcur = dl[l / 2];

                    for (j = 0; j < 8; ++j)
                        y[ib * 32 + l * 8 + j] = dcur * ((float)grid[j] + del[l]);
                }
            }
        }
        return 0;
    }

    case 15: { /* Q8_K: 256 elements, 292 bytes */

        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 292;
            float d = rd_f32(blk);
            const int8_t *qs = (const int8_t *)(blk + 4);
            int j;
            for (j = 0; j < 256; ++j)
                dst[i * 256 + j] = d * (float)qs[j];
        }
        return 0;
    }

    /* ---- binary / ternary formats (PrismML Bonsai family) ---------- */

    case 41: { /* Q1_0: 128 elements, 18 bytes, one sign bit per weight */
        uint64_t nb = ne0 / 128;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float d = rd_f16(blk);
            const uint8_t *qs = blk + 2;
            int j;
            for (j = 0; j < 128; ++j) {
                int bit = (qs[j / 8] >> (j % 8)) & 1;
                dst[i * 128 + j] = bit ? d : -d;
            }
        }
        return 0;
    }

    case 42: { /* Q2_0: 64 elements, 18 bytes, ternary in 2-bit slots */
        uint64_t nb = ne0 / 64;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float d = rd_f16(blk);
            int j;
            for (j = 0; j < 64; ++j) {
                int q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                dst[i * 64 + j] = (float)(q - 1) * d;
            }
        }
        return 0;
    }

    case 142: { /* PQ2_0: 128 elements, 34 bytes, ternary group 128 */
        uint64_t nb = ne0 / 128;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 34;
            float d = rd_f16(blk);
            int j;
            for (j = 0; j < 128; ++j) {
                int q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                dst[i * 128 + j] = (float)(q - 1) * d;
            }
        }
        return 0;
    }

    case 143: { /* PTQ1_0: 128 elements, 28 bytes, dense base-3 trits */
        static const size_t  stages[3] = { 32, 16, 8 };
        static const uint8_t pow3[6]    = { 1, 3, 9, 27, 81, 243 };
        uint64_t nb = ne0 / 128;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 28;
            const uint8_t *qs  = blk;        /* 24 bytes: 5 trits each */
            const uint8_t *qh  = blk + 24;   /*  2 bytes: 4 trits each */
            float d = rd_f16(blk + 26);
            float *y = dst + i * 128;
            size_t j = 0, s, m, n;

            for (s = 0; s < 3; ++s) {
                size_t c = stages[s];
                for (; j + c <= 24; j += c) {
                    for (n = 0; n < 5; ++n) {
                        for (m = 0; m < c; ++m) {
                            /* truncating the product to 8 bits then ceiling-
                             * dividing by 243 isolates the n-th trit */
                            uint8_t q  = (uint8_t)(qs[j + m] * pow3[n]);
                            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                            *y++ = (float)(xi - 1) * d;
                        }
                    }
                }
            }
            for (n = 0; n < 4; ++n) {
                for (m = 0; m < 2; ++m) {
                    uint8_t q  = (uint8_t)(qh[m] * pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        return 0;
    }

    case 34: { /* TQ1_0: 256 elements, 54 bytes, dense base-3 trits, group 256 */
        static const size_t  stages[3] = { 32, 16, 8 };
        static const uint8_t pow3[6]    = { 1, 3, 9, 27, 81, 243 };
        uint64_t nb = ne0 / 256;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 54;
            const uint8_t *qs  = blk;        /* 48 bytes: 5 trits each */
            const uint8_t *qh  = blk + 48;   /*  4 bytes: 4 trits each */
            float d = rd_f16(blk + 52);
            float *y = dst + i * 256;
            size_t j = 0, s, m, n;

            /* the same staged walk PTQ1_0 uses: 32 bytes (160 values) then
             * 16 bytes (80 values), which is exactly upstream's hard-coded
             * 32-then-16 layout for a 48-byte qs */
            for (s = 0; s < 3; ++s) {
                size_t c = stages[s];
                for (; j + c <= 48; j += c) {
                    for (n = 0; n < 5; ++n) {
                        for (m = 0; m < c; ++m) {
                            uint8_t q  = (uint8_t)(qs[j + m] * pow3[n]);
                            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                            *y++ = (float)(xi - 1) * d;
                        }
                    }
                }
            }
            for (n = 0; n < 4; ++n) {
                for (m = 0; m < 4; ++m) {
                    uint8_t q  = (uint8_t)(qh[m] * pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        return 0;
    }

    case 35: { /* TQ2_0: 256 elements, 66 bytes, 2-bit slots, group 256 */
        uint64_t nb = ne0 / 256;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 66;
            const uint8_t *qs  = blk;        /* 64 bytes, 4 slots each */
            float d = rd_f16(blk + 64);      /* scale last, unlike Q2_0 */
            float *y = dst + i * 256;
            size_t g, l, m;

            /* slot l is a bit plane across 32 stored bytes, so it covers 32
             * consecutive elements: upstream calls this "along 32 bytes" */
            for (g = 0; g < 2; ++g)
                for (l = 0; l < 4; ++l)
                    for (m = 0; m < 32; ++m)
                        *y++ = (float)(((qs[g * 32 + m] >> (2 * l)) & 3) - 1) * d;
        }
        return 0;
    }

    default:
        return -1;
    }
}
