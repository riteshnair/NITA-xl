/*
 * quant.h - quantised-domain dot products.
 *
 * The rule this file exists to enforce: weights are never expanded to fp32.
 * Every kernel here consumes the quantised block format directly, does the
 * multiply-accumulate in integer registers, and pays one float multiply-add
 * per 32 (or per super-block sub-block) to apply the block scale to the int32
 * accumulator.
 *
 * Why that matters: converting eight int8 weights to eight floats costs
 * cvt(2 uops) + cvtdq2ps + mul + fmadd = five vector ops per eight weights,
 * all of them on the critical path. The integer form is one maddubs plus one
 * madd for the same eight weights, with the scale hoisted out of the element
 * loop entirely, so the arithmetic is no longer instruction bound.
 *
 * Activations are quantised to int8 once per row (Q8_0 style, one scale per
 * 32 elements) and reused by every weight row, so the quantisation cost is
 * paid once per matvec rather than once per row.
 */
#ifndef NITA_QUANT_H
#define NITA_QUANT_H

#include <stdint.h>

/* True when the engine has a decoder for this GGUF tensor type: refusing an
 * unknown quant at load time is the contract, never partial output. */
int nita_type_supported(uint32_t type);

/* One quantised 32-element activation block. `sum` is kept because the
 * K-quants carry a per-sub-block minimum, and the min term needs the sum of
 * the activations over the same block. */
typedef struct {
    int8_t  qs[32];
    float   d;
    int32_t sum;        /* sum of qs over the whole 32-element block */
    int32_t sums[4];    /* sum of qs over each 8-element group (IQ1_M) */
} nita_aq8_t;

/* Quantise `k` floats (k % 32 == 0) into k/32 blocks. */
void nita_aq8_quantise(const float *x, int k, nita_aq8_t *out);

/* Return non-zero when this type has an integer kernel. Types without one
 * must keep using the dequantise-then-float path. */
int nita_quant_has_int_dot(uint32_t type);

/* y = W . x for one weight row of `type`, already quantised activations.
 *
 * `nb` is the number of 32-element blocks in a row (ne0 / 32). The result is
 * accumulated as int32 per block and scaled once per block, so the block
 * scale never multiplies an individual weight.
 */
float nita_quant_dot(uint32_t type, const uint8_t *row, const nita_aq8_t *a,
                     int nb);

/* Same, for `t` activation rows at once (prefill): out[i] = W . x_i.
 *
 * The point of this entry point is that the weights are expanded to int8 once
 * per block and then dotted against every token while still in registers, so
 * the tile keeps its single pass over the weight bytes. `stride` is the number
 * of blocks per activation row. */
void nita_quant_dot_tile(uint32_t type, const uint8_t *row,
                         const nita_aq8_t *a, int stride, int t, int nb,
                         float *out, int out_stride);

#endif /* NITA_QUANT_H */
