/*
 * qwen35.c – gated delta-net ("linear attention") block for the qwen35
 * hybrid architecture used by the PrismML Bonsai checkpoints.
 *
 * 64 blocks, with full attention every `full_attention_interval` blocks and
 * this gated-delta-net mixer in all the others.  Tensor layout, verified
 * against Bonsai-27B-Q1_0.gguf:
 *
 *   blk.N.attn_norm.weight             [n_embd]                     F32
 *   blk.N.attn_qkv.weight              [n_embd, conv_dim]           quantised
 *   blk.N.attn_gate.weight             [n_embd, inner]              quantised
 *   blk.N.ssm_conv1d.weight            [conv_kernel, conv_dim]      F32
 *   blk.N.ssm_alpha.weight             [n_embd, dt_rank]            quantised
 *   blk.N.ssm_beta.weight              [n_embd, dt_rank]            quantised
 *   blk.N.ssm_dt.bias                  [dt_rank]                    F32
 *   blk.N.ssm_a                        [dt_rank]                    F32
 *   blk.N.ssm_norm.weight              [head_v_dim]                 F32
 *   blk.N.ssm_out.weight               [inner, n_embd]              quantised
 *   blk.N.post_attention_norm.weight   [n_embd]                     F32
 *
 * The mixer runs the delta rule with a per-value-head state matrix and adds
 * its output to the residual stream.  The feed-forward half of the block is
 * left to the caller (transformer.c) so both block kinds share it.
 *
 * Weights stay packed; each row is dequantised immediately before use.
 */

#include "nita.h"
#include "quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Small kernels                                                       */
/* ------------------------------------------------------------------ */

/* y = W x with W stored row-major as ne1 rows of ne0 elements. */
/* Whole activation quantised once per matvec; 256 blocks covers ne0 up to 8192,
 * which is every Bonsai width. Wider than that falls back to the float path. */
#define Q35_MAX_AQ_BLOCKS 256

typedef struct {
    const nita_tensor_t *t;
    const float         *x;
    float               *y;
    size_t               row_bytes;
    float               *row;      /* only the serial path uses this */
    const nita_aq8_t    *aq;       /* integer domain, NULL for the float path */
    int                  nb;
    int                  rc;
} mvrow_job_t;

static void mvrow_rows(void *vctx, int lo, int hi, int thread)
{
    mvrow_job_t *j    = (mvrow_job_t *)vctx;
    float       *row  = nita_thread_rowbuf(thread);
    int          r;

    /* Integer domain: the weights are never expanded to float. This is the same
     * kernel transformer.c uses; the mixer was the one place still doing it the
     * slow way, and it was 83% of a 27B token. */
    if (j->aq) {
        for (r = lo; r < hi; ++r)
            j->y[r] = nita_quant_dot(j->t->type,
                                     j->t->data + (size_t)r * j->row_bytes,
                                     j->aq, j->nb);
        return;
    }

    if (!row) row = j->row;
    if (!row) { j->rc = -3; return; }

    for (r = lo; r < hi; ++r) {
        const uint8_t *src = j->t->data + (size_t)r * j->row_bytes;
        uint64_t       i;
        float          sum = 0.0f;

        if (nita_dequant_row(src, j->t->type, j->t->ne0, row) != 0) {
            j->rc = -3;
            return;
        }
        for (i = 0; i < j->t->ne0; ++i) sum += row[i] * j->x[i];
        j->y[r] = sum;
    }
}

/* Rows are independent, so splitting them across the pool gives the identical
 * per-row arithmetic - the same sum in the same order - on every thread. */
static int mv_row(const nita_tensor_t *t, const float *x, float *y,
                  int expect_in, int expect_out, float *row, nita_aq8_t *aqbuf)
{
    mvrow_job_t j;

    if (!t->bound) return -1;
    if ((int)t->ne0 != expect_in || (int)t->ne1 != expect_out) return -2;

    j.t         = t;
    j.x         = x;
    j.y         = y;
    j.row_bytes = nita_row_bytes(t->type, t->ne0);
    j.row       = row;
    j.aq        = NULL;
    j.nb        = 0;
    j.rc        = 0;

    if (aqbuf && (t->ne0 % 32u) == 0 && nita_quant_has_int_dot(t->type) &&
        (int)(t->ne0 / 32u) <= Q35_MAX_AQ_BLOCKS) {
        j.nb = (int)(t->ne0 / 32u);
        nita_aq8_quantise(x, (int)t->ne0, aqbuf);
        j.aq = aqbuf;
    }

    nita_parallel_for((int)t->ne1, mvrow_rows, &j);
    return j.rc;
}

/* Dequantise a 1-D tensor into dst. */
static int load_vec(const nita_tensor_t *t, float *dst, int n)
{
    if (!t->bound) return -1;
    return nita_dequant_row(t->data, t->type, (uint64_t)n, dst);
}

static void rmsnorm_w(float *out, const float *x, const float *w, int n, float eps)
{
    double ss = 0.0;
    int    i;
    float  scale;

    for (i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    ss /= (double)n;
    scale = 1.0f / sqrtf((float)ss + eps);
    for (i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

/*
 * L2 normalise each head of a [n_heads][dim] vector in place.
 *
 * The delta rule's q/k are direction vectors, so the normalisation is over
 * the head width and the epsilon guards a zero head.  Note this differs from
 * an RMS norm: there is no sqrt(width) rescaling and no learned weight.
 */
static void l2norm_heads(float *v, int n_heads, int dim, float eps)
{
    int h, d;

    for (h = 0; h < n_heads; ++h) {
        float *p = v + (size_t)h * (size_t)dim;
        double ss = 0.0;
        float  norm;

        for (d = 0; d < dim; ++d) ss += (double)p[d] * (double)p[d];
        norm = (float)sqrt(ss);
        if (norm < eps) norm = eps;
        {
            float inv = 1.0f / norm;
            for (d = 0; d < dim; ++d) p[d] *= inv;
        }
    }
}

/* Numerically stable softplus; expf overflows past ~88. */
static float softplus_f(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* PrismML folds a Hadamard rotation into some matmuls: the activation that
 * feeds the weight is rotated first, and the stored weight already contains
 * the inverse. Returns x itself when the weight is not folded. */
static const float *rot_in(const nita_hadamard_t *xf, float *buf,
                           const float *x, int n)
{
    if (!xf || !buf) return x;
    memcpy(buf, x, (size_t)n * sizeof(float));
    nita_hadamard_forward(buf, n, xf);
    return buf;
}

/* Debug tap: appends "name-<layer> n v0 v1 ..." to NITA_Q35_DUMP, using the
 * same names the reference graph tags, so the two can be compared directly. */
static void dump_vec(const q35_gdn_cfg_t *c, const char *nm, const float *v, int n)
{
    FILE *f = nita_q35_dump_file();
    int   i;

    if (!f) return;
    fprintf(f, "%s-%d %d", nm, c->layer, n);
    for (i = 0; i < n; ++i) fprintf(f, " %.6g", v[i]);
    fputc('\n', f);
    fflush(f);
}

static size_t scratch_elems(const q35_gdn_cfg_t *c)
{
    size_t ne    = (size_t)c->n_embd;
    size_t cd    = (size_t)c->conv_dim;
    size_t nv    = (size_t)c->n_v_heads;
    size_t inner = (size_t)c->n_v_heads * (size_t)c->head_v_dim;
    size_t hv    = (size_t)c->head_v_dim;
    size_t row   = ne > inner ? ne : inner;

    /* xh, qkv, conv, alpha, beta, g, gate, attn, nrm, proj, row */
    return ne + cd + cd + nv + nv + nv + inner + inner + hv + ne + row;
}

int q35_gdn_prepare(const q35_gdn_cfg_t *c, const q35_gdn_weights_t *w,
                    q35_gdn_state_t *s, char *err, size_t errlen)
{
    size_t k = (size_t)c->conv_kernel;
    size_t cd = (size_t)c->conv_dim;
    size_t nv = (size_t)c->n_v_heads;

    memset(s, 0, sizeof *s);
    if (err && errlen) err[0] = '\0';

    if (c->n_k_heads <= 0 || c->head_k_dim <= 0 ||
        c->n_v_heads <= 0 || c->head_v_dim <= 0 || c->conv_kernel < 2) {
        snprintf(err, errlen, "qwen35: invalid gated delta-net dimensions");
        return -1;
    }
    if (c->n_v_heads % c->n_k_heads != 0) {
        snprintf(err, errlen,
                 "qwen35: time_step_rank %d is not a multiple of group_count %d",
                 c->n_v_heads, c->n_k_heads);
        return -1;
    }
    /* conv_dim must be exactly q + k + v. */
    if (c->conv_dim != 2 * c->n_k_heads * c->head_k_dim +
                       c->n_v_heads * c->head_v_dim) {
        snprintf(err, errlen,
                 "qwen35: conv_dim %d != 2*q %d + v %d",
                 c->conv_dim, 2 * c->n_k_heads * c->head_k_dim,
                 c->n_v_heads * c->head_v_dim);
        return -1;
    }

    /* Causal-conv ring buffer: conv_kernel-1 previous frames per channel. */
    s->conv_state = (float *)nita_calloc((k - 1) * cd, sizeof(float));
    /* Delta-rule state matrix per value head. */
    s->delta = (float *)nita_calloc(nv * (size_t)c->head_k_dim *
                                    (size_t)c->head_v_dim, sizeof(float));
    /* Conv kernel dequantised once: it is only 4 * conv_dim values. */
    s->conv_w = (float *)nita_calloc(k * cd, sizeof(float));
    s->buf_n  = scratch_elems(c);
    s->buf    = (float *)nita_calloc(s->buf_n, sizeof(float));

    if (!s->conv_state || !s->delta || !s->conv_w || !s->buf) {
        snprintf(err, errlen, "qwen35: out of memory for gated delta-net state");
        q35_gdn_state_free(s);
        return -1;
    }

    if (load_vec(&w->conv1d, s->conv_w, (int)(k * cd)) != 0) {
        snprintf(err, errlen, "qwen35: cannot read ssm_conv1d.weight");
        q35_gdn_state_free(s);
        return -1;
    }
    return 0;
}

void q35_gdn_state_reset(const q35_gdn_cfg_t *c, q35_gdn_state_t *s)
{
    size_t k  = (size_t)c->conv_kernel;
    size_t cd = (size_t)c->conv_dim;
    size_t nv = (size_t)c->n_v_heads;

    if (!s) return;
    if (s->conv_state) memset(s->conv_state, 0, (k - 1) * cd * sizeof(float));
    if (s->delta)      memset(s->delta, 0,
                              nv * (size_t)c->head_k_dim *
                              (size_t)c->head_v_dim * sizeof(float));
}

void q35_gdn_state_free(q35_gdn_state_t *s)
{
    if (!s) return;
    free(s->conv_state); s->conv_state = NULL;
    free(s->delta);      s->delta      = NULL;
    free(s->conv_w);     s->conv_w     = NULL;
    free(s->buf);        s->buf        = NULL;
    s->buf_n = 0;
}

/* ------------------------------------------------------------------ */
/* Forward                                                             */
/* ------------------------------------------------------------------ */

/* Largest value-head width the delta rule accepts (a stack buffer per head). */
#define Q35_MAX_HEAD_V 256

typedef struct {
    const float *q, *k, *v;      /* [nk*hk], [nk*hk], [nv*hv]                 */
    float       *attn;           /* [nv*hv] output                            */
    const float *g, *beta;       /* [nv] per-head decay and step size         */
    float       *state;          /* [nv][hk][hv] delta-rule state             */
    int          nk, hk, hv;
    float        ascale;
} gdn_job_t;

/* The three passes used to walk the state one column at a time, which reads a
 * strided element out of every one of the 128 rows for each output element: 128
 * passes over 64 KB, all from L2, and a scalar FMA chain per column. Walking a
 * row at a time into a vector of column accumulators instead does one pass over
 * the matrix and vectorises across d.
 *
 * Splitting a head's columns between threads was tried and measured worse: 64
 * items of 32 columns ran at 201 us against 134 for 16 items of 128, because a
 * partial row leaves gaps in the stream that cost more than the four threads a
 * thirteen-item-per-thread split leaves idle. */
static void gdn_heads(void *vctx, int lo, int hi, int thread)
{
    const gdn_job_t *j = (const gdn_job_t *)vctx;
    int              h;

    (void)thread;
    for (h = lo; h < hi; ++h) {
        int          kh  = nita_q35_k_head(h, j->nk);   /* see src/nita.h */
        const float *qh  = j->q + (size_t)kh * (size_t)j->hk;
        const float *khv = j->k + (size_t)kh * (size_t)j->hk;
        const float *vh  = j->v + (size_t)h * (size_t)j->hv;
        float       *S   = j->state + (size_t)h * (size_t)j->hk * (size_t)j->hv;
        float       *oh  = j->attn + (size_t)h * (size_t)j->hv;
        float        gv  = j->g[h];
        float        bv  = j->beta[h];
        int          i, d;
        float        delta[Q35_MAX_HEAD_V];

        /* delta = (v - g * S^T k) * beta, then the rank-1 state update. Both
         * sums still run over i in ascending order, so every column adds the
         * same terms in the same order as before. */
        for (d = 0; d < j->hv; ++d) delta[d] = 0.0f;
        for (i = 0; i < j->hk; ++i) {
            const float *Sr = S + (size_t)i * (size_t)j->hv;
            float        ki = khv[i];

            for (d = 0; d < j->hv; ++d) delta[d] += Sr[d] * ki;
        }
        for (d = 0; d < j->hv; ++d) delta[d] = (vh[d] - gv * delta[d]) * bv;
        for (i = 0; i < j->hk; ++i) {
            float  ki = khv[i];
            float *Sr = S + (size_t)i * (size_t)j->hv;

            for (d = 0; d < j->hv; ++d) Sr[d] = gv * Sr[d] + ki * delta[d];
        }
        /* Output uses the updated state, scaled by 1/sqrt(head width) - the
         * fused GDN kernel applies exactly this (gated_delta_net, scale =
         * 1/sqrt(S_v)) before the gated norm. */
        for (d = 0; d < j->hv; ++d) delta[d] = 0.0f;
        for (i = 0; i < j->hk; ++i) {
            const float *Sr = S + (size_t)i * (size_t)j->hv;
            float        qi = qh[i];

            for (d = 0; d < j->hv; ++d) delta[d] += Sr[d] * qi;
        }
        for (d = 0; d < j->hv; ++d) oh[d] = delta[d] * j->ascale;
    }
}

int q35_gdn_layer(const q35_gdn_cfg_t *c, const q35_gdn_weights_t *w,
                  q35_gdn_state_t *s, const float *x, float *out)
{
    int      ne    = c->n_embd;
    int      nk    = c->n_k_heads;
    int      hk    = c->head_k_dim;
    int      nv    = c->n_v_heads;
    int      hv    = c->head_v_dim;
    int      inner = nv * hv;
    int      cd    = c->conv_dim;
    int      kk    = c->conv_kernel;
    int      qdim  = nk * hk;
    int      i, h, t;
    float    ascale;
    const float *xin;

    float *xh    = s->buf;
    float *qkv   = xh + ne;
    float *cv    = qkv + cd;
    float *alpha = cv + cd;
    float *beta  = alpha + nv;
    float *g     = beta + nv;
    float *gate  = g + nv;
    float *attn  = gate + inner;
    float *nrm   = attn + inner;
    float *proj  = nrm + hv;
    float *row   = proj + ne;

    float *q = cv;                 /* q and k alias the conv output */
    float *k = cv + qdim;
    float *v = cv + 2 * qdim;

    nita_aq8_t aq[Q35_MAX_AQ_BLOCKS];   /* activation, quantised per matvec */

    if (!s || !s->buf) return -1;

/* Runtime-gated: the disabled probe is one predictable not-taken branch,
 * so a plain run measures the mixer, not the probe. */
#define Q35_CLOCK() (nita_trace_enabled() ? nita_prof_now() : 0.0)
#define Q35_PHASE(part, t0) do { if (nita_trace_enabled()) nita_prof_mix((part), nita_prof_now() - (t0)); } while (0)
    {
        double t0 = Q35_CLOCK();

        /* ---- pre-mixer norm and the fused q/k/v projection ---- */
        if (load_vec(&w->in_norm, row, ne) != 0) return -2;
        rmsnorm_w(xh, x, row, ne, c->eps);

        xin = rot_in(w->rot_qkv, w->rot_buf, xh, ne);
        if (mv_row(&w->qkv, xin, qkv, ne, cd, row, aq) != 0) return -3;
        Q35_PHASE(NITA_MIX_NORM, t0);
    }

    /* ---- causal depthwise conv, then SiLU ---- */
    {
    double t0 = Q35_CLOCK();

    for (i = 0; i < cd; ++i) {
        float *hist = s->conv_state + (size_t)i * (size_t)(kk - 1);
        const float *kw = s->conv_w + (size_t)i * (size_t)kk;
        float acc = kw[kk - 1] * qkv[i];

        for (t = 0; t < kk - 1; ++t)
            acc += kw[t] * hist[t];
        cv[i] = acc;

        /* Shift the window: drop the oldest, append the current sample. */
        for (t = 0; t < kk - 2; ++t) hist[t] = hist[t + 1];
        hist[kk - 2] = qkv[i];
    }

    /* SiLU over the whole convolution at once: the per-channel call was a libm
     * expf plus a divide, about 190 cycles each, for conv_dim values a layer. */
    nita_silu_inplace(cv, cd);
    Q35_PHASE(NITA_MIX_CONV, t0);
    }

    dump_vec(c, "conv_output_silu", cv, cd);
    dump_vec(c, "v_conv_predelta", v, inner);

    /* ---- gates ---- */
    {
    double t0 = Q35_CLOCK();

    if (mv_row(&w->alpha, xh, alpha, ne, nv, row, aq) != 0) return -4;
    if (mv_row(&w->beta,  xh, beta,  ne, nv, row, aq) != 0) return -4;

    {
        float dtb[128];
        float sa[128];
        if (nv > 128) return -5;
        if (load_vec(&w->dt_bias, dtb, nv) != 0) return -5;
        if (load_vec(&w->ssm_a,   sa,  nv) != 0) return -5;

        for (h = 0; h < nv; ++h) {
            float dt = softplus_f(alpha[h] + dtb[h]);

            /* ssm_a already holds -exp(A_log) - the checkpoint stores it that
             * way and upstream multiplies it straight in - so g = exp(A*dt)
             * is the decay, in (0,1). Exponentiating ssm_a again here would
             * leave the state undamped and the mixer would not forget. */
            g[h] = expf(sa[h] * dt);
            beta[h] = 1.0f / (1.0f + expf(-beta[h]));
        }
    }
    Q35_PHASE(NITA_MIX_GATES, t0);
    }

    /* ---- delta rule, one 128x128 state matrix per value head ---- */
    {
    double t0 = Q35_CLOCK();

    l2norm_heads(q, nk, hk, 1e-6f);
    l2norm_heads(k, nk, hk, 1e-6f);

    dump_vec(c, "q_conv_predelta", q, nk * hk);
    dump_vec(c, "k_conv_predelta", k, nk * hk);

    ascale = 1.0f / sqrtf((float)hv);

    /* (the per-head loop below runs through gdn_heads, defined above) */

    {
        gdn_job_t jj;

        if (hv > Q35_MAX_HEAD_V) return -6;
        jj.q      = q;
        jj.k      = k;
        jj.v      = v;
        jj.attn   = attn;
        jj.g      = g;
        jj.beta   = beta;
        jj.state  = s->delta;
        jj.nk     = nk;
        jj.hk     = hk;
        jj.hv     = hv;
        jj.ascale = ascale;
        /* Heads are independent - own slice of every input, own state matrix,
         * own output slice - so this splits with no locking, and 16 to 48
         * heads is worth a barrier even though it is under the default.
         * 4 heads is roughly where a split pays for itself. */
        nita_parallel_for_min(nv, 4, gdn_heads, &jj);
    }
    Q35_PHASE(NITA_MIX_DELTA, t0);
    }

    dump_vec(c, "gdn_output", attn, inner);

    /* ---- gated RMSNorm over the value head width ---- */
    {
    double t0 = Q35_CLOCK();

    xin = rot_in(w->rot_gate, w->rot_buf, xh, ne);
    if (mv_row(&w->gate, xin, gate, ne, inner, row, aq) != 0) return -7;

    dump_vec(c, "z", gate, inner);
    if (load_vec(&w->ssm_norm, nrm, hv) != 0) return -8;
    dump_vec(c, "ssm_norm", nrm, hv);

    {
        float norm_w[256];
        if (hv > 256) return -9;
        memcpy(norm_w, nrm, (size_t)hv * sizeof(float));

        for (h = 0; h < nv; ++h) {
            float *oh = attn + (size_t)h * (size_t)hv;
            float *gh = gate + (size_t)h * (size_t)hv;
            rmsnorm_w(oh, oh, norm_w, hv, 1e-6f);
            nita_silu_mul(oh, gh, oh, hv);
        }
    }

    Q35_PHASE(NITA_MIX_GATED_NORM, t0);
    }

    dump_vec(c, "final_output", attn, inner);

    /* ---- output projection and residual ---- */
    {
    double t0 = Q35_CLOCK();

    xin = rot_in(w->rot_out, w->rot_buf, attn, inner);
    if (mv_row(&w->out, xin, proj, inner, ne, row, aq) != 0) return -10;
    for (i = 0; i < ne; ++i) out[i] = x[i] + proj[i];
    Q35_PHASE(NITA_MIX_OUT, t0);
    }

#undef Q35_PHASE
#undef Q35_CLOCK

    return 0;
}
