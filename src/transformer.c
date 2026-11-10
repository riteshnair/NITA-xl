#define _POSIX_C_SOURCE 199309L   /* clock_gettime()/CLOCK_MONOTONIC on POSIX */

/*
 * transformer.c – llama-style decoder forward pass.
 *
 * Architecture implemented (GGUF tensor naming):
 *
 *   token_embd.weight                    [n_embd, n_vocab]
 *   output_norm.weight                   [n_embd]
 *   output.weight                        [n_embd, n_vocab]   (optional; tied to token_embd)
 *   blk.N.attn_norm.weight               [n_embd]
 *   blk.N.attn_q.weight                  [n_embd, n_head*head_dim]
 *   blk.N.attn_k.weight                  [n_embd, n_head_kv*head_dim]
 *   blk.N.attn_v.weight                  [n_embd, n_head_kv*head_dim]
 *   blk.N.attn_output.weight             [n_head*head_dim, n_embd]
 *   blk.N.ffn_norm.weight                [n_embd]
 *   blk.N.ffn_gate.weight                [n_embd, n_ff]
 *   blk.N.ffn_up.weight                  [n_embd, n_ff]
 *   blk.N.ffn_down.weight                [n_ff, n_embd]
 *
 * Per token: embedding lookup, RMSNorm, single/multi-head attention with a
 * KV cache and grouped-query heads, RoPE on Q and K, SwiGLU feed-forward,
 * final RMSNorm and the output projection.
 *
 * Weights stay in their GGUF quantised form; each row is dequantised into a
 * reusable buffer immediately before use, so no full f32 matrix is created.
 */

#include "nita.h"
#include "quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#endif
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#include <dirent.h>
#include <ctype.h>
/* The pool speaks LONG everywhere; on Windows that is a 32-bit type, on
 * POSIX a long is at least as wide, which is all the atomics need. */
typedef long LONG;
#endif

/* ------------------------------------------------------------------ */
/* Component profiler.                                                 */
/* On for a run when NITA_TRACE=1 in the environment, or always in a     */
/* -DSHERLOCK_LITE build (NITA_TRACE=0 forces a lit build off). With the */
/* trace disabled the probes are one predictable not-taken branch - no   */
/* clock read, no counter update, nothing printed - so a plain benchmark */
/* measures the engine, not the probe. nita_prof_now() costs ~100ns      */
/* (QPC / CLOCK_MONOTONIC) per call, so even the lit build carries       */
/* ~70 probes per token.                                                */
/* ------------------------------------------------------------------ */
typedef struct { double us; long calls; } prof_t;

static prof_t P_proj;    /* all matvecs (qkv, attn_out, ffn)            */
static prof_t P_out;     /* output projection, a subset of P_proj      */

/* Per-projection split of P_proj. The three FFN matvecs have identical MAC
 * counts but not identical shapes: gate and up are 8192 rows by K=2048, while
 * down is 2048 rows by K=8192, so its activation tile is eight times larger for
 * the same work. These counters exist to price that difference rather than
 * assume it. All are subsets of P_proj. */
static prof_t P_pq, P_pk, P_pv, P_po, P_pg, P_pu, P_pd;
static prof_t P_attn;    /* scores + softmax + context mix             */
static prof_t P_mix;     /* qwen35 gated delta-net layer, whole        */
static prof_t P_mixp[NITA_MIX_PARTS];   /* and its six phases              */
static prof_t P_norm;    /* rmsnorm bodies                             */
static prof_t P_embed;   /* token embedding dequant                    */
static prof_t P_ffnact;  /* silu * up                                  */
static prof_t P_rope;
static prof_t P_aq8;        /* aq8 activation quantise, paid before     */
                            /* the mv bracket - not a subset of P_proj  */
static prof_t P_kq4K, P_kq5K, P_kq6K, P_q50, P_q51, P_q80, P_iq4nl, P_kother;
                            /* per-kernel scratch: which quant type ate  */
                            /* the projections bracket (subsets of        */
                            /* P_proj)                                   */

double nita_prof_now(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER        c;

    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e6 / (double)freq.QuadPart;
#else
    /* clock() reports CPU time across all threads, not wall time, so a
     * multithreaded run would read several times the actual latency and an
     * idle-looking report would still show microseconds everywhere. Use the
     * monotonic wall clock instead. */
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
    return (double)clock() * 1e6 / (double)CLOCKS_PER_SEC;   /* fallback */
#endif
}

static double prof_now(void)
{
    /* The disabled probe is one predictable not-taken branch. */
    return nita_trace_enabled() ? nita_prof_now() : 0.0;
}

static void prof_add(prof_t *p, double dt, long calls)
{
    if (!nita_trace_enabled()) return;
    p->us += dt;
    p->calls += calls;
}

/* Attribute one projection bracket to the quantisation kernel inside it,
 * so a trace can name which kernel ate the time. */
static void prof_kernel(uint32_t type, double dt)
{
    switch (type) {
    case 6:  prof_add(&P_q50,  dt, 1); break;
    case 7:  prof_add(&P_q51,  dt, 1); break;
    case 8:  prof_add(&P_q80,  dt, 1); break;
    case 12: prof_add(&P_kq4K, dt, 1); break;
    case 13: prof_add(&P_kq5K, dt, 1); break;
    case 14: prof_add(&P_kq6K, dt, 1); break;
    case 20: prof_add(&P_iq4nl, dt, 1); break;
    default: prof_add(&P_kother, dt, 1); break;
    }
}

void nita_prof_mix(int part, double us)
{
    if (part < 0 || part >= NITA_MIX_PARTS) return;
    prof_add(&P_mixp[part], us, 1);
}

void nita_prof_reset(void)
{
    memset(&P_aq8, 0, sizeof P_aq8);
    memset(&P_kq4K, 0, sizeof P_kq4K);   memset(&P_kq5K, 0, sizeof P_kq5K);
    memset(&P_kq6K, 0, sizeof P_kq6K);   memset(&P_q50,  0, sizeof P_q50);
    memset(&P_q51,  0, sizeof P_q51);    memset(&P_q80,  0, sizeof P_q80);
    memset(&P_iq4nl, 0, sizeof P_iq4nl); memset(&P_kother, 0, sizeof P_kother);
    memset(&P_proj, 0, sizeof P_proj);   memset(&P_out, 0, sizeof P_out);
    memset(&P_pq, 0, sizeof P_pq);  memset(&P_pk, 0, sizeof P_pk);
    memset(&P_pv, 0, sizeof P_pv);  memset(&P_po, 0, sizeof P_po);
    memset(&P_pg, 0, sizeof P_pg);  memset(&P_pu, 0, sizeof P_pu);
    memset(&P_pd, 0, sizeof P_pd);
    memset(&P_attn, 0, sizeof P_attn);   memset(&P_mix, 0, sizeof P_mix);
    memset(P_mixp, 0, sizeof P_mixp);
    memset(&P_norm, 0, sizeof P_norm);
    memset(&P_embed, 0, sizeof P_embed); memset(&P_ffnact, 0, sizeof P_ffnact);
    memset(&P_rope, 0, sizeof P_rope);
}

int nita_prof_rows(nita_prof_row_t *out, int max_rows)
{
    static const char *names[]  = { "projections", "out-proj(sub)", "attention", "mixer",
                                    "norms", "ffn-activate", "rope", "embed",
                                    "  q-proj", "  k-proj", "  v-proj", "  o-proj",
                                    "  ffn-gate", "  ffn-up", "  ffn-down",
                                    "  mix norm+qkv", "  mix conv", "  mix gates",
                                    "  mix delta", "  mix gated-norm", "  mix out-proj",
                                    "aq8-quantise",
                                    "  kernel:q4_K", "  kernel:q5_K", "  kernel:q6_K",
                                    "  kernel:q5_0", "  kernel:q5_1", "  kernel:q8_0",
                                    "  kernel:iq4_nl", "  kernel:other" };
    static const int   subset[] = { 0, 1, 0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1,
                                    0,
                                    1, 1, 1, 1, 1, 1, 1, 1 };
    const prof_t      *src[]    = { &P_proj, &P_out, &P_attn, &P_mix,
                                    &P_norm, &P_ffnact, &P_rope, &P_embed,
                                    &P_pq, &P_pk, &P_pv, &P_po,
                                    &P_pg, &P_pu, &P_pd,
                                    &P_mixp[0], &P_mixp[1], &P_mixp[2],
                                    &P_mixp[3], &P_mixp[4], &P_mixp[5],
                                    &P_aq8,
                                    &P_kq4K, &P_kq5K, &P_kq6K,
                                    &P_q50, &P_q51, &P_q80,
                                    &P_iq4nl, &P_kother };
    const int          n = (int)(sizeof src / sizeof src[0]);
    int                i;

    if (!out || max_rows <= 0) return 0;
    for (i = 0; i < n && i < max_rows; ++i) {
        out[i].name   = names[i];
        out[i].us     = src[i]->us;
        out[i].calls  = src[i]->calls;
        out[i].subset = subset[i];
    }
    return i;
}

void nita_prof_dump(void)
{
    nita_prof_row_t rows[NITA_PROF_MAX_ROWS];
    int             n = nita_prof_rows(rows, NITA_PROF_MAX_ROWS);
    double          tot = 0.0;
    int             i;

    for (i = 0; i < n; ++i)
        if (!rows[i].subset) tot += rows[i].us;
    if (tot <= 0.0) tot = 1.0;

    printf("%-16s%8s%14s%12s%8s\n", "component", "calls", "total_us", "us/call", "%");
    for (i = 0; i < n; ++i) {
        printf("%-16s%8ld%14.1f%12.2f%7.1f%%\n",
               rows[i].name, rows[i].calls, rows[i].us,
               rows[i].calls ? rows[i].us / (double)rows[i].calls : 0.0,
               100.0 * rows[i].us / tot);
    }
    printf("%-16s%8s%14.1f\n", "sum", "", tot);
    fflush(stdout);
}

/* Which of a qwen35 block's matmuls consume a rotated activation. NULL when
 * prism.hadamard does not fold that weight. */
typedef struct {
    const nita_hadamard_t *q, *k, *v, *o;      /* dense-attention projections */
    const nita_hadamard_t *gate, *up, *down;   /* shared SwiGLU            */
} q35_rot_t;

struct nita_model {
    /* Hyperparameters */
    int   n_embd;
    int   n_layer;
    int   n_head;
    int   n_head_kv;
    int   n_ff;
    int   n_rot;
    int   n_vocab;
    int   ctx;
    int   head_dim;
    int   kv_dim;
    float rms_eps;
    float rope_theta;
    bool  tied_output;
    bool  ready;

    /* Weights */
    nita_tensor_t  tok_embd;
    nita_tensor_t  out_norm;
    nita_tensor_t  out_proj;
    nita_tensor_t *attn_norm;
    nita_tensor_t *attn_q;
    nita_tensor_t *attn_k;
    nita_tensor_t *attn_v;
    nita_tensor_t *attn_out;
    nita_tensor_t *ffn_norm;
    nita_tensor_t *ffn_gate;
    nita_tensor_t *ffn_up;
    nita_tensor_t *ffn_down;

    /* Mixture of experts. Two checkpoint flavours reach this path: llama-arch
     * MoE (Mixtral-style, e.g. Tiny-Moe) which renormalises the selected
     * expert weights, and qwen2moe (Qwen2.5-MoE) which does not and carries a
     * sigmoid-gated shared expert. Everything is derived from the GGUF: the
     * expert count from metadata, the per-expert FFN width from the expert
     * tensor shapes. A model with expert_count == 0 leaves all of this NULL. */
    bool           is_moe;
    int            n_expert;
    int            expert_used;
    int            n_ff_exp;
    int            n_ff_shexp;      /* 0 when there is no shared expert    */
    int            moe_norm_w;      /* llama-style weight renormalisation   */
    float          exp_w_scale;     /* expert_weights_scale; 0 = unset      */
    nita_tensor_t *ffn_gate_inp;    /* [n_layer] router   [n_embd, n_expert] */
    nita_tensor_t *ffn_gate_exps;   /* [n_layer] [n_embd, n_ff_exp, n_expert] */
    nita_tensor_t *ffn_up_exps;     /* [n_layer] [n_embd, n_ff_exp, n_expert] */
    nita_tensor_t *ffn_down_exps;   /* [n_layer] [n_ff_exp, n_embd, n_expert] */
    nita_tensor_t *shexp_gate_inp;  /* [n_layer] 1-D [n_embd]                */
    nita_tensor_t *shexp_gate;      /* [n_layer] [n_embd, n_ff_shexp]        */
    nita_tensor_t *shexp_up;        /* [n_layer] [n_embd, n_ff_shexp]        */
    nita_tensor_t *shexp_down;      /* [n_layer] [n_ff_shexp, n_embd]        */
    /* Optional attention / output biases (Qwen2 carries q/k/v biases). */
    nita_tensor_t *attn_q_bias;     /* [n_layer] or NULL when the arch has none */
    nita_tensor_t *attn_k_bias;
    nita_tensor_t *attn_v_bias;
    nita_tensor_t *attn_out_bias;
    nita_tensor_t  out_bias;        /* optional output.bias (tied: unused)  */
    float         *acc;              /* [n_embd] MoE expert accumulation     */
    float         *router;           /* [n_expert] router logits then probs  */
    int           *sel;              /* [expert_used] selected expert ids    */
    float         *sel_w;            /* [expert_used] their gate weights     */
    unsigned char *sel_taken;        /* [n_expert] selection marker          */
    float         *biasbuf;          /* scratch for 1-D vectors (biases,     */
                                    /* shared-expert gate input)            */
    float         *out_bias_f32;     /* dequantised once at load, or NULL   */

    /* KV cache: one contiguous buffer per layer */
    float **k_cache;
    float **v_cache;
    int    pos;

    /* Scratch */
    float  *rowbuf;
    size_t  rowcap;
    float  *nbuf;    /* dequantised norm vector      */
    float  *xb;      /* residual stream              */
    float  *hb;      /* normalised hidden            */
    float  *qbuf;    /* queries                      */
    float  *kbuf;    /* keys for this token          */
    float  *vbuf;    /* values for this token        */
    float  *abuf;    /* attention output             */
    float  *gate;
    float  *up;
    float  *ff;
    float  *tmp;
    float  *scores;
    float  *rope_tab;   /* [ctx][n_rot/2][cos,sin], or NULL if not built */

    /* Batched prefill scratch: tile rows of each per-token buffer. Allocated
     * on first use, so a model that only decodes never pays for them. */
    int    tile;        /* rows per tile: 0 = not tried yet, <0 = unavailable */
    float *pxb;         /* residual streams   [tile][n_embd] */
    float *phb;         /* normalised hidden  [tile][n_embd] */
    float *ptmp;        /* projection output  [tile][n_embd] */
    float *pq;          /* queries            [tile][q_dim]  */
    float *pa;          /* attention output   [tile][q_dim]  */
    float *pk;          /* keys for the tile  [tile][kv_dim] */
    float *pv;          /* values for the tile[tile][kv_dim] */
    float *pgate;       /* ffn gate           [tile][max(n_ff, n_ff_exp)] */
    float *pup;         /* ffn up             [tile][max(n_ff, n_ff_exp)] */
    float *pff;         /* ffn activated      [tile][max(n_ff, n_ff_exp, n_ff_shexp)] */
    float *pscores;     /* per-row attention scores [tile][ctx] */
    /* MoE gather/scatter scratch (allocated only for MoE models). */
    float *prouter;     /* [tile][n_expert] router logits, then probs */
    float *pacc;        /* [tile][n_embd] weighted expert accumulation */
    float *pgath;       /* [tile][n_embd] gathered rows for one expert */
    int   *psel;        /* [tile][expert_used] selected expert ids     */
    float *psw;         /* [tile][expert_used] gate weights             */
    int   *prow_tok;    /* [tile] source token row, per gathered row    */
    float *prow_w;      /* [tile] gate weight, per gathered row         */

    /* Row-parallel worker pool (NULL or 1 participant = serial). */
    void   *pool;

    /* Quantised activations for the integer kernels: tile rows of blocks,
     * filled once per projection by the calling thread. */
    nita_aq8_t *qact;
    size_t      qact_cap;      /* blocks allocated */

    /* qwen35 hybrid architecture (PrismML Bonsai): most blocks are gated
     * delta-net mixers, every full_attention_interval-th one is dense
     * attention. Loaded per layer; inference is wired in stage 2. */
    bool               is_qwen35;
    int                full_attn_interval;
    unsigned char     *layer_recr;   /* 1 = gated delta-net mixer */
    q35_gdn_cfg_t      q35_cfg;
    q35_gdn_weights_t *q35_w;        /* n_layer entries */
    q35_gdn_state_t   *q35_state;    /* n_layer entries */
    nita_tensor_t     *post_norm;    /* blk.N.post_attention_norm.weight */
    nita_tensor_t     *attn_q_norm;
    nita_tensor_t     *attn_k_norm;
    int                q35_gated_attn;   /* attn_q carries query + gate */

    /* prism.hadamard.*: a rotation folded into the matmuls listed in
     * weight_names, applied to the activation feeding them. One transform per
     * distinct input width, shared by every folded weight of that width. */
    struct {
        int             width;
        nita_hadamard_t xf;
    }                      hd[4];
    int                    hd_n;
    char                 **hd_names;
    int                    hd_n_names;
    const nita_hadamard_t *hd_tok_xf;   /* transform of the rotated embedding */
    float                 *hdbuf;       /* rotated activation copy            */
    float                 *qg;          /* attn_q output, query + gate        */
    float                 *qgate;       /* the gate half                      */
    float                 *qnormv;      /* attn_q_norm, dequantised           */
    float                 *knormv;
    float                 *headbuf;     /* one head, for the per-head norm    */
    q35_rot_t             *rot;         /* n_layer entries                    */
};

/* Transform applied to the activation feeding `name`, or NULL when the matmul
 * is not folded. The width selects which sign vector applies. */
static const nita_hadamard_t *hd_for(const nita_model_t *m, const char *name,
                                     uint64_t width)
{
    int i, w;

    if (m->hd_n == 0 || !m->hd_names) return NULL;
    for (i = 0; i < m->hd_n_names; ++i) {
        if (strcmp(m->hd_names[i], name) != 0) continue;
        for (w = 0; w < m->hd_n; ++w)
            if ((uint64_t)m->hd[w].width == width) return &m->hd[w].xf;
        return NULL;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Tensor binding                                                      */
/* ------------------------------------------------------------------ */
static bool bound_in_file(const gguf_file_t *g, const nita_tensor_t *t)
{
    (void)g;
    return t->bound && t->data != NULL;
}

/* The GGUF table entry for `name`, or NULL. Used to tell "bias tensor is
 * absent" (fine) from "bias tensor is present but unusable" (fatal). */
static const gguf_tensor_t *tensor_named(const gguf_file_t *g, const char *name)
{
    uint64_t i;

    if (!g || !name) return NULL;
    for (i = 0; i < g->tensor_count; ++i)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

static bool bind_tensor(const gguf_file_t *g, nita_tensor_t *t, const char *name)
{
    uint64_t i;

    memset(t, 0, sizeof *t);
    if (!g || !name) return false;

    for (i = 0; i < g->tensor_count; ++i) {
        const gguf_tensor_t *src = &g->tensors[i];
        size_t               row;
        uint64_t             need, mult, k;
        uint64_t             abs_off;

        if (strcmp(src->name, name) != 0) continue;

        row  = nita_row_bytes(src->type, src->n_dims > 0 ? src->dims[0] : 0);
        mult = 1;
        for (k = 1; k < src->n_dims; ++k)
            if (src->dims[k]) mult *= src->dims[k];
        need = row * (size_t)mult;
        abs_off = g->data_offset + src->offset;

        if (abs_off > g->file_size || need > g->file_size - abs_off) {
            BREADCRUMBF("model", "tensor '%s' runs past end of file", name);
            return false;
        }

        t->data  = g->data + abs_off;
        t->type  = src->type;
        t->ne0   = src->n_dims > 0 ? src->dims[0] : 0;
        t->ne1   = src->n_dims > 1 ? src->dims[1] : 1;
        t->ne2   = src->n_dims > 2 ? src->dims[2] : 1;
        t->bound = true;
        snprintf(t->name, sizeof t->name, "%s", name);
        return true;
    }
    return false;
}

/* One line for a weight the model actually holds: which field it is, which
 * tensor it came from, where that pointer lands and what is there. Aliases (a
 * tied output projection, qwen35's reused post-attention norm) are named the
 * same tensor twice on purpose, so a repeated name is reported once. */
static void report_weight(const gguf_file_t *g, const char *role,
                          const nita_tensor_t *t, char **seen, int *nseen, int cap)
{
    uint64_t abs_off, n;
    int      i;

    if (!t->bound || !t->data || !t->name[0]) return;
    for (i = 0; i < *nseen; ++i)
        if (strcmp(seen[i], t->name) == 0) return;
    if (*nseen >= cap) return;

    seen[(*nseen)++] = (char *)t->name;
    abs_off = (uint64_t)(t->data - g->data);
    n = g->file_size - abs_off < NITA_TENSOR_CHECK_BYTES
        ? (size_t)(g->file_size - abs_off)
        : NITA_TENSOR_CHECK_BYTES;

    printf("%s\t%s\t%llu\t%016llx\n", role, t->name,
           (unsigned long long)abs_off, (unsigned long long)nita_fnv1a64(t->data, n));
}

int nita_model_tensor_report(const nita_model_t *m, const gguf_file_t *g)
{
    char  **seen;
    char    role[64];
    int     nseen = 0, l;

    if (!m || !g || !m->ready) return -1;

    seen = (char **)calloc((size_t)m->n_layer * 24 + 8, sizeof *seen);
    if (!seen) return -1;

    printf("layers\t%d\n", m->n_layer);

    report_weight(g, "tok_embd", &m->tok_embd, seen, &nseen, m->n_layer * 24 + 8);
    report_weight(g, "out_norm", &m->out_norm, seen, &nseen, m->n_layer * 24 + 8);
    if (!m->tied_output)
        report_weight(g, "out_proj", &m->out_proj, seen, &nseen, m->n_layer * 24 + 8);

    for (l = 0; l < m->n_layer; ++l) {
        const int cap = m->n_layer * 24 + 8;

        snprintf(role, sizeof role, "attn_norm[%d]", l);
        report_weight(g, role, &m->attn_norm[l], seen, &nseen, cap);
        if (m->is_qwen35) {
            /* These arrays only exist for qwen35 hybrids; a llama-style model
             * leaves them NULL and &NULL[l] would segfault (it did). */
            snprintf(role, sizeof role, "post_norm[%d]", l);
            report_weight(g, role, &m->post_norm[l], seen, &nseen, cap);
        }
        snprintf(role, sizeof role, "ffn_norm[%d]", l);
        report_weight(g, role, &m->ffn_norm[l], seen, &nseen, cap);
        snprintf(role, sizeof role, "attn_q[%d]", l);
        report_weight(g, role, &m->attn_q[l], seen, &nseen, cap);
        snprintf(role, sizeof role, "attn_k[%d]", l);
        report_weight(g, role, &m->attn_k[l], seen, &nseen, cap);
        snprintf(role, sizeof role, "attn_v[%d]", l);
        report_weight(g, role, &m->attn_v[l], seen, &nseen, cap);
        snprintf(role, sizeof role, "attn_out[%d]", l);
        report_weight(g, role, &m->attn_out[l], seen, &nseen, cap);
        if (m->is_qwen35) {
            snprintf(role, sizeof role, "attn_q_norm[%d]", l);
            report_weight(g, role, &m->attn_q_norm[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "attn_k_norm[%d]", l);
            report_weight(g, role, &m->attn_k_norm[l], seen, &nseen, cap);
        }
        snprintf(role, sizeof role, "ffn_gate[%d]", l);
        report_weight(g, role, &m->ffn_gate[l], seen, &nseen, cap);
        snprintf(role, sizeof role, "ffn_up[%d]", l);
        report_weight(g, role, &m->ffn_up[l], seen, &nseen, cap);
        snprintf(role, sizeof role, "ffn_down[%d]", l);
        report_weight(g, role, &m->ffn_down[l], seen, &nseen, cap);

        if (m->is_moe) {
            snprintf(role, sizeof role, "ffn_gate_inp[%d]", l);
            report_weight(g, role, &m->ffn_gate_inp[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "ffn_gate_exps[%d]", l);
            report_weight(g, role, &m->ffn_gate_exps[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "ffn_up_exps[%d]", l);
            report_weight(g, role, &m->ffn_up_exps[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "ffn_down_exps[%d]", l);
            report_weight(g, role, &m->ffn_down_exps[l], seen, &nseen, cap);
            if (m->n_ff_shexp > 0) {
                snprintf(role, sizeof role, "shexp_gate_inp[%d]", l);
                report_weight(g, role, &m->shexp_gate_inp[l], seen, &nseen, cap);
                snprintf(role, sizeof role, "shexp_gate[%d]", l);
                report_weight(g, role, &m->shexp_gate[l], seen, &nseen, cap);
                snprintf(role, sizeof role, "shexp_up[%d]", l);
                report_weight(g, role, &m->shexp_up[l], seen, &nseen, cap);
                snprintf(role, sizeof role, "shexp_down[%d]", l);
                report_weight(g, role, &m->shexp_down[l], seen, &nseen, cap);
            }
        }
        if (!m->is_qwen35) {
            snprintf(role, sizeof role, "attn_q_bias[%d]", l);
            report_weight(g, role, &m->attn_q_bias[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "attn_k_bias[%d]", l);
            report_weight(g, role, &m->attn_k_bias[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "attn_v_bias[%d]", l);
            report_weight(g, role, &m->attn_v_bias[l], seen, &nseen, cap);
            snprintf(role, sizeof role, "attn_out_bias[%d]", l);
            report_weight(g, role, &m->attn_out_bias[l], seen, &nseen, cap);
        }
        if (!m->tied_output) { } /* out_bias reported once, below */

        if (m->is_qwen35 && m->layer_recr[l]) {
            snprintf(role, sizeof role, "ssm_qkv[%d]", l);
            report_weight(g, role, &m->q35_w[l].qkv, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_gate[%d]", l);
            report_weight(g, role, &m->q35_w[l].gate, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_conv1d[%d]", l);
            report_weight(g, role, &m->q35_w[l].conv1d, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_alpha[%d]", l);
            report_weight(g, role, &m->q35_w[l].alpha, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_beta[%d]", l);
            report_weight(g, role, &m->q35_w[l].beta, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_dt[%d]", l);
            report_weight(g, role, &m->q35_w[l].dt_bias, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_a[%d]", l);
            report_weight(g, role, &m->q35_w[l].ssm_a, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_norm[%d]", l);
            report_weight(g, role, &m->q35_w[l].ssm_norm, seen, &nseen, cap);
            snprintf(role, sizeof role, "ssm_out[%d]", l);
            report_weight(g, role, &m->q35_w[l].out, seen, &nseen, cap);
        }
    }

    printf("model\t%d\n", nseen);
    free(seen);
    return nseen;
}

int nita_bind_all(const gguf_file_t *g, nita_tensor_t *out, size_t cap)
{
    uint64_t i;

    if (!g || !out) return -1;
    if (g->tensor_count > (uint64_t)cap) return -1;

    for (i = 0; i < g->tensor_count; ++i)
        if (!bind_tensor(g, &out[i], g->tensors[i].name)) return -1;

    return (int)g->tensor_count;
}

/* F-02: every bound weight tensor must have a decoder in this build.
 * Anything else is refused here, loudly, instead of reaching the kernels
 * where an unknown type dequantises to zeros and the model emits garbage
 * with no signal. Returns the first offending tensor, or NULL. */
static const nita_tensor_t *unsupported_tensor(const nita_model_t *m)
{
    int l;

    if (m->tok_embd.bound && !nita_type_supported(m->tok_embd.type)) return &m->tok_embd;
    if (m->out_norm.bound && !nita_type_supported(m->out_norm.type)) return &m->out_norm;
    if (m->out_bias.bound && !nita_type_supported(m->out_bias.type)) return &m->out_bias;
    if (!m->tied_output && m->out_proj.bound &&
        !nita_type_supported(m->out_proj.type))
        return &m->out_proj;

    for (l = 0; l < m->n_layer; ++l) {
#define NITA_CHECK_TENSOR(T)                                            \
        do {                                                            \
            const nita_tensor_t *_t = (T);                              \
            if (_t && _t->bound && !nita_type_supported(_t->type))      \
                return _t;                                              \
        } while (0)
        NITA_CHECK_TENSOR(&m->attn_norm[l]);
        NITA_CHECK_TENSOR(&m->attn_q[l]);
        NITA_CHECK_TENSOR(&m->attn_k[l]);
        NITA_CHECK_TENSOR(&m->attn_v[l]);
        NITA_CHECK_TENSOR(&m->attn_out[l]);
        NITA_CHECK_TENSOR(&m->ffn_gate[l]);
        NITA_CHECK_TENSOR(&m->ffn_up[l]);
        NITA_CHECK_TENSOR(&m->ffn_down[l]);
        if (m->is_moe) {
            NITA_CHECK_TENSOR(&m->ffn_gate_inp[l]);
            NITA_CHECK_TENSOR(&m->ffn_gate_exps[l]);
            NITA_CHECK_TENSOR(&m->ffn_up_exps[l]);
            NITA_CHECK_TENSOR(&m->ffn_down_exps[l]);
            if (m->n_ff_shexp > 0) {
                NITA_CHECK_TENSOR(&m->shexp_gate_inp[l]);
                NITA_CHECK_TENSOR(&m->shexp_gate[l]);
                NITA_CHECK_TENSOR(&m->shexp_up[l]);
                NITA_CHECK_TENSOR(&m->shexp_down[l]);
            }
        }
        if (!m->is_qwen35) {
            NITA_CHECK_TENSOR(&m->attn_q_bias[l]);
            NITA_CHECK_TENSOR(&m->attn_k_bias[l]);
            NITA_CHECK_TENSOR(&m->attn_v_bias[l]);
            NITA_CHECK_TENSOR(&m->attn_out_bias[l]);
        }
        if (m->is_qwen35) {
            NITA_CHECK_TENSOR(&m->attn_q_norm[l]);
            NITA_CHECK_TENSOR(&m->attn_k_norm[l]);
            NITA_CHECK_TENSOR(&m->post_norm[l]);
            NITA_CHECK_TENSOR(&m->q35_w[l].qkv);
            NITA_CHECK_TENSOR(&m->q35_w[l].gate);
            NITA_CHECK_TENSOR(&m->q35_w[l].conv1d);
            NITA_CHECK_TENSOR(&m->q35_w[l].alpha);
            NITA_CHECK_TENSOR(&m->q35_w[l].beta);
            NITA_CHECK_TENSOR(&m->q35_w[l].dt_bias);
            NITA_CHECK_TENSOR(&m->q35_w[l].ssm_a);
            NITA_CHECK_TENSOR(&m->q35_w[l].ssm_norm);
            NITA_CHECK_TENSOR(&m->q35_w[l].out);
        } else {
            NITA_CHECK_TENSOR(&m->ffn_norm[l]);
        }
#undef NITA_CHECK_TENSOR
    }
    return NULL;
}

/* Read a hyperparameter, trying the model's own architecture prefix first. */
static int64_t hparam_int(const gguf_file_t *g, const char *arch, const char *suffix,
                          int64_t fallback)
{
    char key[160];
    if (arch && *arch) {
        snprintf(key, sizeof key, "%s.%s", arch, suffix);
        if (gguf_has_key(g, key)) return gguf_get_int(g, key, fallback);
    }
    snprintf(key, sizeof key, "llama.%s", suffix);
    if (gguf_has_key(g, key)) return gguf_get_int(g, key, fallback);
    return fallback;
}

static double hparam_float(const gguf_file_t *g, const char *arch, const char *suffix,
                           double fallback)
{
    char key[160];
    if (arch && *arch) {
        snprintf(key, sizeof key, "%s.%s", arch, suffix);
        if (gguf_has_key(g, key)) return gguf_get_float(g, key, fallback);
    }
    snprintf(key, sizeof key, "llama.%s", suffix);
    if (gguf_has_key(g, key)) return gguf_get_float(g, key, fallback);
    return fallback;
}

/* ------------------------------------------------------------------ */
/* Small numeric kernels                                               */
/* ------------------------------------------------------------------ */
/* Dequantise a 1-D tensor (a norm weight vector) into dst. */
static int load_vector(const nita_tensor_t *t, float *dst, int n)
{
    if (!bound_in_file(NULL, t)) return -1;
    return nita_dequant_row(t->data, t->type, (uint64_t)n, dst);
}

/* y = W x, where W is stored row-major as ne1 rows of ne0 elements. */
/* ------------------------------------------------------------------ */
/* Row-parallel matvec.                                                */
/*                                                                     */
/* Every matvec is a set of independent rows, so the whole projection  */
/* work of the forward pass splits cleanly across cores. Workers are    */
/* persistent and spin on a generation counter, which keeps dispatch    */
/* latency at a few hundred nanoseconds instead of the tens of          */
/* microseconds a thread creation would cost on every call.            */
/* ------------------------------------------------------------------ */

#if defined(__AVX2__)
#  define NITA_PAUSE() _mm_pause()
#else
#  define NITA_PAUSE() ((void)0)
#endif

#define NITA_MAX_THREADS 32

typedef struct {
    const nita_tensor_t *t;
    const float         *x;
    float               *y;
    float               *rowbuf;   /* generic path only, per thread */
    size_t               row_bytes;
    uint64_t             lo, hi;

    /* Quantised activations, one row of blocks per tile row. NULL keeps the
     * dequantise-then-float path, which is what types without an integer
     * kernel use and what NITA_INT8_DOT=0 selects for an A/B. */
    const nita_aq8_t    *aq;
    int                  aq_stride;   /* blocks per activation row */

    /* Tile mode (prefill). With ncol > 0 the caller is doing ncol tokens at
     * once: x and y are ncol rows of kstride and ystride floats, and the job
     * computes y[i][r] = W[r] . x[i] for every row r in [lo,hi). The reduction
     * per (i,r) keeps the single-vector kernel's accumulation order, so a
     * tiled result is bit-identical to the sequential one. */
    int                  ncol;
    int                  kstride;
    int                  ystride;
} mv_range_t;

static void mv_rows_tile(const mv_range_t *j);

/* Tile attention over a row range; defined with the other attention code. */
static void attention_rows(nita_model_t *m, int layer, int pos0, int lo, int hi);

/* y[lo..hi) = W[lo..hi) . x */

/* AVX2 float dot for the generic path (F16/F32/BF16 weights): four
 * accumulators, eight floats each, fixed reduction tree. Both the sequential
 * (mv_rows) and tile (mv_rows_tile) tails call it, so the two stay
 * bit-identical. Documented float-order change vs the old scalar loop -
 * measured in the commit message, not claimed away. */
#if defined(__AVX2__)
static float fdot_avx2(const float *a, const float *b, uint64_t n)
{
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    uint64_t i = 0;

    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    acc0 = _mm256_hadd_ps(acc0, acc0);
    acc0 = _mm256_hadd_ps(acc0, acc0);
    {
        float sum = _mm_cvtss_f32(_mm256_castps256_ps128(acc0));
        for (; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
}
#endif

static void mv_rows(const mv_range_t *j)
{
    const nita_tensor_t *t = j->t;
    uint64_t r;

    if (j->ncol > 0) { mv_rows_tile(j); return; }

    /* Integer domain: the weights are never expanded to float here. */
    if (j->aq) {
        const int nb = j->aq_stride;

        for (r = j->lo; r < j->hi; ++r)
            j->y[r] = nita_quant_dot(t->type, t->data + (size_t)r * j->row_bytes,
                                     j->aq, nb);
        return;
    }

#if defined(__AVX2__) && !defined(NITA_NO_Q8_FAST)
    if (t->type == 8 && (t->ne0 % 32u) == 0 && j->row_bytes == (size_t)(t->ne0 / 32u) * 34) {
        const uint64_t nb = t->ne0 / 32u;

        for (r = j->lo; r < j->hi; ++r) {
            const uint8_t *blk = t->data + (size_t)r * j->row_bytes;
            __m256 acc = _mm256_setzero_ps();
            uint64_t b;

            for (b = 0; b < nb; ++b) {
                const uint8_t *q  = blk + b * 34;
                const int8_t  *qs = (const int8_t *)(q + 2);
                const float   *xp = j->x + b * 32;
                __m256 vd = _mm256_set1_ps(nita_f16_to_f32((uint16_t)(q[0] | (q[1] << 8))));

                acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                          _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs))))),
                          _mm256_loadu_ps(xp), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                          _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 8))))),
                          _mm256_loadu_ps(xp + 8), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                          _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 16))))),
                          _mm256_loadu_ps(xp + 16), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                          _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 24))))),
                          _mm256_loadu_ps(xp + 24), acc);
            }
            {
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 s  = _mm_add_ps(lo, hi);
                s = _mm_hadd_ps(s, s);
                s = _mm_hadd_ps(s, s);
                j->y[r] = _mm_cvtss_f32(s);
            }
        }
        return;
    }
#endif

    for (r = j->lo; r < j->hi; ++r) {
        const uint8_t *src = t->data + (size_t)r * j->row_bytes;
        float          sum = 0.0f;
        uint64_t       i;

        if (nita_dequant_row(src, t->type, t->ne0, j->rowbuf) != 0) {
            j->y[r] = 0.0f;
            continue;
        }
#if defined(__AVX2__)
        sum = fdot_avx2(j->rowbuf, j->x, t->ne0);
#else
        for (i = 0; i < t->ne0; ++i) sum += j->rowbuf[i] * j->x[i];
#endif
        j->y[r] = sum;
    }
}

/* Same horizontal reduction as the single-vector kernel, so the float sums
 * line up exactly rather than merely closely. AVX2-only helper. */
#if defined(__AVX2__)
static float hsum256(__m256 acc)
{
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

/*
 * Tile kernel: one weight row, several tokens.
 *
 * Decode is memory bound - all 1.24 GB of weights is streamed for every single
 * token. Prompt processing re-read that same data once per token for nothing:
 * here each row is dequantised once and multiplied into all T input rows while
 * it is still in registers, so the weight traffic is paid once per tile rather
 * than once per token. Prompt cost per token falls from "stream the whole
 * model" to "one T-th of it plus the arithmetic".
 */
static void mv_rows_tile(const mv_range_t *j)
{
    const nita_tensor_t *t = j->t;
    const int  T = j->ncol;
    const int  K = j->kstride;
    const int  N = j->ystride;
    uint64_t   r;

    /* Integer domain, tile form: one expansion of each weight block feeds
     * every token in the tile. */
    if (j->aq) {
        for (r = j->lo; r < j->hi; ++r)
            nita_quant_dot_tile(t->type, t->data + (size_t)r * j->row_bytes,
                                j->aq, j->aq_stride, T, j->aq_stride,
                                j->y + (size_t)r, N);
        return;
    }

#if defined(__AVX2__) && !defined(NITA_NO_Q8_FAST)
    if (t->type == 8 && (t->ne0 % 32u) == 0 && j->row_bytes == (size_t)(t->ne0 / 32u) * 34) {
        const uint64_t nb = t->ne0 / 32u;

        for (r = j->lo; r < j->hi; ++r) {
            const uint8_t *blk = t->data + (size_t)r * j->row_bytes;
            int            i = 0;

            /* Four tokens at a time: four weight vectors, four accumulators
             * and the block scale fit in the sixteen ymm registers, so the
             * dequantisation is shared by four tokens with no spills. */
            for (; i + 4 <= T; i += 4) {
                const float *x0 = j->x + (size_t)(i + 0) * (size_t)K;
                const float *x1 = j->x + (size_t)(i + 1) * (size_t)K;
                const float *x2 = j->x + (size_t)(i + 2) * (size_t)K;
                const float *x3 = j->x + (size_t)(i + 3) * (size_t)K;
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                uint64_t b;

                for (b = 0; b < nb; ++b) {
                    const uint8_t *q  = blk + b * 34;
                    const int8_t  *qs = (const int8_t *)(q + 2);
                    const size_t   o  = (size_t)b * 32u;
                    __m256 vd = _mm256_set1_ps(nita_f16_to_f32((uint16_t)(q[0] | (q[1] << 8))));
                    __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs)))));
                    __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 8)))));
                    __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 16)))));
                    __m256 w3 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 24)))));
                    const float *p;

                    p = x0 + o;
                    a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p),      a0);
                    a0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p + 8),  a0);
                    a0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p + 16), a0);
                    a0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p + 24), a0);

                    p = x1 + o;
                    a1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p),      a1);
                    a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p + 8),  a1);
                    a1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p + 16), a1);
                    a1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p + 24), a1);

                    p = x2 + o;
                    a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p),      a2);
                    a2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p + 8),  a2);
                    a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p + 16), a2);
                    a2 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p + 24), a2);

                    p = x3 + o;
                    a3 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p),      a3);
                    a3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p + 8),  a3);
                    a3 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p + 16), a3);
                    a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p + 24), a3);
                }
                j->y[(size_t)(i + 0) * (size_t)N + r] = hsum256(a0);
                j->y[(size_t)(i + 1) * (size_t)N + r] = hsum256(a1);
                j->y[(size_t)(i + 2) * (size_t)N + r] = hsum256(a2);
                j->y[(size_t)(i + 3) * (size_t)N + r] = hsum256(a3);
            }

            /* Tail rows of the tile: one accumulator each, same shape. */
            for (; i < T; ++i) {
                const float *x0 = j->x + (size_t)i * (size_t)K;
                __m256 acc = _mm256_setzero_ps();
                uint64_t b;

                for (b = 0; b < nb; ++b) {
                    const uint8_t *q  = blk + b * 34;
                    const int8_t  *qs = (const int8_t *)(q + 2);
                    const float   *p  = x0 + (size_t)b * 32u;
                    __m256 vd = _mm256_set1_ps(nita_f16_to_f32((uint16_t)(q[0] | (q[1] << 8))));

                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                              _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs))))),
                              _mm256_loadu_ps(p), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                              _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 8))))),
                              _mm256_loadu_ps(p + 8), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                              _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 16))))),
                              _mm256_loadu_ps(p + 16), acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(
                              _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 24))))),
                              _mm256_loadu_ps(p + 24), acc);
                }
                j->y[(size_t)i * (size_t)N + r] = hsum256(acc);
            }
        }
        return;
    }
#endif

    /* Generic path: dequantise the row once, then dot it with each token. */
    for (r = j->lo; r < j->hi; ++r) {
        const uint8_t *src = t->data + (size_t)r * j->row_bytes;
        int            i;

        if (nita_dequant_row(src, t->type, t->ne0, j->rowbuf) != 0) {
            for (i = 0; i < T; ++i) j->y[(size_t)i * (size_t)N + r] = 0.0f;
            continue;
        }
        for (i = 0; i < T; ++i) {
#if defined(__AVX2__)
            j->y[(size_t)i * (size_t)N + r] =
                fdot_avx2(j->rowbuf, j->x + (size_t)i * (size_t)K, t->ne0);
#else
            const float *xi = j->x + (size_t)i * (size_t)K;
            float        sum = 0.0f;
            uint64_t     k;

            for (k = 0; k < t->ne0; ++k) sum += j->rowbuf[k] * xi[k];
            j->y[(size_t)i * (size_t)N + r] = sum;
#endif
        }
    }
}

/* ------------------------------------------------------------------ */
/* Row-parallel worker pool.                                           */
/*                                                                     */
/* One implementation for every platform. Workers are persistent, spin  */
/* briefly on a generation counter, then park; the only platform seams  */
/* are the wait primitive (Win32 events, or a pthread mutex/cond pair   */
/* on POSIX) and thread create/join. The stall discipline, the          */
/* claim-in-blocks row stealing and every ordering rule are identical   */
/* on both.                                                             */
/* ------------------------------------------------------------------ */
struct mt_pool;
typedef struct { struct mt_pool *pool; int id; } mt_arg_t;

typedef struct mt_pool {
    int           n;          /* total participants, master included */
    int           spin;       /* pause iterations before blocking; 0 = never */
    volatile LONG gen;
    volatile LONG completed;
    volatile LONG have_job;    /* job is valid for the current gen */
    volatile LONG shutdown;
    volatile LONG waiting[NITA_MAX_THREADS];
    mv_range_t    job;

    /* A job is either a matvec row range (kind 0, all of job is used) or an
     * attention row range (kind 1, only job.lo/job.hi are used and the rest of
     * the description lives here). Written before the generation bump, so the
     * interlocked increment publishes it to the workers. */
    volatile LONG kind;
    nita_model_t *att_m;
    int           att_layer;
    int           att_pos0;
    float        *rowbufs[NITA_MAX_THREADS];
    /* kind 2: a generic range job published by nita_parallel_for */
    nita_range_fn fn;
    void         *fn_ctx;
#if defined(_WIN32)
    HANDLE        th[NITA_MAX_THREADS];
    HANDLE        ev[NITA_MAX_THREADS];
#else
    pthread_t       th[NITA_MAX_THREADS];
    pthread_mutex_t mu[NITA_MAX_THREADS];
    pthread_cond_t  cv[NITA_MAX_THREADS];
#endif
    mt_arg_t      args[NITA_MAX_THREADS];

    /* Matvec rows are claimed in blocks rather than split once up front: a
     * participant that falls behind (memory latency, a preempted thread, a
     * slower CCD) stops being the critical path, because the others keep
     * taking blocks. Only assignment changes - every row is still reduced in
     * its own order - so results stay bit-identical to the fixed split. */
    volatile LONG row_next;
    volatile LONG row_end;
    volatile LONG row_step;

} mt_pool_t;

/* Atomic helpers with identical semantics on both platforms. InterlockedAdd
 * returns the NEW value; __atomic_add_fetch matches that, unlike the
 * fetch-add form. */
#if defined(_WIN32)
static LONG pool_atomic_add(volatile LONG *p, LONG v) { return InterlockedAdd(p, v); }
static void pool_atomic_store(volatile LONG *p, LONG v) { InterlockedExchange(p, v); }
static LONG pool_atomic_inc(volatile LONG *p) { return InterlockedIncrement(p); }
#else
static LONG pool_atomic_add(volatile LONG *p, LONG v) { return __atomic_add_fetch(p, v, __ATOMIC_SEQ_CST); }
static void pool_atomic_store(volatile LONG *p, LONG v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
static LONG pool_atomic_inc(volatile LONG *p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
#endif

/* Ask one worker to sleep until the next generation (or a spurious wake,
 * bounded to 5 ms). The waiting[] publication and the generation re-check
 * both happen before/after the block respectively, so a job published to an
 * about-to-sleep worker cannot be lost: it will re-check gen under the same
 * protocol before its first sleep. */
static void worker_block(mt_pool_t *pool, int id, LONG seen)
{
#if defined(_WIN32)
    pool_atomic_store(&pool->waiting[id], 1);
    if (pool->gen == seen) WaitForSingleObject(pool->ev[id], 5);
    pool_atomic_store(&pool->waiting[id], 0);
#else
    struct timespec ts;

    /* Publishing intent before locking (both sides ordered by an atomic op)
     * is what stops a lost wake-up here. */
    pool_atomic_store(&pool->waiting[id], 1);
    pthread_mutex_lock(&pool->mu[id]);
    while (pool->gen == seen && !pool->shutdown) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 5L * 1000 * 1000;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec += 1; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&pool->cv[id], &pool->mu[id], &ts);
    }
    pthread_mutex_unlock(&pool->mu[id]);
    pool_atomic_store(&pool->waiting[id], 0);
#endif
}

/* Wake one worker if (and only if) it is parked. The gen bump already
 * precedes this call, so an unparked worker sees the job on its own and a
 * signal to a parked worker is always matched by a generation re-check. */
static void worker_wake(mt_pool_t *pool, int id)
{
#if defined(_WIN32)
    if (pool->waiting[id]) SetEvent(pool->ev[id]);
#else
    if (pool->waiting[id]) {
        pthread_mutex_lock(&pool->mu[id]);
        pthread_cond_signal(&pool->cv[id]);
        pthread_mutex_unlock(&pool->mu[id]);
    }
#endif
}

/* The pool model_open builds, visible to code that has no model handle (the
 * qwen35 mixer) but must run inside the same threads. Cleared when the model
 * that owns it is freed. */
static mt_pool_t *g_pool;

#if defined(_WIN32)
#  define NITA_WORKER_RET 0
static DWORD WINAPI mt_worker(LPVOID p)
#else
#  define NITA_WORKER_RET NULL
static void *mt_worker(void *p)
#endif
{
    mt_arg_t  *a    = (mt_arg_t *)p;
    mt_pool_t *pool = a->pool;
    int        id   = a->id;
    LONG       seen = 0;

    for (;;) {
        LONG spins = 0;

        if (pool->shutdown) return NITA_WORKER_RET;

        /* Spin briefly - back-to-back matvecs arrive with no gap at all - and
         * then park, so the other cores are not burnt while the master runs
         * the serial parts of a layer. */
        while (pool->gen == seen) {
            if (pool->shutdown) return NITA_WORKER_RET;
            if (++spins < pool->spin) { NITA_PAUSE(); continue; }
            worker_block(pool, id, seen);
            if (pool->shutdown) return NITA_WORKER_RET;
        }
        seen = pool->gen;

        /* Teardown bumps the generation purely to wake us. Never run a job the
         * master has invalidated: its buffers may already be freed. */
        if (pool->shutdown || !pool->have_job) continue;

        if (pool->kind == 2 && pool->row_step > 0) {
            for (;;) {
                LONG lo = pool_atomic_add(&pool->row_next, pool->row_step) - pool->row_step;
                LONG hi;

                if (lo >= pool->row_end) break;
                hi = lo + pool->row_step;
                if (hi > pool->row_end) hi = pool->row_end;
                pool->fn(pool->fn_ctx, (int)lo, (int)hi, id);
            }
        } else if (pool->kind == 0 && pool->row_step > 0) {
            mv_range_t j = pool->job;

            j.rowbuf = pool->rowbufs[id];
            for (;;) {
                LONG lo = pool_atomic_add(&pool->row_next, pool->row_step) - pool->row_step;
                LONG hi;

                if (lo >= pool->row_end) break;
                hi = lo + pool->row_step;
                if (hi > pool->row_end) hi = pool->row_end;
                j.lo = (uint64_t)lo;
                j.hi = (uint64_t)hi;
                mv_rows(&j);
            }
        } else {
            mv_range_t j = pool->job;
            uint64_t   total = j.hi - j.lo;
            uint64_t   lo = j.lo + total * (uint64_t)id / (uint64_t)pool->n;
            uint64_t   hi = j.lo + total * (uint64_t)(id + 1) / (uint64_t)pool->n;

            if (pool->kind == 2) {
                pool->fn(pool->fn_ctx, (int)lo, (int)hi, id);
            } else if (pool->kind) {
                attention_rows(pool->att_m, pool->att_layer, pool->att_pos0,
                               (int)lo, (int)hi);
            } else {
                j.lo = lo;
                j.hi = hi;
                j.rowbuf = pool->rowbufs[id];
                mv_rows(&j);
            }
        }
        pool_atomic_inc(&pool->completed);
    }
}

/* NITA_MV_STEAL=0 restores the fixed row split, for A/B measurement. */
static int steal_mode(void)
{
    static int mode = -1;

    if (mode < 0) {
        const char *e = getenv("NITA_MV_STEAL");
        mode = (e && *e == '0') ? 0 : 1;
    }
    return mode;
}

static mt_pool_t *pool_create(int n, size_t rowcap)
{
    mt_pool_t *pool;
    int        i;

    if (n > NITA_MAX_THREADS) n = NITA_MAX_THREADS;
    if (n < 1) n = 1;
    pool = (mt_pool_t *)nita_calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->n = n;

    /* How long a worker spins before parking. Back-to-back matvecs have no
     * gap, so a short spin is nearly free; the knob exists because the right
     * value is machine and workload dependent.
     * NITA_SPIN=0 means never block (maximum throughput, burns the cores). */
    {
        const char *e = getenv("NITA_SPIN");
        long        v = e ? strtol(e, NULL, 10) : 1000;
        if (v <= 0) v = 0x7FFFFFFF;
        if (v > 0x7FFFFFFF) v = 0x7FFFFFFF;
        pool->spin = (int)v;
    }

    for (i = 0; i < n; ++i) {
        pool->rowbufs[i] = (float *)nita_calloc(rowcap, sizeof(float));
        if (!pool->rowbufs[i]) { pool->n = 1; break; }
    }
    for (i = 1; i < n; ++i) {
        pool->args[i].pool = pool;
        pool->args[i].id   = i;
#if defined(_WIN32)
        pool->ev[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
        if (!pool->ev[i]) { pool->n = i; break; }
        pool->th[i] = CreateThread(NULL, 0, mt_worker, &pool->args[i], 0, NULL);
        if (!pool->th[i]) { pool->n = i; break; }   /* degrade gracefully */
#else
        if (pthread_mutex_init(&pool->mu[i], NULL) != 0) { pool->n = i; break; }
        if (pthread_cond_init(&pool->cv[i], NULL) != 0) {
            pthread_mutex_destroy(&pool->mu[i]);
            pool->n = i;
            break;
        }
        if (pthread_create(&pool->th[i], NULL, mt_worker, &pool->args[i]) != 0) {
            pthread_mutex_destroy(&pool->mu[i]);
            pthread_cond_destroy(&pool->cv[i]);
            pool->n = i;
            break;                                  /* degrade gracefully */
        }
#endif
    }
    return pool;
}

static void pool_destroy(mt_pool_t *pool)
{
    int i;
    if (!pool) return;
    if (g_pool == pool) g_pool = NULL;

    pool_atomic_store(&pool->have_job, 0);   /* no stale job may run */
    pool_atomic_store(&pool->shutdown, 1);
    pool_atomic_inc(&pool->gen);

    for (i = 1; i < pool->n; ++i) {
#if defined(_WIN32)
        if (pool->ev[i]) SetEvent(pool->ev[i]);
        if (pool->th[i]) {
            WaitForSingleObject(pool->th[i], 2000);
            CloseHandle(pool->th[i]);
        }
#else
        pthread_mutex_lock(&pool->mu[i]);
        pthread_cond_signal(&pool->cv[i]);
        pthread_mutex_unlock(&pool->mu[i]);
        pthread_join(pool->th[i], NULL);
        pthread_mutex_destroy(&pool->mu[i]);
        pthread_cond_destroy(&pool->cv[i]);
#endif
    }
#if defined(_WIN32)
    for (i = 1; i < pool->n; ++i)
        if (pool->ev[i]) CloseHandle(pool->ev[i]);
#endif
    for (i = 0; i < pool->n; ++i) free(pool->rowbufs[i]);
    free(pool);
}

int nita_thread_count(void)
{
    return g_pool ? g_pool->n : 1;
}

float *nita_thread_rowbuf(int thread)
{
    if (!g_pool || thread < 0 || thread >= g_pool->n) return NULL;
    return g_pool->rowbufs[thread];
}

void nita_parallel_for(int n, nita_range_fn fn, void *ctx)
{
    nita_parallel_for_min(n, 64, fn, ctx);
}

void nita_parallel_for_min(int n, int min_items, nita_range_fn fn, void *ctx)
{
    mt_pool_t *pool = g_pool;
    int        i;

    if (!fn || n <= 0) return;

    /* Small ranges lose more to the barrier than they save in parallelism. */
    if (!pool || pool->n <= 1 || n < min_items) {
        fn(ctx, 0, n, 0);
        return;
    }

    pool->completed = 0;
    pool->kind   = 2;
    pool->fn     = fn;
    pool->fn_ctx = ctx;
    pool->job.lo = 0;
    pool->job.hi = (uint64_t)n;
    /* Same claim-in-blocks scheme as the matvec rows. These ranges hold up to
     * a few thousand uniform items, so the blocks are bigger, but the reason is
     * the same: a fixed n-way split charges the whole phase for whichever
     * participant is slowest. Published before the bump like the rest. */
    {
        int step = n / (pool->n * 4);

        if (step < 1)  step = 1;
        if (step > 64) step = 64;
        pool->row_step = step;
        pool->row_end  = n;
        pool->row_next = 0;
    }
    pool_atomic_store(&pool->have_job, 1);
    pool_atomic_inc(&pool->gen);          /* full barrier: job visible */

    for (i = 1; i < pool->n; ++i) worker_wake(pool, i);

    for (;;) {
        LONG lo = pool_atomic_add(&pool->row_next, pool->row_step) - pool->row_step;
        LONG hi;

        if (lo >= pool->row_end) break;
        hi = lo + pool->row_step;
        if (hi > pool->row_end) hi = pool->row_end;
        fn(ctx, (int)lo, (int)hi, 0);
    }
    while (pool->completed < pool->n - 1) NITA_PAUSE();
}

/* Split the row range across the pool; the calling thread takes slice 0. */
static void mv_dispatch(mt_pool_t *pool, mv_range_t *job)
{
    int i;

    if (!pool || pool->n <= 1 || (job->hi - job->lo) < 32) {
        mv_rows(job);
        return;
    }

    pool->completed = 0;
    pool->kind = 0;
    pool->job = *job;                          /* published before the bump */
    if (steal_mode()) {
        LONG total = (LONG)(job->hi - job->lo);
        LONG step  = total / (LONG)(pool->n * 4);
        if (step < 8)  step = 8;
        if (step > 64) step = 64;
        pool->row_step = step;
        pool->row_end  = (LONG)job->hi;
        pool->row_next = (LONG)job->lo;
    } else {
        pool->row_step = 0;
    }
    pool_atomic_store(&pool->have_job, 1);
    pool_atomic_inc(&pool->gen);          /* full barrier: job visible */

    for (i = 1; i < pool->n; ++i) worker_wake(pool, i);

    if (pool->row_step > 0) {
        mv_range_t mine = *job;

        for (;;) {
            LONG lo = pool_atomic_add(&pool->row_next, pool->row_step) - pool->row_step;
            LONG hi;

            if (lo >= pool->row_end) break;
            hi = lo + pool->row_step;
            if (hi > pool->row_end) hi = pool->row_end;
            mine.lo = (uint64_t)lo;
            mine.hi = (uint64_t)hi;
            mv_rows(&mine);
        }
    } else {
        mv_range_t mine = *job;
        uint64_t   total = job->hi - job->lo;
        mine.hi = job->lo + total / (uint64_t)pool->n;
        mv_rows(&mine);
    }
    while (pool->completed < pool->n - 1) NITA_PAUSE();
}
/*
 * Attention for tile rows [lo,hi) across the pool.
 *
 * Rows are independent: each owns one score row of m->pscores and its own
 * slice of m->pa, so the same row-range split the matvec kernel uses is safe
 * here with no locking and no extra buffers. Splitting by row rather than by
 * head is what keeps that true - two heads working on one row would collide
 * on its scores.
 */
static void att_dispatch(nita_model_t *m, int layer, int pos0, int ncol)
{
    static int seqmode = -1;                   /* master thread only */
    mt_pool_t *pool = (mt_pool_t *)m->pool;
    int        i;

    /* NITA_ATTN_SEQ=1 keeps attention on the calling thread. It is the A/B
     * baseline for this dispatch and the fallback if a machine behaves badly
     * with one more barrier per layer. */
    if (seqmode < 0) {
        const char *e = getenv("NITA_ATTN_SEQ");

        seqmode = (e && *e && *e != '0') ? 1 : 0;
    }
    if (seqmode || !pool || pool->n <= 1 || ncol < 2) {
        attention_rows(m, layer, pos0, 0, ncol);
        return;
    }

    pool->completed = 0;
    pool->kind      = 1;
    pool->att_m     = m;
    pool->att_layer = layer;
    pool->att_pos0  = pos0;
    pool->job.lo    = 0;
    pool->job.hi    = (uint64_t)ncol;          /* published before the bump */
    pool_atomic_store(&pool->have_job, 1);
    pool_atomic_inc(&pool->gen);          /* full barrier: job visible */

    for (i = 1; i < pool->n; ++i) worker_wake(pool, i);

    /* The calling thread takes slice 0, exactly as mv_dispatch does. With a
     * tile of 8 rows and 16 participants that slice is empty, which is fine:
     * the workers' slices tile [0, ncol) between them. */
    attention_rows(m, layer, pos0, 0, (int)((uint64_t)ncol / (uint64_t)pool->n));
    while (pool->completed < pool->n - 1) NITA_PAUSE();
}

/* Integer kernels are the default; NITA_INT8_DOT=0 restores the
 * dequantise-then-float path for an A/B on the same binary. */
static int int8_dot_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *e = getenv("NITA_INT8_DOT");

        on = !(e && *e && *e == '0');
    }
    return on;
}

/* Blocks of quantised activations, grown on demand. Called from the master
 * thread only, before a job is published, so the workers never allocate. */
static nita_aq8_t *aq_scratch(nita_model_t *m, size_t blocks)
{
    if (blocks > m->qact_cap) {
        nita_aq8_t *p = (nita_aq8_t *)nita_calloc(blocks, sizeof *p);

        if (!p) return NULL;
        free(m->qact);
        m->qact     = p;
        m->qact_cap = blocks;
    }
    return m->qact;
}

static int matvec(nita_model_t *m, const nita_tensor_t *t,
                  const float *x, float *y, int expect_in, int expect_out,
                  prof_t *slot)
{
    mv_range_t job;

    if (!bound_in_file(NULL, t)) return -1;
    if ((int)t->ne0 != expect_in || (int)t->ne1 != expect_out) return -2;

    job.aq        = NULL;
    job.aq_stride = 0;
    if (int8_dot_on() && (t->ne0 % 32u) == 0 && nita_quant_has_int_dot(t->type)) {
        int         nb = (int)(t->ne0 / 32u);
        nita_aq8_t *aq = aq_scratch(m, (size_t)nb);

        if (aq) {
            double _t0 = prof_now();

            nita_aq8_quantise(x, (int)t->ne0, aq);
            prof_add(&P_aq8, prof_now() - _t0, 1);
            job.aq        = aq;
            job.aq_stride = nb;
        }
    }

    job.t         = t;
    job.x         = x;
    job.y         = y;
    job.rowbuf    = m->rowbuf;
    job.row_bytes = nita_row_bytes(t->type, t->ne0);
    job.lo        = 0;
    job.hi        = t->ne1;
    job.ncol      = 0;
    job.kstride   = 0;
    job.ystride   = 0;

    {
        double _t0 = prof_now();
        double dt;

        mv_dispatch((mt_pool_t *)m->pool, &job);
        dt = prof_now() - _t0;
        prof_add(&P_proj, dt, (long)t->ne1);
        if (slot) prof_add(slot, dt, 1);
        prof_kernel(t->type, dt);
    }
    return 0;
}

/* Y[i] = W . X[i] for a whole tile of rows: the same job as matvec, except the
 * weight row is dequantised once and used for every token in the tile. */
static int matvec_tile(nita_model_t *m, const nita_tensor_t *t,
                       const float *X, int ncol, float *Y,
                       int expect_in, int expect_out, prof_t *slot)
{
    mv_range_t job;

    if (!bound_in_file(NULL, t)) return -1;
    if ((int)t->ne0 != expect_in || (int)t->ne1 != expect_out) return -2;
    if (ncol <= 0) return -3;

    job.aq        = NULL;
    job.aq_stride = 0;
    if (int8_dot_on() && (t->ne0 % 32u) == 0 && nita_quant_has_int_dot(t->type)) {
        int         nb = (int)(t->ne0 / 32u);
        nita_aq8_t *aq = aq_scratch(m, (size_t)nb * (size_t)ncol);

        if (aq) {
            double _t0 = prof_now();
            int    i;

            for (i = 0; i < ncol; ++i)
                nita_aq8_quantise(X + (size_t)i * (size_t)expect_in, (int)t->ne0,
                                  aq + (size_t)i * (size_t)nb);
            prof_add(&P_aq8, prof_now() - _t0, (long)ncol);
            job.aq        = aq;
            job.aq_stride = nb;
        }
    }

    job.t         = t;
    job.x         = X;
    job.y         = Y;
    job.rowbuf    = m->rowbuf;
    job.row_bytes = nita_row_bytes(t->type, t->ne0);
    job.lo        = 0;
    job.hi        = t->ne1;
    job.ncol      = ncol;
    job.kstride   = expect_in;
    job.ystride   = expect_out;


    {
        double _t0 = prof_now();
        double dt;

        mv_dispatch((mt_pool_t *)m->pool, &job);
        dt = prof_now() - _t0;
        prof_add(&P_proj, dt, (long)t->ne1 * (long)ncol);
        if (slot) prof_add(slot, dt, 1);
        prof_kernel(t->type, dt);
    }
    return 0;
}


/* y += dequantised bias vector b (1-D). Absent bias is not an error. */
static int add_bias(nita_model_t *m, const nita_tensor_t *b, float *y, int n)
{
    if (!b || !b->bound) return 0;
    if (load_vector(b, m->biasbuf, n) != 0) return -1;
    for (int i = 0; i < n; ++i) y[i] += m->biasbuf[i];
    return 0;
}

/* One expert's slice of a packed 3-D expert weight. */
static int matvec_expert(nita_model_t *m, const nita_tensor_t *t, int e,
                         const float *x, float *y, int expect_in, int expect_out,
                         prof_t *slot)
{
    nita_tensor_t v;
    size_t        slice;

    if (!bound_in_file(NULL, t)) return -1;
    if ((int)t->ne0 != expect_in || (int)t->ne1 != expect_out) return -2;

    slice = (size_t)t->ne1 * nita_row_bytes(t->type, t->ne0);
    v     = *t;
    v.data = t->data + (size_t)e * slice;
    v.ne2  = 1;
    return matvec(m, &v, x, y, expect_in, expect_out, slot);
}

/* Same for the tiled (batched-prefill) kernel: X holds ncol rows of
 * expect_in floats, Y ncol rows of expect_out floats. */
static int matvec_tile_expert(nita_model_t *m, const nita_tensor_t *t, int e,
                              const float *X, int ncol, float *Y,
                              int expect_in, int expect_out, prof_t *slot)
{
    nita_tensor_t v;
    size_t        slice;

    if (!bound_in_file(NULL, t)) return -1;
    if ((int)t->ne0 != expect_in || (int)t->ne1 != expect_out) return -2;
    if (ncol <= 0) return 0;

    slice = (size_t)t->ne1 * nita_row_bytes(t->type, t->ne0);
    v     = *t;
    v.data = t->data + (size_t)e * slice;
    v.ne2  = 1;
    return matvec_tile(m, &v, X, ncol, Y, expect_in, expect_out, slot);
}

/* Top-k expert selection from a probs vector. Shared by the sequential and
 * tiled MoE paths so they can never disagree about ties or ordering. */
static void moe_topk(const float *probs, int E, int K,
                     int *sel, float *sel_w, unsigned char *taken)
{
    int j, e;

    memset(taken, 0, (size_t)E);
    for (j = 0; j < K; ++j) {
        int best = -1;

        for (e = 0; e < E; ++e) {
            if (taken[e]) continue;
            if (best < 0 || probs[e] > probs[best]) best = e;
        }
        sel[j]  = best;
        taken[best] = 1;
        sel_w[j] = probs[best];
    }
}

/* Expert-weight post-processing: renormalise when the arch calls for it
 * (llama-arch MoE), then apply expert_weights_scale when it is set. */
static void moe_weight_finish(nita_model_t *m, float *sel_w, int K)
{
    if (m->moe_norm_w) {
        float s = 0.0f;
        int   j;

        for (j = 0; j < K; ++j) s += sel_w[j];
        /* llama.cpp clamps the divisor to the smallest F16 normal. */
        if (s < 6.103515625e-5f) s = 6.103515625e-5f;
        for (j = 0; j < K; ++j) sel_w[j] /= s;
    }
    if (m->exp_w_scale != 0.0f && m->exp_w_scale != 1.0f) {
        int j;

        for (j = 0; j < K; ++j) sel_w[j] *= m->exp_w_scale;
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps)
{
    double ss = 0.0;
    float  scale;
    double _t0 = prof_now();

#if defined(__AVX2__)
    {
        /* Same double-precision accumulation, four doubles per lane pair,
         * two independent accumulators; reorder is below the double rounding
         * noise the scalar form already carries. */
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        int     i2   = 0;

        for (; i2 + 16 <= n; i2 += 16) {
            __m256  xf0 = _mm256_loadu_ps(x + i2);
            __m256  xf1 = _mm256_loadu_ps(x + i2 + 8);
            __m256d a0  = _mm256_cvtps_pd(_mm256_castps256_ps128(xf0));
            __m256d b0  = _mm256_cvtps_pd(_mm256_extractf128_ps(xf0, 1));
            __m256d a1  = _mm256_cvtps_pd(_mm256_castps256_ps128(xf1));
            __m256d b1  = _mm256_cvtps_pd(_mm256_extractf128_ps(xf1, 1));

            acc0 = _mm256_fmadd_pd(a0, a0, acc0);
            acc0 = _mm256_fmadd_pd(b0, b0, acc0);
            acc1 = _mm256_fmadd_pd(a1, a1, acc1);
            acc1 = _mm256_fmadd_pd(b1, b1, acc1);
        }
        {
            __m256d acc = _mm256_add_pd(acc0, acc1);
            __m128d s   = _mm_hadd_pd(_mm256_castpd256_pd128(acc),
                                      _mm256_extractf128_pd(acc, 1));
            ss         += _mm_cvtsd_f64(s) + _mm_cvtsd_f64(_mm_unpackhi_pd(s, s));
        }
        for (; i2 < n; ++i2) ss += (double)x[i2] * (double)x[i2];
    }
#else
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
#endif
    ss /= (double)n;
    scale = 1.0f / sqrtf((float)ss + eps);
#if defined(__AVX2__)
    {
        const __m256 vs = _mm256_set1_ps(scale);
        int          i3 = 0;

        for (; i3 + 8 <= n; i3 += 8) {
            __m256 vx = _mm256_loadu_ps(x + i3);
            __m256 vw = _mm256_loadu_ps(w + i3);

            _mm256_storeu_ps(out + i3, _mm256_mul_ps(_mm256_mul_ps(vx, vs), vw));
        }
        for (; i3 < n; ++i3) out[i3] = x[i3] * scale * w[i3];
    }
#else
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
#endif
    prof_add(&P_norm, prof_now() - _t0, 1);
}

static float silu(float x)
{
    return x * (1.0f / (1.0f + expf(-x)));
}

/*
 * ff[i] = silu(gate[i]) * up[i], vectorised.
 *
 * This loop was 14% of a token (445 us per call) purely because expf() is
 * called once per element and nothing overlaps its latency. exp2 is rebuilt
 * from an integer and a fractional part so eight lanes run at once:
 *
 *     t = -x*log2e,  k = round(t),  r = t - k
 *     exp(-x) = 2^r * 2^k
 *
 * 2^r comes from a degree-5 minimax polynomial (relative error under 1e-8,
 * i.e. below float rounding) and 2^k is produced by writing the exponent
 * field directly. t is clamped to +-126 so an extreme activation saturates
 * the sigmoid instead of corrupting the exponent field.
 */
#if defined(__AVX2__) && !defined(NITA_NO_SILU_FAST)
/*
 * exp(x) on eight lanes.
 *
 * x*log2e is split into an integer part k and a fraction r in [-0.5,0.5];
 * 2^r comes from a degree-5 minimax polynomial (relative error below 1e-8,
 * i.e. under float rounding) and 2^k is produced by writing the exponent
 * field directly. The input is clamped so the exponent field can never be
 * corrupted; the callers here only use it where saturating is harmless.
 */
static float hmax256(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 m  = _mm_max_ps(lo, hi);

    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    return _mm_cvtss_f32(m);
}

static __m256 vexp256(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504f));
    __m256 k, r, p;

    t = _mm256_min_ps(_mm256_max_ps(t, _mm256_set1_ps(-126.0f)),
                                   _mm256_set1_ps(126.0f));
    k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    r = _mm256_sub_ps(t, k);

    p = _mm256_fmadd_ps(_mm256_set1_ps(1.33335581e-3f), r,
                        _mm256_set1_ps(9.61812911e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.55041087e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(2.40226507e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(6.93147181e-1f));
    p = _mm256_fmadd_ps(p, r, one);

    return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(
               _mm256_add_epi32(_mm256_cvtps_epi32(k),
                                _mm256_set1_epi32(127)), 23)));
}

static void silu_mul(float *dst, const float *g, const float *u, int n)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 x  = _mm256_loadu_ps(g + i);
        __m256 e  = vexp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
        __m256 sg = _mm256_div_ps(one, _mm256_add_ps(one, e));

        _mm256_storeu_ps(dst + i,
                         _mm256_mul_ps(_mm256_mul_ps(x, sg), _mm256_loadu_ps(u + i)));
    }
    for (; i < n; ++i) dst[i] = silu(g[i]) * u[i];
}

void nita_silu_mul(float *dst, const float *g, const float *u, int n)
{
    silu_mul(dst, g, u, n);
}

/* The mixer needs the same kernel twice: SiLU in place over a convolution,
 * and a gated multiply. Both go through vexp256 above so there is one vector
 * exp in the engine rather than one per caller. */
void nita_silu_inplace(float *v, int n)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;

    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(v + i);
        const __m256 e = vexp256(_mm256_sub_ps(_mm256_setzero_ps(), x));

        _mm256_storeu_ps(v + i,
                         _mm256_mul_ps(x, _mm256_div_ps(
                             one, _mm256_add_ps(one, e))));
    }
    for (; i < n; ++i) v[i] = silu(v[i]);
}

/* sum of exp((v[i]-bias)*scale) over n elements, eight lanes at a time. */
static double exp_total(const float *v, int n, float bias, float scale)
{
    const __m256 b   = _mm256_set1_ps(bias);
    const __m256 s   = _mm256_set1_ps(scale);
    __m256       acc = _mm256_setzero_ps();
    float        lanes[8];
    double       tot = 0.0;
    int          i = 0, l;

    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, vexp256(_mm256_mul_ps(
                  _mm256_sub_ps(_mm256_loadu_ps(v + i), b), s)));

    _mm256_storeu_ps(lanes, acc);
    for (l = 0; l < 8; ++l) tot += (double)lanes[l];
    for (; i < n; ++i) tot += (double)expf((v[i] - bias) * scale);
    return tot;
}
#else
void nita_silu_inplace(float *v, int n)
{
    int i;

    for (i = 0; i < n; ++i) v[i] = silu(v[i]);
}

void nita_silu_mul(float *dst, const float *g, const float *u, int n)
{
    int i;

    for (i = 0; i < n; ++i) dst[i] = silu(g[i]) * u[i];
}

static void silu_mul(float *dst, const float *g, const float *u, int n)
{
    int i;
    for (i = 0; i < n; ++i) dst[i] = silu(g[i]) * u[i];
}

static double exp_total(const float *v, int n, float bias, float scale)
{
    double tot = 0.0;
    int    i;
    for (i = 0; i < n; ++i) tot += (double)expf((v[i] - bias) * scale);
    return tot;
}
#endif

static void softmax_inplace(float *v, int n)
{
    float maxv;
    float sum = 0.0f;
    int   i;

    if (n <= 0) return;

#if defined(__AVX2__) && !defined(NITA_NO_SILU_FAST)
    /*
     * The scalar version calls libm expf once per score. Measured at 512
     * context that was 174 cycles an element and 82% of the entire attention
     * pass - far more than the q.k dots it runs on. After the max subtraction
     * every input is <= 0, so vexp256's clamping never triggers and its
     * degree-5 minimax exp is exact to well under float rounding here.
     * Gated on the same switch as silu_mul so the exact-math build still gets
     * the libm path on both.
     */
    {
        __m256 mx  = _mm256_set1_ps(v[0]);
        __m256 acc = _mm256_setzero_ps();
        __m256 b, rs;

        for (i = 0; i + 8 <= n; i += 8)
            mx = _mm256_max_ps(mx, _mm256_loadu_ps(v + i));
        maxv = hmax256(mx);
        for (; i < n; ++i) if (v[i] > maxv) maxv = v[i];

        b = _mm256_set1_ps(maxv);
        for (i = 0; i + 8 <= n; i += 8) {
            const __m256 e = vexp256(_mm256_sub_ps(_mm256_loadu_ps(v + i), b));

            _mm256_storeu_ps(v + i, e);
            acc = _mm256_add_ps(acc, e);
        }
        sum = hsum256(acc);
        for (; i < n; ++i) { v[i] = expf(v[i] - maxv); sum += v[i]; }
        if (sum <= 0.0f) sum = 1.0f;

        rs = _mm256_set1_ps(1.0f / sum);
        for (i = 0; i + 8 <= n; i += 8)
            _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), rs));
        for (; i < n; ++i) v[i] *= 1.0f / sum;
        return;
    }
#else
    maxv = v[0];
    for (i = 1; i < n; ++i) if (v[i] > maxv) maxv = v[i];
    for (i = 0; i < n; ++i) { v[i] = expf(v[i] - maxv); sum += v[i]; }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < n; ++i) v[i] /= sum;
#endif
}

/*
 * Rotary position embedding, adjacent-pair layout.
 *
 * GGUF llama checkpoints are stored with Q/K rows permuted by the converter
 * precisely so that this interleaved form (rotating x[2i] against x[2i+1])
 * matches the original model.
 */
/* Fallback: compute the angles for the current position on the spot. */
static void rope_apply_slow(float *vec, int n_heads, int head_dim, int n_rot,
                            int pos, float theta_base)
{
    int h, i;

    for (h = 0; h < n_heads; ++h) {
        float *p = vec + (size_t)h * (size_t)head_dim;
        for (i = 0; i < n_rot / 2; ++i) {
            float freq  = powf(theta_base, -2.0f * (float)i / (float)n_rot);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = p[2 * i];
            float x1 = p[2 * i + 1];
            p[2 * i]     = x0 * c - x1 * s;
            p[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/*
 * Build cos/sin for every position once, at load time.
 *
 * The rotation angles depend only on (position, pair index) - not on the
 * layer, the head or the token - yet the old code recomputed powf, cosf and
 * sinf for every head of every layer of every token: 2048 transcendental
 * calls per token for 82 us of pure recomputation.
 *
 * Cost is ctx * n_rot/2 * 2 floats (512 KiB for 2048 context, 8 MiB at 32k).
 * If that is ever too large the caller leaves the table NULL and the slow
 * path above runs instead.
 */
static float *rope_build(int ctx, int n_rot, float theta_base)
{
    const int half = n_rot / 2;
    float    *tab;
    int       pos, i;

    if (ctx <= 0 || half <= 0) return NULL;
    tab = (float *)nita_calloc((size_t)ctx * (size_t)half * 2, sizeof(float));
    if (!tab) return NULL;

    for (pos = 0; pos < ctx; ++pos) {
        float *row = tab + (size_t)pos * (size_t)half * 2;
        for (i = 0; i < half; ++i) {
            float freq  = powf(theta_base, -2.0f * (float)i / (float)n_rot);
            float angle = (float)pos * freq;
            row[2 * i]     = cosf(angle);
            row[2 * i + 1] = sinf(angle);
        }
    }
    return tab;
}

/* tab may be NULL, in which case the angles are recomputed. */
static void rope_apply(const float *tab, float *vec, int n_heads, int head_dim,
                       int n_rot, int pos, float theta_base)
{
    const int half = n_rot / 2;
    const float *cs;
    int h, i;

    if (!tab) {
        rope_apply_slow(vec, n_heads, head_dim, n_rot, pos, theta_base);
        return;
    }

    cs = tab + (size_t)pos * (size_t)half * 2;
    for (h = 0; h < n_heads; ++h) {
        float *p = vec + (size_t)h * (size_t)head_dim;
        for (i = 0; i < half; ++i) {
            float c  = cs[2 * i];
            float s  = cs[2 * i + 1];
            float x0 = p[2 * i];
            float x1 = p[2 * i + 1];
            p[2 * i]     = x0 * c - x1 * s;
            p[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/*
 * Rotate-half variant, used by architectures whose ggml rope mode is NEOX or
 * MROPE (qwen35 reports IMROPE): pair p rotates (p, p + n_rot/2) instead of
 * the two adjacent elements the llama-style NORMAL mode pairs. The angle for
 * pair p is the same in both, so the table is shared.
 */
static void rope_apply_half(const float *tab, float *vec, int n_heads, int head_dim,
                            int n_rot, int pos, float theta_base)
{
    const int    half = n_rot / 2;
    const float *cs;
    int          h, i;

    if (!tab) return;   /* qwen35 always has a table built */

    cs = tab + (size_t)pos * (size_t)half * 2;
    for (h = 0; h < n_heads; ++h) {
        float *p = vec + (size_t)h * (size_t)head_dim;
        for (i = 0; i < half; ++i) {
            float c  = cs[2 * i];
            float s  = cs[2 * i + 1];
            float x0 = p[i];
            float x1 = p[i + half];

            p[i]        = x0 * c - x1 * s;
            p[i + half] = x0 * s + x1 * c;
        }
    }
    (void)theta_base;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
static void prefill_free(nita_model_t *m);

static void free_model(nita_model_t *m)
{
    int l;
    if (!m) return;

    /* Stop the workers first: everything below is memory they can reach. */
    pool_destroy((mt_pool_t *)m->pool);
    m->pool = NULL;
    free(m->qact);
    m->qact = NULL;
    m->qact_cap = 0;

    if (m->k_cache) {
        for (l = 0; l < m->n_layer; ++l) free(m->k_cache[l]);
        free(m->k_cache);
    }
    if (m->v_cache) {
        for (l = 0; l < m->n_layer; ++l) free(m->v_cache[l]);
        free(m->v_cache);
    }

    free(m->attn_norm); free(m->attn_q); free(m->attn_k); free(m->attn_v);
    free(m->attn_out);  free(m->ffn_norm); free(m->ffn_gate);
    free(m->ffn_up);    free(m->ffn_down);
    free(m->ffn_gate_inp); free(m->ffn_gate_exps); free(m->ffn_up_exps);
    free(m->ffn_down_exps);
    free(m->shexp_gate_inp); free(m->shexp_gate); free(m->shexp_up); free(m->shexp_down);
    free(m->attn_q_bias); free(m->attn_k_bias); free(m->attn_v_bias); free(m->attn_out_bias);
    free(m->acc); free(m->router); free(m->sel); free(m->sel_w); free(m->sel_taken);
    free(m->biasbuf); free(m->out_bias_f32);

    if (m->q35_state) {
        for (l = 0; l < m->n_layer; ++l)
            if (m->layer_recr && m->layer_recr[l])
                q35_gdn_state_free(&m->q35_state[l]);
        free(m->q35_state);
    }
    free(m->layer_recr); free(m->q35_w);
    free(m->post_norm);  free(m->attn_q_norm); free(m->attn_k_norm);

    for (l = 0; l < m->hd_n; ++l) nita_hadamard_free(&m->hd[l].xf);
    if (m->hd_names) {
        for (l = 0; l < m->hd_n_names; ++l) free(m->hd_names[l]);
        free(m->hd_names);
    }
    free(m->hdbuf); free(m->qg); free(m->qgate);
    free(m->qnormv); free(m->knormv); free(m->headbuf); free(m->rot);

    free(m->rowbuf); free(m->nbuf); free(m->xb); free(m->hb);
    free(m->qbuf); free(m->kbuf); free(m->vbuf); free(m->abuf);
    free(m->gate); free(m->up); free(m->ff); free(m->tmp); free(m->scores);
    free(m->rope_tab);
    prefill_free(m);

    free(m);
}

void nita_model_close(nita_model_t *m)
{
    free_model(m);
}

/* How many threads the pool should start with.
 *
 * A projection splits by row into memory-bound work, so the useful ceiling is
 * the physical core count: the SMT siblings share a core's clocks and add
 * little to a kernel that is already streaming. Measured on a 16-core/32-thread
 * part, 27B, milliseconds a token: 12 -> 152, 14 -> 149, 16 -> 150, 18 -> 152,
 * 20 -> 153, 24 -> 153, 32 -> 157.
 *
 * Which is not the same as half the logical processors. A CPU without SMT
 * reports one logical processor per core, and halving that would leave half the
 * machine idle; ask for the core count itself and fall back to the old estimate
 * only if the query fails. */
#if defined(_WIN32)
static int default_threads(int logical)
{
    DWORD                                     len = 0;
    int                                       cores = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX  *buf;

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len) &&
        GetLastError() == ERROR_INSUFFICIENT_BUFFER && len > 0) {
        buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)malloc(len);
        if (buf) {
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
                DWORD off = 0;

                while (off < len) {
                    const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e =
                        (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)((const char *)buf + off);

                    if (e->Relationship == RelationProcessorCore) cores++;
                    off += e->Size;
                }
            }
            free(buf);
        }
    }
    if (cores > 0) return cores > logical ? logical : cores;
    return logical > 1 ? logical / 2 : 1;
}
#else
/* Physical cores via sysfs topology, the same target the Windows path aims
 * at: SMT siblings share a core's issue width and add little to a
 * memory-bound matvec. Count unique (package, core) pairs; any failure
 * falls back to the old logical/2 estimate. */
static int default_threads(int logical)
{
    DIR           *d = opendir("/sys/devices/system/cpu");
    struct dirent *e;
    int            cores = 0, nseen = 0;
    int            seen_pkg[4096], seen_core[4096];

    if (!d) goto fallback;
    while ((e = readdir(d))) {
        char  path[512], buf[64];
        FILE *f;
        int   pkg = -1, core = -1, i, dup = 0;

        if (strncmp(e->d_name, "cpu", 3) != 0 || !isdigit((unsigned char)e->d_name[3]))
            continue;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/%s/topology/physical_package_id",
                 e->d_name);
        if ((f = fopen(path, "r"))) {
            if (fgets(buf, sizeof buf, f)) pkg = atoi(buf);
            fclose(f);
        }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/%s/topology/core_id", e->d_name);
        if ((f = fopen(path, "r"))) {
            if (fgets(buf, sizeof buf, f)) core = atoi(buf);
            fclose(f);
        }
        if (core < 0) continue;
        for (i = 0; i < nseen; ++i)
            if (seen_pkg[i] == pkg && seen_core[i] == core) { dup = 1; break; }
        if (!dup && nseen < (int)(sizeof seen_pkg / sizeof seen_pkg[0])) {
            seen_pkg[nseen] = pkg;
            seen_core[nseen] = core;
            ++nseen;
        }
    }
    closedir(d);
    cores = nseen;
    if (cores > 0) return cores > logical ? logical : cores;
fallback:
    return logical > 1 ? logical / 2 : 1;
}
#endif

nita_model_t *nita_model_open(const gguf_file_t *g, int ctx, char *err, size_t errlen)
{
    nita_model_t *m;
    const char   *arch;
    int           l, max_row;
    int           q_dim;

    if (err && errlen) err[0] = '\0';

    if (!g) {
        snprintf(err, errlen, "no model file");
        return NULL;
    }

    m = (nita_model_t *)nita_calloc(1, sizeof *m);
    arch = gguf_get_str(g, "general.architecture", "llama");

    m->n_layer   = (int)hparam_int(g, arch, "block_count", 0);
    m->n_embd    = (int)hparam_int(g, arch, "embedding_length", 0);
    m->n_head    = (int)hparam_int(g, arch, "attention.head_count", 0);
    m->n_head_kv = (int)hparam_int(g, arch, "attention.head_count_kv", m->n_head);
    m->n_ff      = (int)hparam_int(g, arch, "feed_forward_length", 0);
    m->rms_eps   = (float)hparam_float(g, arch, "attention.layer_norm_rms_epsilon", 1e-5);
    m->rope_theta= (float)hparam_float(g, arch, "rope.freq_base", 10000.0);

    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_head <= 0) {
        snprintf(err, errlen,
                 "not a transformer: block_count=%d embedding_length=%d head_count=%d",
                 m->n_layer, m->n_embd, m->n_head);
        free_model(m);
        return NULL;
    }
    if (m->n_head_kv <= 0) m->n_head_kv = m->n_head;
    if (m->n_head_kv > m->n_head) m->n_head_kv = m->n_head;
    {
        long klen = (long)hparam_int(g, arch, "attention.key_length", 0);
        long vlen = (long)hparam_int(g, arch, "attention.value_length", 0);

        if (klen > 0) {
            /* Explicit head width. Bonsai/qwen35 keep n_embd != n_head*head_dim
             * (27B: 5120 embedding against 24 heads of 256). */
            if (vlen > 0 && vlen != klen) {
                snprintf(err, errlen,
                         "attention.key_length %ld and value_length %ld differ; "
                         "this engine keeps one width for both", klen, vlen);
                free_model(m);
                return NULL;
            }
            m->head_dim = (int)klen;
        } else {
            if (m->n_embd % m->n_head != 0) {
                snprintf(err, errlen,
                         "embedding_length %d is not divisible by head_count %d "
                         "and attention.key_length is absent",
                         m->n_embd, m->n_head);
                free_model(m);
                return NULL;
            }
            m->head_dim = m->n_embd / m->n_head;
        }
    }

    m->kv_dim   = m->head_dim * m->n_head_kv;
    m->n_rot    = (int)hparam_int(g, arch, "rope.dimension_count", m->head_dim);
    if (m->n_rot <= 0 || m->n_rot > m->head_dim) m->n_rot = m->head_dim;
    if (m->n_rot % 2 != 0) m->n_rot -= 1;

    if (m->n_ff <= 0) {
        snprintf(err, errlen, "feed_forward_length missing");
        free_model(m);
        return NULL;
    }

    m->ctx = ctx > 0 ? ctx : 2048;

    /* Vocabulary size comes from the embedding matrix; fall back to metadata. */
    if (!bind_tensor(g, &m->tok_embd, "token_embd.weight")) {
        snprintf(err, errlen, "missing tensor 'token_embd.weight'");
        free_model(m);
        return NULL;
    }
    m->n_vocab = (int)m->tok_embd.ne1;
    if (m->n_vocab <= 0)
        m->n_vocab = (int)hparam_int(g, arch, "vocab_size", 0);
    if (m->n_vocab <= 0) {
        snprintf(err, errlen, "cannot determine vocabulary size");
        free_model(m);
        return NULL;
    }

    if (!bind_tensor(g, &m->out_norm, "output_norm.weight")) {
        snprintf(err, errlen, "missing tensor 'output_norm.weight'");
        free_model(m);
        return NULL;
    }

    m->tied_output = !bind_tensor(g, &m->out_proj, "output.weight");
    if (m->tied_output) m->out_proj = m->tok_embd;
    if (tensor_named(g, "output.bias")) {
        if (!bind_tensor(g, &m->out_bias, "output.bias")) {
            snprintf(err, errlen, "tensor 'output.bias' failed the bounds check");
            free_model(m);
            return NULL;
        }
        if ((int)m->out_bias.ne0 != m->n_vocab) {
            snprintf(err, errlen, "output.bias has %llu elements, expected %d",
                     (unsigned long long)m->out_bias.ne0, m->n_vocab);
            free_model(m);
            return NULL;
        }
        m->out_bias_f32 = (float *)nita_calloc((size_t)m->n_vocab, sizeof(float));
        if (!m->out_bias_f32 ||
            load_vector(&m->out_bias, m->out_bias_f32, m->n_vocab) != 0) {
            snprintf(err, errlen, "could not dequantise output.bias");
            free_model(m);
            return NULL;
        }
    }

    /* Per-layer tensors */
    m->attn_norm = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_norm);
    m->attn_q    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_q);
    m->attn_k    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_k);
    m->attn_v    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_v);
    m->attn_out  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_out);
    m->ffn_norm  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_norm);
    m->ffn_gate  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_gate);
    m->ffn_up    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_up);
    m->ffn_down  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_down);

    /* qwen35: which blocks are gated delta-net mixers, and their weights.
     * Upstream decides by position: a layer is recurrent unless
     * (layer + 1) % full_attention_interval == 0. */
    m->is_qwen35 = (strcmp(arch, "qwen35") == 0);
    /* MoE: expert_count in metadata (llama-arch MoE uses the llama.* keys,
     * qwen2moe uses qwen2moe.*). qwen35 hybrids are not MoE. */
    m->is_moe = !m->is_qwen35 &&
        hparam_int(g, arch, "expert_count", 0) > 0;
    if (m->is_moe) {
        m->n_expert    = (int)hparam_int(g, arch, "expert_count", 0);
        m->expert_used = (int)hparam_int(g, arch, "expert_used_count", 1);
        m->exp_w_scale = (float)hparam_float(g, arch, "expert_weights_scale", 0.0);
        /* llama.cpp hard-codes weight renormalisation for llama-arch MoE and
         * leaves it off for qwen2moe; both honour expert_weights_scale. */
        m->moe_norm_w  = (strcmp(arch, "qwen2moe") != 0) ? 1 : 0;
        if (m->expert_used <= 0 || m->expert_used > m->n_expert) {
            snprintf(err, errlen,
                     "expert_used_count %d out of range for %d experts",
                     m->expert_used, m->n_expert);
            free_model(m);
            return NULL;
        }
        m->ffn_gate_inp  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_gate_inp);
        m->ffn_gate_exps = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_gate_exps);
        m->ffn_up_exps   = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_up_exps);
        m->ffn_down_exps = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_down_exps);
        m->shexp_gate_inp= (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->shexp_gate_inp);
        m->shexp_gate    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->shexp_gate);
        m->shexp_up      = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->shexp_up);
        m->shexp_down    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->shexp_down);
        if (!m->ffn_gate_inp || !m->ffn_gate_exps || !m->ffn_up_exps ||
            !m->ffn_down_exps || !m->shexp_gate_inp || !m->shexp_gate ||
            !m->shexp_up || !m->shexp_down) {
            snprintf(err, errlen, "out of memory for MoE layer tables");
            free_model(m);
            return NULL;
        }
    }
    if (!m->is_qwen35) {
        /* Optional attention biases (Qwen2/Qwen2.5 carry q/k/v biases).
         * Bind is best-effort; absent tensors stay unbound. */
        m->attn_q_bias   = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_q_bias);
        m->attn_k_bias   = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_k_bias);
        m->attn_v_bias   = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_v_bias);
        m->attn_out_bias = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_out_bias);
        if (!m->attn_q_bias || !m->attn_k_bias || !m->attn_v_bias || !m->attn_out_bias) {
            snprintf(err, errlen, "out of memory for bias tables");
            free_model(m);
            return NULL;
        }
    }
    if (m->is_qwen35) {
        int64_t interval = hparam_int(g, arch, "full_attention_interval", 4);

        if (interval <= 0) interval = 4;
        m->full_attn_interval = (int)interval;
        m->layer_recr = (unsigned char *)nita_calloc((size_t)m->n_layer, 1);
        m->q35_w      = (q35_gdn_weights_t *)nita_calloc((size_t)m->n_layer,
                                                        sizeof *m->q35_w);
        m->q35_state  = (q35_gdn_state_t *)nita_calloc((size_t)m->n_layer,
                                                       sizeof *m->q35_state);
        m->post_norm   = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->post_norm);
        m->attn_q_norm = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_q_norm);
        m->attn_k_norm = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_k_norm);
        if (!m->layer_recr || !m->q35_w || !m->q35_state ||
            !m->post_norm || !m->attn_q_norm || !m->attn_k_norm) {
            snprintf(err, errlen, "out of memory for qwen35 layer tables");
            free_model(m);
            return NULL;
        }
        for (l = 0; l < m->n_layer; ++l)
            m->layer_recr[l] = ((l + 1) % m->full_attn_interval != 0) ? 1 : 0;
    }

    for (l = 0; l < m->n_layer; ++l) {
        char name[96];

        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", l);
        if (!bind_tensor(g, &m->attn_norm[l], name)) goto missing_layer;

        if (m->is_qwen35 && m->layer_recr[l]) {
            /* gated delta-net block: no attention tensors at all, and the
             * post-attention norm is its own tensor */
            snprintf(name, sizeof name, "blk.%d.attn_qkv.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].qkv, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.attn_gate.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].gate, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_conv1d.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].conv1d, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_alpha.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].alpha, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_beta.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].beta, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_dt.bias", l);
            if (!bind_tensor(g, &m->q35_w[l].dt_bias, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_a", l);
            if (!bind_tensor(g, &m->q35_w[l].ssm_a, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_norm.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].ssm_norm, name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ssm_out.weight", l);
            if (!bind_tensor(g, &m->q35_w[l].out, name)) goto missing_layer;
            m->q35_w[l].in_norm = m->attn_norm[l];
            snprintf(name, sizeof name, "blk.%d.post_attention_norm.weight", l);
            if (!bind_tensor(g, &m->post_norm[l], name)) goto missing_layer;
            m->q35_w[l].post_norm = m->post_norm[l];
            m->ffn_norm[l] = m->post_norm[l];      /* FFN follows the mixer */
        } else {
            snprintf(name, sizeof name, "blk.%d.attn_q.weight", l);
            if (!bind_tensor(g, &m->attn_q[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.attn_k.weight", l);
            if (!bind_tensor(g, &m->attn_k[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.attn_v.weight", l);
            if (!bind_tensor(g, &m->attn_v[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.attn_output.weight", l);
            if (!bind_tensor(g, &m->attn_out[l], name)) goto missing_layer;

            if (m->is_qwen35) {
                /* dense-attention blocks of a hybrid model carry qk norms */
                snprintf(name, sizeof name, "blk.%d.attn_q_norm.weight", l);
                if (!bind_tensor(g, &m->attn_q_norm[l], name)) goto missing_layer;
                snprintf(name, sizeof name, "blk.%d.attn_k_norm.weight", l);
                if (!bind_tensor(g, &m->attn_k_norm[l], name)) goto missing_layer;
                snprintf(name, sizeof name, "blk.%d.post_attention_norm.weight", l);
                if (!bind_tensor(g, &m->post_norm[l], name)) goto missing_layer;
                m->ffn_norm[l] = m->post_norm[l];
            } else {
                snprintf(name, sizeof name, "blk.%d.ffn_norm.weight", l);
                if (!bind_tensor(g, &m->ffn_norm[l], name)) goto missing_layer;
            }
        }
        if (!m->is_qwen35) {
            /* Optional attention biases: absent is fine, present-but-unbound
             * is fatal (never run with a silent zero-bias fallback). */
            nita_tensor_t *bias_dst[4];
            const char    *bias_name[4];
            int             bi;

            bias_dst[0] = &m->attn_q_bias[l];
            bias_dst[1] = &m->attn_k_bias[l];
            bias_dst[2] = &m->attn_v_bias[l];
            bias_dst[3] = &m->attn_out_bias[l];
            bias_name[0] = "attn_q.bias";
            bias_name[1] = "attn_k.bias";
            bias_name[2] = "attn_v.bias";
            bias_name[3] = "attn_output.bias";
            for (bi = 0; bi < 4; ++bi) {
                char bias_full[96];

                snprintf(bias_full, sizeof bias_full, "blk.%d.%s", l, bias_name[bi]);
                if (tensor_named(g, bias_full) && !bind_tensor(g, bias_dst[bi], bias_full)) {
                    snprintf(err, errlen,
                             "layer %d '%s' is present but failed the bounds check",
                             l, bias_full);
                    free_model(m);
                    return NULL;
                }
            }
        }
        if (m->is_moe) {
            snprintf(name, sizeof name, "blk.%d.ffn_gate_inp.weight", l);
            if (!bind_tensor(g, &m->ffn_gate_inp[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ffn_gate_exps.weight", l);
            if (!bind_tensor(g, &m->ffn_gate_exps[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ffn_up_exps.weight", l);
            if (!bind_tensor(g, &m->ffn_up_exps[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ffn_down_exps.weight", l);
            if (!bind_tensor(g, &m->ffn_down_exps[l], name)) goto missing_layer;
            /* Shared expert (Qwen2.5-MoE always has one; llama-arch MoE files
             * may not). All-or-nothing: a gate input without its projections
             * is a malformed checkpoint, not a smaller model. */
            snprintf(name, sizeof name, "blk.%d.ffn_gate_inp_shexp.weight", l);
            if (bind_tensor(g, &m->shexp_gate_inp[l], name)) {
                snprintf(name, sizeof name, "blk.%d.ffn_gate_shexp.weight", l);
                if (!bind_tensor(g, &m->shexp_gate[l], name)) goto missing_layer;
                snprintf(name, sizeof name, "blk.%d.ffn_up_shexp.weight", l);
                if (!bind_tensor(g, &m->shexp_up[l], name)) goto missing_layer;
                snprintf(name, sizeof name, "blk.%d.ffn_down_shexp.weight", l);
                if (!bind_tensor(g, &m->shexp_down[l], name)) goto missing_layer;
            } else if (strcmp(arch, "qwen2moe") == 0) {
                snprintf(err, errlen,
                         "qwen2moe layer %d is missing 'blk.%d.ffn_gate_inp_shexp.weight'",
                         l, l);
                free_model(m);
                return NULL;
            }
        } else {
            snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", l);
            if (!bind_tensor(g, &m->ffn_gate[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ffn_up.weight", l);
            if (!bind_tensor(g, &m->ffn_up[l], name)) goto missing_layer;
            snprintf(name, sizeof name, "blk.%d.ffn_down.weight", l);
            if (!bind_tensor(g, &m->ffn_down[l], name)) goto missing_layer;
        }
    }

    /* MoE geometry validation: everything is checked against the tensor
     * shapes, not against hope. Experts must be uniform across layers and
     * match the router width. */
    if (m->is_moe) {
        int ff_exp = -1, ff_shexp = -1, has_shexp = -1;

        for (l = 0; l < m->n_layer; ++l) {
            const nita_tensor_t *gi = &m->ffn_gate_inp[l];
            const nita_tensor_t *ge = &m->ffn_gate_exps[l];
            const nita_tensor_t *ue = &m->ffn_up_exps[l];
            const nita_tensor_t *de = &m->ffn_down_exps[l];

            if ((int)gi->ne0 != m->n_embd || (int)gi->ne1 != m->n_expert) {
                snprintf(err, errlen, "layer %d ffn_gate_inp is %llux%llu, expected %dx%d",
                         l, (unsigned long long)gi->ne0, (unsigned long long)gi->ne1,
                         m->n_embd, m->n_expert);
                free_model(m); return NULL;
            }
            if ((int)ge->ne0 != m->n_embd || (int)ge->ne2 != m->n_expert ||
                (int)ue->ne0 != m->n_embd || (int)ue->ne2 != m->n_expert ||
                ue->ne1 != ge->ne1) {
                snprintf(err, errlen, "layer %d expert gate/up shapes are inconsistent", l);
                free_model(m); return NULL;
            }
            if ((int)de->ne0 != (int)ge->ne1 || (int)de->ne1 != m->n_embd ||
                (int)de->ne2 != m->n_expert) {
                snprintf(err, errlen, "layer %d expert down shape is inconsistent", l);
                free_model(m); return NULL;
            }
            if (ff_exp == -1) ff_exp = (int)ge->ne1;
            else if (ff_exp != (int)ge->ne1) {
                snprintf(err, errlen, "layer %d expert FFN width %d differs from %d",
                         l, (int)ge->ne1, ff_exp);
                free_model(m); return NULL;
            }
            if (m->shexp_gate_inp[l].bound) {
                const nita_tensor_t *sg = &m->shexp_gate_inp[l];
                const nita_tensor_t *sgt = &m->shexp_gate[l];
                const nita_tensor_t *sup = &m->shexp_up[l];
                const nita_tensor_t *sdn = &m->shexp_down[l];

                if (sg->ne1 != 1 || (int)sg->ne0 != m->n_embd) {
                    snprintf(err, errlen, "layer %d ffn_gate_inp_shexp is not a 1-D %d-vector",
                             l, m->n_embd);
                    free_model(m); return NULL;
                }
                if ((int)sgt->ne0 != m->n_embd || sgt->ne1 != sup->ne1 ||
                    (int)sup->ne0 != m->n_embd) {
                    snprintf(err, errlen, "layer %d shared expert gate/up shapes are inconsistent", l);
                    free_model(m); return NULL;
                }
                if (sdn->ne0 != sgt->ne1 || (int)sdn->ne1 != m->n_embd) {
                    snprintf(err, errlen, "layer %d shared expert down shape is inconsistent", l);
                    free_model(m); return NULL;
                }
                if (ff_shexp == -1) ff_shexp = (int)sgt->ne1;
                else if (ff_shexp != (int)sgt->ne1) {
                    snprintf(err, errlen, "layer %d shared expert FFN width %d differs from %d",
                             l, (int)sgt->ne1, ff_shexp);
                    free_model(m); return NULL;
                }
                if (has_shexp == 0) {
                    snprintf(err, errlen, "layer %d has a shared expert but earlier layers do not", l);
                    free_model(m); return NULL;
                }
                has_shexp = 1;
            } else {
                if (has_shexp == 1) {
                    snprintf(err, errlen, "layer %d is missing the shared expert other layers have", l);
                    free_model(m); return NULL;
                }
                has_shexp = 0;
            }
        }
        m->n_ff_exp   = ff_exp;
        m->n_ff_shexp = has_shexp ? ff_shexp : 0;
        if (strcmp(arch, "qwen2moe") == 0 && !has_shexp) {
            snprintf(err, errlen, "qwen2moe checkpoint has no shared expert tensors");
            free_model(m); return NULL;
        }
        printf("moe             : %d experts, top-%d, expert ff=%d, %s%d\n",
               m->n_expert, m->expert_used, m->n_ff_exp,
               has_shexp ? "shared expert ff=" : "no shared expert, norm_w=",
               has_shexp ? m->n_ff_shexp : m->moe_norm_w);
    }

    /* Attention bias shapes must match the projections they amend. */
    if (!m->is_qwen35 && m->attn_q_bias) {
        const int q_dim_  = m->n_head * m->head_dim;

        for (l = 0; l < m->n_layer; ++l) {
            if (m->attn_q_bias[l].bound && (int)m->attn_q_bias[l].ne0 != q_dim_) {
                snprintf(err, errlen, "layer %d attn_q.bias has %llu elements, expected %d",
                         l, (unsigned long long)m->attn_q_bias[l].ne0, q_dim_);
                free_model(m); return NULL;
            }
            if (m->attn_k_bias[l].bound && (int)m->attn_k_bias[l].ne0 != m->kv_dim) {
                snprintf(err, errlen, "layer %d attn_k.bias has %llu elements, expected %d",
                         l, (unsigned long long)m->attn_k_bias[l].ne0, m->kv_dim);
                free_model(m); return NULL;
            }
            if (m->attn_v_bias[l].bound && (int)m->attn_v_bias[l].ne0 != m->kv_dim) {
                snprintf(err, errlen, "layer %d attn_v.bias has %llu elements, expected %d",
                         l, (unsigned long long)m->attn_v_bias[l].ne0, m->kv_dim);
                free_model(m); return NULL;
            }
            if (m->attn_out_bias[l].bound && (int)m->attn_out_bias[l].ne0 != m->n_embd) {
                snprintf(err, errlen, "layer %d attn_output.bias has %llu elements, expected %d",
                         l, (unsigned long long)m->attn_out_bias[l].ne0, m->n_embd);
                free_model(m); return NULL;
            }
        }
    }

    /* F-02 gate: refuse unsupported quants before any kernel sees them. */
    {
        const nita_tensor_t *bad = unsupported_tensor(m);

        if (bad) {
            snprintf(err, errlen,
                     "tensor '%s' uses quant type %u which this build cannot "
                     "decode; refusing to load rather than emit zeros",
                     bad->name, (unsigned)bad->type);
            free_model(m);
            return NULL;
        }
    }

    /* qwen35 mixer geometry and per-layer recurrent state */
    if (m->is_qwen35) {
        const int64_t inner = hparam_int(g, arch, "ssm.inner_size", 0);
        const int64_t state = hparam_int(g, arch, "ssm.state_size", 0);
        const int64_t groups = hparam_int(g, arch, "ssm.group_count", 0);
        const int64_t rank  = hparam_int(g, arch, "ssm.time_step_rank", 0);

        m->q35_cfg.n_embd       = m->n_embd;
        m->q35_cfg.n_k_heads    = (int)groups;
        m->q35_cfg.head_k_dim   = (int)state;
        m->q35_cfg.n_v_heads    = (int)rank;
        m->q35_cfg.head_v_dim   = rank > 0 ? (int)(inner / rank) : 0;
        m->q35_cfg.conv_kernel  = (int)hparam_int(g, arch, "ssm.conv_kernel", 4);
        m->q35_cfg.conv_dim     = 2 * (int)groups * (int)state + (int)inner;
        m->q35_cfg.eps          = m->rms_eps;

        for (l = 0; l < m->n_layer; ++l) {
            if (!m->layer_recr[l]) continue;
            if (q35_gdn_prepare(&m->q35_cfg, &m->q35_w[l], &m->q35_state[l],
                                err, errlen) != 0) {
                snprintf(err, errlen, "layer %d gated delta-net setup failed", l);
                free_model(m);
                return NULL;
            }
        }
        {
            int n_recr = 0;

            for (l = 0; l < m->n_layer; ++l) n_recr += m->layer_recr[l] ? 1 : 0;
            printf("qwen35 hybrid   : %d gated delta-net blocks, %d dense-attention "
                   "blocks (every %d-th), ssm %d state, rank %d, inner %d\n",
                   n_recr, m->n_layer - n_recr, m->full_attn_interval,
                   (int)state, (int)rank, (int)inner);
        }
    }

    /* Shape consistency */
    q_dim = m->n_head * m->head_dim;
    for (l = 0; l < m->n_layer; ++l) {
        if (m->is_qwen35 && m->layer_recr[l]) continue;
        if (m->is_qwen35) {
            /* Bonsai 2 fuses the attention gate into attn_q, so that matrix is
             * twice as wide as the query it produces */
            const int q_rows = (int)m->attn_q[l].ne1;

            if (q_rows != q_dim && q_rows != 2 * q_dim) {
                snprintf(err, errlen,
                         "layer %d attn_q has %d rows, expected %d or %d",
                         l, q_rows, q_dim, 2 * q_dim);
                free_model(m);
                return NULL;
            }
            m->q35_gated_attn = (q_rows == 2 * q_dim) ? 1 : 0;
        }
        if ((int)m->attn_q[l].ne0 != m->n_embd ||
            (int)m->attn_k[l].ne0 != m->n_embd || (int)m->attn_k[l].ne1 != m->kv_dim ||
            (int)m->attn_v[l].ne0 != m->n_embd || (int)m->attn_v[l].ne1 != m->kv_dim ||
            (int)m->attn_out[l].ne0 != q_dim || (int)m->attn_out[l].ne1 != m->n_embd ||
            (!m->is_moe &&
             ((int)m->ffn_gate[l].ne0 != m->n_embd || (int)m->ffn_gate[l].ne1 != m->n_ff ||
              (int)m->ffn_up[l].ne0 != m->n_embd || (int)m->ffn_up[l].ne1 != m->n_ff ||
              (int)m->ffn_down[l].ne0 != m->n_ff || (int)m->ffn_down[l].ne1 != m->n_embd))) {
            snprintf(err, errlen, "layer %d tensor shapes are inconsistent", l);
            free_model(m);
            return NULL;
        }
    }

    /* Scratch buffers */
    {
        int ff_scratch = m->n_ff;

        if (m->n_ff_exp > ff_scratch)   ff_scratch = m->n_ff_exp;
        if (m->n_ff_shexp > ff_scratch) ff_scratch = m->n_ff_shexp;

        max_row = m->n_embd > ff_scratch ? m->n_embd : ff_scratch;
        m->rowbuf = (float *)nita_calloc((size_t)max_row, sizeof(float));
        m->rowcap = (size_t)max_row;

        m->nbuf  = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
        m->xb    = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
        m->hb    = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
        m->qbuf  = (float *)nita_calloc((size_t)q_dim, sizeof(float));
        m->kbuf  = (float *)nita_calloc((size_t)m->kv_dim, sizeof(float));
        m->vbuf  = (float *)nita_calloc((size_t)m->kv_dim, sizeof(float));
        m->abuf  = (float *)nita_calloc((size_t)q_dim, sizeof(float));
        m->gate  = (float *)nita_calloc((size_t)ff_scratch, sizeof(float));
        m->up    = (float *)nita_calloc((size_t)ff_scratch, sizeof(float));
        m->ff    = (float *)nita_calloc((size_t)ff_scratch, sizeof(float));
        m->tmp   = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
        m->scores= (float *)nita_calloc((size_t)m->ctx, sizeof(float));
        m->acc   = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
        m->router= (float *)nita_calloc((size_t)(m->n_expert > 0 ? m->n_expert : 1), sizeof(float));
        m->sel   = (int *)nita_calloc((size_t)(m->expert_used > 0 ? m->expert_used : 1), sizeof(int));
        m->sel_w = (float *)nita_calloc((size_t)(m->expert_used > 0 ? m->expert_used : 1), sizeof(float));
        m->sel_taken = (unsigned char *)nita_calloc((size_t)(m->n_expert > 0 ? m->n_expert : 1), 1);
        {
            int wide = m->n_embd;

            if (q_dim > wide)   wide = q_dim;
            if (m->kv_dim > wide) wide = m->kv_dim;
            m->biasbuf = (float *)nita_calloc((size_t)wide, sizeof(float));
        }
        if (!m->rowbuf || !m->nbuf || !m->xb || !m->hb || !m->qbuf || !m->kbuf ||
            !m->vbuf || !m->abuf || !m->gate || !m->up || !m->ff || !m->tmp ||
            !m->scores || !m->acc || !m->router || !m->sel || !m->sel_w ||
            !m->sel_taken || !m->biasbuf) {
            snprintf(err, errlen, "out of memory for scratch buffers");
            free_model(m);
            return NULL;
        }
    }
#if !defined(NITA_NO_ROPE_TAB)
    m->rope_tab = rope_build(m->ctx, m->n_rot, m->rope_theta);
#endif

    /* Worker pool. Projections are the whole cost of a token and split
     * perfectly by row, so this is where the cores go. */
    {
        const char *env = getenv("NITA_THREADS");
        int         nt;

#if defined(_WIN32)
        SYSTEM_INFO si;

        GetSystemInfo(&si);
        nt = default_threads((int)si.dwNumberOfProcessors);
#else
        {
            long np = sysconf(_SC_NPROCESSORS_ONLN);

            nt = default_threads(np > 0 ? (int)np : 1);
        }
#endif
        if (env) { int v = atoi(env); if (v > 0) nt = v; }
        if (nt < 1) nt = 1;
        m->pool = pool_create(nt, m->rowcap);
        g_pool  = (mt_pool_t *)m->pool;
    }

    /* KV cache */
    m->k_cache = (float **)nita_calloc((size_t)m->n_layer, sizeof(float *));
    m->v_cache = (float **)nita_calloc((size_t)m->n_layer, sizeof(float *));
    for (l = 0; l < m->n_layer; ++l) {
        size_t elems = (size_t)m->ctx * (size_t)m->kv_dim;
        m->k_cache[l] = (float *)nita_calloc(elems, sizeof(float));
        m->v_cache[l] = (float *)nita_calloc(elems, sizeof(float));
    }

    /* prism.hadamard.*: a Bonsai checkpoint folds a rotation into some matmuls.
     * The activation feeding a folded weight is rotated first; the stored
     * weight already holds the inverse, so the integer dot product and its
     * block scales are used exactly as stored. */
    if (m->is_qwen35) {
        const uint64_t nnames = gguf_array_count(g, "prism.hadamard.weight_names");

        m->qg      = (float *)nita_calloc((size_t)2 * (size_t)q_dim, sizeof(float));
        m->qgate   = (float *)nita_calloc((size_t)q_dim, sizeof(float));
        m->qnormv  = (float *)nita_calloc((size_t)m->head_dim, sizeof(float));
        m->knormv  = (float *)nita_calloc((size_t)m->head_dim, sizeof(float));
        m->headbuf = (float *)nita_calloc((size_t)m->head_dim, sizeof(float));
        m->rot     = (q35_rot_t *)nita_calloc((size_t)m->n_layer, sizeof(q35_rot_t));
        m->hdbuf   = (float *)nita_calloc((size_t)max_row, sizeof(float));

        if (!m->qg || !m->qgate || !m->qnormv || !m->knormv || !m->headbuf ||
            !m->rot || !m->hdbuf) {
            snprintf(err, errlen, "out of memory allocating qwen35 scratch");
            free_model(m);
            return NULL;
        }

        if (nnames > 0) {
            char           name[160];
            const int      block = (int)gguf_get_int(g, "prism.hadamard.block_size", 0);
            const uint64_t nw    = gguf_array_count(g, "prism.hadamard.sign_widths");
            const uint64_t nsign = gguf_array_count(g, "prism.hadamard.sign_values");
            uint64_t       off = 0, w, k;
            int            j;

            m->hd_names = (char **)nita_calloc((size_t)nnames, sizeof(char *));
            for (j = 0; j < (int)nnames; ++j) {
                const char *nm = gguf_array_str(g, "prism.hadamard.weight_names", (uint64_t)j);
                m->hd_names[j] = nita_strdup(nm ? nm : "");
            }
            m->hd_n_names = (int)nnames;

            if (block <= 0 || nw == 0 || nw > 4 || nsign == 0) {
                snprintf(err, errlen, "prism.hadamard geometry is unusable "
                         "(block %d, %llu widths, %llu signs)", block,
                         (unsigned long long)nw, (unsigned long long)nsign);
                free_model(m);
                return NULL;
            }
            for (w = 0; w < nw; ++w) {
                const int width = (int)gguf_array_num(g, "prism.hadamard.sign_widths", w);
                float    *signs;

                if (width <= 0 || off + (uint64_t)width > nsign || width % block != 0) {
                    snprintf(err, errlen, "prism.hadamard sign width %d does not fit "
                             "block %d with %llu values", width, block,
                             (unsigned long long)nsign);
                    free_model(m);
                    return NULL;
                }
                signs = (float *)nita_calloc((size_t)width, sizeof(float));
                for (k = 0; k < (uint64_t)width; ++k)
                    signs[k] = (float)gguf_array_num(g, "prism.hadamard.sign_values", off + k);
                j = nita_hadamard_init(&m->hd[w].xf, block, width, signs, err, errlen);
                free(signs);
                if (j != 0) {
                    free_model(m);
                    return NULL;
                }
                m->hd[w].width = width;
                off += (uint64_t)width;
                m->hd_n = (int)w + 1;
            }

            if (gguf_array_count(g, "prism.hadamard.inverse_weight_names") > 0) {
                const char *nm = gguf_array_str(g, "prism.hadamard.inverse_weight_names", 0);

                if (nm && strcmp(nm, "token_embd.weight") == 0) {
                    int w2;
                    for (w2 = 0; w2 < m->hd_n; ++w2)
                        if (m->hd[w2].width == m->n_embd) m->hd_tok_xf = &m->hd[w2].xf;
                }
            }

            /* Wire each block's matmuls to the rotation of its input width. */
            for (l = 0; l < m->n_layer; ++l) {
                const int inner = (int)m->q35_cfg.head_v_dim * m->q35_cfg.n_v_heads;

                if (m->layer_recr[l]) {
                    snprintf(name, sizeof name, "blk.%d.attn_qkv.weight", l);
                    m->q35_w[l].rot_qkv  = hd_for(m, name, (uint64_t)m->n_embd);
                    snprintf(name, sizeof name, "blk.%d.attn_gate.weight", l);
                    m->q35_w[l].rot_gate = hd_for(m, name, (uint64_t)m->n_embd);
                    snprintf(name, sizeof name, "blk.%d.ssm_out.weight", l);
                    m->q35_w[l].rot_out  = hd_for(m, name, (uint64_t)inner);
                    m->q35_w[l].rot_buf  = m->hdbuf;
                } else {
                    snprintf(name, sizeof name, "blk.%d.attn_q.weight", l);
                    m->rot[l].q = hd_for(m, name, (uint64_t)m->n_embd);
                    snprintf(name, sizeof name, "blk.%d.attn_k.weight", l);
                    m->rot[l].k = hd_for(m, name, (uint64_t)m->n_embd);
                    snprintf(name, sizeof name, "blk.%d.attn_v.weight", l);
                    m->rot[l].v = hd_for(m, name, (uint64_t)m->n_embd);
                    snprintf(name, sizeof name, "blk.%d.attn_output.weight", l);
                    m->rot[l].o = hd_for(m, name, (uint64_t)q_dim);
                }
                snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", l);
                m->rot[l].gate = hd_for(m, name, (uint64_t)m->n_embd);
                snprintf(name, sizeof name, "blk.%d.ffn_up.weight", l);
                m->rot[l].up   = hd_for(m, name, (uint64_t)m->n_embd);
                snprintf(name, sizeof name, "blk.%d.ffn_down.weight", l);
                m->rot[l].down = hd_for(m, name, (uint64_t)m->n_ff);
            }
            printf("prism.hadamard  : block %d, %d signed widths, %d folded matmuls, "
                   "embedding %s\n", block, m->hd_n, m->hd_n_names,
                   m->hd_tok_xf ? "inverse-rotated" : "plain");
        }
    }

    m->pos   = 0;
    m->ready = true;

    BREADCRUMBF("model",
                "transformer: n_embd=%d n_layer=%d n_head=%d n_head_kv=%d "
                "n_ff=%d n_rot=%d n_vocab=%d ctx=%d tied_output=%d",
                m->n_embd, m->n_layer, m->n_head, m->n_head_kv,
                m->n_ff, m->n_rot, m->n_vocab, m->ctx, (int)m->tied_output);
    return m;

missing_layer:
    snprintf(err, errlen, "layer %d is missing required tensors", l);
    free_model(m);
    return NULL;
}

void nita_model_reset(nita_model_t *m)
{
    int l;
    if (!m) return;
    for (l = 0; l < m->n_layer; ++l) {
        size_t bytes = (size_t)m->ctx * (size_t)m->kv_dim * sizeof(float);
        memset(m->k_cache[l], 0, bytes);
        memset(m->v_cache[l], 0, bytes);
        if (m->is_qwen35 && m->layer_recr && m->layer_recr[l])
            q35_gdn_state_reset(&m->q35_cfg, &m->q35_state[l]);
    }
    m->pos = 0;
}

int nita_model_ctx(const nita_model_t *m)      { return m ? m->ctx : 0; }
int nita_model_n_vocab(const nita_model_t *m)  { return m ? m->n_vocab : 0; }

/* MoE feed-forward for one token. m->hb must hold the post-norm hidden
 * state. Writes the weighted expert sum into m->acc (added to the residual
 * by the caller). Mirrors llama.cpp: softmax router, top-k selection,
 * optional weight renormalisation (llama-arch MoE), optional expert scale,
 * optional sigmoid-gated shared expert (qwen2moe). */
static int moe_ffn(nita_model_t *m, int l)
{
    const int E = m->n_expert, K = m->expert_used;
    int       j, c;

    if (matvec(m, &m->ffn_gate_inp[l], m->hb, m->router, m->n_embd, E, NULL) != 0)
        return -1;
    softmax_inplace(m->router, E);

    moe_topk(m->router, E, K, m->sel, m->sel_w, m->sel_taken);
    moe_weight_finish(m, m->sel_w, K);

    memset(m->acc, 0, (size_t)m->n_embd * sizeof *m->acc);
    for (j = 0; j < K; ++j) {
        const int ee = m->sel[j];
        const int ffw = m->n_ff_exp;

        if (matvec_expert(m, &m->ffn_gate_exps[l], ee, m->hb, m->gate,
                          m->n_embd, ffw, &P_pg) != 0) return -2;
        if (matvec_expert(m, &m->ffn_up_exps[l],   ee, m->hb, m->up,
                          m->n_embd, ffw, &P_pu) != 0) return -2;
        silu_mul(m->ff, m->gate, m->up, ffw);
        if (matvec_expert(m, &m->ffn_down_exps[l], ee, m->ff, m->tmp,
                          ffw, m->n_embd, &P_pd) != 0) return -2;
        for (c = 0; c < m->n_embd; ++c) m->acc[c] += m->sel_w[j] * m->tmp[c];
    }

    if (m->n_ff_shexp > 0) {
        const int ffw = m->n_ff_shexp;
        float     s   = 0.0f;

        /* sigmoid(gate(x)); llama.cpp writes this as silu(x)/x, which is the
         * same value everywhere except exactly at x == 0. */
        if (load_vector(&m->shexp_gate_inp[l], m->biasbuf, m->n_embd) != 0) return -3;
        for (c = 0; c < m->n_embd; ++c) s += m->hb[c] * m->biasbuf[c];
        s = 1.0f / (1.0f + expf(-s));

        if (matvec(m, &m->shexp_gate[l], m->hb, m->gate, m->n_embd, ffw, &P_pg) != 0) return -3;
        if (matvec(m, &m->shexp_up[l],   m->hb, m->up,   m->n_embd, ffw, &P_pu) != 0) return -3;
        silu_mul(m->ff, m->gate, m->up, ffw);
        if (matvec(m, &m->shexp_down[l], m->ff, m->tmp, ffw, m->n_embd, &P_pd) != 0) return -3;
        for (c = 0; c < m->n_embd; ++c) m->acc[c] += s * m->tmp[c];
    }
    return 0;
}

/* Tiled MoE feed-forward for the T tokens in m->phb. Router runs once for
 * the whole tile, then each expert sees only the rows that selected it
 * (gather/scatter by top-k), so sparse routing actually saves work. Writes
 * the weighted expert sum into m->pacc; the caller adds it to the residual.
 * Note: pacc is accumulated expert-by-expert (ascending expert id), which is
 * not the per-token selection order of the sequential path - the projection
 * sums themselves are still exact, so this differs only by float reordering,
 * like the vectorised tile attention. */
static int moe_tile_ffn(nita_model_t *m, int l, int T)
{
    const int E = m->n_expert, K = m->expert_used, N = m->n_embd;
    const int ffw = m->n_ff_exp;
    int       e, i, j, c, r;

    if (matvec_tile(m, &m->ffn_gate_inp[l], m->phb, T, m->prouter, N, E, NULL) != 0)
        return -1;
    for (i = 0; i < T; ++i)
        softmax_inplace(m->prouter + (size_t)i * (size_t)E, E);

    memset(m->pacc, 0, (size_t)T * (size_t)N * sizeof(float));
    for (i = 0; i < T; ++i) {
        float *probs = m->prouter + (size_t)i * (size_t)E;

        moe_topk(probs, E, K, m->psel + (size_t)i * (size_t)K,
                 m->psw + (size_t)i * (size_t)K, m->sel_taken);
        moe_weight_finish(m, m->psw + (size_t)i * (size_t)K, K);
    }

    for (e = 0; e < E; ++e) {
        int cnt = 0;

        for (i = 0; i < T; ++i)
            for (j = 0; j < K; ++j)
                if (m->psel[(size_t)i * (size_t)K + (size_t)j] == e) {
                    memcpy(m->pgath + (size_t)cnt * (size_t)N,
                           m->phb + (size_t)i * (size_t)N,
                           (size_t)N * sizeof(float));
                    m->prow_tok[cnt] = i;
                    m->prow_w[cnt]   = m->psw[(size_t)i * (size_t)K + (size_t)j];
                    ++cnt;
                }
        if (cnt == 0) continue;

        if (matvec_tile_expert(m, &m->ffn_gate_exps[l], e, m->pgath, cnt,
                               m->pgate, N, ffw, &P_pg) != 0) return -2;
        if (matvec_tile_expert(m, &m->ffn_up_exps[l],   e, m->pgath, cnt,
                               m->pup, N, ffw, &P_pu) != 0) return -2;
        {
            double _tf = prof_now();

            for (i = 0; i < cnt; ++i)
                silu_mul(m->pff + (size_t)i * (size_t)ffw,
                         m->pgate + (size_t)i * (size_t)ffw,
                         m->pup + (size_t)i * (size_t)ffw, ffw);
            prof_add(&P_ffnact, prof_now() - _tf, (long)cnt);
        }
        if (matvec_tile_expert(m, &m->ffn_down_exps[l], e, m->pff, cnt,
                               m->ptmp, ffw, N, &P_pd) != 0) return -2;
        for (r = 0; r < cnt; ++r) {
            const int   tok = m->prow_tok[r];
            const float w   = m->prow_w[r];

            for (c = 0; c < N; ++c)
                m->pacc[(size_t)tok * (size_t)N + (size_t)c] +=
                    w * m->ptmp[(size_t)r * (size_t)N + (size_t)c];
        }
    }

    /* sigmoid-gated shared expert: dense over the tile, one matvec each */
    if (m->n_ff_shexp > 0) {
        const int fs = m->n_ff_shexp;

        if (load_vector(&m->shexp_gate_inp[l], m->biasbuf, N) != 0) return -3;
        if (matvec_tile(m, &m->shexp_gate[l], m->phb, T, m->pgate, N, fs, &P_pg) != 0) return -3;
        if (matvec_tile(m, &m->shexp_up[l],   m->phb, T, m->pup,   N, fs, &P_pu) != 0) return -3;
        {
            double _tf = prof_now();

            for (i = 0; i < T; ++i)
                silu_mul(m->pff + (size_t)i * (size_t)fs,
                         m->pgate + (size_t)i * (size_t)fs,
                         m->pup + (size_t)i * (size_t)fs, fs);
            prof_add(&P_ffnact, prof_now() - _tf, (long)T);
        }
        if (matvec_tile(m, &m->shexp_down[l], m->pff, T, m->ptmp, fs, N, &P_pd) != 0) return -3;
        for (i = 0; i < T; ++i) {
            float s = 0.0f;

            for (c = 0; c < N; ++c)
                s += m->phb[(size_t)i * (size_t)N + (size_t)c] * m->biasbuf[c];
            s = 1.0f / (1.0f + expf(-s));
            for (c = 0; c < N; ++c)
                m->pacc[(size_t)i * (size_t)N + (size_t)c] +=
                    s * m->ptmp[(size_t)i * (size_t)N + (size_t)c];
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */
static int attention(nita_model_t *m, int layer, int pos)
{
    int   h;
    float inv = 1.0f / sqrtf((float)m->head_dim);
    float *kcache = m->k_cache[layer];
    float *vcache = m->v_cache[layer];
    double _ta = prof_now();

    for (h = 0; h < m->n_head; ++h) {
        int         kvh = (int)((long)h * m->n_head_kv / m->n_head);
        const float *qh = m->qbuf + (size_t)h * (size_t)m->head_dim;
        float       *oh = m->abuf + (size_t)h * (size_t)m->head_dim;
        int         t, d;

        for (t = 0; t <= pos; ++t) {
            const float *kt = kcache + (size_t)t * (size_t)m->kv_dim
                            + (size_t)kvh * (size_t)m->head_dim;
            float sum = 0.0f;
            for (d = 0; d < m->head_dim; ++d) sum += qh[d] * kt[d];
            m->scores[t] = sum * inv;
        }
        softmax_inplace(m->scores, pos + 1);

        for (d = 0; d < m->head_dim; ++d) oh[d] = 0.0f;
        for (t = 0; t <= pos; ++t) {
            const float *vt = vcache + (size_t)t * (size_t)m->kv_dim
                            + (size_t)kvh * (size_t)m->head_dim;
            float p = m->scores[t];
            for (d = 0; d < m->head_dim; ++d) oh[d] += p * vt[d];
        }
    }
    prof_add(&P_attn, prof_now() - _ta, 1);
    return 0;
}

/* ------------------------------------------------------------------ */
/* qwen35 hybrid forward                                               */
/* ------------------------------------------------------------------ */

/* Activation rotation for a folded weight: x itself when the matmul is not
 * folded, else a rotated copy in m->hdbuf. */
/* Opens NITA_Q35_DUMP once; NULL when dumping is off. Every dump site shares
 * it, so the file is not truncated and writes cannot interleave. */
FILE *nita_q35_dump_file(void)
{
    static FILE *f;

    if (f) return f;
    {
        const char *p = getenv("NITA_Q35_DUMP");

        if (!p || !*p) return NULL;
        f = fopen(p, "w");
    }
    return f;
}

/* Same pattern for the llama path: NITA_DUMP=<file> dumps named intermediate
 * tensors for differential testing. Each line is
 *   <name> <count> <v0> <v1> ...
 * matching tools/reference_logits.py --dump output byte-for-byte on values. */
static FILE *nita_llama_dump_file(void)
{
    static FILE *f;

    if (f) return f;
    {
        const char *p = getenv("NITA_DUMP");

        if (!p || !*p) return NULL;
        f = fopen(p, "w");
    }
    return f;
}

static void llama_dump(const char *name, const float *v, int n)
{
    FILE *f = nita_llama_dump_file();
    int   i;

    if (!f) return;
    fprintf(f, "%s %d", name, n);
    for (i = 0; i < n; ++i) fprintf(f, " %.6g", v[i]);
    fputc('\n', f);
}

#define q35_dump nita_q35_dump_file

static const float *q35_rot_in(nita_model_t *m, const nita_hadamard_t *xf,
                               const float *x, int n)
{
    if (!xf) return x;
    memcpy(m->hdbuf, x, (size_t)n * sizeof(float));
    nita_hadamard_forward(m->hdbuf, n, xf);
    return m->hdbuf;
}

/* The shared SwiGLU half; qwen35 has no separate FFN norm, the block's
 * post_attention_norm is it. Caller has already filled m->hb. */
static int q35_ffn(nita_model_t *m, int l)
{
    int i;

    if (matvec(m, &m->ffn_gate[l], q35_rot_in(m, m->rot[l].gate, m->hb, m->n_embd),
               m->gate, m->n_embd, m->n_ff, &P_pg) != 0) return -1;
    if (matvec(m, &m->ffn_up[l], q35_rot_in(m, m->rot[l].up, m->hb, m->n_embd),
               m->up, m->n_embd, m->n_ff, &P_pu) != 0) return -1;

    silu_mul(m->ff, m->gate, m->up, m->n_ff);

    if (matvec(m, &m->ffn_down[l], q35_rot_in(m, m->rot[l].down, m->ff, m->n_ff),
               m->tmp, m->n_ff, m->n_embd, &P_pd) != 0) return -1;

    if (q35_dump()) {
        char name[32];

        snprintf(name, sizeof name, "ffn_out-%d", l);
        fprintf(q35_dump(), "%s %d", name, m->n_embd);
        for (i = 0; i < m->n_embd; ++i) fprintf(q35_dump(), " %.6g", m->tmp[i]);
        fputc('\n', q35_dump());
    }

    for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
    return 0;
}

/*
 * Dense-attention block of a hybrid model. Upstream order: joint query+gate
 * projection, per-head query norm, K/V projection, per-head key norm, rotary
 * over the first n_rot dims of each head, attention, sigmoid gate, output.
 * The gate is interleaved per head inside attn_q, so one head's rows are
 * [q(head_dim) | gate(head_dim)].
 */
static int q35_attn(nita_model_t *m, int l, int pos)
{
    const int hd    = m->head_dim;
    const int q_dim = m->n_head * hd;
    int       i, h;

    if (load_vector(&m->attn_norm[l], m->nbuf, m->n_embd) != 0) return -1;
    rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);

    if (matvec(m, &m->attn_q[l], q35_rot_in(m, m->rot[l].q, m->hb, m->n_embd),
               m->qg, m->n_embd, 2 * q_dim, &P_pq) != 0) return -1;
    if (matvec(m, &m->attn_k[l], q35_rot_in(m, m->rot[l].k, m->hb, m->n_embd),
               m->kbuf, m->n_embd, m->kv_dim, &P_pk) != 0) return -1;
    if (matvec(m, &m->attn_v[l], q35_rot_in(m, m->rot[l].v, m->hb, m->n_embd),
               m->vbuf, m->n_embd, m->kv_dim, &P_pv) != 0) return -1;

    if (load_vector(&m->attn_q_norm[l], m->qnormv, hd) != 0) return -1;
    if (load_vector(&m->attn_k_norm[l], m->knormv, hd) != 0) return -1;

    for (h = 0; h < m->n_head; ++h) {
        const float *src = m->qg + (size_t)h * (size_t)(2 * hd);

        memcpy(m->headbuf, src, (size_t)hd * sizeof(float));
        rmsnorm(m->qbuf + (size_t)h * (size_t)hd, m->headbuf, m->qnormv, hd, m->rms_eps);
        memcpy(m->qgate + (size_t)h * (size_t)hd, src + hd, (size_t)hd * sizeof(float));
    }
    for (h = 0; h < m->n_head_kv; ++h) {
        memcpy(m->headbuf, m->kbuf + (size_t)h * (size_t)hd, (size_t)hd * sizeof(float));
        rmsnorm(m->kbuf + (size_t)h * (size_t)hd, m->headbuf, m->knormv, hd, m->rms_eps);
    }

    {
        double _tr = prof_now();
        rope_apply_half(m->rope_tab, m->qbuf, m->n_head,    hd, m->n_rot, pos, m->rope_theta);
        rope_apply_half(m->rope_tab, m->kbuf, m->n_head_kv, hd, m->n_rot, pos, m->rope_theta);
        prof_add(&P_rope, prof_now() - _tr, 2);
    }

    memcpy(m->k_cache[l] + (size_t)pos * (size_t)m->kv_dim,
           m->kbuf, (size_t)m->kv_dim * sizeof(float));
    memcpy(m->v_cache[l] + (size_t)pos * (size_t)m->kv_dim,
           m->vbuf, (size_t)m->kv_dim * sizeof(float));

    if (attention(m, l, pos) != 0) return -1;

    for (i = 0; i < q_dim; ++i) m->abuf[i] *= 1.0f / (1.0f + expf(-m->qgate[i]));

    if (matvec(m, &m->attn_out[l], q35_rot_in(m, m->rot[l].o, m->abuf, q_dim),
               m->tmp, q_dim, m->n_embd, &P_po) != 0) return -1;
    for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
    return 0;
}

static int q35_forward(nita_model_t *m, int32_t token, float *logits)
{
    const int pos = m->pos;
    int       l;
    size_t    row_bytes;

    row_bytes = nita_row_bytes(m->tok_embd.type, (uint64_t)m->n_embd);
    if (nita_dequant_row(m->tok_embd.data + (size_t)token * row_bytes,
                         m->tok_embd.type, (uint64_t)m->n_embd, m->xb) != 0)
        return -4;

    {
        FILE *df = q35_dump();

        if (df) {
            {
                int i;

                /* The raw stored row, before any restore, plus the inputs the
                 * restore needs, so a candidate rotation can be checked offline. */
                fprintf(df, "inp_embd_raw %d", m->n_embd);
                for (i = 0; i < m->n_embd; ++i) fprintf(df, " %.6g", m->xb[i]);
                fputc('\n', df);
                if (m->hd_tok_xf && m->hd_tok_xf->signs) {
                    fprintf(df, "hd_signs %d", m->hd_tok_xf->width);
                    for (i = 0; i < m->hd_tok_xf->width; ++i)
                        fprintf(df, " %.6g", m->hd_tok_xf->signs[i]);
                    fputc('\n', df);
                }
                if (load_vector(&m->attn_norm[0], m->nbuf, m->n_embd) == 0) {
                    fprintf(df, "attn_w-0 %d", m->n_embd);
                    for (i = 0; i < m->n_embd; ++i) fprintf(df, " %.6g", m->nbuf[i]);
                    fputc('\n', df);
                }
                fflush(df);
            }
        }
    }

    /* The embedding table holds rotated rows; restore the primal basis. */
    if (m->hd_tok_xf) nita_hadamard_inverse(m->xb, m->n_embd, m->hd_tok_xf);

    for (l = 0; l < m->n_layer; ++l) {
        /* The block's normed input, same tap as the fork's attn_norm-<il>. */
        if (q35_dump()) {
            if (load_vector(&m->attn_norm[l], m->nbuf, m->n_embd) == 0) {
                char name[32];
                int  i;

                rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);
                snprintf(name, sizeof name, "attn_norm-%d", l);
                fprintf(q35_dump(), "%s %d", name, m->n_embd);
                for (i = 0; i < m->n_embd; ++i) fprintf(q35_dump(), " %.6g", m->hb[i]);
                fputc('\n', q35_dump());
            }
        }

        if (m->layer_recr[l]) {
            /* The mixer normalises, mixes and adds its own residual. */
            double t0 = prof_now();
            int    rc;

            m->q35_cfg.layer = l;
            rc = q35_gdn_layer(&m->q35_cfg, &m->q35_w[l], &m->q35_state[l],
                               m->xb, m->xb);
            prof_add(&P_mix, prof_now() - t0, 1);
            if (rc != 0) {
                fprintf(stderr, "qwen35: delta-net mixer returned %d on layer %d\n", rc, l);
                return -20;
            }
        } else if (q35_attn(m, l, pos) != 0) {
            return -21;
        }

        if (q35_dump()) {
            char name[32];
            int  i;

            snprintf(name, sizeof name, "attn_residual-%d", l);
            fprintf(q35_dump(), "%s %d", name, m->n_embd);
            for (i = 0; i < m->n_embd; ++i) fprintf(q35_dump(), " %.6g", m->xb[i]);
            fputc('\n', q35_dump());
        }

        if (getenv("NITA_Q35_TRACE")) {
            double sum = 0.0, mx = 0.0;
            int    i;

            for (i = 0; i < m->n_embd; ++i) {
                double a = fabs((double)m->xb[i]);
                sum += a;
                if (a > mx) mx = a;
            }
            fprintf(stderr, "blk %2d %-7s mean|x| %.4f  max|x| %.3f\n", l,
                    m->layer_recr[l] ? "delta" : "attn", sum / m->n_embd, mx);
        }

        if (load_vector(&m->post_norm[l], m->nbuf, m->n_embd) != 0) return -5;
        rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);
        if (q35_ffn(m, l) != 0) return -22;

        /* Full block output, the fork's l_out-<il>. */
        if (q35_dump()) {
            char name[32];
            int  i;

            snprintf(name, sizeof name, "l_out-%d", l);
            fprintf(q35_dump(), "%s %d", name, m->n_embd);
            for (i = 0; i < m->n_embd; ++i) fprintf(q35_dump(), " %.6g", m->xb[i]);
            fputc('\n', q35_dump());
            fflush(q35_dump());
        }
    }

    if (load_vector(&m->out_norm, m->nbuf, m->n_embd) != 0) return -5;
    rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);

    {
        FILE *df = q35_dump();

        if (df) {
            {
                int i;

                fprintf(df, "result_norm %d", m->n_embd);
                for (i = 0; i < m->n_embd; ++i) fprintf(df, " %.6g", m->hb[i]);
                fputc('\n', df);
                fflush(df);
            }
        }
    }

    {
        /* Tied output: the table rows are rotated, so the hidden state has to
         * be rotated by the same transform before the dot products. */
        const float *src = m->hb;

        if (m->tied_output && m->hd_tok_xf)
            src = q35_rot_in(m, m->hd_tok_xf, m->hb, m->n_embd);
        if (matvec(m, &m->out_proj, src, logits, m->n_embd, m->n_vocab, NULL) != 0)
            return -11;
    }

    m->pos = pos + 1;
    return 0;
}

int nita_forward(nita_model_t *m, int32_t token, float *logits)
{
    int   l;
    int   pos;
    int   q_dim;
    size_t row_bytes;

    if (!m || !m->ready) return -1;
    if (token < 0 || token >= m->n_vocab) return -2;
    if (m->pos >= m->ctx) return -3;
    if (m->is_qwen35) return q35_forward(m, token, logits);

    pos   = m->pos;
    q_dim = m->n_head * m->head_dim;

    /* Token embedding lookup */
    {
        double _te = prof_now();
        row_bytes = nita_row_bytes(m->tok_embd.type, (uint64_t)m->n_embd);
        if (nita_dequant_row(m->tok_embd.data + (size_t)token * row_bytes,
                             m->tok_embd.type, (uint64_t)m->n_embd, m->xb) != 0)
            return -4;
        prof_add(&P_embed, prof_now() - _te, 1);
    }
    llama_dump("embed", m->xb, m->n_embd);

    for (l = 0; l < m->n_layer; ++l) {
        /* ---- attention block ---- */
        if (load_vector(&m->attn_norm[l], m->nbuf, m->n_embd) != 0) return -5;
        rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.attn_in", l); llama_dump(nm, m->hb, m->n_embd); }

        if (matvec(m, &m->attn_q[l], m->hb, m->qbuf, m->n_embd, q_dim, &P_pq) != 0) return -6;
        if (matvec(m, &m->attn_k[l], m->hb, m->kbuf, m->n_embd, m->kv_dim, &P_pk) != 0) return -6;
        if (matvec(m, &m->attn_v[l], m->hb, m->vbuf, m->n_embd, m->kv_dim, &P_pv) != 0) return -6;
        if (!m->is_qwen35) {
            if (add_bias(m, &m->attn_q_bias[l], m->qbuf, q_dim) != 0) return -6;
            if (add_bias(m, &m->attn_k_bias[l], m->kbuf, m->kv_dim) != 0) return -6;
            if (add_bias(m, &m->attn_v_bias[l], m->vbuf, m->kv_dim) != 0) return -6;
        }

        {
            double _tr = prof_now();
            rope_apply(m->rope_tab, m->qbuf, m->n_head,    m->head_dim,
                       m->n_rot, pos, m->rope_theta);
            rope_apply(m->rope_tab, m->kbuf, m->n_head_kv, m->head_dim,
                       m->n_rot, pos, m->rope_theta);
            prof_add(&P_rope, prof_now() - _tr, 2);
        }
        { char nm[32];
          snprintf(nm, sizeof nm, "l%d.q", l); llama_dump(nm, m->qbuf, q_dim);
          snprintf(nm, sizeof nm, "l%d.k", l); llama_dump(nm, m->kbuf, m->kv_dim);
          snprintf(nm, sizeof nm, "l%d.v", l); llama_dump(nm, m->vbuf, m->kv_dim); }

        memcpy(m->k_cache[l] + (size_t)pos * (size_t)m->kv_dim,
               m->kbuf, (size_t)m->kv_dim * sizeof(float));
        memcpy(m->v_cache[l] + (size_t)pos * (size_t)m->kv_dim,
               m->vbuf, (size_t)m->kv_dim * sizeof(float));

        if (attention(m, l, pos) != 0) return -7;
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.abuf", l); llama_dump(nm, m->abuf, q_dim); }
        if (matvec(m, &m->attn_out[l], m->abuf, m->tmp, q_dim, m->n_embd, &P_po) != 0) return -8;
        if (!m->is_qwen35 &&
            add_bias(m, &m->attn_out_bias[l], m->tmp, m->n_embd) != 0) return -8;
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.attn_out", l); llama_dump(nm, m->tmp, m->n_embd); }

        {
            int i;
            for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
        }
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.post_attn", l); llama_dump(nm, m->xb, m->n_embd); }

        /* ---- feed-forward block (SwiGLU, or experts) ---- */
        if (load_vector(&m->ffn_norm[l], m->nbuf, m->n_embd) != 0) return -5;
        rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.ffn_in", l); llama_dump(nm, m->hb, m->n_embd); }

        if (m->is_moe) {
            if (moe_ffn(m, l) != 0) return -12;
            memcpy(m->tmp, m->acc, (size_t)m->n_embd * sizeof(float));
            { char nm[32]; snprintf(nm, sizeof nm, "l%d.ffn_out", l); llama_dump(nm, m->tmp, m->n_embd); }
            {
                int i;
                for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
            }
            { char nm[32]; snprintf(nm, sizeof nm, "l%d.out", l); llama_dump(nm, m->xb, m->n_embd); }
            continue;
        }

        if (matvec(m, &m->ffn_gate[l], m->hb, m->gate, m->n_embd, m->n_ff, &P_pg) != 0) return -9;
        if (matvec(m, &m->ffn_up[l],   m->hb, m->up,   m->n_embd, m->n_ff, &P_pu) != 0) return -9;
        { char nm[32];
          snprintf(nm, sizeof nm, "l%d.gate", l); llama_dump(nm, m->gate, m->n_ff);
          snprintf(nm, sizeof nm, "l%d.up", l); llama_dump(nm, m->up, m->n_ff); }

        {
            double _tf = prof_now();
            silu_mul(m->ff, m->gate, m->up, m->n_ff);
            prof_add(&P_ffnact, prof_now() - _tf, 1);
        }
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.ff", l); llama_dump(nm, m->ff, m->n_ff); }

        if (matvec(m, &m->ffn_down[l], m->ff, m->tmp, m->n_ff, m->n_embd, &P_pd) != 0) return -10;
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.ffn_out", l); llama_dump(nm, m->tmp, m->n_embd); }

        {
            int i;
            for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
        }
        { char nm[32]; snprintf(nm, sizeof nm, "l%d.out", l); llama_dump(nm, m->xb, m->n_embd); }
    }

    /* Final norm and output projection */
    if (load_vector(&m->out_norm, m->nbuf, m->n_embd) != 0) return -5;
    rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);
    llama_dump("final_norm", m->hb, m->n_embd);

    {
        double _to = prof_now();
        if (matvec(m, &m->out_proj, m->hb, logits, m->n_embd, m->n_vocab, NULL) != 0) return -11;
        prof_add(&P_out, prof_now() - _to, (long)m->n_vocab);
    }
    if (m->out_bias_f32) {
        int i;

        for (i = 0; i < m->n_vocab; ++i) logits[i] += m->out_bias_f32[i];
    }

    m->pos = pos + 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Batched prefill                                                     */
/* ------------------------------------------------------------------ */
static void prefill_free(nita_model_t *m)
{
    if (!m) return;
    free(m->pxb);    m->pxb    = NULL;
    free(m->phb);    m->phb    = NULL;
    free(m->ptmp);   m->ptmp   = NULL;
    free(m->pq);     m->pq     = NULL;
    free(m->pa);     m->pa     = NULL;
    free(m->pk);     m->pk     = NULL;
    free(m->pv);     m->pv     = NULL;
    free(m->pgate);  m->pgate  = NULL;
    free(m->pup);    m->pup    = NULL;
    free(m->pff);    m->pff    = NULL;
    free(m->pscores); m->pscores = NULL;
    free(m->prouter);  m->prouter  = NULL;
    free(m->pacc);     m->pacc     = NULL;
    free(m->pgath);    m->pgath    = NULL;
    free(m->psel);     m->psel     = NULL;
    free(m->psw);      m->psw      = NULL;
    free(m->prow_tok); m->prow_tok = NULL;
    free(m->prow_w);   m->prow_w   = NULL;
    m->tile = 0;
}

/*
 * Allocate the tile scratch on first use.
 *
 * About 1.2 MB for a 1B model at tile 8 and 2k context; a model that only ever
 * decodes never allocates it. A failure is remembered as tile = -1 so a doomed
 * allocation is not retried on every prompt.
 *
 * Sixty-four is the measured default on the one-core Xeon + Tiny-Moe
 * (Q4_K_M): 128 tok 10.4 -> 9.5 ms/tok, 1024 tok 12.05 -> 10.79 ms/tok
 * (+12%), 2048 tok 17.53 -> 16.36 ms/tok (+7%), so the earlier "tile 16 loses
 * at 64+" claim does not reproduce on this model. Attention cost grows with
 * position and very long prompts or much larger models may still prefer less;
 * NITA_TILE overrides the default and remains capped at 64.
 */
static int prefill_alloc(nita_model_t *m)
{
    const char *e;
    int         t, q_dim;

    if (m->tile > 0) return 0;
    if (m->tile < 0) return -1;

    e = getenv("NITA_TILE");
    t = e ? atoi(e) : 64;
    if (t < 1)  t = 1;
    if (t > 64) t = 64;

    q_dim = m->n_head * m->head_dim;

    int ffs = m->n_ff;
    if (m->n_ff_exp > ffs)   ffs = m->n_ff_exp;
    if (m->n_ff_shexp > ffs) ffs = m->n_ff_shexp;

    m->pxb     = (float *)nita_calloc((size_t)t * (size_t)m->n_embd, sizeof(float));
    m->phb     = (float *)nita_calloc((size_t)t * (size_t)m->n_embd, sizeof(float));
    m->ptmp    = (float *)nita_calloc((size_t)t * (size_t)m->n_embd, sizeof(float));
    m->pq      = (float *)nita_calloc((size_t)t * (size_t)q_dim, sizeof(float));
    m->pa      = (float *)nita_calloc((size_t)t * (size_t)q_dim, sizeof(float));
    m->pk      = (float *)nita_calloc((size_t)t * (size_t)m->kv_dim, sizeof(float));
    m->pv      = (float *)nita_calloc((size_t)t * (size_t)m->kv_dim, sizeof(float));
    m->pgate   = (float *)nita_calloc((size_t)t * (size_t)ffs, sizeof(float));
    m->pup     = (float *)nita_calloc((size_t)t * (size_t)ffs, sizeof(float));
    m->pff     = (float *)nita_calloc((size_t)t * (size_t)ffs, sizeof(float));
    m->pscores = (float *)nita_calloc((size_t)t * (size_t)m->ctx, sizeof(float));

    if (m->is_moe) {
        const int E = m->n_expert, K = m->expert_used;

        m->prouter  = (float *)nita_calloc((size_t)t * (size_t)E, sizeof(float));
        m->pacc     = (float *)nita_calloc((size_t)t * (size_t)m->n_embd, sizeof(float));
        m->pgath    = (float *)nita_calloc((size_t)t * (size_t)m->n_embd, sizeof(float));
        m->psel     = (int *)nita_calloc((size_t)t * (size_t)K, sizeof(int));
        m->psw      = (float *)nita_calloc((size_t)t * (size_t)K, sizeof(float));
        m->prow_tok = (int *)nita_calloc((size_t)t, sizeof(int));
        m->prow_w   = (float *)nita_calloc((size_t)t, sizeof(float));
        if (!m->prouter || !m->pacc || !m->pgath || !m->psel || !m->psw ||
            !m->prow_tok || !m->prow_w) {
            prefill_free(m);
            m->tile = -1;
            return -1;
        }
    }

    if (!m->pxb || !m->phb || !m->ptmp || !m->pq || !m->pa || !m->pk ||
        !m->pv || !m->pgate || !m->pup || !m->pff || !m->pscores) {
        prefill_free(m);
        m->tile = -1;
        return -1;
    }
    m->tile = t;

    /* TEMP trap: NITA_DEBUG_ALLOC=1 prints where the tile buffers landed, so a
     * run whose timings look wrong can be correlated with their addresses. */
    if (getenv("NITA_DEBUG_ALLOC")) {
        fprintf(stderr,
                "alloc tile=%d embd=%d ff=%d qdim=%d\n"
                "  xb=%p hb=%p tmp=%p q=%p a=%p k=%p v=%p gate=%p up=%p ff=%p sc=%p\n",
                t, m->n_embd, m->n_ff, q_dim,
                (void *)m->pxb, (void *)m->phb, (void *)m->ptmp, (void *)m->pq,
                (void *)m->pa, (void *)m->pk, (void *)m->pv, (void *)m->pgate,
                (void *)m->pup, (void *)m->pff, (void *)m->pscores);
    }
    return 0;
}

#if defined(__AVX2__)
/*
 * Plain float dot product, eight lanes wide.
 *
 * Unlike the projection kernels this does NOT reproduce the scalar summation
 * order: eight partial sums are reduced as a tree instead of one running sum,
 * so a tiled head is no longer bit-identical to the decode path - it is the
 * same computation to within float rounding. With the softmax below also on
 * the vector exp kernel, the tile attention pass differs from the sequential
 * one by rounding alone; the projections are still exact.
 */
static float dot_f32(const float *a, const float *b, int n)
{
    __m256 acc = _mm256_setzero_ps();
    float  s;
    int    i = 0;

    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);

    s = hsum256(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/*
 * Four q.k dots against four different keys at once.
 *
 * The naive loop is bound by latency, not bandwidth or issue width: one
 * accumulator means a serial chain of n/8 four-cycle FMAs, and because the
 * next key cannot start until the horizontal reduce has drained, the whole
 * score pass runs at about one eighth of the machine's FMA throughput. Four
 * keys in four independent accumulators gives the scheduler something to
 * interleave, and q is loaded once per four keys instead of once per key.
 * out[0..3] are the unscaled dot products.
 */
static void dot4_f32(const float *q, const float *k0, const float *k1,
                     const float *k2, const float *k3, int n, float *out)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int    i = 0;

    for (; i + 8 <= n; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q + i);

        a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + i), a0);
        a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k1 + i), a1);
        a2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k2 + i), a2);
        a3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k3 + i), a3);
    }

    out[0] = hsum256(a0);
    out[1] = hsum256(a1);
    out[2] = hsum256(a2);
    out[3] = hsum256(a3);
    for (; i < n; ++i) {
        const float qv = q[i];
        out[0] += qv * k0[i];
        out[1] += qv * k1[i];
        out[2] += qv * k2[i];
        out[3] += qv * k3[i];
    }
}
#endif

/* Causal attention for tile rows [lo,hi). The tile's first row sits at pos0,
 * so row i may see positions 0 .. pos0+i; every one of those keys is already
 * in the cache because the whole tile's k/v was written before this runs.
 * Row ranges are independent, which is what lets the pool take slices. */
static void attention_rows(nita_model_t *m, int layer, int pos0, int lo, int hi)
{
    int    h, i;
    float  inv    = 1.0f / sqrtf((float)m->head_dim);
    float *kcache = m->k_cache[layer];
    float *vcache = m->v_cache[layer];
    int    q_dim  = m->n_head * m->head_dim;

    for (h = 0; h < m->n_head; ++h) {
        int kvh = (int)((long)h * m->n_head_kv / m->n_head);

        for (i = lo; i < hi; ++i) {
            const float *qh   = m->pq + (size_t)i * (size_t)q_dim
                              + (size_t)h * (size_t)m->head_dim;
            float       *sc   = m->pscores + (size_t)i * (size_t)m->ctx;
            float       *oh   = m->pa + (size_t)i * (size_t)q_dim
                              + (size_t)h * (size_t)m->head_dim;
            const int    last = pos0 + i;
            int          t, d;

            /* q . k for every key this query may see. */
            t = 0;
#if defined(__AVX2__)
            {
                const int hd    = m->head_dim;
                const int kbase = (int)((size_t)kvh * (size_t)hd);
                const size_t ks = (size_t)m->kv_dim;

                for (; t + 4 <= last + 1; t += 4) {
                    const float *kp = kcache + (size_t)t * ks + kbase;

                    dot4_f32(qh, kp, kp + ks, kp + 2 * ks, kp + 3 * ks, hd, &sc[t]);
                    sc[t + 0] *= inv;
                    sc[t + 1] *= inv;
                    sc[t + 2] *= inv;
                    sc[t + 3] *= inv;
                }
                for (; t <= last; ++t)
                    sc[t] = dot_f32(qh, kcache + (size_t)t * ks + kbase, hd) * inv;
            }
#else
            for (; t <= last; ++t) {
                const float *kt = kcache + (size_t)t * (size_t)m->kv_dim
                                + (size_t)kvh * (size_t)m->head_dim;
                float sum = 0.0f;
                for (d = 0; d < m->head_dim; ++d) sum += qh[d] * kt[d];
                sc[t] = sum * inv;
            }
#endif
            softmax_inplace(sc, last + 1);

            /* oh = sum_t p[t] * v[t], eight output lanes at a time. Working a
             * chunk of the head across every key keeps one accumulator in a
             * register for the whole reduction and writes it out once,
             * instead of touching oh twice per key per lane. */
#if defined(__AVX2__)
            {
                const int hd = m->head_dim;
                const int vbase = (int)((size_t)kvh * (size_t)hd);

                /* A whole 64-dimension slab per pass, eight accumulators.
                 *
                 * Working one 8-lane chunk at a time is exactly as wide
                 * arithmetically and measures no faster: it touches 32 bytes
                 * out of every 128-byte v cache line and then walks the whole
                 * prefix again for the next chunk, so at long context the
                 * reduction is dominated by re-reading the v cache, not by the
                 * multiplies. One slab per pass uses every byte of each line
                 * it pulls in. */
                d = 0;
                for (; d + 64 <= hd; d += 64) {
                    __m256 o0 = _mm256_setzero_ps(), o1 = _mm256_setzero_ps();
                    __m256 o2 = _mm256_setzero_ps(), o3 = _mm256_setzero_ps();
                    __m256 o4 = _mm256_setzero_ps(), o5 = _mm256_setzero_ps();
                    __m256 o6 = _mm256_setzero_ps(), o7 = _mm256_setzero_ps();

                    for (t = 0; t <= last; ++t) {
                        const float  *vt = vcache + (size_t)t * (size_t)m->kv_dim + vbase + d;
                        const __m256  p  = _mm256_set1_ps(sc[t]);

                        o0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt +  0), o0);
                        o1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt +  8), o1);
                        o2 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + 16), o2);
                        o3 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + 24), o3);
                        o4 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + 32), o4);
                        o5 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + 40), o5);
                        o6 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + 48), o6);
                        o7 = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + 56), o7);
                    }
                    _mm256_storeu_ps(oh + d +  0, o0);
                    _mm256_storeu_ps(oh + d +  8, o1);
                    _mm256_storeu_ps(oh + d + 16, o2);
                    _mm256_storeu_ps(oh + d + 24, o3);
                    _mm256_storeu_ps(oh + d + 32, o4);
                    _mm256_storeu_ps(oh + d + 40, o5);
                    _mm256_storeu_ps(oh + d + 48, o6);
                    _mm256_storeu_ps(oh + d + 56, o7);
                }

                /* Head widths that are not a multiple of 64 (rare). */
                for (; d + 8 <= hd; d += 8) {
                    __m256 acc = _mm256_setzero_ps();

                    for (t = 0; t <= last; ++t) {
                        const float *vt = vcache + (size_t)t * (size_t)m->kv_dim
                                        + vbase + d;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(sc[t]),
                                              _mm256_loadu_ps(vt), acc);
                    }
                    _mm256_storeu_ps(oh + d, acc);
                }
                for (; d < hd; ++d) {
                    float s = 0.0f;
                    for (t = 0; t <= last; ++t) {
                        const float *vt = vcache + (size_t)t * (size_t)m->kv_dim + vbase;
                        s += sc[t] * vt[d];
                    }
                    oh[d] = s;
                }
            }
#else
            for (d = 0; d < m->head_dim; ++d) oh[d] = 0.0f;
            for (t = 0; t <= last; ++t) {
                const float *vt = vcache + (size_t)t * (size_t)m->kv_dim
                                + (size_t)kvh * (size_t)m->head_dim;
                const float  p  = sc[t];
                for (d = 0; d < m->head_dim; ++d) oh[d] += p * vt[d];
            }
#endif
        }
    }
}

/* One tile's attention, spread over the pool when there is one. */
static void attention_tile(nita_model_t *m, int layer, int pos0, int ncol)
{
    double _ta = prof_now();

    att_dispatch(m, layer, pos0, ncol);
    prof_add(&P_attn, prof_now() - _ta, (long)ncol);
}

int nita_prefill(nita_model_t *m, const int32_t *ids, int n, float *logits)
{
    const char *seq;
    int         start, i, l, j;

    if (m && m->is_qwen35) {
        /* The hybrid graph decodes a token at a time; prefill by looping so
         * generation works end to end. Only the last logits are wanted. */
        for (i = 0; i < n; ++i)
            if (nita_forward(m, ids[i], logits) != 0) return -1;
        return 0;
    }

    if (!m || !m->ready) return -1;
    if (n <= 0) return 0;
    if (m->pos + n > m->ctx) return -3;
    for (i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= m->n_vocab) return -2;

    /* The one-token-at-a-time path is kept for two reasons: it is the A/B
     * baseline for the tiled path, and it is the fallback when the tile
     * buffers cannot be allocated. NITA_PREFILL_SEQ=1 selects it. */
    seq = getenv("NITA_PREFILL_SEQ");
    if (seq && *seq && *seq != '0') {
        float *scratch = (float *)nita_malloc((size_t)m->n_vocab * sizeof(float));
        int    rc      = 0;

        if (!scratch) return -12;
        for (i = 0; i < n; ++i) {
            float *dst = (i + 1 == n && logits) ? logits : scratch;
            if (nita_forward(m, ids[i], dst) != 0) { rc = -5; break; }
        }
        free(scratch);
        return rc;
    }

    if (prefill_alloc(m) != 0) return -13;

    for (start = 0; start < n; start += m->tile) {
        const int T    = (n - start < m->tile) ? (n - start) : m->tile;
        const int pos0 = m->pos;
        const int N    = m->n_embd;
        const int qdim = m->n_head * m->head_dim;
        const size_t rb = nita_row_bytes(m->tok_embd.type, (uint64_t)N);

        /* ---- embeddings ---- */
        for (i = 0; i < T; ++i) {
            double _te = prof_now();
            if (nita_dequant_row(m->tok_embd.data + (size_t)ids[start + i] * rb,
                                 m->tok_embd.type, (uint64_t)N,
                                 m->pxb + (size_t)i * (size_t)N) != 0)
                return -4;
            prof_add(&P_embed, prof_now() - _te, 1);
        }

        for (l = 0; l < m->n_layer; ++l) {
            /* ---- attention block ---- */
            if (load_vector(&m->attn_norm[l], m->nbuf, N) != 0) return -5;
            for (i = 0; i < T; ++i)
                rmsnorm(m->phb + (size_t)i * (size_t)N,
                        m->pxb + (size_t)i * (size_t)N, m->nbuf, N, m->rms_eps);

            if (matvec_tile(m, &m->attn_q[l], m->phb, T, m->pq, N, qdim, &P_pq)    != 0) return -6;
            if (matvec_tile(m, &m->attn_k[l], m->phb, T, m->pk, N, m->kv_dim, &P_pk) != 0) return -6;
            if (matvec_tile(m, &m->attn_v[l], m->phb, T, m->pv, N, m->kv_dim, &P_pv) != 0) return -6;

            if (!m->is_qwen35) {
                int bi;

                for (bi = 0; bi < 3; ++bi) {
                    nita_tensor_t *b;
                    float         *dst;
                    int            w;

                    b   = bi == 0 ? &m->attn_q_bias[l] : bi == 1 ? &m->attn_k_bias[l] : &m->attn_v_bias[l];
                    w   = bi == 0 ? qdim : m->kv_dim;
                    dst = bi == 0 ? m->pq : bi == 1 ? m->pk : m->pv;
                    if (!b->bound) continue;
                    if (load_vector(b, m->biasbuf, w) != 0) return -6;
                    for (i = 0; i < T; ++i)
                        for (j = 0; j < w; ++j)
                            dst[(size_t)i * (size_t)w + j] += m->biasbuf[j];
                }
            }

            for (i = 0; i < T; ++i) {
                const int pos = pos0 + i;
                double    _tr = prof_now();

                rope_apply(m->rope_tab, m->pq + (size_t)i * (size_t)qdim,
                           m->n_head, m->head_dim, m->n_rot, pos, m->rope_theta);
                rope_apply(m->rope_tab, m->pk + (size_t)i * (size_t)m->kv_dim,
                           m->n_head_kv, m->head_dim, m->n_rot, pos, m->rope_theta);
                prof_add(&P_rope, prof_now() - _tr, 2);

                memcpy(m->k_cache[l] + (size_t)pos * (size_t)m->kv_dim,
                       m->pk + (size_t)i * (size_t)m->kv_dim,
                       (size_t)m->kv_dim * sizeof(float));
                memcpy(m->v_cache[l] + (size_t)pos * (size_t)m->kv_dim,
                       m->pv + (size_t)i * (size_t)m->kv_dim,
                       (size_t)m->kv_dim * sizeof(float));
            }

            attention_tile(m, l, pos0, T);
            if (matvec_tile(m, &m->attn_out[l], m->pa, T, m->ptmp, qdim, N, &P_po) != 0) return -8;
            if (!m->is_qwen35 && m->attn_out_bias[l].bound) {
                if (load_vector(&m->attn_out_bias[l], m->biasbuf, N) != 0) return -8;
                for (i = 0; i < T; ++i)
                    for (j = 0; j < N; ++j)
                        m->ptmp[(size_t)i * (size_t)N + j] += m->biasbuf[j];
            }

            for (i = 0; i < T; ++i)
                for (j = 0; j < N; ++j)
                    m->pxb[(size_t)i * (size_t)N + j] += m->ptmp[(size_t)i * (size_t)N + j];

            /* ---- feed-forward block (SwiGLU, or experts) ---- */
            if (load_vector(&m->ffn_norm[l], m->nbuf, N) != 0) return -5;
            for (i = 0; i < T; ++i)
                rmsnorm(m->phb + (size_t)i * (size_t)N,
                        m->pxb + (size_t)i * (size_t)N, m->nbuf, N, m->rms_eps);

            if (m->is_moe) {
                if (moe_tile_ffn(m, l, T) != 0) return -12;
                for (i = 0; i < T; ++i)
                    for (j = 0; j < N; ++j)
                        m->pxb[(size_t)i * (size_t)N + j] +=
                            m->pacc[(size_t)i * (size_t)N + j];
                goto layer_done;
            }

            if (matvec_tile(m, &m->ffn_gate[l], m->phb, T, m->pgate, N, m->n_ff, &P_pg) != 0) return -9;
            if (matvec_tile(m, &m->ffn_up[l],   m->phb, T, m->pup,   N, m->n_ff, &P_pu) != 0) return -9;

            for (i = 0; i < T; ++i) {
                double _tf = prof_now();
                silu_mul(m->pff + (size_t)i * (size_t)m->n_ff,
                         m->pgate + (size_t)i * (size_t)m->n_ff,
                         m->pup + (size_t)i * (size_t)m->n_ff, m->n_ff);
                prof_add(&P_ffnact, prof_now() - _tf, 1);
            }

            if (matvec_tile(m, &m->ffn_down[l], m->pff, T, m->ptmp, m->n_ff, N, &P_pd) != 0) return -10;

            for (i = 0; i < T; ++i)
                for (j = 0; j < N; ++j)
                    m->pxb[(size_t)i * (size_t)N + j] += m->ptmp[(size_t)i * (size_t)N + j];

layer_done:
            (void)0;
        }

        m->pos = pos0 + T;

        /* Only the last token of the run needs a next-token distribution. */
        if (start + T >= n && logits) {
            if (load_vector(&m->out_norm, m->nbuf, N) != 0) return -5;
            rmsnorm(m->hb, m->pxb + (size_t)(T - 1) * (size_t)N,
                    m->nbuf, N, m->rms_eps);
            {
                double _to = prof_now();
                if (matvec(m, &m->out_proj, m->hb, logits, N, m->n_vocab, NULL) != 0) return -11;
                prof_add(&P_out, prof_now() - _to, (long)m->n_vocab);
            }
            if (m->out_bias_f32)
                for (i = 0; i < m->n_vocab; ++i) logits[i] += m->out_bias_f32[i];
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Sampling                                                            */
/* ------------------------------------------------------------------ */
static uint32_t rng_next(uint32_t *s)
{
    uint32_t x = *s ? *s : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

typedef struct { float v; int32_t i; } cand_t;

/*
 * Sample an id from a full-vocabulary logit vector.
 *
 * The previous version copied all 128k logits into a candidate array, ran
 * qsort over the lot and only then discarded everything past k. That cost
 * 17-22 ms per token - as much again as the forward pass - and never showed
 * up in the component table because it happens outside nita_forward.
 *
 * Two paths now, neither of which sorts the vocabulary:
 *   - no top-k truncation: two linear passes (sum, then walk to the target);
 *   - top-k: one linear pass maintaining a descending k-element array, so a
 *     logit that cannot beat the current k-th best costs a single compare.
 */
int32_t nita_sample(const float *logits, int n_vocab,
                    float temperature, int top_k, uint32_t *rng)
{
    float   maxlogit;
    int32_t best = 0;
    int     i;
    double  target;

    if (!logits || n_vocab <= 0) return 0;

    maxlogit = logits[0];
    for (i = 1; i < n_vocab; ++i)
        if (logits[i] > maxlogit) { maxlogit = logits[i]; best = i; }

    /* Greedy when temperature is zero or negative. */
    if (temperature <= 0.0f) return best;

    target = (double)(rng_next(rng) >> 8) / (double)(1u << 24);

    /* ---- no truncation ---- */
    if (top_k <= 0 || top_k >= n_vocab) {
        const float inv_t = 1.0f / temperature;
        double      total, acc = 0.0;
        int         i;

        total = exp_total(logits, n_vocab, maxlogit, inv_t);
        if (total <= 0.0) total = 1.0;
        target *= total;

        /* Chunks that cannot contain the target are summed vectorised; only
         * the chunk it falls in is walked element by element. */
        for (i = 0; i < n_vocab; i += 256) {
            int    hi = (i + 256 < n_vocab) ? i + 256 : n_vocab;
            double cs = exp_total(logits + i, hi - i, maxlogit, inv_t);

            if (acc + cs >= target) {
                int j;
                for (j = i; j < hi; ++j) {
                    acc += (double)expf((logits[j] - maxlogit) * inv_t);
                    if (acc >= target) return j;
                }
                return hi - 1;
            }
            acc += cs;
        }
        return best;
    }

    /* ---- top-k ---- */
    {
        const int k = top_k;
        cand_t   *cand = (cand_t *)nita_malloc((size_t)k * sizeof *cand);
        int       cnt = 0;
        double    total = 0.0, acc = 0.0;

        if (!cand) return best;

        for (i = 0; i < n_vocab; ++i) {
            float v = logits[i];
            int   j;

            if (cnt == k && v <= cand[k - 1].v) continue;  /* common case */
            j = (cnt < k) ? cnt : k - 1;
            while (j > 0 && cand[j - 1].v < v) { cand[j] = cand[j - 1]; --j; }
            cand[j].v = v;
            cand[j].i = i;
            if (cnt < k) ++cnt;
        }

        for (i = 0; i < cnt; ++i)
            total += exp((double)((cand[i].v - maxlogit) / temperature));
        if (total <= 0.0) total = 1.0;
        target *= total;

        best = cand[cnt - 1].i;
        for (i = 0; i < cnt; ++i) {
            acc += exp((double)((cand[i].v - maxlogit) / temperature));
            if (acc >= target) { best = cand[i].i; break; }
        }
        free(cand);
        return best;
    }
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */
int nita_generate(nita_model_t *m, const int32_t *prompt, size_t n_prompt,
                  int max_tokens, float temperature, int top_k, int32_t eos_id,
                  int32_t **out_ids, size_t *out_count, uint32_t seed,
                  bool *hit_limit)
{
    float   *logits;
    int32_t *ids = NULL;
    size_t   produced = 0;
    size_t   i;
    uint32_t rng = seed ? seed : 0xC0FFEEu;

    *out_ids = NULL;
    *out_count = 0;
    if (hit_limit) *hit_limit = false;

    if (!m || !m->ready) return 0;
    if (n_prompt == 0) return 0;
    if (max_tokens <= 0) return 0;

    nita_model_reset(m);

    /* Keep the most recent context if the prompt is too long. */
    if (n_prompt > (size_t)m->ctx) {
        prompt   += n_prompt - (size_t)m->ctx;
        n_prompt  = (size_t)m->ctx;
    }

    logits = (float *)nita_malloc((size_t)m->n_vocab * sizeof(float));
    ids    = (int32_t *)nita_malloc((size_t)max_tokens * sizeof(int32_t));

    /* Prefill the prompt a tile at a time. If the tiled path cannot be set up
     * (no memory for the tile scratch), fall back to single-token forwards:
     * slower, but the same answer. */
    if (nita_prefill(m, prompt, (int)n_prompt, logits) != 0) {
        nita_model_reset(m);
        for (i = 0; i < n_prompt; ++i) {
            if (nita_forward(m, prompt[i], logits) != 0) {
                free(logits); free(ids);
                return 0;
            }
        }
    }

    for (i = 0; i < (size_t)max_tokens; ++i) {
        int32_t next = nita_sample(logits, m->n_vocab, temperature, top_k, &rng);

        if (eos_id >= 0 && next == eos_id) break;

        ids[produced++] = next;

        /* Stop cleanly at the context boundary; report it as a length stop. */
        if (m->pos >= m->ctx) {
            if (hit_limit) *hit_limit = true;
            break;
        }
        if (nita_forward(m, next, logits) != 0) break;

        if ((size_t)(i + 1) >= (size_t)max_tokens) {
            if (hit_limit) *hit_limit = true;
        }
    }

    free(logits);

    if (produced == 0) {
        free(ids);
        return 0;
    }

    *out_ids = ids;
    *out_count = produced;
    return (int)produced;
}
