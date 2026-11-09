/*
 * nita.h – shared declarations for the NITA engine.
 *
 * Modules:
 *   gguf.c      – GGUF v2/v3 container parser (metadata + tensor table)
 *   json.c      – minimal JSON parser / writer for the HTTP API
 *   tokenizer.c – vocabulary + greedy longest-match encoder/decoder
 *   chat.c      – conversation history + completion generation
 *   http.c      – cross-platform HTTP/1.1 server with OpenAI-style routes
 *   main.c      – CLI entry point
 */

#ifndef NITA_H
#define NITA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "hadamard.h"

#define NITA_VERSION "0.1.0"

/* Opaque transformer handle, defined in transformer.c. */
typedef struct nita_model nita_model_t;

/* ------------------------------------------------------------------ */
/* Breadcrumbs – compile-time traced diagnostics (stderr)              */
/* ------------------------------------------------------------------ */
#ifdef DEBUG_BREADCRUMBS
#  define BREADCRUMB(tag, msg) \
        do { fprintf(stderr, "[breadcrumb] %s %s\n", (tag), (msg)); fflush(stderr); } while (0)
#  define BREADCRUMBF(tag, ...) \
        do { fprintf(stderr, "[breadcrumb] %s ", (tag)); \
             fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } while (0)
#else
#  define BREADCRUMB(tag, msg)      ((void)0)
#  define BREADCRUMBF(tag, ...)     ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* Sherlock-lite trace – tag lines on stdout, nothing else             */
/* A probe exists so a later read can name where time went; it must not */
/* carry payloads the machine never produced (the predecessor's fake    */
/* %dev/dev_us/idle_us/host_us columns did) and must not run at all in  */
/* a default build: with SHERLOCK_LITE off every macro expands to       */
/* ((void)0) and the build carries zero probe code.                     */
/* ------------------------------------------------------------------ */
#ifdef SHERLOCK_LITE
#  define SHERLOCK_HEADER() ((void)0)
#  define SHERLOCK_ROW(tag) printf("[sherlock] %s\n", (tag))
#else
#  define SHERLOCK_HEADER() ((void)0)
#  define SHERLOCK_ROW(tag) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* Growable byte buffer                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} nbuf_t;

void   nbuf_init(nbuf_t *b);
void   nbuf_free(nbuf_t *b);
void   nbuf_clear(nbuf_t *b);
int    nbuf_append(nbuf_t *b, const void *bytes, size_t n);
int    nbuf_puts(nbuf_t *b, const char *s);
int    nbuf_putc(nbuf_t *b, char c);
int    nbuf_printf(nbuf_t *b, const char *fmt, ...);
/* Ensure the buffer is NUL terminated and return the C string. */
char  *nbuf_cstr(nbuf_t *b);

/* ------------------------------------------------------------------ */
/* Utility                                                             */
/* ------------------------------------------------------------------ */
void  *nita_malloc(size_t n);
void  *nita_calloc(size_t n, size_t sz);
void  *nita_realloc(void *p, size_t n);
char  *nita_strdup(const char *s);
/* Path helpers */
const char *nita_basename(const char *path);
/* Format a byte count into a human readable string (static buffer). */
const char *nita_human_bytes(uint64_t bytes);
/* 64-bit FNV-1a over n bytes: short, stable, and cheap enough to hash a slice of
 * every tensor, which is what the tensor and model checks compare. */
uint64_t nita_fnv1a64(const void *p, size_t n);

/* ================================================================== */
/* GGUF                                                                */
/* ================================================================== */

/* GGUF metadata value types (on-disk enumeration). */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};

typedef struct {
    char    *key;
    uint32_t type;          /* GGUF_TYPE_* */
    /* Scalar payload (interpretation depends on type) */
    union {
        uint64_t u64;
        int64_t  i64;
        double   f64;
        struct { char *ptr; uint64_t len; } str;
    } v;
    /* Array payload */
    uint32_t item_type;     /* GGUF_TYPE_* of each element */
    uint64_t count;
    uint8_t *raw;           /* serialized elements */
} gguf_kv_t;

typedef struct {
    char     *name;
    uint32_t  n_dims;
    uint64_t  dims[4];
    uint32_t  type;         /* ggml_type */
    uint64_t  offset;       /* offset from start of data section */
    uint64_t  n_elements;   /* product of dims */
} gguf_tensor_t;

typedef struct {
    char          *path;
    uint32_t       version;
    uint64_t       tensor_count;
    uint64_t       kv_count;
    gguf_kv_t     *kv;
    gguf_tensor_t *tensors;
    uint64_t       data_offset;   /* absolute file offset of tensor data */
    uint64_t       file_size;
    uint8_t       *data;          /* whole file image (owned) */

    /* Sequential cursor for gguf_array_str(). Vocabularies hold hundreds of
     * thousands of entries and are always read in order, so walking from the
     * start on every call would be quadratic. */
    const gguf_kv_t *arr_cursor_kv;
    uint64_t         arr_cursor_i;
    uint64_t         arr_cursor_pos;
} gguf_file_t;

/* Load a whole .gguf file. Returns NULL and fills err on failure. */
gguf_file_t *gguf_load(const char *path, char *err, size_t errlen);
void         gguf_free(gguf_file_t *g);

/* Metadata accessors (NULL / defaults when absent). */
const char *gguf_get_str(const gguf_file_t *g, const char *key, const char *fallback);
int64_t     gguf_get_int(const gguf_file_t *g, const char *key, int64_t fallback);
double      gguf_get_float(const gguf_file_t *g, const char *key, double fallback);
bool        gguf_has_key(const gguf_file_t *g, const char *key);

/* Array accessors. For string arrays, use gguf_array_str().
 * Returns element count, or 0 when the key is missing / wrong type. */
uint64_t    gguf_array_count(const gguf_file_t *g, const char *key);
/* Fetch element index of a string array. Returns NULL if out of range. */
const char *gguf_array_str(const gguf_file_t *g, const char *key, uint64_t index);
/* Fetch element index of a numeric array as double. */
double      gguf_array_num(const gguf_file_t *g, const char *key, uint64_t index);

/* Human readable name for a ggml tensor type id. */
const char *ggml_type_name(uint32_t type);
/* Bytes occupied by one row (one element-block group) for a ggml type. */
uint32_t    ggml_type_size(uint32_t type);
uint32_t    ggml_blck_size(uint32_t type);

/* ================================================================== */
/* JSON                                                                */
/* ================================================================== */

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type_t;

typedef struct json_value json_value_t;

struct json_value {
    json_type_t type;
    union {
        bool     b;
        double   num;
        struct { char *ptr; size_t len; } str;
        struct { json_value_t **items; size_t count; } arr;
        struct { char **keys; json_value_t **vals; size_t count; } obj;
    } u;
};

/* Parse a JSON document. Returns NULL on error and fills err. */
json_value_t *json_parse(const char *text, size_t len, char *err, size_t errlen);
void          json_free(json_value_t *v);

json_value_t *json_object_get(const json_value_t *obj, const char *key);
const char   *json_string(const json_value_t *v);        /* NULL unless JSON_STRING */
double        json_number(const json_value_t *v, double fallback);
bool          json_bool(const json_value_t *v, bool fallback);
size_t        json_array_count(const json_value_t *v);
json_value_t *json_array_at(const json_value_t *v, size_t i);

/* Append a JSON-escaped string (with surrounding quotes) to buf. */
void json_escape(nbuf_t *out, const char *s, size_t len);

/* ================================================================== */
/* Tokenizer                                                           */
/* ================================================================== */

typedef enum {
    TOK_MODE_GREEDY = 0,  /* longest-match fallback            */
    TOK_MODE_BPE    = 1,  /* GPT-2 style byte-level BPE+merges */
    TOK_MODE_SPM    = 2   /* llama/sentencepiece score merges  */
} tok_mode_t;

typedef struct {
    char     **tokens;      /* token id -> token string */
    float     *scores;      /* token id -> score (optional) */
    uint32_t   n_tokens;
    int32_t    id_bos, id_eos, id_unk;
    /* open-addressing hash table: token string -> id */
    int32_t   *slots;       /* id, or -1 */
    uint32_t   n_slots;
    bool       has_vocab;

    tok_mode_t   mode;
    const char  *mode_name;

    /* merge ranks keyed by "left right" -> rank (lower wins) */
    char     **merge_keys;
    uint32_t  *merge_rank;
    uint32_t   merge_cap;
    uint32_t   n_merges;

    /* True when the vocabulary contains the sentencepiece U+2581 marker
     * ("▁"). Only then do the GREEDY encoder/decoder perform the
     * space->▁ and ▁->space transformations; a raw byte-level vocabulary
     * (single-byte tokens, e.g. a byte model) encodes identity instead. */
    bool       has_sp_marker;

    /* Raw byte -> single token id, used when a byte has no vocabulary entry. */
    int32_t    byte_token_id[256];
} tokenizer_t;

/* Build a tokenizer from GGUF metadata. Always returns a valid object
 * (empty/has_vocab=false when the model carries no vocabulary). */
tokenizer_t *tokenizer_from_gguf(const gguf_file_t *g);
void         tokenizer_free(tokenizer_t *t);

/* Encode text to token ids. Returns malloc'd array (caller frees). */
int32_t     *tokenizer_encode(const tokenizer_t *t, const char *text,
                              size_t *out_count);
/* Look up a whole piece in the vocabulary; -1 when absent. */
int32_t      tokenizer_find(const tokenizer_t *t, const char *piece);
/* Decode token ids back to a malloc'd NUL-terminated string. */
char        *tokenizer_decode(const tokenizer_t *t, const int32_t *ids, size_t count);
/* Append the printable form of one token to a text stream. */
void         tokenizer_append_piece(const tokenizer_t *t, int32_t id, nbuf_t *out);

/* ================================================================== */
/* Chat                                                                */
/* ================================================================== */

#define CHAT_MAX_MESSAGES 128

typedef enum { ROLE_SYSTEM = 0, ROLE_USER, ROLE_ASSISTANT } chat_role_t;

typedef struct {
    chat_role_t role;
    char       *content;
} chat_message_t;

typedef struct {
    chat_message_t messages[CHAT_MAX_MESSAGES];
    size_t         count;
} chat_history_t;

void chat_history_init(chat_history_t *h);
void chat_history_free(chat_history_t *h);
void chat_history_clear(chat_history_t *h);
int  chat_history_push(chat_history_t *h, chat_role_t role, const char *content);
/* Render the conversation into a prompt string using the model's template. */
char *chat_render_prompt(const gguf_file_t *g, const tokenizer_t *tok,
                         const chat_history_t *h);

/* Result of a completion request. */
typedef struct {
    char       *content;      /* malloc'd assistant text */
    size_t      prompt_tokens;
    size_t      completion_tokens;
    const char *finish_reason;/* static string: "stop" or "length" */
    const char *fingerprint;  /* "fp_nita" for real generation, "fp_nita_stub" otherwise */
} chat_completion_t;

/* Run a completion over the history using the transformer engine.
 * Falls back to an acknowledgement when no usable weights are loaded. */
int chat_complete(const gguf_file_t *g, const tokenizer_t *tok,
                  nita_model_t *engine,
                  const chat_history_t *h, int max_tokens,
                  float temperature, int top_k,
                  chat_completion_t *out);
void chat_completion_free(chat_completion_t *c);

/* ================================================================== */
/* Dequantisation                                                      */
/* ================================================================== */

float  nita_f16_to_f32(uint16_t h);
/* Bytes consumed by one row of ne0 elements. */
size_t nita_row_bytes(uint32_t type, uint64_t ne0);
/* Expand one row of packed GGUF data into floats. 0 on success, -1 if the
 * tensor type is not supported. */
int    nita_dequant_row(const uint8_t *src, uint32_t type, uint64_t ne0, float *dst);

/* ================================================================== */
/* Transformer                                                         */
/* ================================================================== */

#define NITA_TENSOR_CHECK_BYTES 64

typedef struct {
    const uint8_t *data;
    uint32_t       type;
    uint64_t       ne0;      /* input width  */
    uint64_t       ne1;      /* output height */
    uint64_t       ne2;      /* experts (3-D weights), else 1 */
    bool           bound;
    /* The GGUF tensor this was bound from, so a report can say which weight a
     * field holds instead of only where its pointer landed. */
    char           name[64];
} nita_tensor_t;

/* Bind every tensor in the file through the path the forward pass uses, in file
 * order, so a caller can check where those pointers land. Returns the number
 * bound, or -1 when a tensor fails the bounds check bind_tensor applies - which
 * is the answer a caller about to read bytes through them wants. */
int nita_bind_all(const gguf_file_t *g, nita_tensor_t *out, size_t cap);

/* Build the engine from a parsed GGUF file. Returns NULL and fills err when
 * the file is not a supported llama-style transformer. */
nita_model_t *nita_model_open(const gguf_file_t *g, int ctx, char *err, size_t errlen);
/* Print one line per distinct weight the model holds - role, the tensor it was
 * bound from, the absolute file offset of its pointer and a hash of the first
 * NITA_TENSOR_CHECK_BYTES there - plus the layer and line counts. Returns the
 * number of lines, or -1 if there is nothing to report. */
int nita_model_tensor_report(const nita_model_t *m, const gguf_file_t *g);
void          nita_model_close(nita_model_t *m);
/* Drop the KV cache so a new conversation starts at position 0. */
void          nita_model_reset(nita_model_t *m);
int           nita_model_ctx(const nita_model_t *m);
int           nita_model_n_vocab(const nita_model_t *m);

/* Evaluate one token, appending it to the KV cache. logits must hold
 * n_vocab floats. Returns 0 on success. */
int nita_forward(nita_model_t *m, int32_t token, float *logits);

/* Evaluate a whole run of prompt tokens in one pass. Each weight row is
 * dequantised once per tile instead of once per token, which is several times
 * faster than calling nita_forward in a loop. The projections are bit-identical
 * to the sequential path; the attention pass is vectorised end to end (q.k
 * dots, softmax on the vector exp kernel, value mix) and so differs by float
 * rounding only - measured max |dlogit| 1.1e-05 over a 128-token prompt, with
 * no change to the argmax.
 * The run must fit in the remaining context. logits, if non-NULL, receives the
 * next-token distribution for the last token of the run. */
int nita_prefill(nita_model_t *m, const int32_t *ids, int n, float *logits);

/* Component profiler (transformer.c). Compiles only with -DSHERLOCK_LITE:
 * with the flag off every probe site folds to nothing and the binary
 * carries zero instrumentation, so a plain benchmark cannot be skewed by
 * the probe itself. Reset before a measured window and dump afterwards;
 * both write to stdout. */
void nita_prof_reset(void);
void nita_prof_dump(void);

/* One row of that profile. nita_prof_rows() hands the counters out so a caller
 * can repeat a measurement and report the median of each component instead of
 * trusting whichever single run it happened to get. "out-proj" is a subset of
 * "projections", so a total must skip rows with subset set. */
#define NITA_PROF_MAX_ROWS 24

typedef struct {
    const char *name;
    double      us;
    long        calls;
    int         subset;
} nita_prof_row_t;

int  nita_prof_rows(nita_prof_row_t *out, int max_rows);

/* Sub-parts of the qwen35 mixer, so a profile can say where the time goes
 * inside q35_gdn_layer instead of only what the whole layer costs. */
enum {
    NITA_MIX_NORM = 0,   /* pre-mixer rmsnorm + fused q/k/v projection */
    NITA_MIX_CONV,       /* depthwise causal conv + SiLU               */
    NITA_MIX_GATES,      /* alpha/beta projections and the decay/beta  */
    NITA_MIX_DELTA,      /* L2 norms and the per-head delta rule       */
    NITA_MIX_GATED_NORM, /* gate projection + gated RMSNorm            */
    NITA_MIX_OUT,        /* output projection and residual add         */
    NITA_MIX_PARTS
};

void nita_prof_mix(int part, double us);

/* The profiler's clock, in microseconds, so a phase timer in another
 * translation unit measures with the resolution the profile reports. */
double nita_prof_now(void);

/* x -> x*sigmoid(x) over a whole array, and dst[i] *= silu(src[i]), both on
 * the engine's vector exp (transformer.c). The mixer used to call libm expf
 * once per element, at roughly 190 cycles a call. */
void nita_silu_inplace(float *v, int n);
void nita_silu_mul(float *dst, const float *src, const float *u, int n);

/* Run fn over [0, n) split across the engine's worker pool, one half-open row
 * range per participant; the calling thread takes the first slice.
 *   - every row must be independent, and fn must not recurse into this
 *   - a range below 64 rows, or no pool, runs on the calling thread alone
 * thread is a participant id in [0, nita_thread_count()); use it to index
 * nita_thread_rowbuf(), which is NULL when there is no pool. */
typedef void (*nita_range_fn)(void *ctx, int lo, int hi, int thread);
void   nita_parallel_for(int n, nita_range_fn fn, void *ctx);
/* Same, with the caller deciding how small a range is worth a barrier: for work
 * split into a few dozen chunky items (the mixer's value heads) the 64-item
 * default would keep it serial. */
void   nita_parallel_for_min(int n, int min_items, nita_range_fn fn, void *ctx);
float *nita_thread_rowbuf(int thread);
int    nita_thread_count(void);

/* Sample an id from logits using temperature and optional top-k. */
int32_t nita_sample(const float *logits, int n_vocab,
                    float temperature, int top_k, uint32_t *rng);

/* Prefill the prompt then sample up to max_tokens new ids, stopping at eos.
 * Returns the number of ids produced (0 on failure). */
int nita_generate(nita_model_t *m, const int32_t *prompt, size_t n_prompt,
                  int max_tokens, float temperature, int top_k, int32_t eos_id,
                  int32_t **out_ids, size_t *out_count, uint32_t seed,
                  bool *hit_limit);

/* ================================================================== */
/* qwen35 hybrid architecture – gated delta-net block (qwen35.c)       */
/* ================================================================== */

typedef struct {
    int   n_embd;       /* embedding_length                        */
    int   n_k_heads;    /* ssm.group_count                         */
    int   head_k_dim;   /* ssm.state_size                          */
    int   n_v_heads;    /* ssm.time_step_rank                      */
    int   head_v_dim;   /* ssm.inner_size / ssm.time_step_rank     */
    int   conv_kernel;  /* ssm.conv_kernel                         */
    int   conv_dim;     /* 2*n_k_heads*head_k_dim + inner          */
    float eps;          /* attention.layer_norm_rms_epsilon        */
    int   layer;        /* set by the caller; only names dump lines */
} q35_gdn_cfg_t;

typedef struct {
    nita_tensor_t in_norm;   /* attn_norm.weight           [n_embd]        */
    nita_tensor_t qkv;       /* attn_qkv.weight            [ne, conv_dim]  */
    nita_tensor_t gate;      /* attn_gate.weight           [ne, inner]     */
    nita_tensor_t conv1d;    /* ssm_conv1d.weight          [k, conv_dim]   */
    nita_tensor_t alpha;     /* ssm_alpha.weight           [ne, dt_rank]   */
    nita_tensor_t beta;      /* ssm_beta.weight            [ne, dt_rank]   */
    nita_tensor_t dt_bias;   /* ssm_dt.bias                [dt_rank]       */
    nita_tensor_t ssm_a;     /* ssm_a                      [dt_rank]       */
    nita_tensor_t ssm_norm;  /* ssm_norm.weight            [head_v_dim]    */
    nita_tensor_t out;       /* ssm_out.weight             [inner, ne]     */
    nita_tensor_t post_norm; /* post_attention_norm.weight [n_embd]        */

    /* PrismML folds a Hadamard rotation into some matmuls, so the activation
     * feeding them is rotated first (prism.hadamard.weight_names). NULL when
     * the weight is not folded; rot_buf holds the rotated copy and is owned by
     * the caller. alpha/beta are never folded. */
    const nita_hadamard_t *rot_qkv;
    const nita_hadamard_t *rot_gate;
    const nita_hadamard_t *rot_out;
    float                 *rot_buf;
} q35_gdn_weights_t;

typedef struct {
    float *conv_state;   /* (conv_kernel-1) * conv_dim                */
    float *delta;        /* n_v_heads * head_k_dim * head_v_dim       */
    float *conv_w;       /* dequantised conv kernel, k * conv_dim     */
    float *buf;          /* scratch                                   */
    size_t buf_n;
} q35_gdn_state_t;

/* Validate the dimensions and allocate state. Dequantises the (tiny) conv
 * kernel once. Returns 0 on success. */
int  q35_gdn_prepare(const q35_gdn_cfg_t *c, const q35_gdn_weights_t *w,
                     q35_gdn_state_t *s, char *err, size_t errlen);
/* Zero the recurrent state so a new sequence starts fresh. */
void q35_gdn_state_reset(const q35_gdn_cfg_t *c, q35_gdn_state_t *s);
void q35_gdn_state_free(q35_gdn_state_t *s);

/* Shared NITA_Q35_DUMP handle: one stream for the engine and the mixer, so
 * writes from both cannot interleave into each other's buffer. NULL when the
 * dump is off. */
FILE *nita_q35_dump_file(void);

/* Which key head the value head `h` of the delta rule reads.
 *
 * A value head owns a state matrix but not a query, and value heads can
 * outnumber key heads (Bonsai 27B: 48 value heads over 16 key heads), so each
 * value head has to pick one q/k vector. It reads key head `h % n_k_heads`:
 * the fused GDN kernel indexes its q/k rows as `iv1 % neq1`, and the unfused
 * path gets the same mapping from ggml_repeat, which tiles a tensor. Blocked
 * indexing - `h / (n_v_heads / n_k_heads)` - reads just as plausibly and agrees
 * exactly when the two counts are equal, so only a checkpoint with more value
 * heads than key heads tells them apart. Bonsai 27B is such a checkpoint, and
 * blocked indexing dropped its mixer output to cosine 0.81. */
static inline int nita_q35_k_head(int h, int n_k_heads)
{
    return n_k_heads > 0 ? h % n_k_heads : h;
}

/* Run the mixer for one token and add it to the residual stream:
 * out = x + ssm_out(gated_rmsnorm(delta_rule(conv(qkv(norm(x)))))). */
int  q35_gdn_layer(const q35_gdn_cfg_t *c, const q35_gdn_weights_t *w,
                   q35_gdn_state_t *s, const float *x, float *out);

/* ================================================================== */
/* HTTP server                                                         */
/* ================================================================== */

typedef struct {
    const char     *model_path;
    gguf_file_t    *model;
    tokenizer_t    *tok;
    nita_model_t   *engine;      /* transformer; NULL when unavailable */
    const char     *model_id;    /* served model name */
    const char     *quant_name;  /* selected quantisation type */
    int             ctx_size;
} nita_server_t;

/* Start a blocking HTTP server. Returns 0 on clean shutdown. */
int http_serve(const char *host, int port, nita_server_t *srv);

#endif /* NITA_H */
