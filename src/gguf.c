/*
 * gguf.c – GGUF v2/v3 container parser.
 *
 * Reads the whole file into memory, validates the magic/version, decodes
 * the metadata key-value table (all 13 on-disk value types, including
 * arrays), decodes the tensor information table, and records the absolute
 * offset of the tensor data section.
 *
 * GGUF is little-endian on every platform.
 */

#include "nita.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Little-endian reading cursor                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    const uint8_t *p;
    size_t         len;
    size_t         pos;
    bool           bad;
} cursor_t;

static void cur_init(cursor_t *c, const uint8_t *p, size_t len)
{
    c->p = p; c->len = len; c->pos = 0; c->bad = false;
}

static bool cur_need(cursor_t *c, size_t n)
{
    if (c->bad) return false;
    if (n > c->len || c->pos > c->len - n) {
        c->bad = true;
        return false;
    }
    return true;
}

static uint8_t cur_u8(cursor_t *c)
{
    if (!cur_need(c, 1)) return 0;
    return c->p[c->pos++];
}

static uint16_t cur_u16(cursor_t *c)
{
    uint16_t v = 0;
    if (!cur_need(c, 2)) return 0;
    v = (uint16_t)(c->p[c->pos] | ((uint16_t)c->p[c->pos + 1] << 8));
    c->pos += 2;
    return v;
}

static uint32_t cur_u32(cursor_t *c)
{
    uint32_t v = 0;
    if (!cur_need(c, 4)) return 0;
    v = (uint32_t)c->p[c->pos]
      | ((uint32_t)c->p[c->pos + 1] << 8)
      | ((uint32_t)c->p[c->pos + 2] << 16)
      | ((uint32_t)c->p[c->pos + 3] << 24);
    c->pos += 4;
    return v;
}

static uint64_t cur_u64(cursor_t *c)
{
    uint64_t v = 0;
    int      i;
    if (!cur_need(c, 8)) return 0;
    for (i = 7; i >= 0; --i)
        v = (v << 8) | c->p[c->pos + (size_t)i];
    c->pos += 8;
    return v;
}

static const uint8_t *cur_bytes(cursor_t *c, size_t n)
{
    const uint8_t *p;
    if (!cur_need(c, n)) return NULL;
    p = c->p + c->pos;
    c->pos += n;
    return p;
}

/* A GGUF string is a uint64 length followed by raw bytes (no NUL). */
static char *cur_string(cursor_t *c, uint64_t *out_len)
{
    uint64_t       len = cur_u64(c);
    const uint8_t *bytes;
    char          *s;

    if (c->bad) return NULL;
    if (len > (uint64_t)(c->len)) { c->bad = true; return NULL; }

    bytes = cur_bytes(c, (size_t)len);
    if (!bytes) return NULL;

    s = (char *)nita_malloc((size_t)len + 1);
    memcpy(s, bytes, (size_t)len);
    s[len] = '\0';
    if (out_len) *out_len = len;
    return s;
}

/* ------------------------------------------------------------------ */
/* ggml tensor types                                                   */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t    id;
    const char *name;
    uint32_t    block;   /* elements per block */
    uint32_t    size;    /* bytes per block */
} ggml_type_info_t;

static const ggml_type_info_t g_ggml_types[] = {
    {  0, "F32",       1,   4 },
    {  1, "F16",       1,   2 },
    {  2, "Q4_0",     32,  18 },
    {  3, "Q4_1",     32,  20 },
    {  6, "Q5_0",     32,  22 },
    {  7, "Q5_1",     32,  24 },
    {  8, "Q8_0",     32,  34 },
    {  9, "Q8_1",     32,  36 },
    { 10, "Q2_K",    256,  84 },
    { 11, "Q3_K",    256, 110 },
    { 12, "Q4_K",    256, 144 },
    { 13, "Q5_K",    256, 176 },
    { 14, "Q6_K",    256, 210 },
    { 15, "Q8_K",    256, 292 },
    { 16, "IQ2_XXS", 256,  66 },
    { 17, "IQ2_XS",  256,  74 },
    { 18, "IQ3_XXS", 256,  98 },
    { 19, "IQ1_S",   256,  50 },
    { 20, "IQ4_NL",   32,  18 },
    { 21, "IQ3_S",   256, 110 },
    { 22, "IQ2_S",   256,  82 },
    { 23, "IQ4_XS",  256, 136 },
    { 24, "I8",        1,   1 },
    { 25, "I16",       1,   2 },
    { 26, "I32",       1,   4 },
    { 27, "I64",       1,   8 },
    { 28, "F64",       1,   8 },
    { 29, "IQ1_M",   256,  56 },
    { 30, "BF16",      1,   2 },
    { 34, "TQ1_0",   256,  54 },
    { 35, "TQ2_0",   256,  66 },
    { 39, "MXFP4",    32,  17 },
    { 41, "Q1_0",    128,  18 },   /* binary, one sign bit per weight */
    { 42, "Q2_0",     64,  18 },   /* ternary, 2-bit slots, group 64 */
    { 142, "PQ2_0",  128,  34 },   /* ternary, 2-bit slots, group 128 (PrismML) */
    { 143, "PTQ1_0", 128,  28 }    /* ternary, dense base-3 trits, group 128 (PrismML) */
};

/*
 * Type id 42 has two meanings in the wild.  prism-v5 (and the frozen `prism`
 * branch) stored the ternary group-128 layout under 42; mainline llama.cpp
 * standardised 42 as group 64, and prism-v7 moved the fork's group-128 layout
 * to its own id, PQ2_0 (142).  The deprecated *-Q2_0.gguf files that are still
 * on the HuggingFace repos are the old group-128 layout under 42, so every row
 * of them is 34 bytes per 128 weights where this build expects 18 per 64.  A
 * reader that simply follows the current mapping decodes such a file as
 * garbage instead of refusing it, so check the byte span each type-42 tensor
 * actually occupies: a span too small for group 64 but large enough for group
 * 128 is the legacy layout.
 */
static bool check_q2_0_layout(const gguf_file_t *g, char *err, size_t errlen)
{
    const uint64_t data_bytes = g->file_size - g->data_offset;
    uint64_t       i, j;

    for (i = 0; i < g->tensor_count; ++i) {
        const gguf_tensor_t *t = &g->tensors[i];
        uint64_t             rows, need, legacy, span, next = 0;
        bool                 have_next = false;

        if (t->type != 42 || t->n_dims == 0) continue;
        if ((t->dims[0] % 128u) != 0) continue;  /* only group 128 would fit */
        if (t->offset > data_bytes) {
            snprintf(err, errlen,
                     "tensor '%s' starts past the end of the data section",
                     t->name);
            return false;
        }

        /* the span this tensor occupies is the gap to the next tensor record */
        for (j = 0; j < g->tensor_count; ++j) {
            const uint64_t o = g->tensors[j].offset;
            if (o > t->offset && (!have_next || o < next)) {
                next      = o;
                have_next = true;
            }
        }
        span = (have_next ? next : data_bytes) - t->offset;

        rows   = t->n_dims > 1 ? t->dims[1] : 1;
        need   = (uint64_t)nita_row_bytes(42, t->dims[0]) * rows;
        legacy = (t->dims[0] / 128u) * 34u * rows;

        if (span >= need) continue;     /* consistent with group 64 */

        if (span >= legacy) {
            snprintf(err, errlen,
                     "this file matches the legacy Prism Q2_0 layout "
                     "(group size 128 stored as ggml type id 42): tensor '%s' "
                     "holds %llu bytes, but Q2_0 is read here as the official "
                     "group-64 format and needs %llu. Use the *_g64 or PQ2_0 "
                     "file for this model",
                     t->name, (unsigned long long)span,
                     (unsigned long long)need);
        } else {
            snprintf(err, errlen,
                     "tensor '%s' is truncated: the file gives it %llu bytes "
                     "but ggml type 42 needs %llu",
                     t->name, (unsigned long long)span,
                     (unsigned long long)need);
        }
        return false;
    }

    return true;
}

static const ggml_type_info_t *find_type(uint32_t id)
{
    size_t i;
    for (i = 0; i < sizeof g_ggml_types / sizeof g_ggml_types[0]; ++i)
        if (g_ggml_types[i].id == id)
            return &g_ggml_types[i];
    return NULL;
}

const char *ggml_type_name(uint32_t type)
{
    static char buf[32];
    const ggml_type_info_t *t = find_type(type);
    if (t) return t->name;
    snprintf(buf, sizeof buf, "UNKNOWN(%u)", type);
    return buf;
}

uint32_t ggml_type_size(uint32_t type)
{
    const ggml_type_info_t *t = find_type(type);
    return t ? t->size : 1;
}

uint32_t ggml_blck_size(uint32_t type)
{
    const ggml_type_info_t *t = find_type(type);
    return t ? t->block : 1;
}

/*
 * Load-time sanity check for the whole tensor table: every tensor whose type
 * this build knows must lie fully inside the file. Without this, a corrupt
 * file that the model path never happens to bind (or a tool that reads only
 * the header) would sail through, and a binding attempt or a dequant sweep
 * over it would read out of bounds. Types this build does not know are
 * skipped: nita_row_bytes() cannot size them, and refusing the file would
 * brick checkpoints that are otherwise fine.
 */
static bool check_tensor_spans(const gguf_file_t *g, char *err, size_t errlen)
{
    const uint64_t data_bytes = g->file_size - g->data_offset;
    uint64_t       i, d;

    for (i = 0; i < g->tensor_count; ++i) {
        const gguf_tensor_t *t = &g->tensors[i];
        uint64_t             span;
        size_t               row;

        if (t->n_dims == 0 || !find_type(t->type)) continue;

        if (t->offset > data_bytes) {
            snprintf(err, errlen,
                     "tensor '%s' starts past the end of the data section",
                     t->name);
            return false;
        }

        row  = nita_row_bytes(t->type, t->dims[0]);
        span = (uint64_t)row;
        for (d = 1; d < t->n_dims; ++d) {
            if (t->dims[d] && span > UINT64_MAX / t->dims[d]) {
                snprintf(err, errlen,
                         "tensor '%s' geometry overflows 64 bits", t->name);
                return false;
            }
            span *= t->dims[d];
        }
        if (span > data_bytes - t->offset) {
            snprintf(err, errlen,
                     "tensor '%s' is truncated: file gives it %llu bytes but "
                     "ggml type %u (%s) needs %llu",
                     t->name,
                     (unsigned long long)(data_bytes - t->offset),
                     t->type, ggml_type_name(t->type),
                     (unsigned long long)span);
            return false;
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Value decoding                                                      */
/* ------------------------------------------------------------------ */
static size_t type_fixed_size(uint32_t type)
{
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0; /* string / array */
    }
}

static bool read_array(cursor_t *c, gguf_kv_t *kv, char *err, size_t errlen)
{
    uint32_t item_type = cur_u32(c);
    uint64_t count     = cur_u64(c);
    size_t   total     = 0;
    size_t   pos       = 0;
    uint64_t i;
    uint8_t *raw;

    if (c->bad) { snprintf(err, errlen, "truncated array header"); return false; }
    if (count > (uint64_t)64 * 1024 * 1024) {
        snprintf(err, errlen, "array '%s' too large (%llu entries)",
                 kv->key, (unsigned long long)count);
        return false;
    }

    if (item_type == GGUF_TYPE_STRING) {
        /* First pass: compute serialized size. */
        size_t start = c->pos;
        for (i = 0; i < count; ++i) {
            uint64_t len = cur_u64(c);
            if (c->bad) break;
            if (!cur_need(c, (size_t)len)) break;
            c->pos += (size_t)len;
            total += 8 + (size_t)len;
        }
        if (c->bad || i != count) {
            snprintf(err, errlen, "truncated string array '%s'", kv->key);
            return false;
        }
        raw = (uint8_t *)nita_malloc(total ? total : 1);
        /* Second pass: copy. */
        c->pos = start;
        for (i = 0; i < count; ++i) {
            uint64_t len = cur_u64(c);
            const uint8_t *bytes = cur_bytes(c, (size_t)len);
            if (!bytes) break;
            memcpy(raw + pos, &len, 8);
            pos += 8;
            memcpy(raw + pos, bytes, (size_t)len);
            pos += (size_t)len;
        }
        kv->item_type = item_type;
        kv->count     = count;
        kv->raw       = raw;
        return true;
    } else {
        size_t esz = type_fixed_size(item_type);
        if (esz == 0) {
            snprintf(err, errlen, "unsupported array element type %u for '%s'",
                     item_type, kv->key);
            return false;
        }
        if (count > UINT64_MAX / esz) { snprintf(err, errlen, "array overflow"); return false; }
        total = (size_t)count * esz;
        if (!cur_need(c, total)) {
            snprintf(err, errlen, "truncated numeric array '%s'", kv->key);
            return false;
        }
        raw = (uint8_t *)nita_malloc(total ? total : 1);
        memcpy(raw, c->p + c->pos, total);
        c->pos += total;
        kv->item_type = item_type;
        kv->count     = count;
        kv->raw       = raw;
        return true;
    }
}

static bool read_kv(cursor_t *c, gguf_kv_t *kv, char *err, size_t errlen)
{
    uint64_t key_len = 0;

    memset(kv, 0, sizeof *kv);
    kv->key = cur_string(c, &key_len);
    if (c->bad || !kv->key) {
        snprintf(err, errlen, "truncated metadata key");
        return false;
    }
    kv->type = cur_u32(c);
    if (c->bad) { snprintf(err, errlen, "truncated type for '%s'", kv->key); return false; }

    switch (kv->type) {
        case GGUF_TYPE_STRING: {
            uint64_t n = 0;
            kv->v.str.ptr = cur_string(c, &n);
            kv->v.str.len = n;
            if (c->bad || !kv->v.str.ptr) {
                snprintf(err, errlen, "truncated string value for '%s'", kv->key);
                return false;
            }
            return true;
        }
        case GGUF_TYPE_ARRAY:
            return read_array(c, kv, err, errlen);
        case GGUF_TYPE_UINT8:
            kv->v.u64 = cur_u8(c);  return true;
        case GGUF_TYPE_INT8:
            kv->v.i64 = (int8_t)cur_u8(c); return true;
        case GGUF_TYPE_BOOL:
            kv->v.u64 = cur_u8(c) ? 1u : 0u; return true;
        case GGUF_TYPE_UINT16:
            kv->v.u64 = cur_u16(c); return true;
        case GGUF_TYPE_INT16:
            kv->v.i64 = (int16_t)cur_u16(c); return true;
        case GGUF_TYPE_UINT32:
            kv->v.u64 = cur_u32(c); return true;
        case GGUF_TYPE_INT32:
            kv->v.i64 = (int32_t)cur_u32(c); return true;
        case GGUF_TYPE_FLOAT32: {
            uint32_t bits = cur_u32(c);
            float    f;
            memcpy(&f, &bits, 4);
            kv->v.f64 = (double)f;
            return true;
        }
        case GGUF_TYPE_UINT64:
            kv->v.u64 = cur_u64(c); return true;
        case GGUF_TYPE_INT64:
            kv->v.i64 = (int64_t)cur_u64(c); return true;
        case GGUF_TYPE_FLOAT64: {
            uint64_t bits = cur_u64(c);
            double   d;
            memcpy(&d, &bits, 8);
            kv->v.f64 = d;
            return true;
        }
        default:
            snprintf(err, errlen, "unknown metadata type %u for '%s'",
                     kv->type, kv->key);
            return false;
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */
static uint8_t *read_whole_file(const char *path, uint64_t *out_size,
                                char *err, size_t errlen)
{
    FILE    *f;
    int64_t  end;
    uint8_t *buf;
    size_t   got;

#if defined(_MSC_VER)
    if (fopen_s(&f, path, "rb") != 0) f = NULL;
#else
    f = fopen(path, "rb");
#endif
    if (!f) {
        snprintf(err, errlen, "cannot open '%s'", path);
        return NULL;
    }
    /* long is 32-bit on Windows, so plain ftell() overflows past 2 GiB and
     * these models are routinely larger than that. */
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) != 0) {
        snprintf(err, errlen, "cannot seek '%s'", path);
        fclose(f);
        return NULL;
    }
    end = _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) {
        snprintf(err, errlen, "cannot seek '%s'", path);
        fclose(f);
        return NULL;
    }
    end = (int64_t)ftello(f);
#endif
    if (end < 0) {
        snprintf(err, errlen, "cannot size '%s'", path);
        fclose(f);
        return NULL;
    }
    rewind(f);

    buf = (uint8_t *)nita_malloc((size_t)end + 1);
    got = fread(buf, 1, (size_t)end, f);
    fclose(f);
    if (got != (size_t)end) {
        snprintf(err, errlen, "short read on '%s' (%zu of %lld)", path, got, (long long)end);
        free(buf);
        return NULL;
    }
    buf[got] = 0;
    *out_size = (uint64_t)got;
    return buf;
}

gguf_file_t *gguf_load(const char *path, char *err, size_t errlen)
{
    gguf_file_t *g;
    cursor_t     c;
    uint64_t     i;
    uint64_t     alignment = 32;

    if (err && errlen) err[0] = '\0';

    g = (gguf_file_t *)nita_calloc(1, sizeof *g);
    g->path = nita_strdup(path);

    g->data = read_whole_file(path, &g->file_size, err, errlen);
    if (!g->data) { gguf_free(g); return NULL; }

    cur_init(&c, g->data, (size_t)g->file_size);

    /* Magic */
    {
        const uint8_t *m = cur_bytes(&c, 4);
        if (!m || memcmp(m, "GGUF", 4) != 0) {
            snprintf(err, errlen,
                     "'%s' is not a GGUF file (bad magic)", path);
            gguf_free(g);
            return NULL;
        }
    }

    g->version      = cur_u32(&c);
    g->tensor_count = cur_u64(&c);
    g->kv_count     = cur_u64(&c);

    if (c.bad) {
        snprintf(err, errlen, "truncated GGUF header in '%s'", path);
        gguf_free(g);
        return NULL;
    }
    if (g->version < 2 || g->version > 3) {
        snprintf(err, errlen,
                 "unsupported GGUF version %u (expected 2 or 3)", g->version);
        gguf_free(g);
        return NULL;
    }
    if (g->kv_count > 1000000 || g->tensor_count > 10000000) {
        snprintf(err, errlen, "implausible GGUF counts (kv=%llu tensors=%llu)",
                 (unsigned long long)g->kv_count,
                 (unsigned long long)g->tensor_count);
        gguf_free(g);
        return NULL;
    }

    /* Metadata */
    if (g->kv_count) {
        g->kv = (gguf_kv_t *)nita_calloc((size_t)g->kv_count, sizeof *g->kv);
        for (i = 0; i < g->kv_count; ++i) {
            if (!read_kv(&c, &g->kv[i], err, errlen)) {
                g->kv_count = i;
                gguf_free(g);
                return NULL;
            }
        }
    }

    /* Tensor table */
    if (g->tensor_count) {
        g->tensors = (gguf_tensor_t *)nita_calloc((size_t)g->tensor_count,
                                                  sizeof *g->tensors);
        for (i = 0; i < g->tensor_count; ++i) {
            gguf_tensor_t *t = &g->tensors[i];
            uint64_t       name_len = 0;
            uint32_t       d;

            t->name = cur_string(&c, &name_len);
            if (c.bad || !t->name) {
                snprintf(err, errlen, "truncated tensor name (%llu)",
                         (unsigned long long)i);
                g->tensor_count = i;
                gguf_free(g);
                return NULL;
            }
            t->n_dims = cur_u32(&c);
            if (c.bad || t->n_dims > 4) {
                snprintf(err, errlen, "bad dimension count for tensor '%s'", t->name);
                g->tensor_count = i + 1;
                gguf_free(g);
                return NULL;
            }
            t->n_elements = 1;
            for (d = 0; d < t->n_dims; ++d) {
                t->dims[d] = cur_u64(&c);
                t->n_elements *= t->dims[d];
            }
            t->type   = cur_u32(&c);
            t->offset = cur_u64(&c);

            if (c.bad) {
                snprintf(err, errlen, "truncated tensor record '%s'", t->name);
                g->tensor_count = i + 1;
                gguf_free(g);
                return NULL;
            }
        }
    }

    /* Data section starts after alignment padding. */
    {
        int64_t a = gguf_get_int(g, "general.alignment", 32);
        if (a <= 0 || a > 4096) a = 32;
        alignment = (uint64_t)a;
    }
    g->data_offset = ((c.pos + alignment - 1) / alignment) * alignment;

    if (g->data_offset > g->file_size) {
        snprintf(err, errlen,
                 "tensor data offset %llu beyond end of file (%llu bytes)",
                 (unsigned long long)g->data_offset,
                 (unsigned long long)g->file_size);
        gguf_free(g);
        return NULL;
    }

    if (!check_tensor_spans(g, err, errlen)) {
        gguf_free(g);
        return NULL;
    }

    if (!check_q2_0_layout(g, err, errlen)) {
        gguf_free(g);
        return NULL;
    }

    return g;
}

void gguf_free(gguf_file_t *g)
{
    uint64_t i;
    if (!g) return;

    if (g->kv) {
        for (i = 0; i < g->kv_count; ++i) {
            free(g->kv[i].key);
            if (g->kv[i].type == GGUF_TYPE_STRING) free(g->kv[i].v.str.ptr);
            if (g->kv[i].type == GGUF_TYPE_ARRAY)  free(g->kv[i].raw);
        }
        free(g->kv);
    }
    if (g->tensors) {
        for (i = 0; i < g->tensor_count; ++i) free(g->tensors[i].name);
        free(g->tensors);
    }
    free(g->data);
    free(g->path);
    free(g);
}

static const gguf_kv_t *find_kv(const gguf_file_t *g, const char *key)
{
    uint64_t i;
    if (!g || !key) return NULL;
    for (i = 0; i < g->kv_count; ++i)
        if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0)
            return &g->kv[i];
    return NULL;
}

bool gguf_has_key(const gguf_file_t *g, const char *key)
{
    return find_kv(g, key) != NULL;
}

const char *gguf_get_str(const gguf_file_t *g, const char *key, const char *fallback)
{
    const gguf_kv_t *kv = find_kv(g, key);
    if (kv && kv->type == GGUF_TYPE_STRING && kv->v.str.ptr)
        return kv->v.str.ptr;
    return fallback;
}

int64_t gguf_get_int(const gguf_file_t *g, const char *key, int64_t fallback)
{
    const gguf_kv_t *kv = find_kv(g, key);
    if (!kv) return fallback;
    switch (kv->type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_BOOL:    return (int64_t)kv->v.u64;
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_INT64:   return kv->v.i64;
        case GGUF_TYPE_FLOAT32:
        case GGUF_TYPE_FLOAT64: return (int64_t)kv->v.f64;
        default:                return fallback;
    }
}

double gguf_get_float(const gguf_file_t *g, const char *key, double fallback)
{
    const gguf_kv_t *kv = find_kv(g, key);
    if (!kv) return fallback;
    switch (kv->type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_BOOL:    return (double)kv->v.u64;
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_INT64:   return (double)kv->v.i64;
        case GGUF_TYPE_FLOAT32:
        case GGUF_TYPE_FLOAT64: return kv->v.f64;
        default:                return fallback;
    }
}

uint64_t gguf_array_count(const gguf_file_t *g, const char *key)
{
    const gguf_kv_t *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY) return 0;
    return kv->count;
}

const char *gguf_array_str(const gguf_file_t *g, const char *key, uint64_t index)
{
    static char *last = NULL;
    gguf_file_t     *m  = (gguf_file_t *)g;   /* cursor is scratch state */
    const gguf_kv_t *kv = find_kv(g, key);
    uint64_t         i, pos;

    if (!kv || kv->type != GGUF_TYPE_ARRAY || kv->item_type != GGUF_TYPE_STRING)
        return NULL;
    if (index >= kv->count) return NULL;

    /* Resume from the cached cursor when reading forward, which is how
     * vocabularies and merge lists are always traversed. */
    if (m->arr_cursor_kv == kv && index >= m->arr_cursor_i) {
        i   = m->arr_cursor_i;
        pos = m->arr_cursor_pos;
    } else {
        i   = 0;
        pos = 0;
    }

    for (; i < kv->count; ++i) {
        uint64_t len;
        memcpy(&len, kv->raw + pos, 8);
        pos += 8;
        if (i == index) {
            m->arr_cursor_kv  = kv;
            m->arr_cursor_i   = index + 1;
            m->arr_cursor_pos = pos + (size_t)len;

            free(last);
            last = (char *)nita_malloc((size_t)len + 1);
            memcpy(last, kv->raw + pos, (size_t)len);
            last[len] = '\0';
            return last;
        }
        pos += (size_t)len;
    }
    return NULL;
}

double gguf_array_num(const gguf_file_t *g, const char *key, uint64_t index)
{
    const gguf_kv_t *kv = find_kv(g, key);
    size_t   esz;
    const uint8_t *p;

    if (!kv || kv->type != GGUF_TYPE_ARRAY) return 0.0;
    if (index >= kv->count) return 0.0;
    esz = type_fixed_size(kv->item_type);
    if (esz == 0 || esz > 8) return 0.0;

    p = kv->raw + index * esz;
    switch (kv->item_type) {
        case GGUF_TYPE_UINT8:  { uint8_t  v; memcpy(&v, p, 1); return (double)v; }
        case GGUF_TYPE_INT8:   { int8_t   v; memcpy(&v, p, 1); return (double)v; }
        case GGUF_TYPE_UINT16: { uint16_t v; memcpy(&v, p, 2); return (double)v; }
        case GGUF_TYPE_INT16:  { int16_t  v; memcpy(&v, p, 2); return (double)v; }
        case GGUF_TYPE_UINT32: { uint32_t v; memcpy(&v, p, 4); return (double)v; }
        case GGUF_TYPE_INT32:  { int32_t  v; memcpy(&v, p, 4); return (double)v; }
        case GGUF_TYPE_FLOAT32:{ float    v; memcpy(&v, p, 4); return (double)v; }
        case GGUF_TYPE_UINT64: { uint64_t v; memcpy(&v, p, 8); return (double)v; }
        case GGUF_TYPE_INT64:  { int64_t  v; memcpy(&v, p, 8); return (double)v; }
        case GGUF_TYPE_FLOAT64:{ double   v; memcpy(&v, p, 8); return v; }
        case GGUF_TYPE_BOOL:   { uint8_t  v; memcpy(&v, p, 1); return v ? 1.0 : 0.0; }
        default: return 0.0;
    }
}
