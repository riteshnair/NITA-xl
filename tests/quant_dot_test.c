/*
 * quant_dot_test.c - does the integer kernel agree with the float reference?
 *
 * Reads quantised tensors and compares, row by row,
 *
 *     nita_quant_dot()                        (integer domain)
 *     dequantise the row, multiply in float   (the reference)
 *
 * A layout mistake shows up as an error near 100%; the only difference that
 * should remain is the int8 quantisation of the activations, a fraction of a
 * percent per element.
 *
 *   quant_dot_test <file.gguf> <type-id> [rows]
 *   quant_dot_test --raw <blob>        (blob from tools/extract_quant_rows.py)
 */
#include "../src/nita.h"
#include "../src/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static double now_sec(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static unsigned rng_state = 12345u;

static float frand(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)((rng_state >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
}

/* Discover the quantised types the checkpoint actually stores, asking the
 * file instead of a hand-written list that could drift. Returns every stored
 * 2-d type whose rows are 32-element aligned; the caller filters for kernel
 * support. 0 on success, -1 when the table overflows. */
static int collect_stored_types(const gguf_file_t *g, uint32_t *types,
                                int max_types, int *n_types)
{
    uint64_t ti;
    int      n = 0, i;

    for (ti = 0; ti < g->tensor_count; ++ti) {
        const gguf_tensor_t *t = &g->tensors[ti];
        int                  seen = 0;

        if (t->n_dims < 2) continue;
        if (t->dims[0] % 32 != 0) continue;
        if (nita_row_bytes(t->type, t->dims[0]) == 0) continue;
        for (i = 0; i < n; ++i) if (types[i] == t->type) seen = 1;
        if (seen) continue;
        if (n == max_types) return -1;
        types[n++] = t->type;
    }
    *n_types = n;
    return 0;
}

/* One row: returns the relative error of the integer dot. */
static double one_row(uint32_t type, const uint8_t *src, uint32_t k, int nb,
                      float *row, float *x, nita_aq8_t *aq, double *abs_err)
{
    double ref = 0.0, got, scale = 0.0;
    long   i;

    if (nita_dequant_row(src, type, k, row) != 0) return 0.0;
    for (i = 0; i < (long)k; ++i) {
        ref   += (double)row[i] * (double)x[i];
        scale += fabs((double)row[i] * (double)x[i]);
    }
    got = (double)nita_quant_dot(type, src, aq, nb);
    *abs_err = fabs(got - ref);
    return scale > 0.0 ? fabs(got - ref) / scale : 0.0;
}

static int run_blob(const char *path)
{
    FILE       *f = fopen(path, "rb");
    uint32_t    hdr[4];
    uint8_t    *blob;
    size_t      need;
    int         nb, r;
    float      *row, *x;
    nita_aq8_t *aq;
    double      worst = 0.0, worst_abs = 0.0;
    long        i;

    if (!f) { perror(path); return 2; }
    if (fread(hdr, sizeof hdr, 1, f) != 1) { fclose(f); return 2; }
    need = (size_t)hdr[3] * hdr[2];
    blob = (uint8_t *)malloc(need);
    if (!blob || fread(blob, 1, need, f) != need) {
        fprintf(stderr, "short read: %s\n", path);
        fclose(f);
        free(blob);
        return 2;
    }
    fclose(f);

    nb  = (int)(hdr[1] / 32);
    row = (float *)malloc((size_t)hdr[1] * sizeof(float));
    x   = (float *)malloc((size_t)hdr[1] * sizeof(float));
    aq  = (nita_aq8_t *)malloc((size_t)nb * sizeof *aq);
    if (!row || !x || !aq) { fprintf(stderr, "oom\n"); return 2; }

    for (i = 0; i < (long)hdr[1]; ++i) x[i] = frand();
    nita_aq8_quantise(x, (int)hdr[1], aq);

    for (r = 0; r < (int)hdr[2]; ++r) {
        double ae;
        double rel = one_row(hdr[0], blob + (size_t)r * hdr[3], hdr[1], nb,
                             row, x, aq, &ae);

        if (rel > worst) worst = rel;
        if (ae > worst_abs) worst_abs = ae;
    }

    printf("%-34s type %-2u %-6s k=%-5u rows=%-2u  rel err %8.3e  abs %8.4f\n",
           path, hdr[0], ggml_type_name(hdr[0]), hdr[1], hdr[2], worst, worst_abs);

    free(row); free(x); free(aq); free(blob);
    return worst > 1e-2 ? 1 : 0;
}

/* Cost of one dot: integer path vs the dequantise-then-float path that
 * NITA_INT8_DOT=0 takes.  Same row, same activations, same buffer sizes. */
static int run_bench(const char *path, int iters)
{
    FILE       *f = fopen(path, "rb");
    uint32_t    hdr[4];
    uint8_t    *blob;
    size_t      need;
    int         nb, r, i;
    float      *row, *x, *w;
    nita_aq8_t *aq;
    double      sink = 0.0;
    double      t0, t1, ns_int, ns_flt;
    size_t      rows, k, row_bytes;

    if (!f) { perror(path); return 2; }
    if (fread(hdr, sizeof hdr, 1, f) != 1) { fclose(f); return 2; }
    need = (size_t)hdr[3] * hdr[2];
    blob = (uint8_t *)malloc(need);
    if (!blob || fread(blob, 1, need, f) != need) {
        fprintf(stderr, "short read: %s\n", path);
        fclose(f);
        free(blob);
        return 2;
    }
    fclose(f);

    k = hdr[1]; rows = hdr[2]; row_bytes = hdr[3];
    nb  = (int)(k / 32);
    row = (float *)malloc(k * sizeof(float));
    w   = (float *)malloc(k * sizeof(float));
    x   = (float *)malloc(k * sizeof(float));
    aq  = (nita_aq8_t *)malloc((size_t)nb * sizeof *aq);
    if (!row || !w || !x || !aq) { fprintf(stderr, "oom\n"); return 2; }

    for (i = 0; i < (int)k; ++i) x[i] = frand();
    nita_aq8_quantise(x, (int)k, aq);

    /* touch every block first so page faults are not in the timed loop */
    for (r = 0; r < (int)rows; ++r)
        sink += (double)nita_quant_dot(hdr[0], blob + (size_t)r * row_bytes, aq, nb);

    t0 = now_sec();
    for (i = 0; i < iters; ++i)
        for (r = 0; r < (int)rows; ++r)
            sink += (double)nita_quant_dot(hdr[0], blob + (size_t)r * row_bytes, aq, nb);
    t1 = now_sec();
    ns_int = (t1 - t0) * 1e9 / ((double)iters * (double)rows);

    t0 = now_sec();
    for (i = 0; i < iters; ++i)
        for (r = 0; r < (int)rows; ++r) {
            int j;
            float s = 0.0f;

            nita_dequant_row(blob + (size_t)r * row_bytes, hdr[0], (uint64_t)k, w);
            for (j = 0; j < (int)k; ++j) s += w[j] * x[j];
            sink += (double)s;
        }
    t1 = now_sec();
    ns_flt = (t1 - t0) * 1e9 / ((double)iters * (double)rows);

    printf("%-30s %-7s k=%-5llu int8 %8.1f ns/row  dequant+float %8.1f ns/row"
           "  x%.2f  %5.1f GB/s (sink %.1f)\n",
           path, ggml_type_name(hdr[0]), (unsigned long long)k,
           ns_int, ns_flt, ns_flt / ns_int,
           (double)row_bytes / ns_int, sink);

    free(row); free(w); free(x); free(aq); free(blob);
    return 0;
}

typedef struct {
    int    tested;
    int    failed;
    double worst;
} type_result_t;

/* Compare every tensor of one type against the float reference. One line per
 * tensor when `per_tensor` is set, a single summary line otherwise. */
static int run_type(const gguf_file_t *g, uint32_t type, int max_rows,
                    int per_tensor, type_result_t *res)
{
    uint64_t ti;

    memset(res, 0, sizeof *res);

    for (ti = 0; ti < g->tensor_count; ++ti) {
        const gguf_tensor_t *t = &g->tensors[ti];
        size_t               row_bytes, k, nrows, r;
        float               *row, *x;
        nita_aq8_t          *aq;
        double               worst = 0.0, worst_abs = 0.0;
        int                  nb;

        if (t->type != type || t->n_dims < 2) continue;
        k         = (size_t)t->dims[0];
        nrows     = (size_t)t->dims[1];
        row_bytes = nita_row_bytes(t->type, t->dims[0]);
        if (k % 32 != 0 || row_bytes == 0) continue;

        nb  = (int)(k / 32);
        row = (float *)malloc(k * sizeof(float));
        x   = (float *)malloc(k * sizeof(float));
        aq  = (nita_aq8_t *)malloc((size_t)nb * sizeof *aq);
        if (!row || !x || !aq) { free(row); free(x); free(aq); return 2; }

        for (r = 0; r < k; ++r) x[r] = frand();
        nita_aq8_quantise(x, (int)k, aq);

        for (r = 0; r < (size_t)(max_rows < (int)nrows ? max_rows : nrows); ++r) {
            double ae;
            double rel = one_row(t->type,
                                 g->data + g->data_offset + t->offset + r * row_bytes,
                                 (uint32_t)k, nb, row, x, aq, &ae);

            if (rel > worst) worst = rel;
            if (ae > worst_abs) worst_abs = ae;
        }
        if (per_tensor)
            printf("  %-40s rows=%-5llu k=%-6llu  rel err %8.3e  abs %8.4f\n",
                   t->name, (unsigned long long)nrows, (unsigned long long)k,
                   worst, worst_abs);
        ++res->tested;
        if (worst > 1e-2) ++res->failed;
        if (worst > res->worst) res->worst = worst;
        free(row); free(x); free(aq);
    }
    return 0;
}

/* Tile arbiter: the fused prefill kernels (nita_quant_dot_tile) must agree
 * with the sequential integer dot they accelerate, on the same int8-quantised
 * activations. A layout mistake in a fused kernel shows up as an error near
 * 100% - which is exactly what the Q4_K tile did while decode stayed
 * correct. The remaining difference should be float32 scale-application
 * ordering, i.e. ~1e-6; anything at 1e-3 is a bug. */
#define TILE_T 8
static int run_tile_type(const gguf_file_t *g, uint32_t type, int max_rows,
                         int *n_checked, int *n_failed, double *worst_out)
{
    uint64_t ti;

    *n_checked = 0;
    *n_failed  = 0;
    *worst_out = 0.0;

    for (ti = 0; ti < g->tensor_count; ++ti) {
        const gguf_tensor_t *t = &g->tensors[ti];
        size_t               k, nrows, row_bytes, r, maxr;
        float               *x, *tile_out;
        nita_aq8_t          *aq;
        int                  nb;

        if (t->type != type || t->n_dims < 2) continue;
        k         = (size_t)t->dims[0];
        nrows     = (size_t)t->dims[1];
        row_bytes = nita_row_bytes(t->type, t->dims[0]);
        if (k % 32 != 0 || row_bytes == 0) continue;

        nb       = (int)(k / 32);
        x        = (float *)malloc((size_t)TILE_T * k * sizeof *x);
        tile_out = (float *)malloc((size_t)TILE_T * sizeof *tile_out);
        aq       = (nita_aq8_t *)malloc((size_t)TILE_T * (size_t)nb * sizeof *aq);
        if (!x || !tile_out || !aq) { free(x); free(tile_out); free(aq); return 2; }

        {
            int a, j;

            for (a = 0; a < TILE_T; ++a) {
                for (j = 0; j < (int)k; ++j) x[(size_t)a * k + (size_t)j] = frand();
                nita_aq8_quantise(x + (size_t)a * k, (int)k, aq + (size_t)a * (size_t)nb);
            }
        }

        maxr = (size_t)(max_rows < (int)nrows ? max_rows : (int)nrows);
        for (r = 0; r < maxr; ++r) {
            const uint8_t *src = g->data + g->data_offset + t->offset + r * row_bytes;
            int             a;

            memset(tile_out, 0, (size_t)TILE_T * sizeof *tile_out);
            nita_quant_dot_tile(type, src, aq, nb, TILE_T, nb, tile_out, 1);
            for (a = 0; a < TILE_T; ++a) {
                float  want = nita_quant_dot(type, src, aq + (size_t)a * (size_t)nb, nb);
                double rel  = fabs((double)tile_out[a] - (double)want) /
                              ((double)fabs((double)want) + 1.0);

                if (rel > *worst_out) *worst_out = rel;
                ++*n_checked;
                if (rel > 1e-3) ++*n_failed;
            }
        }
        free(x); free(tile_out); free(aq);
    }
    return 0;
}

static int run_tile(const gguf_file_t *g, int max_rows)
{
    uint32_t types[64];
    int      n_types = 0, total = 0, failed = 0, i;

    if (collect_stored_types(g, types, (int)(sizeof types / sizeof types[0]),
                             &n_types) != 0) {
        fprintf(stderr, "more than %d tensor types; raise the array\n", n_types);
        return 2;
    }
    for (i = 0; i < n_types; ++i) {
        int    n_checked = 0, n_failed = 0;
        double worst = 0.0;

        if (!nita_quant_has_int_dot(types[i])) {
            printf("type %-3u %-7s  no integer kernel, skipped\n",
                   types[i], ggml_type_name(types[i]));
            continue;
        }
        if (run_tile_type(g, types[i], max_rows, &n_checked, &n_failed, &worst) != 0)
            return 2;
        printf("type %-3u %-7s  %5d dot(s), %d over the 1e-3 tolerance, worst rel err %.3e\n",
               types[i], ggml_type_name(types[i]), n_checked, n_failed, worst);
        total  += n_checked;
        failed += n_failed;
    }
    printf("%d type(s), %d dot(s), %d over tolerance\n", n_types, total, failed);
    return failed ? 1 : 0;
}

int main(int argc, char **argv)
{
    gguf_file_t *g;
    char         err[256];
    int          max_rows = 32;

    if (argc >= 3 && strcmp(argv[1], "--raw") == 0)
        return run_blob(argv[2]);

    if (argc >= 3 && strcmp(argv[1], "--bench") == 0)
        return run_bench(argv[2], argc > 3 ? atoi(argv[3]) : 2000);

    if (argc >= 3 && strcmp(argv[1], "--tile") == 0) {
        int t_max = 32, trc;

        if (argc > 3) t_max = atoi(argv[3]);
        g = gguf_load(argv[2], err, sizeof err);
        if (!g) { fprintf(stderr, "load failed: %s\n", err); return 2; }
        trc = run_tile(g, t_max);
        gguf_free(g);
        return trc;
    }

    if (argc >= 3 && strcmp(argv[1], "--all") == 0) {
        /* Discover the types the checkpoint actually stores. Naming one by hand
         * tests nothing at all if the checkpoint uses a different type, and a
         * newly added type would go unnoticed; asking the file cannot drift. */
        uint32_t types[64];
        int      n_types = 0, total = 0, failed = 0;
        uint64_t ti;
        int      i;

        if (argc > 3) max_rows = atoi(argv[3]);
        (void)ti;
        g = gguf_load(argv[2], err, sizeof err);
        if (!g) { fprintf(stderr, "load failed: %s\n", err); return 2; }

        if (collect_stored_types(g, types, (int)(sizeof types / sizeof types[0]),
                                 &n_types) != 0) {
            fprintf(stderr, "more than %d tensor types; raise the array\n", n_types);
            gguf_free(g);
            return 2;
        }

        for (i = 0; i < n_types; ++i) {
            type_result_t res;

            /* Norm weights, biases and the like are plain floats. There is no
             * integer kernel to compare against there - the point of this test
             * is the quantised path - so they are reported and skipped rather
             * than counted as failures. */
            if (!nita_quant_has_int_dot(types[i])) {
                printf("type %-3u %-7s  no integer kernel, skipped\n",
                       types[i], ggml_type_name(types[i]));
                continue;
            }
            if (run_type(g, types[i], max_rows, 0, &res) != 0) { gguf_free(g); return 2; }
            printf("type %-3u %-7s %5d tensor(s), %d over the 1e-2 tolerance,"
                   " worst rel err %.3e\n",
                   types[i], ggml_type_name(types[i]), res.tested, res.failed, res.worst);
            total  += res.tested;
            failed += res.failed;
        }

        printf("%d type(s), %d tensor(s), %d over tolerance\n", n_types, total, failed);
        gguf_free(g);
        return failed ? 1 : 0;
    }

    if (argc < 3) {
        fprintf(stderr, "usage: %s <file.gguf> <type-id> [rows]\n"
                        "       %s --all <file.gguf> [rows]   every type in the file\n"
                        "       %s --raw <blob>\n"
                        "       %s --bench <blob> [iters]\n"
                        "       %s --tile <file.gguf> [rows]  fused tile kernels vs sequential dot\n",
                argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }

    {
        type_result_t res;

        if (argc > 3) max_rows = atoi(argv[3]);
        g = gguf_load(argv[1], err, sizeof err);
        if (!g) { fprintf(stderr, "load failed: %s\n", err); return 2; }
        if (run_type(g, (uint32_t)strtoul(argv[2], NULL, 10), max_rows, 1, &res) != 0) {
            gguf_free(g);
            return 2;
        }
        printf("%d tensor(s) of this type, %d over the 1e-2 tolerance\n",
               res.tested, res.failed);
        gguf_free(g);
        return res.failed ? 1 : 0;
    }
}
