/*
 * main.c – NITA command line entry point.
 *
 *   nita --model <path.gguf> [--port 7777] [--host 0.0.0.0]
 *        [--ctx 2048] [--name <served-model-id>] [--quant-type Q]
 *        [--inspect] [--tensor-table] [--model-table] [--version] [--help]
 *
 * --inspect prints the GGUF metadata and tensor table, then exits.
 * --tensor-table prints the header facts and every tensor as tab-separated
 * fields for tools/gguf_probe.py --engine to compare, then exits.
 * --model-table does the same for the weights the loaded model bound.
 */

#include "nita.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

/* The quantisation preset the binary was configured with.  The CMake flags define each
 * selector as the integer 1, so the human-readable name must be chosen with
 * an explicit #if chain rather than by stringizing QUANT_TYPE.  It is a label
 * for --version and the server log only: the model path uses the quantised
 * dot kernels in src/quant.c, selected by tensor type at load time. */
#ifndef QUANT_TYPE
#  define QUANT_TYPE Q4_K_M
#endif

#if defined(TERNARY) && QUANT_TYPE == TERNARY
#  define NITA_QUANT_NAME "TERNARY"
#elif defined(GPTQ) && QUANT_TYPE == GPTQ
#  define NITA_QUANT_NAME "GPTQ"
#elif defined(AWQ) && QUANT_TYPE == AWQ
#  define NITA_QUANT_NAME "AWQ"
#elif defined(BINARY_NET) && QUANT_TYPE == BINARY_NET
#  define NITA_QUANT_NAME "BINARY_NET"
#elif defined(BF16) && QUANT_TYPE == BF16
#  define NITA_QUANT_NAME "BF16"
#elif defined(FP16) && QUANT_TYPE == FP16
#  define NITA_QUANT_NAME "FP16"
#elif defined(MXFP8) && QUANT_TYPE == MXFP8
#  define NITA_QUANT_NAME "MXFP8"
#else
#  define NITA_QUANT_NAME "Q4_K_M"
#endif

static void print_usage(FILE *f, const char *argv0)
{
    fprintf(f,
        "NITA %s - NC Inference Tool for All\n"
        "\n"
        "Usage:\n"
        "  %s --model <path.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>      GGUF model to load (required)\n"
        "  --host <addr>       bind address (default 0.0.0.0)\n"
        "  --port <n>          listen port (default 7777)\n"
        "  --ctx <n>           context size reported to clients (default 2048)\n"
        "  --name <id>         model id served by /v1/models (default: file stem)\n"
        "  --quant-type <name> label to report for the configured quantisation preset\n"
        "  --inspect           print model metadata and exit\n"
        "  --tensor-table      print every tensor's name, type, dims, offsets\n"
        "                      and a hash of its first bytes, tab-separated\n"
        "  --model-table       print every weight the loaded model holds: field,\n"
        "                      tensor, pointer offset and hash\n"
        "  --tokenize <text>   print token ids and decode round-trip, then exit\n"
        "                      (\"-\" reads the text from stdin)\n"
        "  --logits <text>     run the forward pass on text, print logits\n"
        "  --tokens <a,b,c>    run the forward pass on explicit token ids\n"
        "  --prefill <n>       measure prompt processing: n tokens evaluated in\n"
        "                      one pass (NITA_PREFILL_SEQ=1 for the old\n"
        "                      one-token-at-a-time path)\n"
        "  --bench <n>         measure decode throughput and dump the component\n"
        "                      profile (n single-token forwards)\n"
        "  --reps <n>          runs per benchmark, reported as a median with a\n"
        "                      spread warning (default 5; NITA_REP_SPREAD sets\n"
        "                      the warning threshold in percent)\n"
        "  --version           print version and exit\n"
        "  --help              show this message\n"
        "\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  GET  /v1/models\n"
        "  POST /v1/chat/completions   (supports \"stream\": true)\n",
        NITA_VERSION, argv0);
}

/* Strip a trailing .gguf and return a malloc'd id. */
static char *derive_model_id(const char *path)
{
    const char *base = nita_basename(path);
    size_t      len = strlen(base);
    char       *id;

    if (len > 5 && strcmp(base + len - 5, ".gguf") == 0) len -= 5;
    id = (char *)nita_malloc(len + 1);
    memcpy(id, base, len);
    id[len] = '\0';
    return id;
}

/* Read all of stdin as UTF-8 bytes. */
static char *read_stdin_text(void)
{
    size_t cap = 256, len = 0;
    char  *buf = (char *)nita_malloc(cap);
    int    c;

    while ((c = getchar()) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            buf = (char *)nita_realloc(buf, cap);
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';
    return buf;
}

/* Tokenize a string, print the ids and verify the decode round-trip. */
static int run_tokenize(const tokenizer_t *tok, const char *arg)
{
    size_t   n = 0;
    int32_t *ids;
    char    *back;
    char    *owned = NULL;
    const char *text = arg;
    size_t   i;

    if (!tok || !tok->has_vocab) {
        fprintf(stderr, "nita: model has no vocabulary\n");
        return 1;
    }

    /* "-" reads the text from stdin, which is how non-ASCII input reaches
     * the engine on Windows, where argv is limited to the ANSI code page. */
    if (strcmp(arg, "-") == 0) {
        owned = read_stdin_text();
        text = owned;
    }

    ids = tokenizer_encode(tok, text, &n);

    printf("mode    : %s\n", tok->mode_name);
    printf("tokens  : %zu\n", n);
    printf("ids     :");
    for (i = 0; i < n; ++i) printf(" %d", ids[i]);
    printf("\n");

    back = tokenizer_decode(tok, ids, n);
    printf("decoded : \"%s\"\n", back ? back : "");
    printf("roundtrip: %s\n", (back && strcmp(back, text) == 0) ? "exact" : "DIFFERS");

    free(back);
    free(ids);
    free(owned);
    return 0;
}

/* Run the forward pass over explicit token ids and print the logits. */
static int run_logits_ids(const gguf_file_t *g, const int32_t *ids, size_t n)
{
    char          err[256];
    nita_model_t *m;
    float        *logits;
    int           nv;
    size_t        i;

    m = nita_model_open(g, n + 16 > 256 ? (int)n + 16 : 256, err, sizeof err);
    if (!m) {
        fprintf(stderr, "nita: %s\n", err);
        return 1;
    }

    nv = nita_model_n_vocab(m);
    logits = (float *)calloc((size_t)nv, sizeof(float));

    /* One call for the whole run. The tiled path is the default; with
     * NITA_PREFILL_SEQ=1 the same ids go through nita_forward one at a time,
     * so comparing the two outputs checks the tiled path against the
     * reference (identical up to float rounding in the attention pass). */
    if (nita_prefill(m, ids, (int)n, logits) != 0) {
        fprintf(stderr, "nita: forward failed on %zu tokens\n", n);
        free(logits);
        nita_model_close(m);
        return 1;
    }

    printf("tokens:");
    for (i = 0; i < n; ++i) printf(" %d", ids[i]);
    printf("\nlogits:");
    for (i = 0; i < (size_t)nv; ++i) printf(" %.6f", logits[i]);
    printf("\n");

    free(logits);
    nita_model_close(m);
    return 0;
}

/* Measure prompt processing: the whole run through the engine in one pass.
 * With NITA_PREFILL_SEQ=1 the same tokens go through one forward each, which
 * is the pre-tiling path, so the two runs are directly comparable. */
/* ------------------------------------------------------------------ */
/* Repeated measurement                                                */
/* ------------------------------------------------------------------ */
/*
 * One run of either benchmark is not reproducible on this machine: background
 * load moves the same workload by tens of percent between runs, so a single
 * number is partly a measurement of what else was running. Both modes
 * therefore repeat the measurement and report the median, and a set whose
 * spread is too wide says so instead of being quoted as a result.
 *
 * --reps <n>            runs to take (default 5, clamped to 1..15)
 * NITA_REP_SPREAD=<pct> spread that triggers the warning (default 10,
 *                       0 disables the warning)
 */

#define NITA_MAX_REPS 15

/* Why a single run was slow.
 *
 * A median hides the reason one run drifted, so every run is bracketed by two
 * cheap counter reads. Our own process CPU says how much of the machine we
 * asked for; the system counters say how much everything else did. The gap
 * between them is the load that actually competes with us, and it is usually
 * the whole explanation for a run that was slow in isolation.
 */
typedef struct {
    double proc_s;      /* this process, kernel + user, seconds */
    double busy_s;      /* every process, busy seconds        */
    int    have_sys;
} load_snap_t;

typedef struct {
    int             reps;                              /* requested */
    int             n;                                 /* recorded  */
    double          secs[NITA_MAX_REPS];
    double          cpu[NITA_MAX_REPS];      /* our cores busy, mean */
    double          sys[NITA_MAX_REPS];      /* machine busy, 0..1   */
    double          other[NITA_MAX_REPS];    /* cores busy elsewhere */
    int             have_sys;
    int             ncpu;
    int             keep[NITA_MAX_REPS];     /* 1 = counted in the median */
    int             nkeep;
    double          load_lim;                /* cores of external load */
    nita_prof_row_t prof[NITA_MAX_REPS][NITA_PROF_MAX_ROWS];
    int             nprof;
} rep_set_t;

#if defined(_WIN32)
static double ft_secs(FILETIME t)
{
    return ((double)t.dwHighDateTime * 4294967296.0 +
            (double)t.dwLowDateTime) / 1e7;
}
#endif

static int cpu_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;

    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    return 1;
#endif
}

static void load_snap(load_snap_t *s)
{
    s->proc_s   = 0.0;
    s->busy_s   = 0.0;
    s->have_sys = 0;
#if defined(_WIN32)
    {
        FILETIME cre, ext, kern, user, idle, skern, suser;

        if (GetProcessTimes(GetCurrentProcess(), &cre, &ext, &kern, &user)) {
            s->proc_s = ft_secs(kern) + ft_secs(user);
            if (GetSystemTimes(&idle, &skern, &suser)) {
                /* GetSystemTimes folds idle into the kernel figure. */
                s->busy_s   = ft_secs(skern) - ft_secs(idle) + ft_secs(suser);
                s->have_sys = 1;
            }
        }
    }
#else
    s->proc_s = (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static double median_of(const double *v, int n)
{
    double tmp[NITA_MAX_REPS];
    int    i;

    if (n <= 0) return 0.0;
    if (n > NITA_MAX_REPS) n = NITA_MAX_REPS;
    for (i = 0; i < n; ++i) tmp[i] = v[i];
    qsort(tmp, (size_t)n, sizeof tmp[0], cmp_double);
    return (n & 1) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

static void rep_init(rep_set_t *r, int reps)
{
    memset(r, 0, sizeof *r);
    if (reps < 1) reps = 1;
    if (reps > NITA_MAX_REPS) reps = NITA_MAX_REPS;
    r->reps = reps;
    r->ncpu = cpu_count();
}

/* `a` and `b` bracket the run whose wall time is `secs`. */
static void rep_record(rep_set_t *r, double secs,
                       const load_snap_t *a, const load_snap_t *b)
{
    double wall = secs > 1e-9 ? secs : 1e-9;
    double dproc, dbusy;

    if (r->n >= r->reps) return;
    r->secs[r->n] = secs;
    r->keep[r->n] = 1;

    dproc = b->proc_s - a->proc_s;
    dbusy = b->busy_s - a->busy_s;
    r->cpu[r->n] = dproc / wall;
    if (b->have_sys && r->ncpu > 0) {
        r->sys[r->n]   = dbusy / (wall * (double)r->ncpu);
        r->other[r->n] = r->sys[r->n] * (double)r->ncpu - r->cpu[r->n];
        if (r->other[r->n] < 0.0) r->other[r->n] = 0.0;
        r->have_sys    = 1;
    }
    r->nprof = nita_prof_rows(r->prof[r->n], NITA_PROF_MAX_ROWS);
    ++r->n;
    r->nkeep = r->n;
}

/* Drop runs that ran under far more external load than the rest of the set.
 *
 * The test uses the load counters only, never the timings, so this cannot
 * select on the result: a run is excluded because the machine was busy, not
 * because it came out slow.
 *
 * The baseline is the quietest run in the set, not the middle one. If load
 * arrived halfway through, most runs are polluted and their median would be
 * the polluted level, so the clean minority would look like the outlier and
 * nothing would be discarded. The lightest run is the best available estimate
 * of what this machine costs when the benchmark has it to itself, and the
 * margin below it absorbs ordinary jitter.
 *
 * Two runs always survive, and the report says so, because a median of two is
 * a mean and has no spread of its own.
 * NITA_REP_LOAD=<cores> pins the limit instead of deriving it.
 */
static void rep_mark_clean(rep_set_t *r)
{
    const char *e = getenv("NITA_REP_LOAD");
    double      pinned = e ? atof(e) : 0.0;
    double      base, margin, lim;
    int         i, kept = 0, floor_n;

    if (!r->have_sys || r->n < 3) return;

    base = r->other[0];
    for (i = 1; i < r->n; ++i)
        if (r->other[i] < base) base = r->other[i];

    /* Half the baseline, but never less than two cores: small absolute loads
     * must not be split into "quiet" and "polluted" on noise alone. */
    margin = base * 0.5 > 2.0 ? base * 0.5 : 2.0;
    lim = pinned > 0.0 ? pinned : base + margin;
    r->load_lim = lim;

    for (i = 0; i < r->n; ++i)
        if (r->other[i] > lim) r->keep[i] = 0;

    for (i = 0; i < r->n; ++i) kept += r->keep[i];
    floor_n = 2;                    /* a median needs two numbers */
    while (kept < floor_n) {
        int worst = -1;

        for (i = 0; i < r->n; ++i)
            if (!r->keep[i] && (worst < 0 || r->other[i] < r->other[worst]))
                worst = i;
        if (worst < 0) break;
        r->keep[worst] = 1;
        ++kept;
    }
    r->nkeep = kept;
}

static double rep_median_secs(const rep_set_t *r)
{
    double v[NITA_MAX_REPS];
    int    i, n = 0;

    for (i = 0; i < r->n; ++i)
        if (r->keep[i]) v[n++] = r->secs[i];
    return n ? median_of(v, n) : median_of(r->secs, r->n);
}

/* Says which runs were left out and on what evidence. */
static void rep_report_excluded(const rep_set_t *r)
{
    int i, dropped = 0, worst = -1;

    for (i = 0; i < r->n; ++i) {
        if (r->keep[i]) continue;
        ++dropped;
        if (worst < 0 || r->other[i] > r->other[worst]) worst = i;
    }
    if (!dropped) return;
    printf("  excluded: %d of %d runs, external load above %.1f cores "
           "(worst run %d at %.1f)\n",
           dropped, r->n, r->load_lim, worst + 1, r->other[worst]);
    if (r->nkeep < 3)
        printf("  note: %d runs left; the median is the mean of two and "
               "carries no spread of its own\n", r->nkeep);
    fflush(stdout);
}

static void rep_print_runs(const rep_set_t *r, long units)
{
    int i;

    if (r->n < 2 || units <= 0) return;
    if (r->have_sys)
        printf("  load: cpu = our cores busy, sys = whole machine busy, "
               "other = the rest\n");
    for (i = 0; i < r->n; ++i) {
        printf("  run %d: %.3f s  %.2f ms/token", i + 1, r->secs[i],
               1000.0 * r->secs[i] / (double)units);
        if (r->have_sys)
            printf("   cpu %.1f  sys %2.0f%%  other %.1f",
                   r->cpu[i], 100.0 * r->sys[i], r->other[i]);
        if (!r->keep[i])
            printf("   [excluded: load]");
        printf("\n");
    }
}

/* Gap between the fastest and slowest run, relative to their median. */
static void rep_report_spread(const rep_set_t *r, long units)
{
    const char *e = getenv("NITA_REP_SPREAD");
    double      limit = e ? atof(e) : 10.0;
    double      v[NITA_MAX_REPS];
    double      lo, hi, med, pct;
    int         i, nv = 0;
    int         lo_i = -1, hi_i = -1;    /* run indices, for the load line */
    int         lo_v = 0, hi_v = 0;      /* and their places in v[]      */

    if (r->n < 2 || units <= 0) return;

    /* Spread of the runs that survived, which is the spread of the number
     * being reported above it. The excluded runs are accounted for on their
     * own line, so nothing disappears silently. */
    for (i = 0; i < r->n; ++i) {
        if (!r->keep[i]) continue;
        v[nv] = 1000.0 * r->secs[i] / (double)units;
        if (lo_i < 0 || v[nv] < v[lo_v]) { lo_i = i; lo_v = nv; }
        if (hi_i < 0 || v[nv] > v[hi_v]) { hi_i = i; hi_v = nv; }
        ++nv;
    }
    if (lo_i < 0 || hi_i < 0) return;
    lo = v[lo_v];
    hi = v[hi_v];
    med = median_of(v, nv);
    pct = med > 0.0 ? 100.0 * (hi - lo) / med : 0.0;

    printf("  spread: %.1f%% over %d runs (min %.2f, max %.2f ms/token)\n",
           pct, nv, lo, hi);

    /* Attribute the spread: a slow run under someone else's load is not the
     * same evidence as a slow run on an otherwise idle machine. */
    if (r->have_sys) {
        printf("  load: fastest run cpu %.1f sys %.0f%% other %.1f | "
               "slowest run cpu %.1f sys %.0f%% other %.1f (%d CPUs)\n",
               r->cpu[lo_i], 100.0 * r->sys[lo_i], r->other[lo_i],
               r->cpu[hi_i], 100.0 * r->sys[hi_i], r->other[hi_i], r->ncpu);
    }
    if (limit > 0.0 && pct > limit)
        printf("  WARNING: spread exceeds %.0f%% - this number is not stable on "
               "this machine right now\n", limit);
    fflush(stdout);
}

/* Component table of the run set: each row is the median of that component
 * across runs, so one slow run cannot move any of them. */
static void rep_print_profile(const rep_set_t *r)
{
    nita_prof_row_t med[NITA_PROF_MAX_ROWS];
    double          tot = 0.0;
    int             nprof = r->n ? r->nprof : 0;
    int             i, k;

    if (nprof <= 0) return;

    for (i = 0; i < nprof; ++i) {
        double us[NITA_MAX_REPS], calls[NITA_MAX_REPS];
        int    nk = 0;

        /* Same runs as the reported timing: an excluded run must not drag
         * the component medians either. */
        for (k = 0; k < r->n; ++k) {
            if (!r->keep[k]) continue;
            us[nk]    = r->prof[k][i].us;
            calls[nk] = (double)r->prof[k][i].calls;
            ++nk;
        }
        if (nk == 0) { us[0] = 0.0; calls[0] = 0.0; nk = 1; }
        med[i].name   = r->prof[0][i].name;
        med[i].us     = median_of(us, nk);
        med[i].calls  = (long)median_of(calls, nk);
        med[i].subset = r->prof[0][i].subset;
        if (!med[i].subset) tot += med[i].us;
    }
    if (tot <= 0.0) tot = 1.0;

    printf("%-16s%8s%14s%12s%8s\n", "component", "calls", "total_us", "us/call", "%");
    for (i = 0; i < nprof; ++i) {
        printf("%-16s%8ld%14.1f%12.2f%7.1f%%\n",
               med[i].name, med[i].calls, med[i].us,
               med[i].calls ? med[i].us / (double)med[i].calls : 0.0,
               100.0 * med[i].us / tot);
    }
    printf("%-16s%8s%14.1f\n", "sum", "", tot);
    fflush(stdout);
}

static int run_prefill(const gguf_file_t *g, const tokenizer_t *tok, int n, int reps)
{
    char          err[256];
    nita_model_t *m;
    int32_t      *base = NULL, *ids = NULL;
    size_t        nbase = 0, i;
    float        *logits;
    int           nv, rc = 1;
    clock_t       t0;
    double        secs;
    const char   *seq;
    rep_set_t     set;
    int           rep_ok = 1;
    int           rep_i;

    if (!tok || !tok->has_vocab) {
        fprintf(stderr, "nita: model has no vocabulary; cannot tokenize\n");
        return 1;
    }
    if (n <= 0) n = 32;

    m = nita_model_open(g, n + 8, err, sizeof err);
    if (!m) {
        fprintf(stderr, "nita: %s\n", err);
        return 1;
    }

    base = tokenizer_encode(tok, "The quick brown fox jumps over the lazy dog. "
                                 "Prompt processing should stream the weights "
                                 "once for the whole prompt, not once per token.",
                            &nbase);
    if (!base || nbase == 0) {
        fprintf(stderr, "nita: tokenization produced no tokens\n");
        nita_model_close(m);
        return 1;
    }

    ids = (int32_t *)calloc((size_t)n, sizeof(int32_t));
    for (i = 0; i < (size_t)n; ++i) ids[i] = base[i % nbase];

    nv = nita_model_n_vocab(m);
    logits = (float *)calloc((size_t)nv, sizeof(float));

    /* Warm the tile buffers with a short pass so the first recorded run is not
     * the one paying for their allocation. */
    nita_model_reset(m);
    if (nita_prefill(m, ids, n < 8 ? n : 8, logits) != 0) {
        fprintf(stderr, "nita: prefill failed\n");
        goto done;
    }

    rep_init(&set, reps);
    for (rep_i = 0; rep_i < set.reps; ++rep_i) {
        load_snap_t la, lb;

        nita_model_reset(m);
        nita_prof_reset();
        load_snap(&la);
        t0 = clock();
        if (nita_prefill(m, ids, n, logits) != 0) {
            fprintf(stderr, "nita: prefill failed\n");
            rep_ok = 0;
            break;
        }
        load_snap(&lb);
        rep_record(&set, (double)(clock() - t0) / (double)CLOCKS_PER_SEC, &la, &lb);
    }
    if (!rep_ok) goto done;

    rep_mark_clean(&set);
    secs = rep_median_secs(&set);
    seq = getenv("NITA_PREFILL_SEQ");
    rep_print_runs(&set, n);
    printf("\nprefill: %d tokens in %.3f s -> %.1f tok/s (%.2f ms/token) [%s]%s\n",
           n, secs, secs > 0.0 ? (double)n / secs : 0.0,
           secs > 0.0 ? 1000.0 * secs / (double)n : 0.0,
           (seq && *seq && *seq != '0') ? "sequential" : "tiled",
           set.reps > 1 ? " (median)" : "");
    rep_report_excluded(&set);
    rep_report_spread(&set, n);
    rep_print_profile(&set);
    rc = 0;
done:
    free(logits);
    free(ids);
    free(base);
    nita_model_close(m);
    return rc;
}

/* Measure steady-state decode throughput and dump the component profile.
 * Single-token forwards only, with no per-token output, so the numbers
 * reflect the engine rather than the terminal. */
static int run_bench(const gguf_file_t *g, const tokenizer_t *tok, int n, int reps)
{
    char          err[256];
    nita_model_t *m;
    int32_t      *ids = NULL;
    size_t        ntok = 0, i;
    float        *logits;
    int           nv;
    clock_t       t0;
    double        secs;
    rep_set_t     set;
    int           rep_i;

    if (!tok || !tok->has_vocab) {
        fprintf(stderr, "nita: model has no vocabulary; cannot tokenize\n");
        return 1;
    }
    if (n <= 0) n = 8;

    m = nita_model_open(g, n + 32, err, sizeof err);
    if (!m) {
        fprintf(stderr, "nita: %s\n", err);
        return 1;
    }

    ids = tokenizer_encode(tok, "The quick brown fox jumps over the lazy dog.", &ntok);
    if (!ids || ntok == 0) {
        fprintf(stderr, "nita: tokenization produced no tokens\n");
        nita_model_close(m);
        return 1;
    }

    nv = nita_model_n_vocab(m);
    logits = (float *)calloc((size_t)nv, sizeof(float));

    /* One warm-up forward so first-touch page faults stay out of the window. */
    if (nita_forward(m, ids[0], logits) != 0) {
        fprintf(stderr, "nita: forward failed\n");
        free(logits); free(ids); nita_model_close(m);
        return 1;
    }

    rep_init(&set, reps);
    for (rep_i = 0; rep_i < set.reps; ++rep_i) {
        /* Each run starts from an empty cache so every rep walks the same
         * positions: the second one must not be measuring a longer context. */
        load_snap_t la, lb;

        nita_model_reset(m);
        nita_prof_reset();
        load_snap(&la);
        t0 = clock();
        for (i = 0; i < (size_t)n; ++i) {
            if (nita_forward(m, ids[i % ntok], logits) != 0) {
                fprintf(stderr, "nita: forward failed at %zu\n", i);
                free(logits); free(ids); nita_model_close(m);
                return 1;
            }
        }
        load_snap(&lb);
        rep_record(&set, (double)(clock() - t0) / (double)CLOCKS_PER_SEC, &la, &lb);
    }

    rep_mark_clean(&set);
    secs = rep_median_secs(&set);
    rep_print_runs(&set, n);
    printf("\nbench: %d forwards in %.3f s -> %.2f tok/s (%.1f ms/token)%s\n",
           n, secs, secs > 0.0 ? (double)n / secs : 0.0,
           secs > 0.0 ? 1000.0 * secs / (double)n : 0.0,
           set.reps > 1 ? " (median)" : "");
    rep_report_excluded(&set);
    rep_report_spread(&set, n);
    rep_print_profile(&set);

    /* Sampling happens outside the forward pass, so it never appears in the
     * component table above. Time it with the settings a caller actually
     * uses, otherwise a slow sampler looks like free work. */
    {
        uint32_t rng = 12345u;
        clock_t  s0;
        double   ssecs;
        int32_t  chosen = 0;
        int      k;

        for (k = 40; k >= 0; k -= 40) {
            s0 = clock();
            for (i = 0; i < (size_t)n; ++i)
                chosen = nita_sample(logits, nv, 0.8f, k, &rng);
            ssecs = (double)(clock() - s0) / (double)CLOCKS_PER_SEC;
            printf("sample top_k=%-6d: %.2f ms/call (last id %d)\n",
                   k, 1000.0 * ssecs / (double)n, (int)chosen);
            fflush(stdout);
        }
    }

    free(logits);
    free(ids);
    nita_model_close(m);
    return 0;
}

/* Run the forward pass over a string and print the resulting logits.
 * Used to compare the C engine against the Python reference. */
static int run_logits(const gguf_file_t *g, const tokenizer_t *tok, const char *text)
{
    char          err[256];
    nita_model_t *m;
    int32_t      *ids = NULL;
    size_t        n = 0;
    float        *logits;
    int           nv;
    size_t        i;

    if (!tok || !tok->has_vocab) {
        fprintf(stderr, "nita: model has no vocabulary; cannot tokenize\n");
        return 1;
    }

    m = nita_model_open(g, 256, err, sizeof err);
    if (!m) {
        fprintf(stderr, "nita: %s\n", err);
        return 1;
    }

    ids = tokenizer_encode(tok, text, &n);
    if (!ids || n == 0) {
        fprintf(stderr, "nita: tokenization produced no tokens\n");
        nita_model_close(m);
        return 1;
    }

    nv = nita_model_n_vocab(m);
    logits = (float *)calloc((size_t)nv, sizeof(float));

    for (i = 0; i < n; ++i) {
        if (nita_forward(m, ids[i], logits) != 0) {
            fprintf(stderr, "nita: forward failed at token %zu\n", i);
            free(logits); free(ids); nita_model_close(m);
            return 1;
        }
    }

    printf("tokens:");
    for (i = 0; i < n; ++i) printf(" %d", ids[i]);
    printf("\nlogits:");
    for (i = 0; i < (size_t)nv; ++i) printf(" %.6f", logits[i]);
    printf("\n");

    free(logits);
    free(ids);
    nita_model_close(m);
    return 0;
}

/* One tensor per line: name, type id, dims, offset from the header, the absolute
 * file offset the forward pass will read it at, and a hash of the bytes that
 * pointer leads to. The human-readable --inspect listing stops at 40 tensors and
 * omits offsets, so a mis-read entry in the middle of a 851-tensor file would
 * not show up there, and an offset that parses correctly but points somewhere
 * else only shows up as wrong logits.
 *
 * The pointers come from nita_bind_all, i.e. the same bounds-checked path that
 * binds the weights for the forward pass, not a second copy of the arithmetic.
 * That is what makes the hash a statement about the weights. */
static int tensor_table(const gguf_file_t *g)
{
    nita_tensor_t *bound = NULL;
    uint64_t       i;
    int            nb = 0;

    printf("version\t%u\n", g->version);
    printf("tensors\t%llu\n", (unsigned long long)g->tensor_count);
    printf("keys\t%llu\n", (unsigned long long)g->kv_count);
    printf("data_offset\t%llu\n", (unsigned long long)g->data_offset);

    if (g->tensor_count) {
        bound = (nita_tensor_t *)calloc((size_t)g->tensor_count, sizeof *bound);
        if (!bound) {
            fprintf(stderr, "out of memory for %llu bindings\n",
                    (unsigned long long)g->tensor_count);
            return 1;
        }
        nb = nita_bind_all(g, bound, (size_t)g->tensor_count);
        if (nb != (int)g->tensor_count) {
            fprintf(stderr, "binding failed: a tensor does not fit in the file\n");
            free(bound);
            return 1;
        }
    }

    for (i = 0; i < g->tensor_count; ++i) {
        const gguf_tensor_t *t = &g->tensors[i];
        uint64_t             abs_off = (uint64_t)(bound[i].data - g->data);
        size_t               n = g->file_size - abs_off < NITA_TENSOR_CHECK_BYTES
                                 ? (size_t)(g->file_size - abs_off)
                                 : NITA_TENSOR_CHECK_BYTES;
        uint32_t             d;

        printf("%s\t%u\t", t->name, (unsigned)t->type);
        for (d = 0; d < t->n_dims; ++d)
            printf("%s%llu", d ? "," : "", (unsigned long long)t->dims[d]);
        printf("\t%llu\t%llu\t%016llx\n", (unsigned long long)t->offset,
               (unsigned long long)abs_off,
               (unsigned long long)nita_fnv1a64(bound[i].data, n));
    }

    free(bound);
    return 0;
}

static void inspect(const gguf_file_t *g, const tokenizer_t *tok)
{
    uint64_t i;

    printf("GGUF version    : %u\n", g->version);
    printf("file            : %s\n", g->path);
    printf("size            : %s\n", nita_human_bytes(g->file_size));
    printf("metadata keys   : %llu\n", (unsigned long long)g->kv_count);
    printf("tensors         : %llu\n", (unsigned long long)g->tensor_count);
    printf("data offset     : %llu\n", (unsigned long long)g->data_offset);
    printf("architecture    : %s\n", gguf_get_str(g, "general.architecture", "(none)"));
    printf("name            : %s\n", gguf_get_str(g, "general.name", "(none)"));

    if (tok && tok->has_vocab)
        printf("vocabulary      : %u pieces, mode=%s, merges=%u (bos=%d eos=%d unk=%d)\n",
               tok->n_tokens, tok->mode_name, tok->n_merges,
               tok->id_bos, tok->id_eos, tok->id_unk);
    else
        printf("vocabulary      : (absent)\n");

    printf("\nMetadata:\n");
    for (i = 0; i < g->kv_count; ++i) {
        const gguf_kv_t *kv = &g->kv[i];

        if (kv->type == GGUF_TYPE_STRING) {
            size_t n = (size_t)kv->v.str.len;
            if (n > 60)
                printf("  %-42s = \"<string, %zu bytes>\"\n", kv->key, n);
            else
                printf("  %-42s = \"%s\"\n", kv->key, kv->v.str.ptr);
        } else if (kv->type == GGUF_TYPE_ARRAY) {
            printf("  %-42s = array[%llu] of type %u\n", kv->key,
                   (unsigned long long)kv->count, kv->item_type);
        } else if (kv->type == GGUF_TYPE_FLOAT32 || kv->type == GGUF_TYPE_FLOAT64) {
            printf("  %-42s = %g\n", kv->key, kv->v.f64);
        } else if (kv->type == GGUF_TYPE_INT8 || kv->type == GGUF_TYPE_INT16 ||
                   kv->type == GGUF_TYPE_INT32 || kv->type == GGUF_TYPE_INT64) {
            printf("  %-42s = %lld\n", kv->key, (long long)kv->v.i64);
        } else {
            printf("  %-42s = %llu\n", kv->key, (unsigned long long)kv->v.u64);
        }
    }

    printf("\nTensors (first 40):\n");
    for (i = 0; i < g->tensor_count && i < 40; ++i) {
        const gguf_tensor_t *t = &g->tensors[i];
        uint32_t             d;

        printf("  %-40s %-8s dims=[", t->name, ggml_type_name(t->type));
        for (d = 0; d < t->n_dims; ++d)
            printf("%s%llu", d ? "," : "", (unsigned long long)t->dims[d]);
        printf("]\n");
    }
    if (g->tensor_count > 40)
        printf("  ... %llu more\n", (unsigned long long)(g->tensor_count - 40));
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *host = "0.0.0.0";
    const char *served_name = NULL;
    const char *quant_label = NULL;
    const char *logits_text = NULL;
    const char *tokens_arg = NULL;
    const char *tokenize_text = NULL;
    int         port = 7777;
    int         ctx = 2048;
    int         want_inspect = 0;
    int         want_tensor_table = 0;
    int         want_model_table = 0;
    int         bench_n = 0;
    int         prefill_n = 0;
    int         reps_n = 5;
    int         i;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            print_usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(a, "--version") == 0) {
            printf("NITA %s (kernel: %s)\n", NITA_VERSION, NITA_QUANT_NAME);
            return 0;
        } else if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(a, "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(a, "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            ctx = atoi(argv[++i]);
        } else if (strcmp(a, "--name") == 0 && i + 1 < argc) {
            served_name = argv[++i];
        } else if (strcmp(a, "--quant-type") == 0 && i + 1 < argc) {
            quant_label = argv[++i];
        } else if (strcmp(a, "--inspect") == 0) {
            want_inspect = 1;
        } else if (strcmp(a, "--tensor-table") == 0) {
            want_tensor_table = 1;
        } else if (strcmp(a, "--model-table") == 0) {
            want_model_table = 1;
        } else if (strcmp(a, "--bench") == 0 && i + 1 < argc) {
            bench_n = atoi(argv[++i]);
        } else if (strcmp(a, "--prefill") == 0 && i + 1 < argc) {
            prefill_n = atoi(argv[++i]);
        } else if (strcmp(a, "--reps") == 0 && i + 1 < argc) {
            reps_n = atoi(argv[++i]);
        } else if (strcmp(a, "--logits") == 0 && i + 1 < argc) {
            logits_text = argv[++i];
        } else if (strcmp(a, "--tokens") == 0 && i + 1 < argc) {
            tokens_arg = argv[++i];
        } else if (strcmp(a, "--tokenize") == 0 && i + 1 < argc) {
            tokenize_text = argv[++i];
        } else {
            fprintf(stderr, "nita: unknown argument '%s'\n\n", a);
            print_usage(stderr, argv[0]);
            return 2;
        }
    }

    if (!model_path) {
        fprintf(stderr, "nita: --model is required\n\n");
        print_usage(stderr, argv[0]);
        return 2;
    }
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "nita: --port must be between 1 and 65535\n");
        return 2;
    }
    if (ctx <= 0) ctx = 2048;

    {
        char          err[512];
        gguf_file_t  *g;
        tokenizer_t  *tok;
        nita_model_t *engine = NULL;
        char         *derived = NULL;
        nita_server_t srv;
        int           rc;

        printf("NITA %s (kernel: %s)\n", NITA_VERSION,
               quant_label ? quant_label : NITA_QUANT_NAME);
        printf("loading %s ...\n", model_path);
        fflush(stdout);

        g = gguf_load(model_path, err, sizeof err);
        if (!g) {
            fprintf(stderr, "nita: failed to load model: %s\n", err);
            return 1;
        }

        BREADCRUMB("model", "GGUF parsed");

        printf("loaded %s (%s, GGUF v%u, %llu tensors, %llu metadata keys)\n",
               nita_basename(g->path),
               nita_human_bytes(g->file_size),
               g->version,
               (unsigned long long)g->tensor_count,
               (unsigned long long)g->kv_count);

        tok = tokenizer_from_gguf(g);

        if (tokenize_text) {
            int trc = run_tokenize(tok, tokenize_text);
            tokenizer_free(tok);
            gguf_free(g);
            return trc;
        }

        if (want_tensor_table) {
            int trc = tensor_table(g);
            tokenizer_free(tok);
            gguf_free(g);
            return trc;
        }

        /* Build the engine only far enough to bind every weight: the report is
         * about pointers, so the smallest context keeps the KV cache out of the
         * way (a 64-layer model at the default 2048 is gigabytes of it). */
        if (want_model_table) {
            char          buf[256];
            nita_model_t *engine = nita_model_open(g, 1, buf, sizeof buf);
            int           mrc;

            if (!engine) {
                fprintf(stderr, "model not loadable: %s\n", buf);
                tokenizer_free(tok);
                gguf_free(g);
                return 1;
            }
            mrc = nita_model_tensor_report(engine, g);
            nita_model_close(engine);
            tokenizer_free(tok);
            gguf_free(g);
            return mrc < 0 ? 1 : 0;
        }

        if (want_inspect) {
            inspect(g, tok);
            tokenizer_free(tok);
            gguf_free(g);
            return 0;
        }

        if (bench_n > 0) {
            int brc = run_bench(g, tok, bench_n, reps_n);
            tokenizer_free(tok);
            gguf_free(g);
            return brc;
        }

        if (prefill_n != 0) {
            int prc = run_prefill(g, tok, prefill_n, reps_n);
            tokenizer_free(tok);
            gguf_free(g);
            return prc;
        }

        if (tokens_arg) {
            int32_t ids[512];
            size_t  n = 0;
            char   *copy = nita_strdup(tokens_arg);
            char   *p = copy;
            int     lrc;

            while (*p && n < sizeof ids / sizeof ids[0]) {
                char *end;
                long  v = strtol(p, &end, 10);
                if (end == p) break;
                ids[n++] = (int32_t)v;
                p = end;
                while (*p == ',' || *p == ' ') ++p;
            }
            free(copy);

            if (n == 0) {
                fprintf(stderr, "nita: --tokens expects comma separated ids\n");
                tokenizer_free(tok);
                gguf_free(g);
                return 2;
            }
            lrc = run_logits_ids(g, ids, n);
            tokenizer_free(tok);
            gguf_free(g);
            return lrc;
        }

        if (logits_text) {
            int lrc = run_logits(g, tok, logits_text);
            tokenizer_free(tok);
            gguf_free(g);
            return lrc;
        }

        SHERLOCK_HEADER();
        SHERLOCK_ROW("model-load");

        /* Build the transformer. A metadata-only file has no usable weights,
         * in which case we say so and let the API fall back. */
        {
            char engine_err[256];
            engine = nita_model_open(g, ctx, engine_err, sizeof engine_err);
            if (engine)
                printf("transformer ready: ctx=%d vocab=%d\n",
                       nita_model_ctx(engine), nita_model_n_vocab(engine));
            else
                printf("transformer unavailable: %s\n",
                       engine_err[0] ? engine_err : "unknown reason");
            fflush(stdout);
        }

        memset(&srv, 0, sizeof srv);
        srv.model_path = model_path;
        srv.model      = g;
        srv.tok        = tok;
        srv.engine     = engine;
        srv.ctx_size   = ctx;
        srv.quant_name = quant_label ? quant_label : NITA_QUANT_NAME;

        if (served_name) {
            srv.model_id = served_name;
        } else {
            derived = derive_model_id(model_path);
            srv.model_id = derived;
        }

        printf("serving as model id '%s'\n", srv.model_id);
        fflush(stdout);

        rc = http_serve(host, port, &srv);

        nita_model_close(engine);
        free(derived);
        tokenizer_free(tok);
        gguf_free(g);
        return rc;
    }
}
