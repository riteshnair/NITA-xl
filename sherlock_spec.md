# Sherlock-Lite tracing & breadcrumbs – what is implemented (corrected 2026-11-09)

> Previously this file specified a C++/`chrono`/`SHERLOCK_LITE` RAII timer with
> "device time" and a global destructor footer. This engine is C and has no
> device/runtime to measure; what exists is a lightweight accumulating profiler
> printed on demand. This is how it actually works.

## 1. SHERLOCK_LITE – component timing (stdout)

Compile with `-DSHERLOCK_LITE=ON` (CMake). Defined in `src/nita.h`:

```c
#ifdef SHERLOCK_LITE
#  define SHERLOCK_HEADER() /* prints a header row */
#  define SHERLOCK_ROW(component, ops, dev, dev_us, idle_us, host_us) /* prints a row */
#else
#  define SHERLOCK_HEADER() ((void)0)
#  define SHERLOCK_ROW(...) ((void)0)
#endif
```

`src/transformer.c` accumulates per-component timings into `prof_t` buckets
(`P_embed`, `P_pq`, `P_pk`, `P_pv`, `P_norm`, `P_proj`, …) via `prof_now()` /
`prof_add()` (wall-clock, CPU only — there is no device) and prints them with
`--bench <n>` together with the elapsed time and the implied tok/s. With the
flag off, every macro expands to `((void)0)` and the profiling calls compile
out to nothing measurable.

Example invocation (not captured output — run it for your machine):

```bash
./build/nita --model model.gguf --bench 32 --reps 5
```

Reported counters are wall time of CPU work broken down by phase
(embedding / norms / projections / FFN / sampling-ish path), the number of
operations, and the implied fraction per phase. There is **no** `dev_us`,
`idle_us` or "instrumentation floor" line — those came from the old fictional
document and were never produced by this codebase.

## 2. BREADCRUMBS – debug trace (stderr)

Compile with `-DDEBUG_BREADCRUMBS=ON`. `BREADCRUMB(tag, msg)` /
`BREADCRUMBF(tag, fmt, …)` expand to a stderr line like

```
[nita gguf] tensor-table: 179 entries
```

and to `((void)0)` otherwise — zero cost in release builds, which is the only
build type CMake produces by default (Release). These are used at load time,
at route dispatch (`http.c`), and in the tokenizer/model paths. They are a
diagnostic, not telemetry; nothing aggregates them.

## 3. Caveats

- Wall-clock only; no cycle counters, no per-thread breakdown, no file sink.
- Numbers from `--bench` vary with machine state; the CLI prints a spread
  warning when repeated runs differ by more than `NITA_REP_SPREAD` percent
  (default: see README).
- If you see `dev us` or `idle us` printed anywhere, you are looking at this
  document's predecessor, not at the program.
