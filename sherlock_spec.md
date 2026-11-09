# Sherlock-Lite tracing & breadcrumbs – what is implemented (corrected 2026-11-09)

> Previously this file specified a C++/`chrono`/`SHERLOCK_LITE` RAII timer with
> "device time" and a global destructor footer. This engine is C and has no
> device/runtime to measure; what exists is a lightweight accumulating profiler
> printed on demand. This is how it actually works.

## 1. SHERLOCK_LITE – tag lines and component timing

Compile with `-DSHERLOCK_LITE=ON` (CMake). Defined in `src/nita.h`:

```c
#ifdef SHERLOCK_LITE
#  define SHERLOCK_HEADER() ((void)0)
#  define SHERLOCK_ROW(tag) printf("[sherlock] %s\n", (tag))  /* the tag, nothing else */
#else
#  define SHERLOCK_HEADER() ((void)0)
#  define SHERLOCK_ROW(tag) ((void)0)
#endif
```

A row carries the tag and only the tag — no op counts, no `%dev`, no
`dev_us`/`idle_us`/`host_us`. This CPU has no device to time, and a column
of 0.0 µs next to a 100.0 % would read as a fact the probe never measured;
a trace that can be misread can manufacture the bottleneck it exists to
find. No such columns exist any more.

`src/transformer.c` accumulates per-component timings into `prof_t` buckets
(`P_embed`, `P_pq`, `P_pk`, `P_pv`, `P_norm`, `P_proj`, …) via `prof_now()` /
`prof_add()` (wall-clock, CPU only — there is no device), and `src/qwen35.c`
do the same for its six mixer phases. With the flag on, `--prefill <n>` and
`--bench <n>` print the median component table after the run. With the flag
off, both are empty inline functions: every probe site folds to zero
instructions, `nita_prof_rows()` returns 0, and no component table is
printed. That is deliberate: a probe that costs instructions inside the
measured region can skew the number being read, so a **default build carries
no probe code at all**. Build with the flag when the table is needed.

Example invocation (needs the flag):

```bash
cmake -B build -DSHERLOCK_LITE=ON && cmake --build build -j
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
- A **default build carries no probe code**: no component table is printed
  and no probe instruction exists in it, so a default-build timing cannot be
  skewed by instrumentation. With `-DSHERLOCK_LITE=ON` the ~70 probes per
  token are vDSO `clock_gettime` pairs (tens of ns each), still far below
  measurement noise — but the lit and unlit numbers are not the same build
  and should not be interleaved in one comparison.
- Numbers from `--bench` vary with machine state; the CLI prints a spread
  warning when repeated runs differ by more than `NITA_REP_SPREAD` percent
  (default: see README).
- If you see `dev us` or `idle us` printed anywhere, you are looking at this
  document's predecessor, not at the program.
