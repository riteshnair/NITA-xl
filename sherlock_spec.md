# Sherlock-Lite tracing & breadcrumbs – what is implemented (corrected 2026-11-09)

> Previously this file specified a C++/`chrono`/`SHERLOCK_LITE` RAII timer with
> "device time" and a global destructor footer. This engine is C and has no
> device/runtime to measure; what exists is a lightweight accumulating profiler
> printed on demand. This is how it actually works.

## 1. NITA_TRACE / SHERLOCK_LITE – runtime tag timings and the component table

Two ways to switch the traces on, and one output mode each:

1. **`NITA_TRACE=1` in the environment** (runtime, no rebuild). `--prefill` /
   `--bench` print the **tag-only time trace**: one line per tag, highest
   total first, ms to three decimals. Nothing else — no call counts, no
   us/call, no percentages. `NITA_TRACE=0` silences a lit build too.
2. **`-DSHERLOCK_LITE=ON` at configure time**. The same probes are compiled
   in and always on; with the env var unset the CLI prints the detailed
   component table as before. (With `NITA_TRACE=1` it prints the tag-only
   lines instead.)

Example:

```bash
NITA_TRACE=1 ./build/nita --model model.gguf --prefill 128 --reps 5
```

```
[trace] projections      1187.292 ms
[trace]   kernel:q4_K    826.344 ms
[trace]   ffn-down       425.509 ms
[trace]   kernel:q6_K    359.378 ms
[trace]   ffn-gate       265.108 ms
[trace]   ffn-up         264.600 ms
...
[trace] aq8-quantise     18.049 ms
```

With the trace off, every probe site is one predictable not-taken branch —
no clock reads, no counter updates, nothing printed. That is deliberately
not "the probes compile out": a *runtime* trace requires the probe code to
exist, and the price of that is a single branch per site in every build.
Measured, the price is unmeasurable: interleaved Tiny-Moe `--prefill 128`
runs of the trace-capable build against the pre-trace binary agree within
spread (≈9.94 vs ≈9.99 ms/token, one-core Xeon run). The SHERLOCK header/row
macros still exist for compile-time tag lines (`[sherlock] <tag>`); a row
carries the tag and only the tag — no op counts, no device microseconds
this CPU has no way to produce.

`src/transformer.c` carries the accumulators: the existing component buckets
(`P_proj`, `P_pq`, `P_pg`, `P_attn`, …), plus `aq8-quantise` (the activation
quantiser, bracketed before the projection kernel — not a subset of
`projections`) and per-kernel scratch `kernel:q4_K`, `kernel:q5_K`,
`kernel:q6_K`, `kernel:q5_0`, `kernel:q5_1`, `kernel:q8_0`, `kernel:iq4_nl`,
`kernel:other` (subsets of `projections`, selected by the tensor's quant
type). `src/qwen35.c` keeps its six mixer phase probes on the same switch.

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
- With the trace disabled, probe sites do no work: one cached env-var check
  per site, nothing timed, nothing printed, so a disabled trace cannot skew
  a benchmark. With it enabled, the ~70 probes per token are vDSO
  `clock_gettime` pairs (tens of ns each), still far below measurement noise
  — but traced and untraced numbers are not the same run regime and should
  not be interleaved in one comparison to chase sub-percent effects.
- Numbers from `--bench` vary with machine state; the CLI prints a spread
  warning when repeated runs differ by more than `NITA_REP_SPREAD` percent
  (default: see README).
- If you see `dev us` or `idle us` printed anywhere, you are looking at this
  document's predecessor, not at the program.
