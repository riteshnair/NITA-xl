# Why prefill is "painfully slow" — fishbone & full code/trace dissection

**Date:** 2026-11-09 · **Branch:** `arena/be3bc048-nita-xl` · **Head:** `83d48b6` (post-PR #9)
**Complaint:** "that is still painfully slow!! for prefill!"
**Method:** line-level read of the prefill path (`src/transformer.c`, `src/quant.c`),
the built-in component profiler, `objdump` of the hot kernel, interleaved A/B runs.
No `perf` / `valgrind` / `strace` exist in this sandbox — noted wherever a conclusion
would have needed them. Every number below was measured on this exact box unless marked.

---

## 1. The machine first (often the answer)

| Fact | Value | How known |
|---|---|---|
| CPU | Intel Xeon @ 2.60 GHz, **1 physical core, 2 hyperthreads** (cloud vCPU) | `lscpu`: 1 core, 2 threads, 54 MiB L3 |
| sysfs topology | cpu0 & cpu1 both `physical_package_id=0, core_id=0` | sysfs reads |
| Thread policy | `default_threads()` counts **physical cores** via topology → **1 thread** here | code read |
| HT siblings | `NITA_THREADS=2` → **12.06 ms/token** vs **10.32** with 1 thread (**+17% slower**) | measured A/B |
| Build target | `-march=znver3` (Zen 3): AVX2 + FMA3, **no VNNI** | CMakeLists |
| Sandbox AVX512 | cpuinfo shows avx512/vnni flags, **but the znver3 build does not use them** (target constraint) | cpuinfo + CMake |

The "5.34 ms/token" in the README was measured on a **32-physical-core** machine. This
sandbox is one shared core with two SMT siblings sharing every execution port. Comparing
the two as if they were the same hardware is the single biggest reason prefill *feels*
painfully slow here.

**Effective parallelism for int-dot kernels here: 1 physical core.** Hyperthreading is
measurably negative for this kernel, and the engine already knows that
(`default_threads` counts cores, not siblings) — that design is correct and stays.

## 2. Where the time goes (measured, Tiny-Moe Q4_K_M, `--prefill 128`, this build)

| component | tile 8 | tile 64 | share |
|---|---|---|---|
| projections (all matvecs) | 1341 ms | 1178 ms | **98.2%** |
| ↳ ffn-down (2048×768, mixed Q4_K/Q6_K) | 461 ms | 430 ms | 33.7% |
| ↳ ffn-gate | 315 ms | 259 ms | 23.0% |
| ↳ ffn-up | 290 ms | 257 ms | 21.2% |
| ↳ attn v / q / o / k | 90/66/60/52 ms | 82/50/50/… | ~17% |
| attention_tile | 19 ms | 15 ms | 1.4% |
| norms, rope, silu, embed | < 4 ms | < 4 ms | < 0.3% |
| **wall** | **1.39 s** | **1.22 s** | |

tinyllama-15M (Q3_K_M): 0.39 ms/token, projections ≈ 90%, attention ≈ 6% (context grows).
Note the projection element count (19,667,968) is **identical** in both tile runs — the
kernel does not do less work at tile 64; see §5-F6.

## 3. Line-level path trace (one matvec, the whole story)

1. `nita_prefill` loops tiles of `m->tile` (default 8) — layers strictly serial (dependency), tiles serial too (single core).
2. `matvec_tile` (`transformer.c:1497`): checks bounds, then **re-quantises the tile's X** with `nita_aq8_quantise` into `qact` scratch (reused, no alloc) — *this cost is invisible in the profiler* (happens before `prof_now()`).
3. `mv_dispatch`: `pool->n == 1` → bypasses pool, calls `mv_rows` directly (no futex, no wakeups — verified in code; no strace available).
4. `mv_rows_tile` → `nita_quant_dot_tile` (`quant.c`) → type dispatch: 12→`dot_tile_q4_K`, 14→`dot_tile_q6_K`, 6/7→Q5 tiles, 20→IQ4_NL, 8→Q8_0.
5. Kernel inner loop per 32 elements per token: `vpmaddubsw` (unsigned nibble × signed int8, pairs) → `vpmaddwd` (widen+pair → 8×int32) → `vcvtdq2ps` → `vfmadd…ps` with `d·sc·aⁿ.d` (set1 broadcast) + separate scalar `msum` for Q4_K (`dmin·mn·Σx`). Weights expand once per (row, i0=8-token group); activations hit `a->qs` 32 B loads per token.
6. `hsum256_ps` once per row-tile at the end.
7. Back up: biases (none in these models), residual adds (`pxb += ptmp`), norms (`rmsnorm`), rope, KV memcpys, `attention_tile` (already `dot4_f32`/slab AVX2 — not the problem).
8. MoE: `moe_tile_ffn` — once-per-tile router, top-k gather/scatter into `pgath/prow_tok/prow_w`, expert slices via `matvec_tile_expert` → same tile kernels. For this checkpoint E=2, top-2 ⇒ **every token runs through both experts** (gather = identity permutation). Nothing to skip.
9. Output projection only on the last tile — correct already.

## 4. Findings (each one measured)

**F1. No missing/unfused kernels remain on the prefill path.** PR #9 added the
Q6_K/Q5_0/Q5_1 tile forms; every projection in both checkpoints now runs a fused
int-domain tile kernel. Before that: 23–25 ms/token (seq fallback) → 12.05 (PR #8) →
~10.6 (PR #9). tinyllama v-proj/o-proj went 136→42 / 123→40 µs per call.

**F2. The kernels are int8-issue bound, ~1 physical core.** tiny-moe activates ≈ 125 M
weight elements per token (top-2 experts dominate). The AVX2 inner loop is ≈
3 vector ops + 1 FMA + 1 load per 32 elements; `vpmaddubsw`/`vpmaddwd` are the
throughput-limited pair. Zen-3 target ⇒ **no VNNI** (`vpdpbusd` would fuse them);
AVX-512 exists in this Xeon but the build target forbids it (user constraint: znver3).
The measured ~10.4 ms/token vs a ≈ 7.5 ms int-issue floor ≈ 70% issue efficiency —
the residue is loads, `vcvtdq2ps`, broadcasts and the scalar `msum` chain.

**F3. GCC does not, in a harmful way, spill the accumulators.** `dot_tile_q4_K` keeps
`vacc[8]`/`msum[8]` dynamically indexed; disassembly shows the accumulator being
reloaded/stored from stack per (token, j). A straight-line, fully-named-accumulator
rewrite (no dynamic indexing) measured **10.55 / 10.55 / 10.57 ms/token — neutral**
vs the spill version (~10.6). A 4+4 token split measured **11.34–11.71 — consistently
worse** (halved independent-FMA ILP). Was reverted; PR #9 kernels stay. Code churn with
zero gain is not free.

**F4. Hyperthreads are a pessimisation here, and already disabled by default.**
`NITA_THREADS=2`: 12.06 vs 10.32 ms/token (+17%). Physical-core counting stands.

**F5. Per-call fixed costs ≈ 10%.** Identical kernel + identical element count,
tile 8 → 64: projections 1.34 s → 1.18 s (−12 µs/call-ish attribution; ~1,680 fewer
`matvec_tile` invocations). Wall: 10.66 → 9.51 ms/token (−11%). Tile sweep (this box,
fused kernels): 8 → 10.66 · 16 → 10.03 · 32 → 9.61 · 64 → **9.51** · 128 → 10.04
(row re-reads through L2 start to dominate at 128). **Default kept at 8** — changing it
needs evidence on the 32-core machine (constraint from PR #8 onward). Zero-cost lever
today: `NITA_TILE=64`.

**F6. The −12% per-element gain at tile 64 is mechanism-underdetermined.** Same element
count, same kernel code, smaller sum of per-call overheads (~1,680 calls saved) and
better L1/L3 residency of repeatedly-read weight rows are the plausible drivers;
without `perf` (`cache-misses`, `cycles`) this sandbox cannot separate them further.
Marked: measured, mechanism not fully attributed.

**F7. Weights are re-read T/8 times per row (tc=8 blocking inside kernels).** At
default tile 8 that is exactly once per prefill pass — optimal for traffic; not the
bottleneck. Model (90 MiB) does not fit 54 MiB L3; DRAM demand ≈ 1.4 GB per 128-token
prefill ≈ ~1 GB/s — far below any modern interface; **not memory-bound**.

**F8. X is re-quantised to aq8 once per matvec call** (q/k/v share `phb` → 3×, gate/up
2×). It happens before `prof_now()`, invisible to the profile; wall-clock accounting
(`sum` 1.367 s vs wall 1.372 s) puts it at ≈ 0.4 % of prefill. Skip.

**F9. Attention is fine.** 1.4 % (tiny-moe) / 6.5 % (tinyllama) of wall; already
AVX2 (`dot4_f32`, 64-dim slabs, single-accumulator reductions). Not worth touching
until contexts get long.

**F10. The floor is hardware, not a defect.** On one Xeon core @ 2.6 GHz, ~125 M
int8 MACs/token through AVX2-only kernels ≈ 8–10 ms/token — exactly what we measure
(10.4). The engine is within ~1.4× of the issue-rate floor and within 30–40% of it
after PR #9. Sequential execution (layers, tiles) is inherent to the graph.

## 5. Fishbone

```
                        METHOD                          MACHINE
 ┌─────────────────────────────────────────┐  ┌───────────────────────────────────┐
 │ layers serial (dependency)              │  │ 1 physical Xeon core @2.6GHz      │
 │ tiles serial — no cross-tile overlap    │  │ 2 SMT siblings SHARE issue ports  │
 │ MoE top-2 ⇒ all tokens dense-run both   │  │ znver3 build: no VNNI (Zen 3 has  │
 │   experts (E=2 ⇒ gather=identity)       │  │   none), no AVX-512 in build      │
 │ weight rows re-read t/8 per kernel call │  │ 54 MiB L3 < 90 MiB model          │
 │ default tile 8 tuned for 32-core box    │  │ shared/noisy cloud vCPU           │
 └───────────────────┬─────────────────────┘  └───────────────┬───────────────────┘
                     │                                         │
 MEASUREMENT         │        PREFILL PAINFUL?                │  MATERIAL/DATA
 ┌───────────────────┴────────┐                ┌───────────────┴──────────────────┐
 │ profiler hides aq8 quantise│                │ weights streamed per 8-token grp │
 │ tinyllama spread 28–43%    │                │ X re-quantised per matvec call   │
 │ (shared HT); Tiny-Moe ±5%  │                │ KV memcpys per tile; f16→f32     │
 │ "5.34 ms" = 32-core, other │                │ conversions per block            │
 │ hardware — apples↛oranges  │                │ aq8 struct {i32 sum; f32 d; i8   │
 └───────────────────┬────────┘                │   qs[32]} 40 B stride            │
                     │                         └───────────────┬──────────────────┘
 ┌───────────────────┴────────┐                ┌───────────────┴──────────────────┐
 │ usage: PR #8 numbers quoted│                │ lscpu: HT on, 1 core; thread     │
 │ against 1-core sandbox     │                │ count 1 forced by physical-core  │
 │ decode is 2× SLOWER/token  │                │ policy (NITA_THREADS=2 → +17%)   │
 │ (22.5 ms) — prefill IS the │                │ 2 HT ≠ 2 cores of maddubs        │
 │ fast path already          │                │                                  │
 └────────────────────────────┘                └──────────────────────────────────┘
                     ENVIRONMENT / HUMAN (usage)             MAN  (measurement)
```

## 6. Five whys → root cause

1. **Why is prefill 10.4 ms/token?** — Because every token's ~125 M weight elements
   pass through the AVX2 int8 kernels, single-threaded.
2. **Why single-threaded / int8 kernels only?** — default_threads counts physical
   cores (correct here: 1), and the build targets znver3 (no VNNI by decision).
3. **Why one physical core?** — This sandbox is a cloud vCPU with 2 SMT siblings on
   one core; SMT measurably *hurts* this kernel (+17%).
4. **Why does one core take ~10 ms for 125 M elements?** — `vpmaddubsw`/`vpmaddwd`
   issue-rate × 2.6 GHz ≈ 32–42 GMAC/s ceiling; kernel ≈ 70% of it ≈ 10 ms. Physics.
5. **Why does it *feel* like a bug?** — Because the baseline in the README (5.34 ms)
   was a 32-physical-core machine and PR #8/#9 targets were quoted interchangeably
   with 1-core sandbox numbers. Different hardware class, ~10–30× parallel width.

**Root cause:** there is no prefill defect at HEAD. Cost = int8-issue floor on one
shared physical core, × sequential graph, plus ≈ 10% recoverable per-call overhead
(F5) and ≈ 0 otherwise. The levers are configuration and hardware, not code defects.

## 7. What would actually move the needle (ranked, honest)

| # | Lever | Expected | Constraint |
|---|---|---|---|
| 1 | Run on the 32-core box / any multi-core machine (prefill already parallelises by rows) | near-linear in physical cores | machine, not code |
| 2 | `NITA_TILE=64` on single/few-core machines | −9…−11% (9.5 vs 10.6) here | needs both-machine evidence before becoming default; measured values: 16→10.03, 32→9.61, 64→9.51, 128→10.04 |
| 3 | Zen 4/Zen 5 or AVX512-VNNI target (`vpdpbusd` fuses maddubs+madd+cvt-chain) | est. 1.3–1.6× on the kernel | violates current "zen3 for now" constraint; new target = new validation matrix |
| 4 | Leave everything as merged (PR #9) | 12.05 → ~10.4 vs before optimisation | — |
| — | Attention rewrite, norm/rope fusion, aq8 sharing for q/k/v | ≤ 1–2% | poor ROI |
| — | 4-token straight-line kernel rewrite | measured **worse** (−8%) | dead end, documented |
| — | NITA_THREADS=2 on this box | **+17% slower** | dead end, documented |
| — | VNNI-in-znver3 build | impossible — Zen 3 lacks vpdpbusd | not available |

## 8. Verification status

- Bit-identity: tiled vs `NITA_PREFILL_SEQ=1` logits, max |Δ| = **0.0** both checkpoints.
- ctest 5/5 (before, during, after the reverted rewrite).
- Decode paths untouched (tile kernels are prefill-only).
- Default `m->tile = 8` unchanged; `NITA_TILE` env override available and documented.
- The speculative kernel rewrite was **measured and reverted**; `quant.c` is byte-identical to `83d48b6` (`git diff` empty).

## 9. Open items

1. **Both-machine tile-size evidence** — if the 32-core box also prefers tile 32–64 on
   128-token prompts, propose default change (PR #8 constraint).
2. **MoE shared-expert / bonsai-scale checkpoints** — blocked by >100 MiB Drive limit;
   never numerically validated end-to-end (pre-existing, unchanged).
3. **perf counters unavailable here** — F6 attribution is provisional; worth one run
   with `perf stat -e cycles,instructions,cache-misses` on the 32-core box.
4. **Reference-engine cross-check of MoE numerics** (standing item from PR #7).
