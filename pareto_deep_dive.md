# Pareto Deep-Dive Audit – NITA-xl, 2026-11-09

**Scope:** repository-wide audit of `/home/user/NITA-xl` at base commit
`91a6688` (main, PR #5 merge) on branch `arena/561a643b-nita-xl`. Read-through of
every `src/*.c`, every `tests/*.c`, `CMakeLists.txt`, all root-level docs, and a
smoke pass over `tools/`. **Fixes are included in this branch** (the 2026-11-09
request authorised fixing all issues, including pre-existing ones). Findings
ids: `PD-xx` (this report); `F-xx` ids referenced are from the PR #4/#5 cycle.

**Verdict:** the engine is real and measurably correct on its supported path
(greedy TinyLlama output byte-identical across the changes; logits bit-equal
T=1/T=2/seq), but the repository carried a **fake kernel presented as
compiled proof**, **five spec docs describing a non-existent MLIR engine**, a
**single-threaded HTTP server killable by SIGPIPE**, a **hardcoded AVX2 build
fault**, and **zero compiler-warning hygiene**. All of these are fixed in this
branch and re-verified. Residual risks and unrun gates are listed honestly in
§7/§8.

---

## 1. Executive Pareto ranking

Impact-weighted vital few (impact = correctness/reliability/trust blast radius;
confidence = reproduced, not suspected):

| # | Finding | Class | Sev | Conf | Fixed |
|---|---------|-------|-----|------|-------|
| PD-01 | `src/q4_matmul.c` placeholder "kernel" + `kernel_probe()` announcing `[nita] kernel (Q4_K_M) probe: out=5` as if it proved anything; dead RED-phase test scaffolds; spec docs claimed it was the engine | Integrity / false evidence | P1 | FACT | ✅ removed |
| PD-02 | No SIGPIPE handling → any client that hangs up mid-response kills the single-threaded server process | Reliability | P1 | FACT (design) | ✅ `SIG_IGN` + recv timeout |
| PD-03 | `-march=znver3 -mavx2` hardcoded → SIGILL on non-AVX2 CPUs; no `__AVX2__`-less build existed | Portability | P2→P1 on old HW | FACT | ✅ configure-time detection + `-DNITA_ARCH=none` baseline, verified |
| PD-04 | Five root docs (`engine_quick_start`, `component_spec`, `validation_spec`, `quant_table`, `sherlock_spec`) specified an MLIR/IREE/ORC-lite C++ engine that does not exist (dated 2026-11-06) | Integrity / docs | P2 | FACT | ✅ rewritten to reality |
| PD-05 | Tensor table never validated at load (only bound tensors, only at model open) | Robustness | P2 | FACT | ✅ `check_tensor_spans`, overflow-checked |
| PD-06 | HTTP client receive timeout missing → silent peer wedges the sequential server forever | Reliability | P2 | FACT | ✅ 30 s `SO_RCVTIMEO` |
| PD-07 | JSON parser accepted invalid input (`+5`, `1e`, bare `-`); lone surrogates emitted as CESU-8 | Correctness | P2 | FACT | ✅ strict number grammar, U+FFFD; verified 400 |
| PD-08 | README known-gaps stale ("pool exists on Windows only" — POSIX pool merged in PR #5; hardcode claim; phantom `q4_matmul.c` in Layout; streaming sold as incremental) | Docs | P2 | FACT | ✅ corrected |
| PD-09 | Dead/broken code: `dot_i8_32`, `read_scalar_raw`, AVX2-only build of `q1_table`; 12 `-Wall -Wextra` warning lines incl. sign-compare and maybe-uninitialized | Hygiene | P3 | FACT | ✅ zero warnings |
| PD-10 | `reason_phrase()` defaulted to "OK" for unknown status; `new_id()` seed entropy weak | Polish | P3 | FACT | ✅ |
| PD-11 | `--quant-type` / CMake quant options implied a kernel switch; MXFP8 documented as "forces compile-time error" (true only inside the deleted placeholder) | Integrity (labels) | P3 | FACT | ✅ documented as labels |
| PD-12 | Stale CMake comment referencing removed test files; `tests/quant_dot_test.c` warnings | Hygiene | P3 | FACT | ✅ |

**Trivial many (documented, not all code changes):** PD-13 streaming not
token-incremental (by design, needs a generate() callback); PD-14
`gguf_array_str` returns an internal static buffer (safe today — tokenizer
copies immediately; fragile API); PD-15 `q35_probe@<llama checkpoint>` fails
**by design** (Bonsai-only probe); PD-16 `gguf_get_int` float→int64 truncation
(harmless here); PD-17 `fork_match` never run in this workspace (needs the
PrismML fork — UNVERIFIED); PD-18 no runtime ISA dispatch (build-time only);
PD-19 GO/C++-style BMC/LM-cache/MoE/spec-decode aspirations in old docs (now
explicitly documented as non-existent); PD-20 `tokenizer.merges` 10M cap not
fuzzed; PD-21 `q4_matmul_kernel_decl.h` root header deleted (was test-only).

**The 80/20 claim, honestly:** fixing PD-01..PD-03+PD-04 (four items) removes
the majority of the trust and robustness damage; PD-05..PD-08 are the next
band; the remainder are hygiene. Nothing in the "trivial many" was silently
dropped — each is either fixed or listed above.

### Pareto coverage (impact × frequency)

```
PD-01 ████████████████████  fake proof / false evidence (every serve start)
PD-02 ████████████████      server death on rude client (any client)
PD-03 ██████████████        SIGILL on non-AVX2 host (any such host)
PD-04 ████████████          wrong spec to every future reader/agent
PD-05 ████████              corrupt-file detection at load time
PD-06 ██████                wedged server by slow peer
PD-07 ████                  lenient JSON / CESU-8 emitter
PD-08..PD-12 ███            stale docs & warnings
```

---

## 2. Findings in detail

### PD-01 – placeholder kernel presented as compiled evidence (P1, FACT)
`src/q4_matmul.c` (deleted) contained scalar "placeholder tile-scale lookup"
loops reachable only via `main.c:kernel_probe()`, which printed
`[nita] kernel (Q4_K_M) probe: out=5` on **every** server start and CLI run.
`tests/kernel_test.c` / `tests/model_test.c` (deleted) were RED-phase TDD
scaffolds: no `main()`, assertions replaced by `printf`, a header
(`q4_matmul_kernel_decl.h`) that didn't match the callee's signature — never
built, and the CMake comment about them was itself stale. The quantised path is
real and good (`src/quant.c` integer dots + dequant, covered by
`quant_dot`/`quant_tile`). Fix: files deleted; `kernel_probe()` removed;
`CMakeLists.txt` sources and stale comments updated; README Layout corrected;
`-Wall -Wextra` clean. **Verification:** build+ctest+HTTP smoke below; the
startup banner no longer prints a probe; binary no longer links a placeholder.

### PD-02 – SIGPIPE kills the server (P1, FACT)
`src/http.c:sock_send_all()` sent without `MSG_NOSIGNAL` and the process never
ignored `SIGPIPE`. A client disconnecting during a response (or closing an
SSE stream) delivers SIGPIPE → default action = terminate the (sequential,
one-connection-at-a-time) server. Fix: `signal(SIGPIPE, SIG_IGN)` at
`http_serve()` entry (POSIX only), plus PD-06's timeout. **Verified:** 3
deliberately rude connections (send-half-body-then-close) — server then answered
`/health` 200.

### PD-03 – hardcoded `-march=znver3 -mavx2` (P2, FACT)
CMake had `add_compile_options(-march=znver3 -mavx2)` unconditionally → SIGILL
on pre-AVX2 x86 and non-x86 targets. Fixed: configure-time detection
(`-march=znver3` accepted → else `-mavx2` → else baseline), `-DNITA_ARCH=
znver3|avx2|none` override, all AVX2 intrinsics confirmed behind `__AVX2__`
guards with scalar fallbacks (one unguarded spot — `hsum256` in
`transformer.c` — fixed; `immintrin.h` include made x86-conditional). During
this work I also deleted `q1_table`, which was "unused" only in AVX2 builds —
**the scalar path needs it**; caught by the new baseline build (link error),
restored guarded by `#if !defined(__AVX2__)`.
**Verified:** `-DNITA_ARCH=none` → CTest 5/5 pass; logits vs AVX2 build:
max |Δ| = 7.0e-06, identical argmax (29892); `-Wall -Wextra` scratch build:
zero warnings. Runtime ISA dispatch is **not** implemented — a znver3 binary
still faults elsewhere; documented as residual risk, not claimed fixed.

### PD-04 – spec docs described a fictional engine (P2, FACT)
`engine_quick_start.md`, `component_spec.md`, `validation_spec.md`,
`quant_table.md`, `sherlock_spec.md` (all dated 6 Nov 2026) specified:
`iree-import-gguf` → MLIR → `opt -march=znver3`, an ORC-lite `main.cpp`
driver, `-DBMC`/`-DLM_CACHE`/`-DMEO`/`-DSPEC_WIDTH`/`-DSELECTIVE_DEQUANT`
flags, a C++ `validate_engine_components()` with `static_assert(QUANT_TYPE …)`,
GPTQ/AWQ "kernel lowering passes", a `chrono` RAII `sherlock_timer` with
`dev_us`/`idle_us`, and a `why not` of ignored requirements. **None of it
exists**; the CMake quant flags are labels only (see PD-11). Fix: all five
rewritten 2026-11-09 to describe the real build, the real checks (gguf spans,
Q2_0 refusal, quant_dot/quant_tile, tokenizer pair), the real tracing macros,
and explicitly what does not exist. No fictional behaviour is documented any
more.

### PD-05 – tensor table not validated at load (P2, FACT)
`gguf_load()` checked magic/counts but never that each tensor's declared bytes
fit the file; `bind_tensor()` checked only the tensors the model actually
binds, at model-open time. A corrupt-but-unbound tensor (or a tool reading
only the header) passed silently. Fix: `check_tensor_spans()` in `src/gguf.c` —
per tensor, overflow-checked 64-bit span of `nita_row_bytes(type, dim0) ×
remaining dims`, known ggml types only (unknown types are skipped rather than
mis-sized), runs at every `gguf_load()` before the Q2_0 check. **Verified:**
truncating the TinyLlama checkpoint by 400 KB now fails with
`tensor 'blk.5.attn_v.weight' is truncated: file gives it 31424 bytes but
ggml type 6 (Q5_0) needs 57024` (exit non-zero, no crash). Honest note: the
model path could not be driven out of bounds through bound tensors even before
this (bind_tensor refused them); this closes the whole-table/header-consumer
gap and gives a precise error. Not a new exploit-fix, a robustness gap closed.

### PD-06 – no client receive timeout (P2, FACT)
Sequential accept loop; a peer that connected and went silent (slowloris-style)
held the only worker forever. Fix: 30 s `SO_RCVTIMEO` on every client socket
(POSIX `timeval` / Winsock `DWORD`); on timeout `recv` fails and the request is
dropped. Generation is unaffected (timeout applies to reading the *request*).

### PD-07 – lenient JSON, CESU-8 lone surrogates (P2, FACT)
`parse_number` accepted leading `+`, `-` alone, `1e`, and never validated with
`strtod`'s end pointer. Lone `\uD800`/`\uDCxx` pairs were encoded raw (invalid
UTF-8). Fix: strict grammar (mantissa/frac/exponent digit counts enforced,
`+` rejected, `strtod` end-pointer checked), surrogate logic emits U+FFFD for
unpaired surrogates. **Verified:** `+5`, `1e`, `1.`, trailing garbage → HTTP
400; well-formed bodies unchanged (chat round-trip OK).

### PD-08 – README drift (P2, FACT)
"The pool exists on Windows only" (false since the F-03 POSIX pool, PR #5),
"hardcodes `-march=znver3 -mavx2`" (now configure-detected), Layout listing
`src/q4_matmul.c`, streaming described as if incremental. All corrected; the
remaining runtime-dispatch gap is now stated precisely.

### PD-09 – dead code & 14 warnings (P3, FACT)
Removed `dot_i8_32` (unused), `read_scalar_raw` (unused), RMS-norm unused
`i` (AVX2 build), root `q4_matmul_kernel_decl.h`; `tests/quant_dot_test.c`
sign-compare (`int` vs `size_t` ternary) and two maybe-uninitialised `ae`
(now `= 0.0`); fixed `-Wall -Wextra` to **zero** (verified by scratch build; was 12 warning lines).

### PD-10 – status phrase / id entropy (P3, FACT)
`reason_phrase()` now returns `Unknown Status` (was `OK`); `new_id()` seeds
from time ⊕ pid ⊕ clock ⊕ counter ⊕ pointer (was time ⊕ pointer).

### PD-11 – quant-flag honesty (P3, FACT)
`--quant-type`, `NITA_QUANT_NAME`, `-DTERNARY/-DGPTQ/-DAWQ/-DMXFP8` select a
**label** only (`--version`, served `system_fingerprint`, log line). MXFP8's
old "forces compile-time error" text lived in the deleted placeholder. Help
text and CMake comments now say so; `validation_spec.md`/`quant_table.md`
state exactly what is and isn't supported (int-dot ids: 2,3,6–23,29,34,35,39,
41,42,142,143; everything else = dequant+matmul or unsupported).

### PD-12 – ctest/file hygiene (P3, FACT)
CMake comment about `tests/kernel_test.c`/`model_test.c` updated (files
deleted); `quant_dot_test.c` warnings fixed.

### PD-13..PD-21 – documented, deliberate
See §1 table. Not worked around, not hidden.

---

## 3. Coverage ledger

| Unit | Role | Status | Evidence |
|------|------|--------|----------|
| `src/util.c`, `src/json.c` | buffers, JSON | **INSPECTED, VALIDATED** (fixes PD-07; rebuilt, chat round-trip OK) | build; HTTP smoke |
| `src/gguf.c` | GGUF parser | **INSPECTED, VALIDATED** (PD-05) | truncated-file refusal test |
| `src/tokenizer.c` | vocab/BPE/SPM | **INSPECTED** (no defect found; `gguf_array_str` pattern noted PD-14) | read-through |
| `src/dequant.c`, `src/quant.c` | codecs/dots | **INSPECTED, VALIDATED** (PD-09; kernels exercised by tests) | `quant_dot`, `quant_tile` PASS |
| `src/transformer.c` | model/forward | **INSPECTED, VALIDATED** (PD-03, PD-09; `--model-table` segfault previously fixed in PR #5) | logits identical; `--bench` unchanged |
| `src/qwen35.c` | hybrid mixer | **INSPECTED** (no defect found this pass) | `q35_heads` PASS; `q35_probe` by design |
| `src/hadamard.c` | FWHT | **INSPECTED, VALIDATED** | `hadamard` PASS |
| `src/http.c` | HTTP/OpenAI | **INSPECTED, VALIDATED** (PD-02/06/10/13) | /health, chat, 400s, SSE, rude clients |
| `src/chat.c` | completion | **INSPECTED** (ChatML-vs-plain logic unchanged; noted) | chat round-trip |
| `src/main.c` | CLI | **INSPECTED, VALIDATED** (PD-01, PD-11) | `--help`, `--version`, startup log |
| `src/nita.h`, `quant.h`, `iq_tables.h` | headers/tables | **INSPECTED** | read-through |
| `CMakeLists.txt` | build | **INSPECTED, VALIDATED** (PD-03, PD-11, PD-12) | builds: znver3 / avx2-scratch / none |
| `tests/*.c` | test harness | **INSPECTED** (PD-01 delete, PD-09/PD-12 fixes) | CTest 9/10 (+1 by-design) |
| `tools/*.py, tools/*.c` | reference model/probe | **SPOT-INSPECTED** (not line-audited — see §7) | gguf_probe PASS; fork driver unrun |
| Root docs (9 files) | docs | **INSPECTED, REWRITTEN** (PD-04, PD-08) | see files |
| Prior PRs #4/#5 | fixes | **IN SCOPE context** (Q4_K tile fix; F-02/03/05; segfault) | merged; re-verified here |

Uninspected in depth: `tools/reference_*.py`, `_*.py`, `compare_*.py`,
`extract_quant_rows.py`, `make_*.py` internals (test fixtures/generators);
`build/prism-llama` (absent). Status: **NOT RUN** where executables, else
UNVALIDATED. Not called exhaustive — this ledger has those gaps explicitly.

## 4. Commands executed (evidence)

- `cmake -B build -G Ninja -DNITA_TEST_MODELS=…` → build OK; `ctest` → **9/10**
  (q35_probe fails *by design* on a llama checkpoint; documented in README/tests).
- `cmake -B /tmp/nita-base -G Ninja -DNITA_ARCH=none` → build OK; CTest **5/5**;
  logits max|Δ| vs AVX2 = **7.0e-06**, argmax identical (29892).
- `cmake -B /tmp/nita-warn -G Ninja -DCMAKE_C_FLAGS="-Wall -Wextra -Wno-unused-parameter"`
  → **zero warnings** (before: 14).
- `./build/nita --model tinyllama… --logits "Once upon a time"` → argmax 29892;
  `NITA_PREFILL_SEQ=1` vs tiled vs `NITA_THREADS=2`: **bit-identical**, max|Δ|=0.0.
- HTTP: `/health` 200; greedy 64 tokens **byte-identical** to pre-change output
  (`", there was a little girl named Lily. … but she didn"`, finish=length);
  SSE stream OK; `+5`/`1e`/bad JSON → 400; 3 rude mid-body-close clients →
  server alive; `/v1/models` OK.
- `/tmp/trunc.gguf` (checkpoint minus 400 KB) → refused at load with the
  PD-05 message, exit non-zero.
- `./build/nita --model-table …` → rc 0 (regression test for the PR #5 segfault).

## 5. Not run / blocked (honest list)

1. **fork_match / llama.cpp comparison** — PrismML fork absent from this
   workspace; not rebuilt, not fetched. Quality-vs-reference claim = **UNVERIFIED**.
2. **Bonsai 27B / Ternary-Bonsai checkpoints** — not read from Drive this run;
   qwen35-specific paths re-verified only via `q35_heads`. **NOT RUN.**
3. **ARM / non-x86 compile** — `immintrin.h` now guarded, but no ARM toolchain
   here. **NOT RUN** (capability claimed only for x86-64).
4. **Ad-hoc HTTP fuzzing / TLS / auth** — server is plaintext HTTP on a
   trusted dev box by design; no authn/authz exists (OpenAI clients rarely run
   against localhost with TLS). Documented, not added.
5. **Fork-dependent and Windows paths** — no Windows runner available; the pool
   change (F-03) was compile-checked only on Linux.

## 6. Residual risks

- **Runtime ISA dispatch missing** — an AVX2 binary faults on pre-AVX2 CPUs.
  Build-time `-DNITA_ARCH=none` is the workaround; a proper dispatch (per-TU
  targets + cpuid) is the real fix; not done (touches the numerics hot path,
  needs a Bonsai run to validate).
- **Single-threaded sequential HTTP** — one slow/hung generation blocks all
  clients. Acceptable for a dev engine; a bounded thread pool or per-request
  fork is follow-up work with its own contention questions (KV cache is not
  request-reentrant).
- **Streaming is not token-incremental** — clients see one burst of SSE events.
- **`gguf_array_str` static buffer** — reentrant-hostile API; fine only while
  the tokenizer copies immediately (it does) and the server stays sequential
  (it does).
- **No fuzzing/sanitizers in CI** — ASan found the model-table segfault (PR #5);
  a sanitizer job would have caught it earlier and is cheap to add.
- **BOS mismatch vs llama.cpp** remains parked by the user (NITA adds no BOS;
  llama.cpp SPM BOS=1). Unchanged.
- **TinyLlama checkpoint is a dev proxy** — numerics validated on a 15M story
  model; K-quant correctness for other architectures rests on the reference
  path in `quant_dot`/`quant_tile` (layout-exact) rather than on broad model
  coverage.

## 7. Findings → tasks (NOT created)

Per the audit gate, **no issues, tasks or tickets have been created**. If you
want them, say where (GitHub Issues in `riteshnair/NITA-xl` needs your explicit
OK). Suggested breakdown, dependency-ordered, after your acknowledgement:

1. Runtime ISA dispatch (PD-03 residual) — effort: M; needs Bonsai-class
   validation hardware/checkpoint.
2. Request-scoped generation worker(s) or documented hard limit on concurrent
   clients (PD-02 follow-through) — effort: M.
3. Token-incremental SSE streaming via a `nita_generate` callback — effort: M.
4. Sanitizer CI job (ASan+UBSan on `ctest`) — effort: S.
5. Fork_match in CI where the PrismML fork is available — effort: S.
6. Optional: fuzz `gguf_load`/`json_parse` — effort: M.

## 8. Notes on prior fixes (context, not re-audited)

PR #4 fixed the Q4_K AVX2 tile group-size bug (red→green: 384/384 type-12
failures pre-fix → 11008 dots, worst rel err 0.000e+00). PR #5 fixed F-02
(load-time quant refusal), F-03 (POSIX pthread pool), F-05 (AVX2 rmsnorm), the
`--model-table` qwen35 NULL-array segfault (ASan-pinned), and added the
`quant_tile@<tag>` arbiter. All remain intact in this branch (verified above).

---
*Report generated 2026-11-09 on branch `arena/561a643b-nita-xl`. Evidence over
assumption: every "verified" above names the command or observation. Everything
not verified is listed in §5/§6 as NOT RUN/UNVERIFIED, not as PASS.*
