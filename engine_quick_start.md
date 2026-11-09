# NITA – Quick Start (corrected 2026-11-09)

> This document previously described a non-existent "pure LLVM-IR / MLIR"
> engine with `iree-import-gguf`, an ORC-lite driver, `-DBMC`/`-DLM_CACHE`/
> `-DSPEC_WIDTH` flags and a `validate_engine_components()` startup
> assertion. None of that exists in this repository and it never has. This
> file now documents the engine that is actually here: a plain C11 program
> built with CMake, AVX2 where available, with a scalar fallback path.

## Folder layout (real)

```
NITA-xl/
├─ CMakeLists.txt      build (x86-64; -march=znver3 if accepted, else -mavx2,
│                      else baseline; override with -DNITA_ARCH=znver3|avx2|none)
├─ src/                the engine (see README "Layout")
├─ tests/              CTest entry points
├─ tools/              reference models, gguf_probe.py, comparators
├─ quant_table.md      what the engine can decode, corrected
├─ validation_spec.md  what is actually validated, corrected
├─ component_spec.md   the real components and their checks, corrected
└─ sherlock_spec.md    the real SHERLOCK_LITE / BREADCRUMB macros, corrected
```

## Build

```bash
cmake -B build -G Ninja
cmake --build build -j$(nproc)
ctest --test-dir build            # base suite (5 tests)
# checkpoint tests (Bonsai/TinyLlama files):
cmake -B build -G Ninja -DNITA_TEST_MODELS=/path/to/model.gguf
ctest --test-dir build -L checkpoint
```

## Run

```bash
./build/nita --model tinyllama.gguf --port 7777          # serve (OpenAI API)
./build/nita --model tinyllama.gguf --inspect            # GGUF dump, exit
./build/nita --model tinyllama.gguf --tensor-table       # tab-separated table
./build/nita --model tinyllama.gguf --logits "text"      # forward pass, logits
./build/nita --model tinyllama.gguf --bench 32 --reps 5  # decode timing
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
(`"stream": true` supported; SSE is assembled after generation, not
token-incremental).

## Runtime knobs (real ones)

| Knob | Where | Meaning |
|---|---|---|
| `NITA_THREADS` | env | decode threads (default: auto, ≤32) |
| `NITA_PREFILL_SEQ=1` | env | force one-token-at-a-time prefill (tiled path is default) |
| `NITA_REP_SPREAD` | env | bench spread warning threshold (percent) |
| `-DNITA_ARCH` | CMake | `znver3` (default where accepted) / `avx2` / `none` (portable baseline) |
| `-DNITA_TEST_MODELS` | CMake | checkpoint(s) for the checkpoint test entries |
| `-DNITA_TEST_SLOW=ON` | CMake | register fork_match on checkpoints >1 GiB |
| `-DDEBUG_BREADCRUMBS=ON` | CMake | stderr breadcrumb trace (see sherlock_spec.md) |
| `-DSHERLOCK_LITE=ON` | CMake | tag lines + component timing table on stdout (zero probe code when off) |

The CMake quantisation options (`-DTERNARY`, `-DGPTQ`, `-DAWQ`, …) only set a
**label** reported by `--version` and the served `system_fingerprint`; they do
not select a kernel, there is no other "quantised GEMV" to enable. Weight
layout is chosen per tensor type at load time by the integer dot kernels in
`src/quant.c`.

## What does not exist

MLIR/IREE front-end, `model.o`, ORC-lite driver, `-DBMC`, `-DLM_CACHE`,
`-DMEO`, `-DSPEC_WIDTH`, `-DSELECTIVE_DEQUANT`, `iree-import-gguf`,
`validate_engine_components()`, speculative decode, MoE expert caches, GPTQ/AWQ
kernel lowering passes, `static_assert`-based startup validation. Any document,
script or log line referring to those is stale (see the spec docs, all corrected
2026-11-09) or aspirational, not a description of this codebase.
