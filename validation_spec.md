# Validation – what is enforced, and what does what (corrected 2026-11-09)

> Previously this file specified a C++ `validate_engine_components()` with
> `static_assert(QUANT_TYPE …)` and MLIR pass checks. None of that exists; the
> engine is C, has no MLIR, and fails loudly at load time instead. This file
> lists the checks that actually run.

## 1. At `gguf_load()` (every model open)

- Magic `GGUF`, version 2 or 3, plausible KV/tensor counts (≤1e6 / ≤1e7).
- Every KV value is decoded bounds-checked from the file buffer (13 on-disk
  types; string arrays are counted in a first pass, then copied).
- **Tensor-table span validation (new, 2026-11):** every tensor whose type id
  this build knows must lie fully inside the data section. Overflow-checked
  64-bit geometry (`check_tensor_spans`); unknown types are skipped. Previously
  only bound tensors were individually checked at model-open time; now the
  whole table is validated up front and for every consumer (`--inspect`
  included). A truncated file is refused naming the tensor and byte counts,
  e.g.:
  `tensor 'blk.5.attn_v.weight' is truncated: file gives it 31424 bytes but ggml type 6 (Q5_0) needs 57024`.
- **Legacy/ambigious Q2_0 refusal:** a type-42 tensor whose span is too small
  for group-64 but large enough for the legacy PrismML group-128 layout is
  refused with an explanatory message rather than decoded as garbage.

## 2. At model open (`nita_model_open()`)

- Required weights are bound by name per architecture family (llama-style
  dense, qwen35 hybrid); a missing tensor fails the open with the weight's
  role in the error (`--model-table` / `model_bind@<tag>` enumerate exactly
  which tensors the model holds).
- `q35_probe@<tag>` checks Bonsai gated-delta-net geometry; it **fails by
  design** on llama-family checkpoints (documented, not a defect).

## 3. Numerical verification (tests, `ctest`)

- `quant_dot@<tag>` – every quantised type in the checkpoint vs a float
  reference (worst rel err recorded; must be ≤1e-2).
- `quant_tile@<tag>` – fused prefill tile kernels vs the sequential integer
  dot (the arbiter that catches layout bugs the float reference cannot see;
  regression for the Q4_K AVX2 group-size bug, PR #4).
- `hadamard` – FWHT vs the reference matrix (always registered).
- `tokenizer@spm|bpe` – engine vs the independent Python encoder pair.
- `fork_match@<tag>` – greedy logits vs llama.cpp (PrismML fork). **Not run in
  this workspace today** – the fork is not present. Treat quality claims that
  need it as UNVERIFIED until run where the fork exists.
- `gguf_probe@<tag>` / `model_bind@<tag>` – independent (pure-struct Python)
  check of every tensor's declared geometry and of the engine's bindings.

## 4. Baseline (no-AVX2) build verification

- Configures with `-DNITA_ARCH=none`; all AVX2 intrinsics are behind
  `__AVX2__` (scalar fallbacks). CTest 5/5 passes (base entries); logits on
  TinyLlama match the AVX2 build to 7e-6 with identical argmax (29892).

## 5. Deliberately NOT present

No static asserts, no startup self-test beyond GGUF/model-open checks, no
runtime ISA dispatch (build-time selection only: `-DNITA_ARCH`), no fuzzing
harness, no sanitizer-in-CI. Compiler warnings are checked with a
`-Wall -Wextra` scratch build (currently zero). See the Pareto report
(`pareto_deep_dive.md`) for severity-ranked findings and residual risks.
