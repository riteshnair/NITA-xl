# Component Specification – what this engine actually is (corrected 2026-11-09)

> Previously this file specified an MLIR/LLVM-IR/ORC-lite engine that does not
> exist here. Rewritten to name the real components, their interfaces and
> their in-tree checks. Build system is CMake; language is C11.

| # | Component | Files / interface | Build flag | In-tree check |
|---|-----------|-------------------|------------|---------------|
| C1 | GGUF v2/v3 parser | `src/gguf.c` – magic/version/counts, KV table (all 13 types), tensor table, alignment | – | `tests/…`: `gguf_probe@<tag>` (tools/gguf_probe.py, independent struct parse), load-time span validation (`check_tensor_spans`), legacy Q2_0 layout refusal (`check_q2_0_layout`) |
| C2 | Tensor binding | `src/transformer.c` – `bind_tensor()`, name→weight mapping incl. llama- and qwen35-arch layouts | – | `model_bind@<tag>` (tools/gguf_probe.py --model-bind), `--model-table` |
| C3 | Dequantisation | `src/dequant.c` – block decoders for every type id in `ggml_type_name`, incl. F32/F16/BF16/F64 and all K-quants | – | `quant_dot@<tag>` (vs float reference), fork_match vs llama.cpp (opt-in, needs the PrismML fork) |
| C4 | Integer dot kernels | `src/quant.c` – `nita_quant_dot()`, `nita_quant_has_int_dot()`, `nita_quant_dot_tile()` (fused multi-token prefill) | – | `quant_dot@<tag>` (per type, vs reference), `quant_tile@<tag>` (sequential vs fused arbiter – catches layout slips the reference cannot see) |
| C5 | Forward pass | `src/transformer.c` – embedding, RMSNorm (AVX2 with scalar fallback), RoPE, GQA attention, KV cache, SwiGLU FFN, sampling | – | greedy/logits comparison vs llama.cpp (manual, see README); `--logits`, `--prefill`, `--bench` |
| C5b| qwen35 hybrid mixer | `src/qwen35.c` – gated delta-net for the Bonsai hybrids | – | `q35_heads` (no checkpoint needed), `q35_probe@<tag>` (Bonsai checkpoints only; **fails by design on a llama checkpoint**) |
| C6 | Hadamard rotation | `src/hadamard.c` – block Sylvester FWHT + explicit ±1 sign vector, AVX2 with scalar fallback | – | `hadamard` test entry (always registered) |
| C7 | Tokenizer | `src/tokenizer.c` – BPE (byte-level, `tokenizer.ggml.merges`) + SentencePiece score merge, byte fallback | – | `tokenizer@spm`, `tokenizer@bpe` (tools/reference_tokenizer.py vs engine) |
| C8 | Chat/completion | `src/chat.c` – history, plain vs ChatML prompt rendering (ChatML only when the checkpoint ships a template or ChatML vocab), completion accounting | – | exercised via the HTTP endpoint; `fp_nita_stub` fingerprint when weights are unusable |
| C9 | HTTP/OpenAI API | `src/http.c` – `/health`, `/v1/models`, `/v1/chat/completions` (+SSE) | – | manual smoke (see validation_spec.md); sequential accept loop by design |
| C10| Tracing | `src/nita.h` – `BREADCRUMB[F]` → stderr (`-DDEBUG_BREADCRUMBS`); `NITA_TRACE=1` (runtime, tag-only sorted timings); `-DSHERLOCK_LITE=ON` (detailed component table); disabled probes = one predictable branch per site, no probe work | both off by default | nothing asserted; they are diagnostics, not features |

## Notes

- **No startup self-validation exists** beyond what `gguf_load()` checks (magic,
  version, counts, spans, Q2_0 layout) and what `nita_model_open()` checks
  (required weights bound). A failure is reported with a message and a non-zero
  exit; nothing "recovers" or "retries".
- **No MoE, no speculative decode, no GPTQ/AWQ lowering, no selective
  de-quant, no LM/BMC caches.** The quantised path is the integer dot kernel +
  dequant-on-demand; nothing is materialised to f32 ahead of time.
- The CMake quantisation flags (`-DTERNARY`, `-DGPTQ`, …) are **labels** only.
  MXFP8 in particular sets a label and has no kernel; the model path always
  uses the type's kernel from `src/quant.c` (or dequant+matmul for non-int-dot
  types).
- Thread pool: `src/transformer.c` (`NITA_MAX_THREADS` 32), pthreads on POSIX,
  native threads on Windows; decode is parallel, the 2-vCPU test box shows no
  T=2 speedup (SMT-limited) – see README.
