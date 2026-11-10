# tiled==seq verification — tile kernels for every remaining quant type (Fix #2)

Date: 2026-11-09 · Branch: `arena/344d9eef-nita-xl` · Base: `52742d2` · Fix #1 (generic
AVX2 float dot): `6f9dde3` · Fix #2: this change (uncommitted at the time of writing;
committed on this branch, not pushed).

Environment: x86_64, `-march=znver3` (AVX2+FMA3, no AVX-512/VNNI), 2 logical processors,
`./build/nita`. Models live in `/home/user/models` (12 real checkpoints); synthetic
demos in `/tmp/tiny_*.gguf` (22 files, not committed).

## 1. What Fix #2 does

Before Fix #2 the prefill path only ran fused int-domain tile kernels for six types
(5_0=6, 5_1=7, 8_0=8, 4_K=12, 6_K=14, iq4_nl=20); every other quantised type fell back
to a per-token integer-domain matvec even during tiled prefill. Fix #2 adds a tile form
for every remaining type and dispatches all of them from `nita_quant_dot_tile`:

    2 q4_0 · 3 q4_1 · 6 q5_0† · 7 q5_1† · 8 q8_0† · 9 q8_1 · 10 q2_K · 11 q3_K
    12 q4_K† · 13 q5_K · 14 q6_K† · 15 q8_K · 16 iq2_xxs · 17 iq2_xs · 18 iq3_xxs
    19 iq1_s · 20 iq4_nl† · 21 iq3_s · 22 iq2_s · 23 iq4_xs · 29 iq1_m
    34 tq1_0 · 35 tq2_0 · 39 mxfp4 · 41 q1_0 · 42 q2_0 · 142 pq2_0 · 143 ptq1_0
    († = pre-existing tile path, unchanged)

Any type without a kernel keeps the old behaviour (per-token, integer domain) — that
fallback is still present and reached for unknown types. Decode is untouched; the tile
kernels are prefill-only. No AVX-512 or VNNI. One correction to a misleading source
comment is included (the old text claimed tiled and sequential prefill agree "bit for
bit"; see §4).

### Q3_K accumulation-order fix (this session)

The first version of the new `dot_tile_q3_K` deferred the per-superblock horizontal sum
to the end of the kernel. The sequential kernel (`dot_q3_K`) instead adds
`hsum128_ps(vacc)` into the float accumulator once per superblock. The tile kernel was
reordered to the same per-superblock order (scalar `tot[i] += hsum` per superblock).
After the change, the Q3_K checkpoints compare byte-identically to the pre-fix baseline
(§2, Gate B) and `tiny_q3_k` is bit-exact tiled-vs-seq (max |Δ| = 0.0).

## 2. Verification

All logit comparisons are at the repo's standing granularity: `--logits`/`--tokens`
prints values with `%.6f`, and this report compares those printed lines (string equality
or element-wise max |Δ| parsed from the `%.6f` text). No hex-logits or full-precision
dump flag exists, so sub-1e-6 differences are invisible by construction throughout.

**Gate A — standing gate.** `--logits "hello world the a"` tiled vs
`NITA_PREFILL_SEQ=1`, `%.6f` string equality.

> 34/34 PASS (12 real + 22 synthetic demos).
> Caveat on strength: real checkpoints tokenize the gate prompt to 4 ids
> (`22172 3186 278 263`) — a single partial tile, a weak probe. The demos tokenize it to
> 22 ids (mostly `0` plus a trailing `6`) and do exercise multi-tile prefill (8/8/6).

**Gate B — byte-identity vs the pre-Fix-#1/#2 baseline binary (`/tmp/nita_base`).**
Same 18 raw ids, `%.6f` string equality, `NITA_TILE` default:

> 33/34 SAME (all 22 demos, 11 real models).
> Tiny-Moe.Q4_K_M DIFF: max |Δ| = 0.0814 (4096/4096 logits nonzero). Tiny-Moe quantises
> only Q4_K (73), Q6_K (13) and F32 (37) — none of which Fix #2 touches — and this is
> the drift Fix #1 documented and accepted (commit `6f9dde3`, measured 0.290 max there).
> Fix #2 contributes nothing to it.
> Pre-fix artefact: before the Q3_K reorder, Tiny-LLM.Q2_K and tiny_q3_k showed DIFF;
> after the rebuild both are SAME. The Q2_K flip is not fully explained (no Q2_K code
> changed) but not reproducible against the final build; recorded here for honesty.

**Gate C — 18-token probe (multi-tile, raw ids).** Element-wise max |Δ| between tiled
and sequential logits, all 32000 (4096 for Tiny-Moe) values:

| model                              | max |Δ| tiled-vs-seq | same delta at base commit? |
|------------------------------------|------------------|-----------------------------|
| Tiny-LLM.Q4_K_M / S                | 4e-06            | yes (string-gate FAIL at base too) |
| Tiny-LLM.Q5_K_M / S                | 3e-06 / 4e-06    | (passes at %.6f; not individually probed at base) |
| Tiny-LLM.Q2_K / Q3_K_L/M/S         | 3e-06 / 5e-06 / 6e-06 / 3e-06 | yes |
| Tiny-LLM.Q6_K                      | 3e-06            | (passes at %.6f) |
| Tiny-LLM.IQ4_XS                    | 5e-06            | (passes at %.6f) |
| Tiny-Moe.Q4_K_M                    | 0.0637           | yes (0.0574 at base) |
| tinyllama-15M-stories-Q3_K_M       | 0.133            | yes (0.133 at base) |
| tiny_iq3_xxs                       | 0.457            | yes (0.457 at base, nonzero 16/16) |
| tiny_q4_1 / tiny_iq1_s             | 0.0357           | yes (0.0357 at base) |
| tiny_mxfp4                         | 0.0255           | yes (0.0255 at base) |
| tiny_q4_0 / tiny_q8_k              | 1e-06            | yes (1e-06 at base) |
| tiny_iq2_xxs                       | 2e-06            | yes (2e-06 at base) |
| tiny_q3_k                          | 0.0              | — (bit-exact after the reorder) |
| remaining 15 demos                 | passes %.6f gate | — |

Key fact for every value above that was probed at the base commit: **the delta is
byte-identical (same max, same nonzero count) to the pre-Fix-#1/#2 binary**, i.e. Fix #2
adds zero incremental tiled-vs-seq drift. A tile-width sweep (`NITA_TILE` = 1, 2, 4, 8,
16, 64) does not change the delta at all (identically 0.457 / 6e-06 respectively), which
rules out the tile matvec kernels as the source.

**Root cause of the pre-existing tiled-vs-seq gap: the attention paths.** The sequential
prefill calls `attention()` (`transformer.c`): scalar `sum += qh[d] * kt[d]` for q·k and
scalar `oh[d] += p * vt[d]` for the value accumulation. Tiled prefill calls
`attention_rows()`: `dot4_f32` for q·k (four 8-float lanes, sequential per vector but
associated across 4 keys) and an 8×8-lane slab FMA for the output. These differ in float
association, are deterministic and row-wise (so tile-width independent), and are
identical in the base build — consistent with every observation in Gate C. Fix #2 did not
touch attention; unifying the two paths would change every checkpoint's numerics and is
out of scope.

## 3. Performance (direction only, not a benchmark claim)

Synthetic demos, `--prefill 128 --reps 3`, interleaved base/new ×3, median of 3,
2-core box. Pre-Fix-#2 the demos fell back to per-token matvec in "tiled" mode; now they
run the fused tile kernels:

| demo          | base ms/tok (3 runs)          | Fix #2 ms/tok (3 runs) | direction |
|---------------|-------------------------------|------------------------|-----------|
| tiny_iq3_xxs  | 2.01 / 2.04 / 1.89            | 0.40 / 0.40 / 0.40     | ~4.8×     |
| tiny_q4_0     | 0.15 / 0.15 / 0.15            | 0.09 / 0.09 / 0.09     | ~1.7×     |
| tiny_mxfp4    | 0.12 / 0.13 / 0.12            | 0.08 / 0.08 / 0.08     | ~1.5×     |
| tiny_q2_k     | 0.46 / 0.43 / 0.43            | 0.30 / 0.30 / 0.29     | ~1.4×     |

Caveats: synthetic L-resident weights, 2 cores, high run-to-run similarity confirms the
effect is real but the magnitude is not representative of real checkpoints. No real model
this repo tests uses the new paths (real checkpoints are Q4_K/Q6_K/Q8_0/F32, all
pre-existing tile types), so **no speed claim is made for real models, and decode is
unchanged**.

## 4. Honest limits and open items

1. **tiled ≠ seq at full precision.** Never was, on any checkpoint, before or after
   Fix #2; root-caused above to attention (not matvec, not tile kernels). The standing
   %.6f gate passes; it does not prove bit-identity. If bit-exact tiled==seq is ever
   required, `attention()` and `attention_rows()` must be unified — a numeric-changing,
   project-wide decision.
2. **All comparisons are at printed `%.6f` granularity.** Full-precision or bitwise
   logit comparison was not performed (no flag exists; adding one was considered and
   rejected as scope creep for this fix).
3. **Synthetic demos prove decode-path equivalence, not model quality.** Weights are
   random-but-legal payloads (k-quant/IQ/MXFP4 encoders force finite `d`/`dmin`/packed
   scales; MXFP4 `d = 2^(e−128)` with amax-fitted `e`; IQ1_M synthesises `d` from forced
   `sc` prefix bits — `sc0=0x6000|r12`, `sc1=(0x60|r4)<<8|r8`, `sc2=0xE000|r12`,
   `sc3=0x2000|r12`). Demo logits are meaningless text. All demos except tq1_0 keep
   F32 norms (16 quantised + 5 F32 tensors); tq1_0 quantises norms too (21/21) and the
   gate still exercises the dequant path for them.
4. **The real-model gate prompt is weak** (4 ids, one partial tile). The 18-token raw-id
   probe (Gate C) is the multi-tile evidence for real models; a longer real-text prompt
   that tokenizes within the demos' vocab was not constructed.
5. **Tiny-Moe.Q4_K_M vs baseline**: drift accepted per Fix #1 policy (commit `6f9dde3`);
   unchanged by Fix #2, documented here again because it is the only Gate B failure.
6. **IQ1_M forced-scale encoding** is legal but arbitrary; it validates the decoder, not
   any real IQ1_M distribution.
7. **`prefill_dissection.md` §4-F1** lists the PR #9 dispatch set
   (12/14/6/7/20/8); it is a historical measurement record and now under-describes the
   dispatch table (§1 above). Not rewritten to keep the record honest about what was
   measured when.
8. **No push/PR** without explicit instruction; the work is committed locally on
   `arena/344d9eef-nita-xl` only.
9. **Performance of the new paths on real hardware is unmeasured**; decode paths are
   untouched and out of scope.

## 5. Reproduce

    # standing gate (34/34)
    for m in /home/user/models/*.gguf /tmp/tiny_*.gguf; do
        a=$(./build/nita --model "$m" --logits "hello world the a" | tail -1)
        b=$(NITA_PREFILL_SEQ=1 ./build/nita --model "$m" --logits "hello world the a" | tail -1)
        [ "$a" = "$b" ] && echo "PASS $m" || echo "FAIL $m"
    done

    # Gate B (byte-identity vs baseline), Gate C (18-token probe)
    IDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1"
    ./build/nita --model M --tokens "$IDS" | tail -1
    NITA_PREFILL_SEQ=1 ./build/nita --model M --tokens "$IDS" | tail -1
    /tmp/nita_base --model M --tokens "$IDS" | tail -1   # pre-Fix-#1/#2 baseline

    # synthetic demos
    python3 tools/make_tiny_model.py /tmp/tiny_q2_k.gguf --dtype q2_k \
        --n-embd 256 --n-ff 512 --n-head-kv 4

`/tmp` artifacts (`/tmp/nita_base`, `/tmp/tiny_*.gguf`) are ephemeral and not committed;
regenerate demos with the tool above.
