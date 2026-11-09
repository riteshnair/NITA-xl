# Quantisation support – what the engine decodes (corrected 2026-11-09)

> Previously this table mandated a single "selective de-quant" kernel with
> per-tile flags and `-D` toggles for GPTQ/AWQ/BF16 paths that do not exist.
> This is the actual situation. "Int dot" = integer-domain dot-product kernel
> in `src/quant.c` (`nita_quant_has_int_dot`); the model path always uses the
> int kernel when one exists, otherwise dequant+matmul. Rows are ggml type
> ids/names as used on disk.

| Bit/type | ggml id | bpw-ish | Int dot kernel | Dequant (matmul path) | Notes |
|---|---|---|---|---|---|
| F32 | 0 | 32 | – | yes | reference dtype |
| F16 | 1 | 16 | – | yes | |
| Q4_0 | 2 | 4.5 | yes | yes | |
| Q4_1 | 3 | 5.0 | yes | yes | |
| Q5_0 | 6 | 5.5 | yes | yes | |
| Q5_1 | 7 | 6.0 | yes | yes | |
| Q8_0 | 8 | 8.5 | yes | yes | |
| Q8_1 | 9 | 9.0 | yes | yes | |
| Q2_K | 10 | 2.6 | yes | yes | |
| Q3_K | 11 | 3.4 | yes | yes | K-quant blocks, group 256 |
| Q4_K | 12 | 4.5 | yes | yes | engine default for Q4_K_M checkpoints; fused tile path `nita_quant_dot_tile` |
| Q5_K | 13 | 5.5 | yes | yes | |
| Q6_K | 14 | 6.6 | yes | yes | |
| Q8_K | 15 | 8.5 | yes | yes | activation side only in ggml terms; decoded here |
| IQ2_XXS | 16 | ~2.1 | yes | yes | |
| IQ2_XS | 17 | ~2.3 | yes | yes | |
| IQ3_XXS | 18 | ~3.1 | yes | yes | |
| IQ1_S | 19 | ~1.6 | yes | yes | |
| IQ4_NL | 20 | 4.5 | yes | yes | |
| IQ3_S | 21 | ~3.4 | yes | yes | |
| IQ2_S | 22 | ~2.6 | yes | yes | |
| IQ4_XS | 23 | 4.25 | yes | yes | |
| IQ2_M | 27 | ~2.5 | – | yes (ids 24–28,30,141-family) | dequant only |
| IQ1_M | 29 | ~1.75 | yes | yes | |
| TQ1_0 | 34 | ~1.7 | yes | yes | upstream ternary, group 256 |
| TQ2_0 | 35 | ~2.1 | yes | yes | upstream ternary, group 256 |
| MXFP4 | 39 | 4.25 | yes | yes | E4M3-ish block float |
| Q1_0 | 41 | 1.12 | yes | yes | binary, group 128 (Bonsai Q1_0) |
| Q2_0 | 42 | 2.25 | yes | yes | group 64; the legacy PrismML group-128 file under id 42 is **refused at load** – use PQ2_0 |
| PQ2_0 | 142 | 2.125 | yes | yes | PrismML ternary, group 128 |
| PTQ1_0 | 143 | 1.75 | yes | yes | PrismML ternary dense, group 128 |
| BF16 | – | 16 | – | yes (type id 30) | no storage matmul kernel; dequant only |
| GPTQ/AWQ | – | – | **no** | **no** | no support at all; CMake flags are labels only |
| MXFP8 | – | 8 | **no** | **no** | not implemented; the CMake option changes the reported label only |

Everything is dequantised on demand per row (no f32 materialisation of the
model). Decode is memory-bandwidth bound: a 27B token streams ~3.66 GB at
~30 GB/s on the test machine ≈ 122 ms of matvec (see README).

**Validation rule (actual):** a tensor type works iff it has a row in the
`nita_quant_dot` table (int path) or a decoder in `src/dequant.c`
(dequant+matmul path); otherwise the weight is unbound and the model open
fails with the role name. There is no "selective de-quant" or per-tile flag
metadata; `quant_dot@<tag>`/`quant_tile@<tag>` are the gates.
