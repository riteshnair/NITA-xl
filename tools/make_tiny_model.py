#!/usr/bin/env python3
"""Generate a complete, randomly-initialised tiny llama-style GGUF model.

The model is far too small to be useful for language, but it exercises the
whole forward pass: embeddings, RMSNorm, RoPE, grouped-query attention with a
KV cache, SwiGLU feed-forward and the output projection.

Weights are also written to ``<out>.npz`` in the same orientation the C code
sees them, so ``tools/reference_forward.py`` can compute logits independently.

Usage:
    python3 tools/make_tiny_model.py build/tiny.gguf [--dtype f32|q8_0]

For the 256-element-block k/i-quant types, generate with the larger geometry:
    python3 tools/make_tiny_model.py /tmp/tiny_q2k.gguf --dtype q2_k \
        --n-embd 256 --n-ff 512 --n-head-kv 4
so that every quantised row is a multiple of 256 elements and the kernel
actually covers the row (smaller rows would silently process zero blocks).

k/i-quant and MXFP4 encoders write random-but-legal block payloads (valid
layouts, finite scales) rather than accurate quantisations; the demo model
exercises the kernels for the tiled==seq gate, it is not a usable LLM. The
.npz records the pre-quantisation source matrix for those types, not a
dequantisation. Proper amax encoders are used for q4_0, q4_1, q8_1 and q8_k.
"""

import argparse
import struct

import numpy as np

GGUF_MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32

UINT32, INT32, FLOAT32, STRING, ARRAY = 4, 5, 6, 8, 9

GGML_F32 = 0
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q8_0 = 8
GGML_Q8_1 = 9
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q5_K = 13
GGML_Q8_K = 15
GGML_IQ2_XXS = 16
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ1_S = 19
GGML_IQ3_S = 21
GGML_IQ2_S = 22
GGML_IQ4_XS = 23
GGML_IQ1_M = 29
GGML_MXFP4 = 39
GGML_Q1_0 = 41
GGML_Q2_0 = 42
GGML_PQ2_0 = 142
GGML_TQ1_0 = 34
GGML_TQ2_0 = 35
GGML_PTQ1_0 = 143

VOCAB = [
    "<unk>", "<s>", "</s>",
    "\u2581hello", "\u2581world", "\u2581the", "\u2581a", "\u2581is",
    "\u2581good", "\u2581bad", "\u2581yes", "\u2581no",
    ".", "!", "\u2581ok", "\u2581fine",
]

# Hyperparameters (defaults; --n-embd/--n-ff/--n-head-kv override in main)
N_EMBD = 128
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 2
N_FF = 256
N_VOCAB = len(VOCAB)
CTX = 64
HEAD_DIM = N_EMBD // N_HEAD          # 32
KV_DIM = HEAD_DIM * N_HEAD_KV        # 64
N_ROT = HEAD_DIM
RMS_EPS = 1e-5
ROPE_THETA = 10000.0


def configure(n_embd, n_ff, n_head_kv):
    """Override the geometry globals; must run before build()/kv emission."""
    global N_EMBD, N_FF, N_HEAD_KV, HEAD_DIM, KV_DIM, N_ROT
    N_EMBD = n_embd
    N_FF = n_ff
    N_HEAD_KV = n_head_kv
    HEAD_DIM = N_EMBD // N_HEAD
    KV_DIM = HEAD_DIM * N_HEAD_KV
    N_ROT = HEAD_DIM


def gstr(t: str) -> bytes:
    b = t.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return gstr(k) + struct.pack("<I", STRING) + gstr(v)


def kv_i32(k, v):
    return gstr(k) + struct.pack("<I", INT32) + struct.pack("<i", v)


def kv_u32(k, v):
    return gstr(k) + struct.pack("<I", UINT32) + struct.pack("<I", v)


def kv_f32(k, v):
    return gstr(k) + struct.pack("<I", FLOAT32) + struct.pack("<f", v)


def kv_arr_str(k, values):
    out = gstr(k) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", STRING) + struct.pack("<Q", len(values))
    for v in values:
        out += gstr(v)
    return out


def kv_arr_f32(k, values):
    out = gstr(k) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", FLOAT32) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<f", float(v))
    return out


def quantise_q8_0(mat: np.ndarray) -> bytes:
    """Encode a 2-D array as Q8_0 blocks of 32 elements."""
    flat = mat.reshape(-1)
    assert flat.size % 32 == 0, "rows must be a multiple of 32 elements"
    out = bytearray()
    for start in range(0, flat.size, 32):
        blk = flat[start:start + 32]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        d = amax / 127.0 if amax > 0 else 0.0
        inv = 127.0 / amax if amax > 0 else 0.0
        q = np.clip(np.round(blk.astype(np.float64) * inv), -127, 127).astype(np.int8)
        out += np.float16(d).tobytes()
        out += q.tobytes()
    return bytes(out)


def dequantise_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_q8_0, used to record the effective weights."""
    out = np.empty(n_elements, dtype=np.float32)
    pos = 0
    for blk in range(n_elements // 32):
        d = float(np.frombuffer(data[pos:pos + 2], dtype=np.float16)[0])
        pos += 2
        q = np.frombuffer(data[pos:pos + 32], dtype=np.int8).astype(np.float32)
        pos += 32
        out[blk * 32:(blk + 1) * 32] = q * d
    return out


def quantise_q1_0(mat: np.ndarray) -> bytes:
    """Encode as Q1_0: 128-weight groups, one sign bit per weight, one fp16
    scale per group. Matches quantize_row_q1_0_ref, which uses mean |w|."""
    flat = mat.reshape(-1)
    assert flat.size % 128 == 0, "rows must be a multiple of 128 elements"
    out = bytearray()
    for start in range(0, flat.size, 128):
        blk = flat[start:start + 128]
        d = float(np.mean(np.abs(blk)))
        qs = bytearray(16)
        for j, w in enumerate(blk):
            if w > 0:
                qs[j // 8] |= 1 << (j % 8)
        out += np.float16(d).tobytes()
        out += bytes(qs)
    return bytes(out)


def dequantise_q1_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_q1_0."""
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 128):
        base = b * 18
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        qs = data[base + 2:base + 18]
        for j in range(128):
            bit = (qs[j // 8] >> (j % 8)) & 1
            out[b * 128 + j] = d if bit else -d
    return out


def quantise_pq2_0(mat: np.ndarray, group: int = 128) -> bytes:
    """Encode as PQ2_0 (group 128) or Q2_0 (group 64): one fp16 scale per
    group, each ternary weight in a 2-bit slot."""
    flat = mat.reshape(-1)
    assert flat.size % group == 0, f"rows must be a multiple of {group} elements"
    out = bytearray()
    for start in range(0, flat.size, group):
        blk = flat[start:start + group]
        amax = float(np.max(np.abs(blk)))
        d = amax
        inv = 1.0 / d if d > 0 else 0.0
        qs = bytearray(group // 4)
        for j, w in enumerate(blk):
            q = int(np.rint(float(w) * inv)) + 1
            q = max(0, min(3, q))
            qs[j // 4] |= (q & 3) << ((j % 4) * 2)
        out += np.float16(d).tobytes()
        out += bytes(qs)
    return bytes(out)


def dequantise_pq2_0(data: bytes, n_elements: int, group: int = 128) -> np.ndarray:
    """Inverse of quantise_pq2_0 (also used for group-64 Q2_0)."""
    out = np.empty(n_elements, dtype=np.float32)
    nbytes = 2 + group // 4
    for b in range(n_elements // group):
        base = b * nbytes
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        qs = data[base + 2:base + nbytes]
        for j in range(group):
            q = (qs[j // 4] >> ((j % 4) * 2)) & 3
            out[b * group + j] = float(q - 1) * d
    return out


PTQ1_STAGES = (32, 16, 8)
POW3 = (1, 3, 9, 27, 81, 243)


def quantise_ptq1_0(mat: np.ndarray) -> bytes:
    """Encode as PTQ1_0: 128-weight groups, dense base-3 trits."""
    flat = mat.reshape(-1)
    assert flat.size % 128 == 0, "rows must be a multiple of 128 elements"
    out = bytearray()
    for start in range(0, flat.size, 128):
        blk = flat[start:start + 128]
        amax = float(np.max(np.abs(blk)))
        d = amax
        inv = 1.0 / d if d > 0 else 0.0
        tr = [max(0, min(2, int(np.rint(float(w) * inv)) + 1)) for w in blk]

        qs = bytearray(24)
        qh = bytearray(2)
        # Invert the published decoder: it emits byte (j+m)'s trit n at output
        # position n*c + m, so pack value (xoff + n*c + m) into that slot.
        xoff = 0
        j = 0
        for c in PTQ1_STAGES:
            while j + c <= 24:
                for m in range(c):
                    byte = 0
                    for n in range(5):
                        byte = byte * 3 + tr[xoff + n * c + m]
                    # ceiling-divide by 3^5 so the top trit lands in bit 7
                    qs[j + m] = ((byte * 256 + 242) // 243) & 0xFF
                j += c
                xoff += 5 * c
        for h in range(2):
            byte = 0
            for n in range(4):
                byte = byte * 3 + tr[xoff + n * 2 + h]
            byte = byte * 3
            qh[h] = ((byte * 256 + 242) // 243) & 0xFF

        out += bytes(qs)
        out += bytes(qh)
        out += np.float16(d).tobytes()
    return bytes(out)


def dequantise_ptq1_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_ptq1_0."""
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 128):
        base = b * 28
        qs = data[base:base + 24]
        qh = data[base + 24:base + 26]
        d = float(np.frombuffer(data[base + 26:base + 28], dtype=np.float16)[0])

        vals = []
        j = 0
        for c in PTQ1_STAGES:
            while j + c <= 24:
                for n in range(5):
                    for m in range(c):
                        q = (qs[j + m] * POW3[n]) & 0xFF
                        xi = ((q * 3) & 0xFFFF) >> 8
                        vals.append(float(xi - 1) * d)
                j += c
        for n in range(4):
            for h in range(2):
                q = (qh[h] * POW3[n]) & 0xFF
                xi = ((q * 3) & 0xFFFF) >> 8
                vals.append(float(xi - 1) * d)

        out[b * 128:(b + 1) * 128] = vals
    return out


def quantise_tq1_0(mat: np.ndarray) -> bytes:
    """Encode as TQ1_0: 256-weight groups, dense base-3 trits.

    Loop structure is upstream's quantize_row_tq1_0_ref: a 32-byte stage, then
    a 16-byte stage, then four qh bytes at four trits each, scale last.
    """
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0, "rows must be a multiple of 256 elements"
    out = bytearray()
    for start in range(0, flat.size, 256):
        blk = flat[start:start + 256]
        amax = float(np.max(np.abs(blk)))
        inv = 1.0 / amax if amax > 0 else 0.0
        tr = [max(0, min(2, int(np.rint(float(w) * inv)) + 1)) for w in blk]

        qs = bytearray(48)
        qh = bytearray(4)
        x = 0
        for c in (32, 16):
            j = 0 if c == 32 else 32
            for m in range(c):
                byte = 0
                for n in range(5):
                    byte = byte * 3 + tr[x + n * c + m]
                qs[j + m] = ((byte * 256 + 242) // 243) & 0xFF
            x += 5 * c
        for h in range(4):
            byte = 0
            for n in range(4):
                byte = byte * 3 + tr[x + h + n * 4]
            byte *= 3
            qh[h] = ((byte * 256 + 242) // 243) & 0xFF

        out += bytes(qs)
        out += bytes(qh)
        out += np.float16(amax).tobytes()
    return bytes(out)


def dequantise_tq1_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_tq1_0."""
    out = np.empty(n_elements, dtype=np.float32)
    stages = (32, 16, 8)
    for b in range(n_elements // 256):
        base = b * 54
        qs = data[base:base + 48]
        qh = data[base + 48:base + 52]
        d = float(np.frombuffer(data[base + 52:base + 54], dtype=np.float16)[0])

        vals = []
        j = 0
        for c in stages:
            while j + c <= 48:
                for n in range(5):
                    for m in range(c):
                        q = (qs[j + m] * POW3[n]) & 0xFF
                        xi = ((q * 3) & 0xFFFF) >> 8
                        vals.append(float(xi - 1) * d)
                j += c
        for n in range(4):
            for h in range(4):
                q = (qh[h] * POW3[n]) & 0xFF
                xi = ((q * 3) & 0xFFFF) >> 8
                vals.append(float(xi - 1) * d)

        out[b * 256:(b + 1) * 256] = vals
    return out


def quantise_tq2_0(mat: np.ndarray) -> bytes:
    """Encode as TQ2_0: 256-weight groups, 2-bit slots, scale last.

    Slot l is a bit plane over 32 stored bytes (upstream's "along 32 bytes"),
    so it holds elements m + n*32 of each 128-element half.
    """
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0, "rows must be a multiple of 256 elements"
    out = bytearray()
    for start in range(0, flat.size, 256):
        blk = flat[start:start + 256]
        amax = float(np.max(np.abs(blk)))
        inv = 1.0 / amax if amax > 0 else 0.0
        tr = [max(0, min(2, int(np.rint(float(w) * inv)) + 1)) for w in blk]

        qs = bytearray(64)
        x = 0
        for j in (0, 32):
            for m in range(32):
                q = 0
                for n in range(4):
                    q |= (tr[x + m + n * 32] & 3) << (2 * n)
                qs[j + m] = q
            x += 128

        out += bytes(qs)
        out += np.float16(amax).tobytes()
    return bytes(out)


def dequantise_tq2_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_tq2_0."""
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 256):
        base = b * 66
        qs = data[base:base + 64]
        d = float(np.frombuffer(data[base + 64:base + 66], dtype=np.float16)[0])
        vals = []
        for g in (0, 1):
            for l in range(4):
                for m in range(32):
                    vals.append(float(((qs[g * 32 + m] >> (2 * l)) & 3) - 1) * d)
        out[b * 256:(b + 1) * 256] = vals
    return out



"""Random-but-legal payload encoders for the demo k/i-quant models.

A block is *legal* when the loader can decode it and the dot kernels run:
layouts and sizes match upstream, fp16/q4k scale bytes decode to finite
values, and grid indices stay inside their tables by construction (any byte
value is a legal index into a 256-entry table). Weight magnitudes are
irrelevant for the tiled==seq gate, so payloads are random rather than
accurate. Scales are fixed small constants so decodes stay finite.
"""

_RNG = np.random.default_rng(20261110)


def _f16(x):
    return np.float16(x).tobytes()


def _rand_bytes(n):
    return _RNG.integers(0, 256, size=n, dtype=np.uint8).tobytes()


def quantise_q2_k(mat, n_elements):
    """Q2_K: 84 bytes / 256 weights. Random scales+qs, finite fp16 d/dmin."""
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0
    out = bytearray()
    for _ in range(flat.size // 256):
        out += _rand_bytes(16)          # scales: low nibble sc, high nibble mn
        out += _rand_bytes(64)          # qs: 2-bit at shift 0/2/4/6
        out += _f16(0.02)               # d
        out += _f16(0.01)               # dmin
    return bytes(out)


def quantise_q3_k(mat, n_elements):
    """Q3_K: 110 bytes / 256 weights. Random hmask/qs/packed scales, finite d."""
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0
    out = bytearray()
    for _ in range(flat.size // 256):
        out += _rand_bytes(32)          # hmask
        out += _rand_bytes(64)          # qs
        out += _rand_bytes(12)          # packed scales (q3k_scales decode)
        out += _f16(0.02)               # d
    return bytes(out)


def quantise_q5_k(mat, n_elements):
    """Q5_K: 176 bytes / 256 weights. Random qh/ql/packed scales, finite d/dmin."""
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0
    out = bytearray()
    for _ in range(flat.size // 256):
        out += _f16(0.02)               # d
        out += _f16(0.01)               # dmin
        out += _rand_bytes(12)          # packed scales (q4k_scales decode)
        out += _rand_bytes(32)          # qh
        out += _rand_bytes(128)         # ql
    return bytes(out)


def quantise_q8_k(mat, n_elements):
    """Q8_K: 292 bytes / 256 weights. Proper amax int8 encoding."""
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0
    out = bytearray()
    for start in range(0, flat.size, 256):
        blk = flat[start:start + 256]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        d = amax / 127.0 if amax > 0 else 0.0
        inv = 127.0 / amax if amax > 0 else 0.0
        q = np.clip(np.round(blk.astype(np.float64) * inv), -127, 127).astype(np.int8)
        out += np.float32(d).tobytes()
        out += q.tobytes()
        out += b"\x00" * 32             # bsums (unused by the kernel)
    return bytes(out)


def dequantise_q8_k(data, n_elements):
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 256):
        base = b * 292
        d = float(np.frombuffer(data[base:base + 4], dtype=np.float32)[0])
        q = np.frombuffer(data[base + 4:base + 260], dtype=np.int8).astype(np.float32)
        out[b * 256:(b + 1) * 256] = q * d
    return out


def quantise_q4_0(mat, n_elements):
    """Q4_0: 18 bytes / 32 weights. Proper amax (value = d * (q - 8))."""
    flat = mat.reshape(-1)
    assert flat.size % 32 == 0
    out = bytearray()
    for start in range(0, flat.size, 32):
        blk = flat[start:start + 32]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        d = amax / 7.0 if amax > 0 else 0.0
        inv = 1.0 / d if d > 0 else 0.0
        q = np.clip(np.rint(blk.astype(np.float64) * inv) + 8, 0, 15).astype(np.uint8)
        out += _f16(d)
        packed = np.zeros(16, dtype=np.uint8)
        packed[:16] = (q[:16] | (q[16:] << 4))
        out += packed.tobytes()
    return bytes(out)


def dequantise_q4_0(data, n_elements):
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 32):
        base = b * 18
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        qs = data[base + 2:base + 18]
        for l in range(16):
            out[b * 32 + l] = (float(qs[l] & 0x0F) - 8.0) * d
            out[b * 32 + 16 + l] = (float(qs[l] >> 4) - 8.0) * d
    return out


def quantise_q4_1(mat, n_elements):
    """Q4_1: 20 bytes / 32 weights. Proper min-fit (value = d * q + m)."""
    flat = mat.reshape(-1)
    assert flat.size % 32 == 0
    out = bytearray()
    for start in range(0, flat.size, 32):
        blk = flat[start:start + 32]
        bmax = float(np.max(blk)) if blk.size else 0.0
        bmin = float(np.min(blk)) if blk.size else 0.0
        d = (bmax - bmin) / 15.0 if bmax > bmin else 0.0
        m = bmin if d > 0 else 0.0
        inv = 1.0 / d if d > 0 else 0.0
        q = np.clip(np.rint((blk.astype(np.float64) - m) * inv), 0, 15).astype(np.uint8)
        out += _f16(d)
        out += _f16(m)
        packed = np.zeros(16, dtype=np.uint8)
        packed = (q[:16] | (q[16:] << 4))
        out += packed.tobytes()
    return bytes(out)


def dequantise_q4_1(data, n_elements):
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 32):
        base = b * 20
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        m = float(np.frombuffer(data[base + 2:base + 4], dtype=np.float16)[0])
        qs = data[base + 4:base + 20]
        for l in range(16):
            out[b * 32 + l] = float(qs[l] & 0x0F) * d + m
            out[b * 32 + 16 + l] = float(qs[l] >> 4) * d + m
    return out


def quantise_q8_1(mat, n_elements):
    """Q8_1: 36 bytes / 32 weights. Proper amax int8; sum stored for realism."""
    flat = mat.reshape(-1)
    assert flat.size % 32 == 0
    out = bytearray()
    for start in range(0, flat.size, 32):
        blk = flat[start:start + 32]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        d = amax / 127.0 if amax > 0 else 0.0
        inv = 127.0 / amax if amax > 0 else 0.0
        q = np.clip(np.round(blk.astype(np.float64) * inv), -127, 127).astype(np.int8)
        out += _f16(d)
        out += _f16(0.0)                # s field: unused by the kernel
        out += q.tobytes()
    return bytes(out)


def dequantise_q8_1(data, n_elements):
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 32):
        base = b * 36
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        q = np.frombuffer(data[base + 4:base + 36], dtype=np.int8).astype(np.float32)
        out[b * 32:(b + 1) * 32] = q * d
    return out


def _iq_random(mat, n_elements, block_bytes, fixed_prefix=None):
    """Shared helper for the i-quant family: random payload, finite fp16 d."""
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0
    out = bytearray()
    for _ in range(flat.size // 256):
        if fixed_prefix:
            out += fixed_prefix()
        else:
            out += _f16(0.02)           # d
        out += _rand_bytes(block_bytes - 2)
    return bytes(out)


def quantise_iq2_xxs(mat, n_elements):
    """IQ2_XXS: 66 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 66)


def quantise_iq2_xs(mat, n_elements):
    """IQ2_XS: 74 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 74)


def quantise_iq3_xxs(mat, n_elements):
    """IQ3_XXS: 98 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 98)


def quantise_iq1_s(mat, n_elements):
    """IQ1_S: 50 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 50)


def quantise_iq3_s(mat, n_elements):
    """IQ3_S: 110 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 110)


def quantise_iq2_s(mat, n_elements):
    """IQ2_S: 82 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 82)


def quantise_iq4_xs(mat, n_elements):
    """IQ4_XS: 136 bytes / 256 weights."""
    return _iq_random(mat, n_elements, 136)


def quantise_iq1_m(mat, n_elements):
    """IQ1_M: 56 bytes / 256 weights, d is synthesised from 12 bits spread over
    the four scale u16s; the pattern 0x2E66 (~0.1) is forced finite there."""
    flat = mat.reshape(-1)
    assert flat.size % 256 == 0
    out = bytearray()
    for _ in range(flat.size // 256):
        out += _rand_bytes(32)          # qs grid indices
        out += _rand_bytes(16)          # qh grid/delta bits
        r = _RNG.integers(0, 4096, size=4, dtype=np.uint16)
        sc = [0x6000 | r[0],
              ((0x60 | (_RNG.integers(0, 16))) << 8) | _RNG.integers(0, 256),
              0xE000 | r[2],
              0x2000 | r[3]]
        out += b"".join(np.uint16(s).tobytes() for s in sc)
    return bytes(out)


_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int32)


def quantise_mxfp4(mat, n_elements):
    """MXFP4: 17 bytes / 32 weights, per-block E8M0 exponent, doubled e2m1 codes."""
    flat = mat.reshape(-1)
    assert flat.size % 32 == 0
    out = bytearray()
    for start in range(0, flat.size, 32):
        blk = flat[start:start + 32]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        if amax <= 0.0:
            e = 0
        else:
            e = int(np.floor(np.log2(amax))) + 128
            e = max(1, min(254, e))
        d = 2.0 ** (e - 128)
        q = np.empty(32, dtype=np.uint8)
        for j, w in enumerate(blk):
            tgt = w / d
            q[j] = int(np.argmin(np.abs(_FP4.astype(np.float64) - tgt)))
        out += bytes([e])
        packed = np.zeros(16, dtype=np.uint8)
        packed[:16] = (q[:16] | (q[16:] << 4))
        out += packed.tobytes()
    return bytes(out)


def dequantise_mxfp4(data, n_elements):
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 32):
        base = b * 17
        e = data[base]
        d = float(2.0 ** (e - 128))
        qs = data[base + 1:base + 17]
        for l in range(16):
            out[b * 32 + l] = float(_FP4[qs[l] & 0x0F]) * d
            out[b * 32 + 16 + l] = float(_FP4[qs[l] >> 4]) * d
    return out


def build(dtype: str):
    rng = np.random.default_rng(20261106)
    scale = 0.08
    ternary = dtype in ("ptq1_0", "pq2_0", "q2_0", "tq1_0", "tq2_0")
    binary = dtype == "q1_0"

    def W(n_out, n_in):
        # Shape (n_out, n_in); ne0 = n_in, ne1 = n_out.
        if binary:
            # Q1_0 has no zero state: every weight is +/-d.
            return (rng.integers(0, 2, size=(n_out, n_in)) * 2 - 1).astype(np.float32)
        if ternary:
            # Exactly representable ternary weights, so any decode error shows up
            # in the logits rather than being absorbed by quantisation noise.
            return rng.integers(-1, 2, size=(n_out, n_in)).astype(np.float32)
        return (rng.standard_normal((n_out, n_in)) * scale).astype(np.float32)

    weights = {}
    weights["token_embd.weight"] = W(N_VOCAB, N_EMBD)
    weights["output_norm.weight"] = np.ones(N_EMBD, dtype=np.float32)
    weights["output.weight"] = W(N_VOCAB, N_EMBD)

    for l in range(N_LAYER):
        weights[f"blk.{l}.attn_norm.weight"] = np.ones(N_EMBD, dtype=np.float32)
        weights[f"blk.{l}.attn_q.weight"] = W(N_HEAD * HEAD_DIM, N_EMBD)
        weights[f"blk.{l}.attn_k.weight"] = W(KV_DIM, N_EMBD)
        weights[f"blk.{l}.attn_v.weight"] = W(KV_DIM, N_EMBD)
        weights[f"blk.{l}.attn_output.weight"] = W(N_EMBD, N_HEAD * HEAD_DIM)
        weights[f"blk.{l}.ffn_norm.weight"] = np.ones(N_EMBD, dtype=np.float32)
        weights[f"blk.{l}.ffn_gate.weight"] = W(N_FF, N_EMBD)
        weights[f"blk.{l}.ffn_up.weight"] = W(N_FF, N_EMBD)
        weights[f"blk.{l}.ffn_down.weight"] = W(N_EMBD, N_FF)
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="tiny.gguf")
    ap.add_argument("--dtype", choices=["f32", "q8_0", "q4_0", "q4_1", "q8_1",
                                        "q2_k", "q3_k", "q5_k", "q8_k",
                                        "iq2_xxs", "iq2_xs", "iq3_xxs",
                                        "iq1_s", "iq3_s", "iq2_s", "iq4_xs", "iq1_m",
                                        "mxfp4",
                                        "q1_0", "q2_0", "pq2_0",
                                        "ptq1_0", "tq1_0", "tq2_0"],
                    default="f32")
    ap.add_argument("--n-embd", type=int, default=N_EMBD)
    ap.add_argument("--n-ff", type=int, default=N_FF)
    ap.add_argument("--n-head-kv", type=int, default=N_HEAD_KV)
    args = ap.parse_args()

    configure(args.n_embd, args.n_ff, args.n_head_kv)
    weights = build(args.dtype)

    # Serialise tensors with 32-byte alignment.
    blobs = []
    infos = []
    offset = 0
    effective = {}

    for name, mat in weights.items():
        ne0 = mat.shape[1] if mat.ndim == 2 else mat.shape[0]
        ne1 = mat.shape[0] if mat.ndim == 2 else 1
        dims = [ne0, ne1] if mat.ndim == 2 else [ne0]
        ne0_pad = ne0

        # k/i-quant family requires every row to be a multiple of 256
        kq_ok = ne0_pad % 256 == 0
        # i-quant random payloads: legal decode, source recorded as effective
        iq_random = {
            "iq2_xxs": (quantise_iq2_xxs, GGML_IQ2_XXS),
            "iq2_xs":  (quantise_iq2_xs,  GGML_IQ2_XS),
            "iq3_xxs": (quantise_iq3_xxs, GGML_IQ3_XXS),
            "iq1_s":   (quantise_iq1_s,   GGML_IQ1_S),
            "iq3_s":   (quantise_iq3_s,   GGML_IQ3_S),
            "iq2_s":   (quantise_iq2_s,   GGML_IQ2_S),
            "iq4_xs":  (quantise_iq4_xs,  GGML_IQ4_XS),
            "iq1_m":   (quantise_iq1_m,   GGML_IQ1_M),
        }

        if args.dtype == "q8_0" and ne0_pad % 32 == 0 and ne1 > 1:
            data = quantise_q8_0(mat)
            ttype = GGML_Q8_0
            effective[name] = dequantise_q8_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q4_0" and ne0_pad % 32 == 0 and ne1 > 1:
            data = quantise_q4_0(mat, mat.size)
            ttype = GGML_Q4_0
            effective[name] = dequantise_q4_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q4_1" and ne0_pad % 32 == 0 and ne1 > 1:
            data = quantise_q4_1(mat, mat.size)
            ttype = GGML_Q4_1
            effective[name] = dequantise_q4_1(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q8_1" and ne0_pad % 32 == 0 and ne1 > 1:
            data = quantise_q8_1(mat, mat.size)
            ttype = GGML_Q8_1
            effective[name] = dequantise_q8_1(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q2_k" and kq_ok and ne1 > 1:
            data = quantise_q2_k(mat, mat.size)
            ttype = GGML_Q2_K
            effective[name] = mat.astype(np.float32)
        elif args.dtype == "q3_k" and kq_ok and ne1 > 1:
            data = quantise_q3_k(mat, mat.size)
            ttype = GGML_Q3_K
            effective[name] = mat.astype(np.float32)
        elif args.dtype == "q5_k" and kq_ok and ne1 > 1:
            data = quantise_q5_k(mat, mat.size)
            ttype = GGML_Q5_K
            effective[name] = mat.astype(np.float32)
        elif args.dtype == "q8_k" and kq_ok and ne1 > 1:
            data = quantise_q8_k(mat, mat.size)
            ttype = GGML_Q8_K
            effective[name] = dequantise_q8_k(data, mat.size).reshape(mat.shape)
        elif args.dtype == "mxfp4" and ne0_pad % 32 == 0 and ne1 > 1:
            data = quantise_mxfp4(mat, mat.size)
            ttype = GGML_MXFP4
            effective[name] = dequantise_mxfp4(data, mat.size).reshape(mat.shape)
        elif args.dtype in iq_random and kq_ok and ne1 > 1:
            enc, gt = iq_random[args.dtype]
            data = enc(mat, mat.size)
            ttype = gt
            effective[name] = mat.astype(np.float32)
        elif args.dtype == "q1_0" and ne0_pad % 128 == 0:
            data = quantise_q1_0(mat)
            ttype = GGML_Q1_0
            effective[name] = dequantise_q1_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q2_0" and ne0_pad % 64 == 0:
            data = quantise_pq2_0(mat, 64)
            ttype = GGML_Q2_0
            effective[name] = dequantise_pq2_0(data, mat.size, 64).reshape(mat.shape)
        elif args.dtype == "pq2_0" and ne0_pad % 128 == 0:
            data = quantise_pq2_0(mat)
            ttype = GGML_PQ2_0
            effective[name] = dequantise_pq2_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "ptq1_0" and ne0_pad % 128 == 0:
            data = quantise_ptq1_0(mat)
            ttype = GGML_PTQ1_0
            effective[name] = dequantise_ptq1_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "tq1_0" and ne0_pad % 256 == 0:
            data = quantise_tq1_0(mat)
            ttype = GGML_TQ1_0
            effective[name] = dequantise_tq1_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "tq2_0" and ne0_pad % 256 == 0:
            data = quantise_tq2_0(mat)
            ttype = GGML_TQ2_0
            effective[name] = dequantise_tq2_0(data, mat.size).reshape(mat.shape)
        else:
            data = np.ascontiguousarray(mat, dtype=np.float32).tobytes()
            ttype = GGML_F32
            effective[name] = mat.astype(np.float32)

        pad = (-offset) % ALIGNMENT
        if pad:
            blobs.append(b"\x00" * pad)
            offset += pad

        infos.append((name, dims, ttype, offset))
        blobs.append(data)
        offset += len(data)

        pad = (-offset) % ALIGNMENT
        if pad:
            blobs.append(b"\x00" * pad)
            offset += pad

    kv = [
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "nita-tiny"),
        kv_u32("general.alignment", ALIGNMENT),
        kv_u32("llama.context_length", CTX),
        kv_u32("llama.embedding_length", N_EMBD),
        kv_u32("llama.block_count", N_LAYER),
        kv_u32("llama.attention.head_count", N_HEAD),
        kv_u32("llama.attention.head_count_kv", N_HEAD_KV),
        kv_u32("llama.feed_forward_length", N_FF),
        kv_u32("llama.rope.dimension_count", N_ROT),
        kv_f32("llama.attention.layer_norm_rms_epsilon", RMS_EPS),
        kv_f32("llama.rope.freq_base", ROPE_THETA),
        kv_arr_str("tokenizer.ggml.tokens", VOCAB),
        kv_arr_f32("tokenizer.ggml.scores", [0.0] * N_VOCAB),
        kv_i32("tokenizer.ggml.bos_token_id", 1),
        kv_i32("tokenizer.ggml.eos_token_id", 2),
        kv_i32("tokenizer.ggml.unknown_token_id", 0),
        kv_str("tokenizer.ggml.model", "llama"),
    ]
    metadata = b"".join(kv)

    tensor_table = b""
    for name, dims, ttype, off in infos:
        tensor_table += gstr(name)
        tensor_table += struct.pack("<I", len(dims))
        for d in dims:
            tensor_table += struct.pack("<Q", d)
        tensor_table += struct.pack("<I", ttype)
        tensor_table += struct.pack("<Q", off)

    header = GGUF_MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<Q", len(infos))
    header += struct.pack("<Q", len(kv))

    body = header + metadata + tensor_table
    pad = (-len(body)) % ALIGNMENT
    body += b"\x00" * pad
    body += b"".join(blobs)

    with open(args.out, "wb") as f:
        f.write(body)

    np.savez(args.out + ".npz", **effective)

    print(f"wrote {args.out} ({len(body)} bytes, {len(infos)} tensors, dtype={args.dtype})")
    print(f"wrote {args.out}.npz (reference weights)")


if __name__ == "__main__":
    main()
