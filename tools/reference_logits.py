#!/usr/bin/env python3
"""Independent forward pass for a llama-family GGUF, in numpy.

Prints the final logits for a comma-separated token list, in the same
format as `nita --tokens`, so the two can be compared directly:

    python3 tools/reference_logits.py model.gguf --tokens 5,6,7 > ref.txt
    ./build/nita --model model.gguf --tokens 5,6,7 | grep ^logits > eng.txt
    python3 tools/compare_logits.py ref.txt eng.txt

Implements: token embedding (untied lookup), RMSNorm, RoPE (adjacent-pair,
the GGUF llama NORMAL mode), causal attention with GQA and a KV cache,
SwiGLU FFN, final RMSNorm, tied output projection when output.weight is
absent. Mirrors the reference forward used to validate the qwen35 path,
and shares no code with src/.
"""
import argparse
import sys

import numpy as np

from gguf_numpy import GGUF, load_matrix


def rmsnorm(x, w, eps):
    ss = float(np.mean(x.astype(np.float64) ** 2))
    return (x * np.float32(1.0 / np.sqrt(ss + eps)) * w).astype(np.float32)


def rope(vec, n_heads, head_dim, n_rot, pos, theta):
    out = vec.astype(np.float32).copy()
    for h in range(n_heads):
        base = h * head_dim
        for i in range(n_rot // 2):
            freq = np.float32(theta) ** np.float32(-2.0 * i / n_rot)
            angle = np.float32(pos) * freq
            c, s = np.cos(angle), np.sin(angle)
            x0, x1 = out[base + 2 * i], out[base + 2 * i + 1]
            out[base + 2 * i] = x0 * c - x1 * s
            out[base + 2 * i + 1] = x0 * s + x1 * c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--tokens', required=True)
    args = ap.parse_args()

    ids = [int(t) for t in args.tokens.split(',') if t.strip()]
    g = GGUF(args.model)
    kv = g.kv
    arch = kv.get('general.architecture', 'llama')
    p = arch + '.'

    n_embd = int(kv[p + 'embedding_length'])
    n_layer = int(kv[p + 'block_count'])
    n_head = int(kv[p + 'attention.head_count'])
    n_head_kv = int(kv.get(p + 'attention.head_count_kv', n_head))
    n_ff = int(kv[p + 'feed_forward_length'])
    head_dim = int(kv.get(p + 'attention.key_length', 0)) or n_embd // n_head
    n_rot = int(kv.get(p + 'rope.dimension_count', head_dim))
    theta = float(kv.get(p + 'rope.freq_base', 10000.0))
    eps = float(kv.get(p + 'attention.layer_norm_rms_epsilon', 1e-5))
    kv_dim = n_head_kv * head_dim
    q_dim = n_head * head_dim

    m = {}
    m['tok'] = load_matrix(g, 'token_embd.weight')
    try:
        m['out'] = load_matrix(g, 'output.weight')
    except KeyError:
        m['out'] = m['tok']                      # tied
    m['out_norm'] = load_matrix(g, 'output_norm.weight')[0]
    for l in range(n_layer):
        for key in ('attn_norm', 'attn_q', 'attn_k', 'attn_v', 'attn_output',
                    'ffn_norm', 'ffn_gate', 'ffn_up', 'ffn_down'):
            m[(l, key)] = load_matrix(g, 'blk.%d.%s.weight' % (l, key))

    k_cache = [np.zeros((0, kv_dim), dtype=np.float32) for _ in range(n_layer)]
    v_cache = [np.zeros((0, kv_dim), dtype=np.float32) for _ in range(n_layer)]
    logits = None

    for pos, tok in enumerate(ids):
        x = m['tok'][tok].astype(np.float32).copy()
        for l in range(n_layer):
            h = rmsnorm(x, m[(l, 'attn_norm')][0], eps)
            q = m[(l, 'attn_q')] @ h
            k = m[(l, 'attn_k')] @ h
            v = m[(l, 'attn_v')] @ h
            q = rope(q, n_head, head_dim, n_rot, pos, theta)
            k = rope(k, n_head_kv, head_dim, n_rot, pos, theta)
            k_cache[l] = np.vstack([k_cache[l], k[None, :]])
            v_cache[l] = np.vstack([v_cache[l], v[None, :]])

            attn = np.zeros(q_dim, dtype=np.float32)
            inv = np.float32(1.0 / np.sqrt(head_dim))
            for hd in range(n_head):
                kvh = hd * n_head_kv // n_head
                qh = q[hd * head_dim:(hd + 1) * head_dim]
                kh = k_cache[l][:, kvh * head_dim:(kvh + 1) * head_dim]
                vh = v_cache[l][:, kvh * head_dim:(kvh + 1) * head_dim]
                scores = (kh @ qh) * inv
                scores = scores - np.max(scores)
                probs = np.exp(scores)
                probs = probs / np.sum(probs)
                attn[hd * head_dim:(hd + 1) * head_dim] = probs @ vh

            x = x + m[(l, 'attn_output')] @ attn
            h2 = rmsnorm(x, m[(l, 'ffn_norm')][0], eps)
            gate = m[(l, 'ffn_gate')] @ h2
            up = m[(l, 'ffn_up')] @ h2
            ff = (gate / (1.0 + np.exp(-gate))) * up
            x = x + m[(l, 'ffn_down')] @ ff

        hout = rmsnorm(x, m['out_norm'], eps)
        logits = m['out'] @ hout

    print('tokens:', ' '.join(str(t) for t in ids))
    print('logits:', ' '.join('%.6f' % v for v in logits))


if __name__ == '__main__':
    sys.exit(main())
