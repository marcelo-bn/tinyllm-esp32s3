#!/usr/bin/env python3
"""Converts a llama2.c checkpoint (`stories*.bin`, f32) to `.tlm`:
symmetric int8 per row with one f32 scale per row. Norm weights stay f32.

Usage:
    python tools/export.py models/stories260K.bin models/stories260K.tlm

.tlm layout (little-endian):
    b"TLM1"
    7 x u32   : dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    u32       : flags (bit0 = classifier shares weights with the embedding)
    QMatrix   : embedding            (vocab x dim)
    per layer :
        f32[dim]  rms_att
        QMatrix   wq (dim x dim), wk (kv_dim x dim), wv (kv_dim x dim), wo (dim x dim)
        f32[dim]  rms_ffn
        QMatrix   w1 (hidden x dim), w2 (dim x hidden), w3 (hidden x dim)
    f32[dim]  rms_final
    [QMatrix  wcls (vocab x dim)]   only if flags bit0 == 0

QMatrix = int8[rows*cols] followed by f32[rows] (scales).
"""
import struct
import sys

import numpy as np


def read_config(f):
    dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = struct.unpack("<7i", f.read(28))
    shared_cls = vocab > 0
    return dict(dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
                n_kv_heads=n_kv_heads, vocab_size=abs(vocab), seq_len=seq_len), shared_cls


def read_f32(f, n):
    a = np.frombuffer(f.read(n * 4), dtype="<f4")
    assert a.size == n, "truncated file"
    return a


def quantize_rows(w: np.ndarray):
    """Symmetric int8 per row. Returns (int8[rows, cols], f32[rows])."""
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype("<f4")
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def write_qmatrix(out, w: np.ndarray):
    q, s = quantize_rows(w)
    out.write(q.tobytes())
    out.write(s.tobytes())
    return q, s


def main(src, dst):
    with open(src, "rb") as f:
        cfg, shared_cls = read_config(f)
        d, h, L = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"]
        V, S = cfg["vocab_size"], cfg["seq_len"]
        kv_dim = d * cfg["n_kv_heads"] // cfg["n_heads"]
        head_size = d // cfg["n_heads"]

        # llama2.c read order (all layers of each tensor together)
        emb = read_f32(f, V * d).reshape(V, d)
        rms_att = read_f32(f, L * d).reshape(L, d)
        wq = read_f32(f, L * d * d).reshape(L, d, d)
        wk = read_f32(f, L * d * kv_dim).reshape(L, kv_dim, d)
        wv = read_f32(f, L * d * kv_dim).reshape(L, kv_dim, d)
        wo = read_f32(f, L * d * d).reshape(L, d, d)
        rms_ffn = read_f32(f, L * d).reshape(L, d)
        w1 = read_f32(f, L * h * d).reshape(L, h, d)
        w2 = read_f32(f, L * d * h).reshape(L, d, h)
        w3 = read_f32(f, L * h * d).reshape(L, h, d)
        rms_final = read_f32(f, d)
        _ = read_f32(f, S * head_size // 2)  # freq_cis_real (recomputed at runtime)
        _ = read_f32(f, S * head_size // 2)  # freq_cis_imag
        wcls = None if shared_cls else read_f32(f, V * d).reshape(V, d)

    print(f"config: {cfg}  shared_cls={shared_cls}")

    total_q = 0
    with open(dst, "wb") as out:
        out.write(b"TLM1")
        out.write(struct.pack("<7I", d, h, L, cfg["n_heads"], cfg["n_kv_heads"], V, S))
        out.write(struct.pack("<I", 1 if shared_cls else 0))

        q, _ = write_qmatrix(out, emb); total_q += q.size
        for l in range(L):
            out.write(rms_att[l].astype("<f4").tobytes())
            for w in (wq[l], wk[l], wv[l], wo[l]):
                q, _ = write_qmatrix(out, w); total_q += q.size
            out.write(rms_ffn[l].astype("<f4").tobytes())
            for w in (w1[l], w2[l], w3[l]):
                q, _ = write_qmatrix(out, w); total_q += q.size
        out.write(rms_final.astype("<f4").tobytes())
        if wcls is not None:
            q, _ = write_qmatrix(out, wcls); total_q += q.size

    import os
    size = os.path.getsize(dst)
    print(f"{total_q/1e6:.2f} M quantized parameters -> {size/1024:.0f} KB in {dst}")

    # RAM estimate for the state (the KV cache dominates)
    kv_bytes = 2 * L * S * kv_dim * 4
    print(f"KV cache at seq_len={S}: {kv_bytes/1024:.0f} KB  "
          f"(at seq=256: {2*L*min(S,256)*kv_dim*4/1024:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
