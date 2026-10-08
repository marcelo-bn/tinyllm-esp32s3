#!/usr/bin/env python3
"""Reference implementation in numpy that reads the SAME `.tlm` and uses the
SAME dynamic activation quantization as the Rust code. With temperature 0, its
output must match the desktop `tinyllm` and the firmware token for token.

Usage:
    python tools/reference.py models/stories260K.tlm models/tok512.bin \
        --prompt "Once upon a time" --steps 64
"""
import argparse
import struct

import numpy as np


class Reader:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def take(self, n):
        s = self.buf[self.pos:self.pos + n]
        assert len(s) == n, "truncated file"
        self.pos += n
        return s

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def f32s(self, n):
        return np.frombuffer(self.take(4 * n), dtype="<f4").astype(np.float32)

    def qmatrix(self, rows, cols):
        data = np.frombuffer(self.take(rows * cols), dtype=np.int8).reshape(rows, cols)
        scales = self.f32s(rows)
        return data, scales


def load_model(path):
    r = Reader(open(path, "rb").read())
    assert r.take(4) == b"TLM1"
    keys = ["dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len"]
    cfg = dict(zip(keys, [r.u32() for _ in keys]))
    shared = r.u32() & 1
    d, h = cfg["dim"], cfg["hidden_dim"]
    kv = d * cfg["n_kv_heads"] // cfg["n_heads"]
    m = {"cfg": cfg, "emb": r.qmatrix(cfg["vocab_size"], d), "layers": []}
    for _ in range(cfg["n_layers"]):
        m["layers"].append(dict(
            rms_att=r.f32s(d), wq=r.qmatrix(d, d), wk=r.qmatrix(kv, d), wv=r.qmatrix(kv, d),
            wo=r.qmatrix(d, d), rms_ffn=r.f32s(d), w1=r.qmatrix(h, d), w2=r.qmatrix(d, h),
            w3=r.qmatrix(h, d)))
    m["rms_final"] = r.f32s(d)
    m["wcls"] = m["emb"] if shared else r.qmatrix(cfg["vocab_size"], d)
    return m


def load_tokenizer(path, vocab_size):
    r = Reader(open(path, "rb").read())
    max_len = r.u32()
    vocab, scores = [], []
    for _ in range(vocab_size):
        scores.append(struct.unpack("<f", r.take(4))[0])
        n = r.u32()
        vocab.append(bytes(r.take(n)))
    return vocab, scores


def quantize(x):
    amax = np.abs(x).max()
    if amax == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = amax / 127.0
    q = np.rint(x / scale).clip(-127, 127).astype(np.int8)
    return q, np.float32(scale)


def matmul(w, xq, xs):
    data, scales = w
    acc = data.astype(np.int32) @ xq.astype(np.int32)
    return (acc.astype(np.float32) * xs * scales).astype(np.float32)


def rmsnorm(x, w):
    ss = np.float32(np.mean(x * x) + 1e-5)
    return (w * (x / np.sqrt(ss))).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def encode(vocab, scores, text, bos=True):
    lookup = {v: i for i, v in enumerate(vocab)}
    toks = [1] if bos else []
    if text:
        toks.append(lookup[b" "])
    for ch in text:
        b = ch.encode()
        toks.append(lookup[b]) if b in lookup else toks.extend(x + 3 for x in b)
    while True:
        best = None
        for i in range(len(toks) - 1):
            merged = vocab[toks[i]] + vocab[toks[i + 1]]
            if merged in lookup:
                sc = scores[lookup[merged]]
                if best is None or sc > best[0]:
                    best = (sc, lookup[merged], i)
        if best is None:
            break
        toks[best[2]] = best[1]
        del toks[best[2] + 1]
    return toks


def decode(vocab, prev, t):
    p = vocab[t]
    if prev == 1 and p.startswith(b" "):
        p = p[1:]
    if len(p) == 6 and p.startswith(b"<0x") and p.endswith(b">"):
        return bytes([int(p[3:5], 16)])
    return p


def forward(m, st, token, pos):
    c = m["cfg"]
    d, hs = c["dim"], c["dim"] // c["n_heads"]
    kv = d * c["n_kv_heads"] // c["n_heads"]
    kv_mul = c["n_heads"] // c["n_kv_heads"]
    emb_data, emb_s = m["emb"]
    x = emb_data[token].astype(np.float32) * emb_s[token]

    # RoPE: same formula as the Rust code (powers in f32)
    i = np.arange(0, d, 2)
    freq = 1.0 / np.power(np.float32(10000.0), (i % hs).astype(np.float32) / np.float32(hs))
    ang = (pos * freq).astype(np.float32)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def rope(v, n):
        v = v.copy()
        a, b = v[0:n:2].copy(), v[1:n:2].copy()
        v[0:n:2] = a * cos[: n // 2] - b * sin[: n // 2]
        v[1:n:2] = a * sin[: n // 2] + b * cos[: n // 2]
        return v

    for l, L in enumerate(m["layers"]):
        xb = rmsnorm(x, L["rms_att"])
        xq, xs = quantize(xb)
        q = rope(matmul(L["wq"], xq, xs), d)
        k = rope(matmul(L["wk"], xq, xs), kv)
        v = matmul(L["wv"], xq, xs)
        st["k"][l, pos] = k
        st["v"][l, pos] = v

        out = np.zeros(d, dtype=np.float32)
        for h in range(c["n_heads"]):
            qh = q[h * hs:(h + 1) * hs]
            kvh = (h // kv_mul) * hs
            K = st["k"][l, : pos + 1, kvh:kvh + hs]
            V = st["v"][l, : pos + 1, kvh:kvh + hs]
            att = softmax((K @ qh) / np.sqrt(np.float32(hs)))
            out[h * hs:(h + 1) * hs] = att @ V

        xq, xs = quantize(out)
        x = x + matmul(L["wo"], xq, xs)

        xb = rmsnorm(x, L["rms_ffn"])
        xq, xs = quantize(xb)
        hb = matmul(L["w1"], xq, xs)
        hb2 = matmul(L["w3"], xq, xs)
        hb = (hb * (1.0 / (1.0 + np.exp(-hb)))) * hb2
        hq, hs_ = quantize(hb.astype(np.float32))
        x = x + matmul(L["w2"], hq, hs_)

    xb = rmsnorm(x, m["rms_final"])
    xq, xs = quantize(xb)
    return matmul(m["wcls"], xq, xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokenizer")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--show-tokens", action="store_true")
    a = ap.parse_args()

    m = load_model(a.model)
    c = m["cfg"]
    vocab, scores = load_tokenizer(a.tokenizer, c["vocab_size"])
    steps = min(a.steps, c["seq_len"])
    kv = c["dim"] * c["n_kv_heads"] // c["n_heads"]
    st = {"k": np.zeros((c["n_layers"], steps, kv), np.float32),
          "v": np.zeros((c["n_layers"], steps, kv), np.float32)}

    toks = encode(vocab, scores, a.prompt)
    token, out, ids = toks[0], bytearray(), []
    for pos in range(steps):
        logits = forward(m, st, token, pos)
        nxt = toks[pos + 1] if pos + 1 < len(toks) else int(np.argmax(logits))
        if nxt in (1, 2):
            break
        out += decode(vocab, token, nxt)
        ids.append(nxt)
        token = nxt
    print(out.decode("utf-8", errors="replace"))
    if a.show_tokens:
        print(ids)


if __name__ == "__main__":
    main()
