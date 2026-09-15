"""
A Llama-style model built from the autodiff primitives in autodiff.py.
Every formula here matches the book's actual C++ source (src/model.cpp,
src/ops.cpp) exactly -- see the comments citing what each line mirrors.
"""
import numpy as np
from autodiff import T, add, mul, matmul, transpose, silu, rmsnorm, softmax_lastdim, embed_rows

class Config:
    def __init__(self, vocab_size, hidden_size, n_layers, n_heads, n_kv_heads,
                 intermediate_size, rope_theta=10000.0, rms_norm_eps=1e-5):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

def init_layer(cfg, rng):
    H, KVH, D, hid, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.hidden_size, cfg.intermediate_size
    def lin(o, i): return T(rng.standard_normal((o, i)) * (1.0 / np.sqrt(i)))  # [out,in]
    return {
        "attn_norm": T(np.ones(hid)),
        "wq": lin(H * D, hid), "wk": lin(KVH * D, hid), "wv": lin(KVH * D, hid), "wo": lin(hid, H * D),
        "ffn_norm": T(np.ones(hid)),
        "w_gate": lin(I, hid), "w_up": lin(I, hid), "w_down": lin(hid, I),
    }

def init_model(cfg, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "token_embedding": T(rng.standard_normal((cfg.vocab_size, cfg.hidden_size)) * 0.02),
        "layers": [init_layer(cfg, rng) for _ in range(cfg.n_layers)],
        "final_norm": T(np.ones(cfg.hidden_size)),
        "lm_head": T(rng.standard_normal((cfg.vocab_size, cfg.hidden_size)) * (1.0 / np.sqrt(cfg.hidden_size))),
    }

def linear(x, w):
    # book (src/model.cpp `linear`): out[i,o] = sum_k x[i,k]*w[o,k]  ==  x @ w.T
    return matmul(x, transpose(w))

def rope_cos_sin(seq, head_dim, theta):
    # EXACTLY src/model.cpp apply_rope: inv_freq[i] = theta^(-2i/head_dim), i in [0,half)
    half = head_dim // 2
    inv_freq = theta ** (-2.0 * np.arange(half) / head_dim)          # [half]
    pos = np.arange(seq)                                              # [seq]
    ang = np.outer(pos, inv_freq)                                     # [seq, half]
    return np.cos(ang), np.sin(ang)                                   # each [seq, half]

def apply_rope(x, cos, sin, n_heads, head_dim):
    # x: T [seq, n_heads*head_dim]. Matches apply_rope: v[i]=x0*c-x1*s, v[i+half]=x1*c+x0*s.
    seq = x.data.shape[0]
    half = head_dim // 2
    xr = x.data.reshape(seq, n_heads, head_dim)
    x0 = xr[:, :, :half]; x1 = xr[:, :, half:]
    c = cos[:, None, :]; s = sin[:, None, :]
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    y = np.concatenate([y0, y1], axis=-1).reshape(seq, n_heads * head_dim)
    out = T(y, (x,))
    def bw():
        g = out.grad.reshape(seq, n_heads, head_dim)
        g0 = g[:, :, :half]; g1 = g[:, :, half:]
        gx0 = g0 * c + g1 * s          # adjoint of the 2x2 rotation
        gx1 = -g0 * s + g1 * c
        x.grad += np.concatenate([gx0, gx1], axis=-1).reshape(seq, n_heads * head_dim)
    out._backward = bw
    return out

def gqa_causal_attention_core(q, k, v, n_heads, n_kv_heads, head_dim):
    """
    q: T [S, H*D] (already RoPE'd)   k: T [S, KVH*D] (already RoPE'd)   v: T [S, KVH*D]
    -> T [S, H*D], the per-head weighted sums concatenated (BEFORE the wo projection).
    One self-contained op with a hand-derived backward (verified by grad_check_attention.py
    before this is ever used for training) -- mirrors src/model.cpp `attention` exactly:
    scale = 1/sqrt(D); causal softmax over j<=i; kvh = h // (H/KVH).
    """
    S = q.data.shape[0]; H = n_heads; KVH = n_kv_heads; D = head_dim; group = H // KVH
    scale = 1.0 / np.sqrt(D)

    qh = q.data.reshape(S, H, D)
    kh = k.data.reshape(S, KVH, D)
    vh = v.data.reshape(S, KVH, D)
    kh_full = np.repeat(kh, group, axis=1)     # [S,H,D]; kh_full[:,h,:] == kh[:, h//group, :]
    vh_full = np.repeat(vh, group, axis=1)

    scores = np.einsum('ihd,jhd->hij', qh, kh_full) * scale         # [H,S,S]
    causal = np.triu(np.full((S, S), -np.inf), k=1)                  # 0 for j<=i, -inf for j>i
    scores = scores + causal[None, :, :]
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)                    # [H,S,S]

    out_h = np.einsum('hij,jhd->ihd', probs, vh_full)                # [S,H,D]
    out = out_h.reshape(S, H * D)

    result = T(out, (q, k, v))
    def bw():
        g = result.grad.reshape(S, H, D)
        g_probs = np.einsum('ihd,jhd->hij', g, vh_full)
        g_vh_full = np.einsum('hij,ihd->jhd', probs, g)

        dot = np.sum(g_probs * probs, axis=-1, keepdims=True)
        g_scores = probs * (g_probs - dot)                            # softmax backward (masked entries: probs=0)

        g_qh = np.einsum('hij,jhd->ihd', g_scores, kh_full) * scale
        g_kh_full = np.einsum('hij,ihd->jhd', g_scores, qh) * scale

        g_kh = g_kh_full.reshape(S, KVH, group, D).sum(axis=2)        # adjoint of np.repeat(..., group, axis=1)
        g_vh = g_vh_full.reshape(S, KVH, group, D).sum(axis=2)

        q.grad += g_qh.reshape(S, H * D)
        k.grad += g_kh.reshape(S, KVH * D)
        v.grad += g_vh.reshape(S, KVH * D)
    result._backward = bw
    return result

def attention(x, w, cfg):
    q = linear(x, w["wq"]); k = linear(x, w["wk"]); v = linear(x, w["wv"])
    cos, sin = rope_cos_sin(x.data.shape[0], cfg.head_dim, cfg.rope_theta)
    q = apply_rope(q, cos, sin, cfg.n_heads, cfg.head_dim)
    k = apply_rope(k, cos, sin, cfg.n_kv_heads, cfg.head_dim)          # values are NOT rotated
    ctx = gqa_causal_attention_core(q, k, v, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
    return linear(ctx, w["wo"])

def ffn(x, w):
    # book (src/model.cpp `ffn`): SiLU(gate)*up, then down-projection.
    gate = linear(x, w["w_gate"])
    up = linear(x, w["w_up"])
    act = mul(silu(gate), up)
    return linear(act, w["w_down"])

def transformer_block(x, w, cfg):
    # book (src/model.cpp `transformer_block`): pre-norm residual, attn then ffn.
    normed = rmsnorm(x, w["attn_norm"], cfg.rms_norm_eps)
    x = add(x, attention(normed, w, cfg))
    normed = rmsnorm(x, w["ffn_norm"], cfg.rms_norm_eps)
    x = add(x, ffn(normed, w))
    return x

def forward(model, cfg, ids):
    x = embed_rows(model["token_embedding"], ids)
    for w in model["layers"]:
        x = transformer_block(x, w, cfg)
    x = rmsnorm(x, model["final_norm"], cfg.rms_norm_eps)
    return linear(x, model["lm_head"])       # T [S, vocab]

def all_params(model):
    ps = [model["token_embedding"], model["final_norm"], model["lm_head"]]
    for w in model["layers"]:
        ps += [w["attn_norm"], w["wq"], w["wk"], w["wv"], w["wo"],
               w["ffn_norm"], w["w_gate"], w["w_up"], w["w_down"]]
    return ps
