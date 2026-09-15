import numpy as np
from autodiff import T
from tinyllama import gqa_causal_attention_core, apply_rope, rope_cos_sin

rng = np.random.default_rng(1)

def numeric_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps; fp = f()
        x[idx] = orig - eps; fm = f()
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
    return g

def check(name, build_fn, leaves):
    out = build_fn()
    out.backward()
    ok = True
    for i, leaf in enumerate(leaves):
        def f():
            return float(build_fn().data)
        ng = numeric_grad(f, leaf.data)
        ag = leaf.grad.copy()
        err = np.max(np.abs(ng - ag)) / (np.max(np.abs(ng)) + 1e-8)
        status = "OK" if err < 1e-4 else "FAIL"
        if status == "FAIL": ok = False
        print(f"  [{name}] leaf{i} shape={leaf.data.shape} max_rel_err={err:.2e}  {status}")
        leaf.grad[...] = 0
    return ok

def wsum(x, w):
    out = T(np.sum(x.data * w), (x,))
    def bw(): x.grad += out.grad * w
    out._backward = bw
    return out

all_ok = True
S, H, KVH, D = 5, 4, 2, 6   # small but with GQA (H != KVH) and S>1 for a real causal mask

# --- 1) the attention core alone (q,k already given directly, no rope) ---
q = T(rng.standard_normal((S, H * D)))
k = T(rng.standard_normal((S, KVH * D)))
v = T(rng.standard_normal((S, KVH * D)))
w = rng.standard_normal((S, H * D))
all_ok &= check("gqa_causal_attention_core",
                lambda: wsum(gqa_causal_attention_core(q, k, v, H, KVH, D), w),
                [q, k, v])

# --- 2) apply_rope alone ---
x = T(rng.standard_normal((S, H * D)))
cos, sin = rope_cos_sin(S, D, 10000.0)
w2 = rng.standard_normal((S, H * D))
all_ok &= check("apply_rope", lambda: wsum(apply_rope(x, cos, sin, H, D), w2), [x])

# --- 3) the two composed together (rope feeding into attention), the way `attention()` uses them ---
q0 = T(rng.standard_normal((S, H * D)))
k0 = T(rng.standard_normal((S, KVH * D)))
v0 = T(rng.standard_normal((S, KVH * D)))
w3 = rng.standard_normal((S, H * D))
def composed():
    cos_, sin_ = rope_cos_sin(S, D, 10000.0)
    qr = apply_rope(q0, cos_, sin_, H, D)
    kr = apply_rope(k0, cos_, sin_, KVH, D)
    return wsum(gqa_causal_attention_core(qr, kr, v0, H, KVH, D), w3)
all_ok &= check("rope+attention composed", composed, [q0, k0, v0])

print()
print("ALL ATTENTION CHECKS OK" if all_ok else "*** ATTENTION CHECK FAILED ***")
