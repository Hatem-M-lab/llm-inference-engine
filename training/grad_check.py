import numpy as np
from autodiff import T, add, mul, matmul, transpose, silu, rmsnorm, softmax_lastdim, embed_rows

rng = np.random.default_rng(0)

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

def wsum(x, w):
    """Fixed-weight scalar reduction sum(x*w) -- non-degenerate, w is a plain
    constant ndarray captured by closure (stable across every call)."""
    out = T(np.sum(x.data * w), (x,))
    def bw(): x.grad += out.grad * w
    out._backward = bw
    return out

def check(name, build_fn, leaves):
    out = build_fn()
    out.backward()
    ok = True
    for i, leaf in enumerate(leaves):
        def f():
            o = build_fn()
            return float(o.data)
        ng = numeric_grad(f, leaf.data)
        ag = leaf.grad.copy()
        err = np.max(np.abs(ng - ag)) / (np.max(np.abs(ng)) + 1e-8)
        status = "OK" if err < 1e-4 else "FAIL"
        if status == "FAIL": ok = False
        print(f"  [{name}] leaf{i} shape={leaf.data.shape} max_rel_err={err:.2e}  {status}")
        leaf.grad[...] = 0
    return ok

all_ok = True

a = T(rng.standard_normal((3, 4))); b = T(rng.standard_normal((4,)))
w = rng.standard_normal((3, 4))
all_ok &= check("add(broadcast)", lambda: wsum(add(a, b), w), [a, b])

a = T(rng.standard_normal((3, 4))); b = T(rng.standard_normal((3, 4)))
w = rng.standard_normal((3, 4))
all_ok &= check("mul(elementwise)", lambda: wsum(mul(a, b), w), [a, b])

a = T(rng.standard_normal((5, 6))); b = T(rng.standard_normal((6, 3)))
w = rng.standard_normal((5, 3))
all_ok &= check("matmul", lambda: wsum(matmul(a, b), w), [a, b])

a = T(rng.standard_normal((4, 5)))
w = rng.standard_normal((5, 4))
all_ok &= check("transpose", lambda: wsum(transpose(a), w), [a])

a = T(rng.standard_normal((3, 5)))
w = rng.standard_normal((3, 5))
all_ok &= check("silu", lambda: wsum(silu(a), w), [a])

x = T(rng.standard_normal((3, 8))); wt = T(rng.standard_normal((8,)))
w = rng.standard_normal((3, 8))
all_ok &= check("rmsnorm", lambda: wsum(rmsnorm(x, wt, 1e-5), w), [x, wt])

a = T(rng.standard_normal((3, 6)))
w = rng.standard_normal((3, 6))
all_ok &= check("softmax_lastdim", lambda: wsum(softmax_lastdim(a), w), [a])

table = T(rng.standard_normal((10, 4)))
ids = [2, 5, 2, 9]
w = rng.standard_normal((4, 4))
all_ok &= check("embed_rows", lambda: wsum(embed_rows(table, ids), w), [table])

print()
print("ALL PRIMITIVES OK" if all_ok else "*** SOME PRIMITIVES FAILED ***")
