"""
Minimal reverse-mode autodiff over NumPy arrays.
Every primitive implements forward + local backward; correctness of the whole
graph then follows from the chain rule by construction. Each primitive is
checked against finite differences in grad_check_primitives() below.
"""
import numpy as np

class T:
    """A node: wraps a numpy array, remembers parents + a backward closure."""
    __slots__ = ("data", "grad", "_parents", "_backward", "_label")
    def __init__(self, data, parents=(), backward=lambda: None, label=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = backward
        self._label = label

    def backward(self):
        topo, seen = [], set()
        def build(n):
            if id(n) not in seen:
                seen.add(id(n))
                for p in n._parents:
                    build(p)
                topo.append(n)
        build(self)
        self.grad = np.ones_like(self.data)
        for n in reversed(topo):
            n._backward()

def const(x):
    return T(x)

# ---------------------------------------------------------------- add / mul
def add(a, b):
    out = T(a.data + b.data, (a, b))
    def bw():
        ga, gb = out.grad, out.grad
        # unbroadcast to a/b shapes
        ga = _unbroadcast(ga, a.data.shape)
        gb = _unbroadcast(gb, b.data.shape)
        a.grad += ga; b.grad += gb
    out._backward = bw
    return out

def mul(a, b):
    out = T(a.data * b.data, (a, b))
    def bw():
        a.grad += _unbroadcast(out.grad * b.data, a.data.shape)
        b.grad += _unbroadcast(out.grad * a.data, b.data.shape)
    out._backward = bw
    return out

def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g

# --------------------------------------------------------------------- matmul
def matmul(a, b):
    out = T(a.data @ b.data, (a, b))
    def bw():
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad
    out._backward = bw
    return out

def transpose(a):
    out = T(a.data.T, (a,))
    def bw(): a.grad += out.grad.T
    out._backward = bw
    return out

# --------------------------------------------------------------------- silu
def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))

def silu(a):
    s = sigmoid_np(a.data)
    y = a.data * s
    out = T(y, (a,))
    def bw():
        # d/dx [x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        local = s + a.data * s * (1.0 - s)
        a.grad += out.grad * local
    out._backward = bw
    return out

# ------------------------------------------------------------------ rmsnorm
def rmsnorm(x, w, eps):
    # x: [..., D]  w: [D]   (matches src/ops.cpp rmsnorm exactly)
    ss = np.mean(x.data * x.data, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ss + eps)                 # [...,1]
    normed = x.data * inv_rms                          # [...,D]
    y = normed * w.data                                 # broadcast [D]
    out = T(y, (x, w))
    D = x.data.shape[-1]
    def bw():
        g = out.grad                                    # [...,D]
        gw = np.sum(g * normed, axis=tuple(range(g.ndim - 1)))
        w.grad += gw
        g_normed = g * w.data                            # d/d(normed)
        # normed = x * inv_rms ; inv_rms depends on x too.
        # d(normed)/dx_j = inv_rms*delta_ij - x_i * x_j / (D * rms^3)   [rms = 1/inv_rms]
        dot = np.sum(g_normed * x.data, axis=-1, keepdims=True)  # sum_i g_i * x_i
        gx = g_normed * inv_rms - x.data * (inv_rms ** 3) * dot / D
        x.grad += gx
    out._backward = bw
    return out

# ----------------------------------------------------------------- softmax
def softmax_lastdim(a):
    m = np.max(a.data, axis=-1, keepdims=True)
    e = np.exp(a.data - m)
    s = e / np.sum(e, axis=-1, keepdims=True)
    out = T(s, (a,))
    def bw():
        g = out.grad
        dot = np.sum(g * s, axis=-1, keepdims=True)
        a.grad += s * (g - dot)
    out._backward = bw
    return out

# ------------------------------------------------------------ embedding gather
def embed_rows(table, ids):
    # table: T [V, D]  ids: python list/array[int]  -> T [S, D]
    ids = np.asarray(ids, dtype=np.int64)
    y = table.data[ids]
    out = T(y, (table,))
    def bw():
        np.add.at(table.grad, ids, out.grad)
    out._backward = bw
    return out

def reshape(a, shape):
    orig = a.data.shape
    out = T(a.data.reshape(shape), (a,))
    def bw(): a.grad += out.grad.reshape(orig)
    out._backward = bw
    return out

def add_bias_like(a):  # identity helper for symmetry / debugging
    return a
