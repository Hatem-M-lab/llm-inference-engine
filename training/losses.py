import numpy as np
from autodiff import T

def cross_entropy(logits, targets):
    # logits: T [S, V]   targets: list[int] len S  -> scalar T (mean NLL over S positions)
    x = logits.data
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    S = x.shape[0]
    nll = -np.log(probs[np.arange(S), targets] + 1e-12)
    out = T(float(np.mean(nll)), (logits,))
    def bw():
        g = probs.copy()
        g[np.arange(S), targets] -= 1.0
        g /= S
        logits.grad += g * out.grad
    out._backward = bw
    return out
