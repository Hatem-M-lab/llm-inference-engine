import numpy as np
from autodiff import T
from tinyllama import Config, init_model, forward, all_params

rng = np.random.default_rng(2)

from losses import cross_entropy

def numeric_grad_scalar_param(f, x, eps=1e-4, n_samples=6, rng=None):
    """Check a handful of random coordinates of a (possibly large) param array."""
    flat = x.reshape(-1)
    idxs = rng.choice(len(flat), size=min(n_samples, len(flat)), replace=False)
    g = {}
    for idx in idxs:
        orig = flat[idx]
        flat[idx] = orig + eps; fp = f()
        flat[idx] = orig - eps; fm = f()
        flat[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
    return g, idxs

cfg = Config(vocab_size=13, hidden_size=16, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=20)
model = init_model(cfg, seed=3)
ids = [1, 4, 2, 7, 9, 3]         # toy sequence, len 6 < vocab 13
targets = [4, 2, 7, 9, 3, 1]     # arbitrary "next token" targets, same length

def loss_fn():
    logits = forward(model, cfg, ids)
    return cross_entropy(logits, targets)

loss = loss_fn()
loss.backward()

all_ok = True
params = all_params(model)
names = (["token_embedding", "final_norm", "lm_head"] +
         sum([[f"L{i}.attn_norm", f"L{i}.wq", f"L{i}.wk", f"L{i}.wv", f"L{i}.wo",
               f"L{i}.ffn_norm", f"L{i}.w_gate", f"L{i}.w_up", f"L{i}.w_down"]
              for i in range(cfg.n_layers)], []))

for name, p in zip(names, params):
    g_num, idxs = numeric_grad_scalar_param(lambda: float(loss_fn().data), p.data, rng=rng)
    ag = p.grad.reshape(-1)
    errs = []
    for idx in idxs:
        num = g_num[idx]; ana = ag[idx]
        err = abs(num - ana) / (abs(num) + 1e-6)
        errs.append(err)
    maxerr = max(errs)
    status = "OK" if maxerr < 2e-2 else "FAIL"   # looser tol: eps=1e-4 + full 2-layer model compounding
    if status == "FAIL": all_ok = False
    print(f"  [{name}] shape={p.data.shape} sampled_max_rel_err={maxerr:.2e}  {status}")
    p.grad[...] = 0

print()
print(f"loss = {loss.data:.6f}")
print("FULL MODEL GRADIENT CHECK: " + ("ALL OK" if all_ok else "*** FAILED ***"))
