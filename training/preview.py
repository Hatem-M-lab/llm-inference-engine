import sys, numpy as np
from tinyllama import Config, init_model, forward
from train import cfg, build_model, load_ckpt
from adam import Adam
from tinyllama import all_params

model = build_model()
opt = Adam(all_params(model))
step = load_ckpt(model, opt)

def generate_greedy(prompt: str, n=120, temperature=0.8, seed=0):
    rng = np.random.default_rng(seed)
    ids = [0] + [b for b in prompt.encode('utf-8')]
    out_bytes = bytearray()
    for _ in range(n):
        logits = forward(model, cfg, ids)
        last = logits.data[-1] / temperature
        last = last - last.max()
        p = np.exp(last); p /= p.sum()
        nxt = int(rng.choice(256, p=p))
        out_bytes.append(nxt)
        ids.append(nxt)
        if len(ids) > 200:  # keep it bounded for this quick preview
            ids = ids[:1] + ids[-199:]
    try:
        return out_bytes.decode('utf-8', errors='replace')
    except Exception:
        return repr(out_bytes)

print(f"=== checkpoint at step {step} ===")
for prompt in ["ROMEO:", "The king", "To be"]:
    print(f"--- prompt: {prompt!r} ---")
    print(generate_greedy(prompt, n=150, temperature=0.7))
    print()
