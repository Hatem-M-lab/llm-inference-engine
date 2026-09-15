import sys, time, numpy as np
from tinyllama import Config, init_model, forward, all_params
from losses import cross_entropy
from adam import Adam

import os
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT   = os.path.join(HERE, "ckpt.npz")
LOG    = os.path.join(HERE, "train_log.txt")
CORPUS = os.environ.get("CORPUS", os.path.join(HERE, "shakespeare.txt"))

cfg = Config(vocab_size=256, hidden_size=128, n_layers=4, n_heads=4, n_kv_heads=2, intermediate_size=256)
PARAM_NAMES = (["token_embedding", "final_norm", "lm_head"] +
    sum([[f"L{i}.attn_norm", f"L{i}.wq", f"L{i}.wk", f"L{i}.wv", f"L{i}.wo",
          f"L{i}.ffn_norm", f"L{i}.w_gate", f"L{i}.w_up", f"L{i}.w_down"]
         for i in range(cfg.n_layers)], []))

def build_model(seed=42):
    return init_model(cfg, seed=seed)

def save_ckpt(model, opt, step):
    params = all_params(model)
    d = {f"p_{n}": p.data for n, p in zip(PARAM_NAMES, params)}
    d.update({f"m_{n}": m for n, m in zip(PARAM_NAMES, opt.m)})
    d.update({f"v_{n}": v for n, v in zip(PARAM_NAMES, opt.v)})
    d["step"] = np.array(step); d["t"] = np.array(opt.t)
    np.savez(CKPT, **d)

def load_ckpt(model, opt):
    d = np.load(CKPT)
    params = all_params(model)
    for n, p in zip(PARAM_NAMES, params):
        p.data[...] = d[f"p_{n}"]
    opt.m = [d[f"m_{n}"].copy() for n in PARAM_NAMES]
    opt.v = [d[f"v_{n}"].copy() for n in PARAM_NAMES]
    opt.t = int(d["t"])
    return int(d["step"])

def load_corpus():
    if not os.path.exists(CORPUS):
        raise SystemExit(
            f"corpus not found: {CORPUS}\n"
            "fetch it with:\n"
            "  curl -o training/shakespeare.txt \\\n"
            "    https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")
    with open(CORPUS, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def sample_batch(corpus, S, rng):
    s = int(rng.integers(0, len(corpus) - S - 1))
    window = corpus[s:s + S]
    ids = [0] + list(window[:-1].astype(int))
    targets = list(window.astype(int))
    return ids, targets

def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    S = 96
    resume = len(sys.argv) > 2 and sys.argv[2] == "resume"

    model = build_model()
    params = all_params(model)
    opt = Adam(params, lr=3e-3)
    start_step = 0
    if resume:
        start_step = load_ckpt(model, opt)
        print(f"resumed from step {start_step}")

    corpus = load_corpus()
    rng = np.random.default_rng(1234 + start_step)

    t0 = time.time()
    losses = []
    with open(LOG, "a") as logf:
        for i in range(n_steps):
            step = start_step + i
            ids, targets = sample_batch(corpus, S, rng)
            opt.zero_grad()
            logits = forward(model, cfg, ids)
            loss = cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            losses.append(loss.data)
            if step % 50 == 0 or i == n_steps - 1:
                avg = np.mean(losses[-50:])
                line = f"step {step:6d}  loss {loss.data:.4f}  avg50 {avg:.4f}  elapsed {time.time()-t0:6.1f}s"
                print(line); logf.write(line + "\n"); logf.flush()

    save_ckpt(model, opt, start_step + n_steps)
    print(f"checkpoint saved at step {start_step + n_steps} -> {CKPT}")

if __name__ == "__main__":
    main()
