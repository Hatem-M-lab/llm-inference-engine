# LLM Inference Engine — Reference Code

Companion reference implementation for **Build an LLM Inference Engine in C++ — Through Challenges**.
The book teaches you to build this yourself, challenge by challenge; this tree is the assembled,
compile-verified reference.

## Build & test
```bash
./build_and_test.sh        # or: CXX=g++-13 ./build_and_test.sh
```
Requirements: a C++20 compiler (verified on **g++ 13.3.0**) and a CPU with **AVX2 + FMA**.

## Generate text with a real model
A small model trained for the book's appendix ships in `training/`:

```bash
g++ -std=c++20 -Iinclude -O2 -pthread tools/cli/main.cpp \
    src/version.cpp src/tensor.cpp src/ops.cpp src/gemm.cpp \
    src/tokenizer.cpp src/model.cpp -o llm_cli

./llm_cli --model training/tinyshake.gguf --prompt "ROMEO:" --max-new 150
```
```
ROMEO:
Way, he wos more may sir, and of a bod, porson, for were and more word,
and him to with in faor ham to longosed,
```
It is a 656k-parameter, 4-layer model trained on tiny-shakespeare — rough, but genuinely
learned, and loaded entirely by this engine's own GGUF reader and forward pass.
See `training/README.md` to train your own or verify the math.

## Verification status
All runnable units (0–14) compile cleanly (`-std=c++20 -Wall`) and pass their tests:
**~121 tests / ~19,000 assertions, 0 failures.** Highlights: a real ~14× SIMD matmul speedup,
a deterministic causal Transformer forward pass, GGUF load→forward roundtrip, and an
end-to-end "engine is alive" test.

The appendix model is cross-validated: the same weights run through this C++ engine and
through an independent Python implementation agree to **3.7e-5** on all 256 output logits,
with identical top-5 predictions.

- **Unit 15 (CUDA)** requires an NVIDIA GPU and is **not** verified here.
- Two *performance* tests (SIMD-faster, threaded-faster) assert speedups and may flake under
  heavy CPU contention; all *correctness* tests are deterministic.

## Layout
- `include/llm/` — headers   `src/` — library sources   `tests/` — unit tests
- `tools/cli/` — the `llm_cli` text-generation tool
- `training/` — trains and exports the appendix model (Python + NumPy)
- `tests/test_helpers.hpp` — shared test fixtures
