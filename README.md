# LLM Inference Engine — Reference Code

Companion reference implementation for **Build an LLM Inference Engine in C++ — Through Challenges**.
The book teaches you to build this yourself, challenge by challenge; this tree is the assembled,
compile-verified reference.

## Build & test
```bash
./build_and_test.sh        # or: CXX=g++-13 ./build_and_test.sh
```
Requirements: a C++20 compiler (verified on **g++ 13.3.0**) and a CPU with **AVX2 + FMA**.

## Verification status
All runnable units (0–14) compile cleanly (`-std=c++20 -Wall`) and pass their tests:
**~121 tests / ~19,000 assertions, 0 failures.** Highlights: a real ~14× SIMD matmul speedup,
a deterministic causal Transformer forward pass, GGUF load→forward roundtrip, and an
end-to-end "engine is alive" test.

- **Unit 15 (CUDA)** requires an NVIDIA GPU and is **not** verified here.
- Two *performance* tests (SIMD-faster, threaded-faster) assert speedups and may flake under
  heavy CPU contention; all *correctness* tests are deterministic.

## Layout
- `include/llm/` — headers   `src/` — library sources   `tests/` — unit tests   `tools/` — CLI
- `tests/test_helpers.hpp` — shared test fixtures (collected so multiple test files can reuse them)
