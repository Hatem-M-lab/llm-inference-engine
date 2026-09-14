#!/usr/bin/env bash
# Build & test the LLM inference engine reference code on g++ (C++20).
# Verified on g++ 13.3.0. Requires a CPU with AVX2 + FMA.
set -e
CXX="${CXX:-g++}"
FLAGS="-std=c++20 -Iinclude -Itests -O2 -Wall -mavx2 -mfma -pthread"

echo "== Units 0-9 (core engine: tensor, ops, gemm/SIMD, tokenizer, Transformer, MoE, KV cache, Flash, quant) =="
$CXX $FLAGS \
  src/version.cpp src/tensor.cpp src/ops.cpp src/gemm.cpp src/tokenizer.cpp src/model.cpp \
  tests/test_smoke.cpp tests/test_common.cpp tests/test_tensor.cpp tests/test_ops.cpp \
  tests/test_gemm.cpp tests/test_tokenizer.cpp tests/test_model.cpp tests/test_moe.cpp \
  tests/test_kvcache.cpp tests/test_flash.cpp tests/test_quant.cpp tests/test_main.cpp \
  -o llm_tests
./llm_tests

echo ""
echo "== Units 10-14 (header-only modules: each test is its own binary) =="
mkdir -p obj
for s in version tensor ops gemm tokenizer model test_main; do
  $CXX $FLAGS -c $( [ "$s" = test_main ] && echo tests/test_main.cpp || echo src/$s.cpp ) -o obj/$s.o
done
for t in gguf sampler speculative serving profile threadpool engine; do
  $CXX $FLAGS tests/test_$t.cpp obj/test_main.o obj/version.o obj/tensor.o obj/ops.o \
    obj/gemm.o obj/tokenizer.o obj/model.o -o bin_$t
  echo "-- $t --"; ./bin_$t
done
echo ""
echo "Note: Unit 15 (CUDA) requires nvcc + an NVIDIA GPU and is not built here."
