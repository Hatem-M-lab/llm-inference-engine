#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/engine.hpp"
#include "llm/model.hpp"
#include <vector>

// tests/test_engine.cpp
// (reuses tiny_model/tiny_cfg, a byte-level Tokenizer from Unit 4, generate)
#include "test_framework.hpp"
#include "llm/engine.hpp"
#include "llm/threadpool.hpp"
#include <cstdio>
#include <chrono>
using namespace llm;

TEST(boss_engine_alive_end_to_end) {
    ModelConfig c = tiny_cfg(); c.vocab_size = 256;        // match a byte-level tokenizer
    const ModelWeights m = tiny_model(c, 4);
    Tokenizer tok = Tokenizer::with_byte_vocab();               // 256 single-byte tokens
    SamplingParams sp; sp.greedy = true;

    set_num_threads(4);
    RNG r1(0), r2(0);
    std::string a = run_engine(m, c, tok, "hi", 8, sp, r1);
    std::string b = run_engine(m, c, tok, "hi", 8, sp, r2);
    CHECK(a == b);                                         // deterministic end to end

    std::vector<int> ids = tok.encode("hi", /*bos=*/true);  // == encode + generate + decode
    RNG r3(0);
    auto ref = generate(m, c, ids, 8, tok.eos_id(), sp, r3);
    CHECK(tok.decode(ref) == a);
    set_num_threads(1);
}

TEST(boss_optimized_path_faster_same_output) {
    ModelConfig c = tiny_cfg();                            // a model big enough to time
    c.hidden_size = 512; c.n_heads = 8; c.head_dim = 64; c.n_kv_heads = 8;
    c.intermediate_size = 1376; c.n_layers = 4; c.vocab_size = 4096;
    const ModelWeights m = tiny_model(c, 7);
    const std::vector<int> prompt = {3, 7, 1, 5, 2};
    SamplingParams sp; sp.greedy = true;

    auto run = [&](int threads) {
        set_num_threads(threads); RNG rng(0);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = generate(m, c, prompt, /*max_new=*/32, /*eos=*/-1, sp, rng);
        double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
        return std::make_pair(out, 32.0 / secs);          // tokens, tokens/sec
    };
    auto serial = run(1);
    auto threaded = run(8);
    set_num_threads(1);

    CHECK(threaded.first == serial.first);                // byte-identical output
    CHECK(threaded.second > serial.second);               // higher tokens/sec
    std::printf("decode: %.1f tok/s serial, %.1f tok/s threaded (%.2fx)\n",
                serial.second, threaded.second, threaded.second / serial.second);
}
