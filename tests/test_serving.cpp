#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/serving.hpp"
#include "llm/sampler.hpp"
#include "llm/model.hpp"
#include <vector>

// tests/test_serving.cpp
// (reuses tiny_model/tiny_cfg, forward_cached, argmax)
#include "test_framework.hpp"
#include "llm/serving.hpp"
#include "llm/model.hpp"
#include "llm/sampler.hpp"
using namespace llm;

TEST(batched_decode_equals_independent_decode) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    const std::vector<std::vector<int>> prompts = {{3,7,1}, {5,2}, {9,4,6,8}};
    const int B = 3, V = c.vocab_size;

    std::vector<KVCache> bcache, rcache;     // batched caches and reference caches
    std::vector<int> feed(B), ref_next(B);
    for (int b = 0; b < B; ++b) {
        bcache.push_back(KVCache::allocate(c, 64));
        rcache.push_back(KVCache::allocate(c, 64));
        Tensor lb = forward_cached(m, c, prompts[b], bcache[b]);     // prefill both caches
        Tensor lr = forward_cached(m, c, prompts[b], rcache[b]);
        feed[b] = argmax(lb.data_ptr<float>() + (int64_t)(prompts[b].size()-1)*V, V);
        Tensor rn = forward_cached(m, c, {feed[b]}, rcache[b]);      // reference: decode alone
        ref_next[b] = argmax(rn.data_ptr<float>(), V);
    }

    std::vector<KVCache*> ptrs;
    for (auto& ca : bcache) ptrs.push_back(&ca);
    std::vector<int> bnext = batched_decode(m, c, ptrs, feed);       // one batched step

    for (int b = 0; b < B; ++b) CHECK(bnext[b] == ref_next[b]);      // batch == independent
}

// tests/test_serving.cpp  (continued)

TEST(paged_cache_stores_and_reads_back) {
    const ModelConfig c = tiny_cfg();                    // n_layers=2, kvd = 1*4 = 4
    PagedKVCache pool = PagedKVCache::create(c, /*page_size=*/4, /*num_pages=*/8);
    SequenceCache seq;
    const int kvd = c.n_kv_heads * c.head_dim;
    for (int t = 0; t < 6; ++t) {                        // 6 tokens -> 2 pages of size 4
        REQUIRE(ensure_page(pool, seq));
        std::vector<float> k0(kvd, (float)t),      v0(kvd, (float)(100 + t));
        std::vector<float> k1(kvd, (float)(t+10)), v1(kvd, (float)(200 + t));
        write_kv(pool, seq, 0, k0.data(), v0.data());
        write_kv(pool, seq, 1, k1.data(), v1.data());
        seq.length++;
    }
    CHECK((int)seq.block_table.size() == 2);
    for (int t = 0; t < 6; ++t) {                        // read back, both layers
        CHECK_CLOSE(read_k(pool, seq, 0, t)[0], (float)t,       1e-6);
        CHECK_CLOSE(read_v(pool, seq, 0, t)[0], (float)(100+t), 1e-6);
        CHECK_CLOSE(read_k(pool, seq, 1, t)[0], (float)(t+10),  1e-6);
        CHECK_CLOSE(read_v(pool, seq, 1, t)[0], (float)(200+t), 1e-6);
    }
}

TEST(paged_cache_reuses_freed_pages) {
    const ModelConfig c = tiny_cfg();
    PagedKVCache pool = PagedKVCache::create(c, 2, 4);
    int a = pool.alloc_page(), b = pool.alloc_page();    // 0, 1
    CHECK(a == 0); CHECK(b == 1);
    pool.free_page(a);
    CHECK(pool.alloc_page() == a);                       // freed page reused (LIFO)
}

TEST(paged_cache_two_sequences_share_pool) {
    const ModelConfig c = tiny_cfg();
    PagedKVCache pool = PagedKVCache::create(c, 2, 8);
    SequenceCache s1, s2;
    const int kvd = c.n_kv_heads * c.head_dim;
    for (int t = 0; t < 3; ++t) {                        // interleaved growth
        ensure_page(pool, s1);
        std::vector<float> a(kvd, (float)t);      write_kv(pool, s1, 0, a.data(), a.data()); s1.length++;
        ensure_page(pool, s2);
        std::vector<float> b(kvd, (float)(t+50)); write_kv(pool, s2, 0, b.data(), b.data()); s2.length++;
    }
    for (int t = 0; t < 3; ++t) {
        CHECK_CLOSE(read_k(pool, s1, 0, t)[0], (float)t,      1e-6);
        CHECK_CLOSE(read_k(pool, s2, 0, t)[0], (float)(t+50), 1e-6);   // no cross-contamination
    }
    for (int p1 : s1.block_table) for (int p2 : s2.block_table) CHECK(p1 != p2);  // disjoint
}

// tests/test_serving.cpp  (continued; reuses generate)

TEST(boss_serve_output_matches_standalone) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    std::vector<Request> reqs = {
        {{3,7,1}, 5, -1, {}, false},
        {{5,2},   3, -1, {}, false},
        {{9,4,6}, 6, -1, {}, false},
        {{1,1},   4, -1, {}, false},
    };
    serve(m, c, reqs, /*max_batch=*/2, nullptr);            // continuous admission at batch 2
    for (auto& r : reqs) {
        SamplingParams sp; sp.greedy = true; RNG rng(0);
        auto solo = generate(m, c, r.prompt, r.max_new, r.eos, sp, rng);
        CHECK(r.output == solo);                            // identical to running alone
    }
}

TEST(boss_serve_fewer_iterations_than_tokens) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    std::vector<Request> reqs = {
        {{3,7,1}, 6, -1, {}, false}, {{5,2}, 6, -1, {}, false},
        {{9,4,6}, 6, -1, {}, false}, {{1,1}, 6, -1, {}, false},
    };
    int iters = 0;
    serve(m, c, reqs, /*max_batch=*/4, &iters);
    int total = 0; for (auto& r : reqs) total += (int)r.output.size();
    CHECK(total == 24);                                     // 4 requests x 6 tokens
    REQUIRE(iters > 0);
    CHECK(iters < total);                                   // < 1 iteration per token
}
