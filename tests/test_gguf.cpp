#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/gguf.hpp"
#include "llm/model.hpp"
#include <vector>
#include <cstring>
#include <cstdint>

// tests/test_gguf.cpp
#include "test_framework.hpp"
#include "llm/gguf.hpp"
#include <vector>
#include <cstring>
using namespace llm;

// Helpers to build a little-endian GGUF blob by hand.
static void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i=0;i<4;++i) b.push_back(v>>(8*i)); }
static void put_u64(std::vector<uint8_t>& b, uint64_t v) { for (int i=0;i<8;++i) b.push_back(v>>(8*i)); }
static void put_f32(std::vector<uint8_t>& b, float f)    { uint32_t u; std::memcpy(&u,&f,4); put_u32(b,u); }
static void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size()); b.insert(b.end(), s.begin(), s.end());
}

TEST(gguf_parses_header_and_scalars) {
    std::vector<uint8_t> b;
    put_u32(b, 0x46554747);          // magic "GGUF"
    put_u32(b, 3);                   // version
    put_u64(b, 0);                   // tensor_count
    put_u64(b, 2);                   // metadata_kv_count
    put_str(b, "llama.block_count"); put_u32(b, GGUF_U32); put_u32(b, 24);
    put_str(b, "llama.rope.freq_base"); put_u32(b, GGUF_F32); put_f32(b, 10000.0f);

    GGUFReader r;
    REQUIRE(r.parse(b.data(), b.size()));
    CHECK(r.meta_u32("llama.block_count") == 24);
    CHECK_CLOSE(r.meta_f32("llama.rope.freq_base"), 10000.0f, 1e-3);
}

TEST(gguf_parses_string_array) {
    std::vector<uint8_t> b;
    put_u32(b, 0x46554747); put_u32(b, 3); put_u64(b, 0); put_u64(b, 1);
    put_str(b, "tokenizer.ggml.tokens");
    put_u32(b, GGUF_ARRAY); put_u32(b, GGUF_STRING); put_u64(b, 3);
    put_str(b, "<bos>"); put_str(b, "he"); put_str(b, "llo");

    GGUFReader r;
    REQUIRE(r.parse(b.data(), b.size()));
    const auto& toks = r.meta_str_array("tokenizer.ggml.tokens");
    REQUIRE(toks.size() == 3);
    CHECK(toks[0] == "<bos>");
    CHECK(toks[2] == "llo");
}

// add to tests/test_gguf.cpp  (reuses put_u32/put_u64/put_f32/put_str)

TEST(gguf_dequant_q8_0_block) {
    std::vector<uint8_t> blk;
    blk.push_back(0x00); blk.push_back(0x3C);          // f16 1.0 (0x3C00), little-endian
    for (int i = 0; i < 32; ++i) blk.push_back((uint8_t)(int8_t)(i - 16));   // -16..15
    std::vector<float> out(32);
    dequant_gguf_q8_0(blk.data(), 32, out.data());
    for (int i = 0; i < 32; ++i) CHECK_CLOSE(out[i], (float)(i - 16), 1e-3);  // scale 1.0
}

TEST(gguf_parses_tensor_and_loads_f32) {
    std::vector<uint8_t> b;
    put_u32(b, 0x46554747); put_u32(b, 3);
    put_u64(b, 1); put_u64(b, 0);                      // 1 tensor, 0 metadata
    put_str(b, "token_embd.weight");
    put_u32(b, 2);                                     // n_dims
    put_u64(b, 4); put_u64(b, 3);                      // dims {in=4, out=3} -> shape {3,4}
    put_u32(b, GGML_F32); put_u64(b, 0);               // type, offset
    while (b.size() % 32 != 0) b.push_back(0);         // align data section
    for (int i = 0; i < 12; ++i) put_f32(b, (float)i); // 3x4 data

    GGUFReader r;
    REQUIRE(r.parse(b.data(), b.size()));
    const auto* ti = r.tensor("token_embd.weight");
    REQUIRE(ti != nullptr);
    CHECK(ti->dims[0] == 4 && ti->dims[1] == 3 && ti->type == GGML_F32);
    Tensor t = load_tensor_f32(r, "token_embd.weight");
    CHECK(t.shape()[0] == 3 && t.shape()[1] == 4);     // reversed dims
    CHECK_CLOSE(t.at<float>({0, 0}), 0.0f, 1e-6);
    CHECK_CLOSE(t.at<float>({2, 3}), 11.0f, 1e-6);
}

// add to tests/test_gguf.cpp  (reuses put_*, tiny_model/tiny_cfg from model tests)
#include "llm/model.hpp"

// Minimal F32 GGUF writer for fixtures.
struct GgufWriter {
    std::vector<uint8_t> meta; int n_meta = 0;
    struct T { std::string name; std::vector<int64_t> dims; const float* data; int64_t n; };
    std::vector<T> ts;
    void u32(const std::string& k, uint32_t v){ put_str(meta,k); put_u32(meta,GGUF_U32); put_u32(meta,v); ++n_meta; }
    void f32(const std::string& k, float v){ put_str(meta,k); put_u32(meta,GGUF_F32); put_f32(meta,v); ++n_meta; }
    void str(const std::string& k, const std::string& s){ put_str(meta,k); put_u32(meta,GGUF_STRING); put_str(meta,s); ++n_meta; }
    void strs(const std::string& k, const std::vector<std::string>& a){
        put_str(meta,k); put_u32(meta,GGUF_ARRAY); put_u32(meta,GGUF_STRING); put_u64(meta,a.size());
        for (auto& s : a) put_str(meta,s); ++n_meta; }
    void tensor(const std::string& name, const Tensor& t){
        ts.push_back({name, std::vector<int64_t>(t.shape().rbegin(), t.shape().rend()),
                      t.data_ptr<float>(), t.numel()}); }
    std::vector<uint8_t> build(){
        std::vector<uint8_t> b;
        put_u32(b,0x46554747); put_u32(b,3); put_u64(b,ts.size()); put_u64(b,n_meta);
        b.insert(b.end(), meta.begin(), meta.end());
        std::vector<int64_t> off(ts.size()); int64_t cur = 0;
        for (size_t i=0;i<ts.size();++i){ off[i]=cur; cur += ts[i].n*4; cur=(cur+31)/32*32; }
        for (size_t i=0;i<ts.size();++i){
            put_str(b, ts[i].name); put_u32(b,(uint32_t)ts[i].dims.size());
            for (int64_t d : ts[i].dims) put_u64(b,(uint64_t)d);
            put_u32(b, GGML_F32); put_u64(b,(uint64_t)off[i]);
        }
        while (b.size()%32) b.push_back(0);
        size_t ds = b.size(); b.resize(ds + cur);
        for (size_t i=0;i<ts.size();++i) std::memcpy(b.data()+ds+off[i], ts[i].data, ts[i].n*4);
        return b;
    }
};

TEST(boss_gguf_roundtrip_forward) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);

    GgufWriter w;
    w.str("general.architecture", "llama");
    w.u32("llama.embedding_length", c.hidden_size);
    w.u32("llama.block_count", c.n_layers);
    w.u32("llama.attention.head_count", c.n_heads);
    w.u32("llama.attention.head_count_kv", c.n_kv_heads);
    w.u32("llama.feed_forward_length", c.intermediate_size);
    w.f32("llama.attention.layer_norm_rms_epsilon", c.rms_norm_eps);
    w.f32("llama.rope.freq_base", c.rope_theta);
    std::vector<std::string> toks(c.vocab_size);
    for (int i = 0; i < c.vocab_size; ++i) toks[i] = "t" + std::to_string(i);
    w.strs("tokenizer.ggml.tokens", toks);

    w.tensor("token_embd.weight", m.token_embedding);
    w.tensor("output_norm.weight", m.final_norm);
    w.tensor("output.weight", m.lm_head);
    for (int i = 0; i < c.n_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        w.tensor(p+"attn_norm.weight", m.layers[i].attn_norm);
        w.tensor(p+"attn_q.weight",    m.layers[i].wq);
        w.tensor(p+"attn_k.weight",    m.layers[i].wk);
        w.tensor(p+"attn_v.weight",    m.layers[i].wv);
        w.tensor(p+"attn_output.weight", m.layers[i].wo);
        w.tensor(p+"ffn_norm.weight",  m.layers[i].ffn_norm);
        w.tensor(p+"ffn_gate.weight",  m.layers[i].w_gate);
        w.tensor(p+"ffn_up.weight",    m.layers[i].w_up);
        w.tensor(p+"ffn_down.weight",  m.layers[i].w_down);
    }
    const std::vector<uint8_t> blob = w.build();

    GGUFReader r; REQUIRE(r.parse(blob.data(), blob.size()));
    const ModelConfig lc = load_config(r);
    CHECK(lc.hidden_size == c.hidden_size && lc.n_layers == c.n_layers);
    CHECK(lc.n_heads == c.n_heads && lc.n_kv_heads == c.n_kv_heads);
    CHECK(lc.intermediate_size == c.intermediate_size && lc.vocab_size == c.vocab_size);

    const ModelWeights lm = load_model(r, lc);
    const std::vector<int> ids = {3, 7, 1, 9, 2};
    Tensor a = forward(m,  c,  ids);
    Tensor b = forward(lm, lc, ids);
    for (int i = 0; i < a.numel(); ++i)
        CHECK_CLOSE(a.data_ptr<float>()[i], b.data_ptr<float>()[i], 1e-4);   // faithful round-trip
}
