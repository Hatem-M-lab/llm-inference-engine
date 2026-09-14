// tests/test_tokenizer.cpp
#include "test_framework.hpp"
#include "llm/tokenizer.hpp"
using namespace llm;

TEST(byte_vocab_has_256_tokens) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    CHECK(t.vocab_size() == 256);
    CHECK(t.id_of(std::string(1, 'A')) == 65);     // 'A' == 0x41
    CHECK(t.token_of(65) == "A");
}

TEST(every_byte_value_is_representable) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    for (int b = 0; b < 256; ++b) {
        const std::string one(1, static_cast<char>(static_cast<unsigned char>(b)));
        CHECK(t.id_of(one) == b);                  // byte b -> id b
    }
}

TEST(add_token_is_idempotent_and_extends) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    const int id1 = t.add_token("the");
    const int id2 = t.add_token("the");            // same token -> same id
    CHECK(id1 == id2);
    CHECK(id1 == 256);                             // first id past the byte base
    CHECK(t.vocab_size() == 257);
}

// add to tests/test_tokenizer.cpp

TEST(bpe_merges_in_rank_order) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    t.add_merge("l", "l", 0);                 // "ll" has priority over "he"
    t.add_merge("h", "e", 1);                 // "he"
    auto ids = t.bpe_encode_bytes("hello");
    REQUIRE(ids.size() == 3);
    CHECK(t.token_of(ids[0]) == "he");
    CHECK(t.token_of(ids[1]) == "ll");
    CHECK(t.token_of(ids[2]) == "o");
}

TEST(bpe_applies_chained_merges) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    t.add_merge("a", "a", 0);                 // "aa"
    t.add_merge("aa", "a", 1);                // "aa" + "a" -> "aaa"
    auto ids = t.bpe_encode_bytes("aaa");
    REQUIRE(ids.size() == 1);
    CHECK(t.token_of(ids[0]) == "aaa");
}

TEST(bpe_without_rules_is_raw_bytes) {
    Tokenizer t = Tokenizer::with_byte_vocab();   // no merges
    auto ids = t.bpe_encode_bytes("hi");
    REQUIRE(ids.size() == 2);
    CHECK(ids[0] == static_cast<int>('h'));
    CHECK(ids[1] == static_cast<int>('i'));
}

// add to tests/test_tokenizer.cpp

TEST(encode_runs_bpe_per_piece) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    t.add_merge("l", "l", 0);  t.add_merge("e", "ll", 1);     // "ell"
    t.add_merge("h", "ell", 2); t.add_merge("hell", "o", 3);  // -> "hello"
    auto ids = t.encode("hello world");
    // "hello" -> 1 token,  " " -> 1 token,  "world" -> 5 raw bytes
    REQUIRE(ids.size() == 7);
    CHECK(t.token_of(ids[0]) == "hello");
    CHECK(t.token_of(ids[1]) == " ");
    CHECK(t.token_of(ids[2]) == "w");
}

TEST(encode_prepends_bos_when_asked) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    const int bos = t.add_token("<bos>");
    t.set_special_tokens(bos, /*eos=*/-1);
    auto ids = t.encode("hi", /*add_bos=*/true);
    REQUIRE(ids.size() == 3);                  // <bos> h i
    CHECK(ids[0] == bos);
    CHECK(t.token_of(ids[1]) == "h");
}

// add to tests/test_tokenizer.cpp

TEST(decode_concatenates_token_bytes) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    t.add_merge("l", "l", 0);
    const auto ids = t.bpe_encode_bytes("hello");
    CHECK(t.decode(ids) == "hello");
}

TEST(decode_skips_specials_by_default) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    const int bos = t.add_token("<bos>"), eos = t.add_token("<eos>");
    t.set_special_tokens(bos, eos);
    const std::vector<int> ids = { bos, int('h'), int('i'), eos };
    CHECK(t.decode(ids) == "hi");                              // specials dropped
    CHECK(t.decode(ids, /*skip_special=*/false) == "<bos>hi<eos>");
}

// add to tests/test_tokenizer.cpp
#include <cstdint>

TEST(boss_roundtrip_byte_for_byte) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    t.add_merge("l", "l", 0);                 // a few merges; round-trip must hold anyway
    t.add_merge("e", "ll", 1);
    t.add_merge("s", "s", 2);

    const std::vector<std::string> corpus = {
        "Hello, world!",
        "",                                          // empty
        " ",                                         // lone space
        "   leading and  internal   spaces   ",
        "tabs\tand\nnewlines\r\n",
        "caf\xC3\xA9 d\xC3\xA9j\xC3\xA0 na\xC3\xAFve",   // 2-byte UTF-8 (café déjà naïve)
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",           // 3-byte UTF-8 (日本語)
        "emoji \xF0\x9F\x98\x80 \xF0\x9F\x9A\x80",        // 4-byte UTF-8 (emoji)
        "mixed ABC 123 !@#$ ",
        std::string("\x00\x01\x02\xFE\xFF", 5),          // arbitrary non-UTF-8 bytes
    };
    for (const std::string& s : corpus) {
        const std::vector<int> ids  = t.encode(s);
        const std::string      back = t.decode(ids);
        CHECK(back == s);                            // byte-for-byte
    }
}

TEST(boss_roundtrip_fuzz) {
    Tokenizer t = Tokenizer::with_byte_vocab();
    t.add_merge("a", "a", 0);  t.add_merge("aa", "a", 1);
    uint32_t s = 0x12345;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return s; };
    for (int trial = 0; trial < 500; ++trial) {
        const int len = static_cast<int>(rnd() % 64);
        std::string str;
        for (int i = 0; i < len; ++i)
            str.push_back(static_cast<char>(rnd() & 0xFFu));     // any byte 0..255
        CHECK(t.decode(t.encode(str)) == str);                  // exact round-trip
    }
}
