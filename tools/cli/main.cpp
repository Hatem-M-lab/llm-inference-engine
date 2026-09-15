// tools/cli/main.cpp
#include "llm/model.hpp"
#include "llm/gguf.hpp"
#include "llm/tokenizer.hpp"
#include "llm/engine.hpp"
#include <cstdio>
#include <cstring>
#include <string>

namespace llm { const char* version(); }

using namespace llm;

static void print_usage(const char* argv0) {
    std::printf("llm-engine %s\n\n", version());
    std::printf("usage: %s --model FILE.gguf --prompt \"TEXT\" [options]\n\n", argv0);
    std::printf("options:\n");
    std::printf("  --model FILE       GGUF model file to load (required)\n");
    std::printf("  --prompt TEXT      prompt to generate from (required)\n");
    std::printf("  --max-new N        tokens to generate (default 200)\n");
    std::printf("  --temperature F    sampling temperature (default 0.7)\n");
    std::printf("  --top-p F          nucleus sampling threshold (default 0.95)\n");
    std::printf("  --greedy           ignore temperature/top-p, always pick the top token\n");
    std::printf("  --seed N           RNG seed (default 42)\n");
}

int main(int argc, char** argv) {
    std::string model_path, prompt;
    int max_new = 200;
    float temperature = 0.7f, top_p = 0.95f;
    bool greedy = false;
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(1); }
            return argv[++i];
        };
        if (a == "--model") model_path = next("--model");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--max-new") max_new = std::stoi(next("--max-new"));
        else if (a == "--temperature") temperature = std::stof(next("--temperature"));
        else if (a == "--top-p") top_p = std::stof(next("--top-p"));
        else if (a == "--greedy") greedy = true;
        else if (a == "--seed") seed = static_cast<uint64_t>(std::stoll(next("--seed")));
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 0;
    }

    GGUFReader r;
    if (!r.load(model_path)) {
        std::fprintf(stderr, "error: could not load '%s'\n", model_path.c_str());
        return 1;
    }
    ModelConfig cfg = load_config(r);
    ModelWeights model = load_model(r, cfg);

    // Byte-level tokenizer: works with any model exported the way the book's
    // training appendix does (vocab_size 256, BOS = byte 0). A model trained
    // with a different tokenizer would need the matching Tokenizer here instead.
    Tokenizer tok = Tokenizer::with_byte_vocab();
    tok.set_special_tokens(/*bos=*/0, /*eos=*/-1);

    SamplingParams sp;
    sp.greedy = greedy;
    sp.temperature = temperature;
    sp.top_p = top_p;
    RNG rng(seed);

    std::string out = run_engine(model, cfg, tok, prompt, max_new, sp, rng);
    std::printf("%s%s\n", prompt.c_str(), out.c_str());
    return 0;
}
