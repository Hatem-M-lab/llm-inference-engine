// tools/cli/main.cpp
#include <cstdio>

namespace llm { const char* version(); }

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("llm-engine %s\n", llm::version());
    // Future subcommands: run, tokenize, perplexity, serve, ...
    return 0;
}
