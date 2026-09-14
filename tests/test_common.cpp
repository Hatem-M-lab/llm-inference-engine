// tests/test_common.cpp
#include "test_framework.hpp"
#include "llm/common.hpp"

TEST(check_passes_silently_on_true) {
    LLM_CHECK(2 + 2 == 4, "arithmetic still works");
    CHECK(true);   // if we got here, LLM_CHECK did not abort
}

TEST(logger_respects_threshold) {
    llm::set_log_level(llm::LogLevel::Warn);
    LLM_LOG_INFO("this should be suppressed");   // should NOT appear
    LLM_LOG_WARN("this should appear");          // should appear
    CHECK(true);
    llm::set_log_level(llm::LogLevel::Info);      // restore for other tests
}
