// include/llm/common.hpp
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <string_view>

namespace llm {

const char* version();

// ---- Panic & checks ----
[[noreturn]] inline void panic(
    std::string_view msg,
    const std::source_location loc = std::source_location::current()) {
    std::fprintf(stderr, "\n[llm:PANIC] %s:%u in %s\n           %.*s\n",
                 loc.file_name(), loc.line(), loc.function_name(),
                 static_cast<int>(msg.size()), msg.data());
    std::abort();
}

// ---- Logging ----
enum class LogLevel : int { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4 };

inline LogLevel& log_threshold() { static LogLevel level = LogLevel::Info; return level; }
inline void set_log_level(LogLevel l) { log_threshold() = l; }

inline const char* level_name(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}

inline void log_message(LogLevel l, std::string_view msg) {
    if (static_cast<int>(l) < static_cast<int>(log_threshold())) return;
    std::fprintf(stderr, "[llm:%s] %.*s\n", level_name(l),
                 static_cast<int>(msg.size()), msg.data());
}

}  // namespace llm

// ---- Macros (outside any namespace) ----

// Always-on: enforces conditions that depend on external input.
#define LLM_CHECK(cond, msg)                                                   \
    do { if (!(cond)) ::llm::panic(msg); } while (0)

// Debug-only invariant checks; zero cost in Release.
#if defined(LLM_ENABLE_ASSERTS)
  #define LLM_ASSERT(cond, msg)                                                \
      do { if (!(cond)) ::llm::panic(msg); } while (0)
#else
  #define LLM_ASSERT(cond, msg)                                                \
      do { (void)sizeof(cond); /* type-checks without evaluating */ } while (0)
#endif

#define LLM_UNREACHABLE() ::llm::panic("unreachable code reached")

#define LLM_LOG_TRACE(msg) ::llm::log_message(::llm::LogLevel::Trace, msg)
#define LLM_LOG_DEBUG(msg) ::llm::log_message(::llm::LogLevel::Debug, msg)
#define LLM_LOG_INFO(msg)  ::llm::log_message(::llm::LogLevel::Info,  msg)
#define LLM_LOG_WARN(msg)  ::llm::log_message(::llm::LogLevel::Warn,  msg)
#define LLM_LOG_ERROR(msg) ::llm::log_message(::llm::LogLevel::Error, msg)
