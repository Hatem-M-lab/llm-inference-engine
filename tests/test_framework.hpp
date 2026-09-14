// tests/test_framework.hpp
#pragma once
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace llmtest {

// ---- Registry ----
struct TestCase { std::string name; std::function<void()> fn; };

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;   // first-use init: no order fiasco
    return r;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> fn) {
        registry().push_back({std::move(name), std::move(fn)});
    }
};

// ---- Failure bookkeeping ----
struct Stats { int checks = 0; int check_failures = 0; };
inline Stats& stats() { static Stats s; return s; }

struct RequireFailed {};   // thrown to abort the current test only

inline void report_failure(const char* file, int line, const std::string& what) {
    ++stats().check_failures;
    std::fprintf(stderr, "    FAIL  %s:%d   %s\n", file, line, what.c_str());
}

// ---- Runner ----
inline int run_all() {
    int tests_failed = 0;
    for (const auto& tc : registry()) {
        const int before = stats().check_failures;
        std::printf("[ RUN  ] %s\n", tc.name.c_str());
        try {
            tc.fn();
        } catch (const RequireFailed&) {
            // failure already recorded; just stop this test
        } catch (const std::exception& e) {
            report_failure("<exception>", 0, std::string("uncaught: ") + e.what());
        } catch (...) {
            report_failure("<exception>", 0, "uncaught non-std exception");
        }
        if (stats().check_failures > before) {
            std::printf("[ FAIL ] %s\n", tc.name.c_str());
            ++tests_failed;
        } else {
            std::printf("[  OK  ] %s\n", tc.name.c_str());
        }
    }
    std::printf("\n%zu test(s) run, %d failed | %d checks, %d failed\n",
                registry().size(), tests_failed,
                stats().checks, stats().check_failures);
    return tests_failed == 0 ? 0 : 1;
}

}  // namespace llmtest

// ---- Macros ----
#define TEST(test_name)                                                        \
    static void test_name();                                                   \
    static ::llmtest::Registrar registrar_##test_name(#test_name, test_name);  \
    static void test_name()

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++::llmtest::stats().checks;                                           \
        if (!(cond)) ::llmtest::report_failure(__FILE__, __LINE__, #cond);     \
    } while (0)

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        ++::llmtest::stats().checks;                                           \
        if (!(cond)) {                                                         \
            ::llmtest::report_failure(__FILE__, __LINE__, #cond);              \
            throw ::llmtest::RequireFailed{};                                  \
        }                                                                      \
    } while (0)

// pass iff |a - b| <= tol + tol*|b|   (absolute + relative tolerance)
#define CHECK_CLOSE(a, b, tol)                                                  \
    do {                                                                       \
        ++::llmtest::stats().checks;                                           \
        const double _a = (a), _b = (b), _t = (tol);                          \
        if (!(std::fabs(_a - _b) <= _t + _t * std::fabs(_b)))                  \
            ::llmtest::report_failure(__FILE__, __LINE__,                      \
                                      #a " ~= " #b " (tol " #tol ")");         \
    } while (0)
