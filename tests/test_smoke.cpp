// tests/test_smoke.cpp
#include "test_framework.hpp"

TEST(smoke_basic_checks) {
    CHECK(1 + 1 == 2);
    REQUIRE(true);                       // does not abort
    CHECK_CLOSE(0.1 + 0.2, 0.3, 1e-9);   // != in binary fp, but close
}

TEST(smoke_close_handles_zero) {
    CHECK_CLOSE(0.0, 1e-12, 1e-6);       // absolute term saves us near zero
}
