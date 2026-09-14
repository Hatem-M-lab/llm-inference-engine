cmake_minimum_required(VERSION 3.20)
project(llm_engine LANGUAGES CXX VERSION 0.1.0)

# ---- Global language settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd / tooling

# Default to Release for single-config generators if unset.
get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---- Centralized warnings ----
add_library(llm_warnings INTERFACE)
option(LLM_WERROR "Treat warnings as errors" OFF)
if(MSVC)
  target_compile_options(llm_warnings INTERFACE /W4 /permissive-)
  if(LLM_WERROR)
    target_compile_options(llm_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(llm_warnings INTERFACE -Wall -Wextra -Wpedantic)
  if(LLM_WERROR)
    target_compile_options(llm_warnings INTERFACE -Werror)
  endif()
endif()

# ---- Core library ----
add_library(llm_core STATIC
  src/version.cpp
  src/tensor.cpp
)
target_include_directories(llm_core PUBLIC include)
target_compile_features(llm_core PUBLIC cxx_std_20)
target_link_libraries(llm_core PRIVATE llm_warnings)
target_compile_definitions(llm_core PUBLIC
  $<$<CONFIG:Debug>:LLM_ENABLE_ASSERTS>)

# ---- Command-line tool ----
add_executable(llm tools/cli/main.cpp)
target_link_libraries(llm PRIVATE llm_core llm_warnings)

# ---- Tests ----
enable_testing()
add_executable(llm_tests
  tests/test_main.cpp
  tests/test_smoke.cpp
  tests/test_common.cpp
  tests/test_tensor.cpp
)
target_include_directories(llm_tests PRIVATE tests)
target_link_libraries(llm_tests PRIVATE llm_core llm_warnings)
add_test(NAME unit COMMAND llm_tests)
