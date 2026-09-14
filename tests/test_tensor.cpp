// tests/test_tensor.cpp
#include "test_framework.hpp"
#include "llm/tensor.hpp"
using namespace llm;

TEST(tensor_shape_strides_numel) {
    Tensor t = Tensor::empty({2, 3, 4}, DType::F32);
    CHECK(t.ndim() == 3);
    CHECK(t.numel() == 24);
    CHECK(t.nbytes() == 24u * 4u);
    CHECK(t.strides()[0] == 12);   // row-major strides for [2,3,4] are [12,4,1]
    CHECK(t.strides()[1] == 4);
    CHECK(t.strides()[2] == 1);
    CHECK(t.is_contiguous());
    CHECK(t.dtype() == DType::F32);
}

TEST(tensor_set_get_roundtrip) {
    Tensor t = Tensor::empty({2, 2}, DType::F32);
    t.at<float>({0, 0}) = 1.0f;  t.at<float>({0, 1}) = 2.0f;
    t.at<float>({1, 0}) = 3.0f;  t.at<float>({1, 1}) = 4.0f;
    CHECK_CLOSE(t.at<float>({1, 0}), 3.0f, 1e-6);
    const float* p = t.data_ptr<float>();   // contiguous order is 1,2,3,4
    CHECK_CLOSE(p[0], 1.0f, 1e-6);
    CHECK_CLOSE(p[3], 4.0f, 1e-6);
}

TEST(dtype_sizes) {
    CHECK(dtype_size(DType::F32) == 4);
    CHECK(dtype_size(DType::F16) == 2);
    CHECK(dtype_size(DType::I8)  == 1);
}

// add to tests/test_tensor.cpp (or a new tests/test_memory.cpp)
#include "llm/memory.hpp"

TEST(aligned_alloc_returns_aligned_pointer) {
    void* p = aligned_malloc(1000, 64);
    REQUIRE(p != nullptr);
    CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);
    aligned_free(p);
}

TEST(arena_aligned_and_nonoverlapping) {
    Arena a(1u << 20, 64);                 // 1 MiB
    void* x = a.allocate(100);
    void* y = a.allocate(100);
    CHECK(reinterpret_cast<uintptr_t>(x) % 64 == 0);
    CHECK(reinterpret_cast<uintptr_t>(y) % 64 == 0);
    CHECK(x != y);
    // 100 bytes rounds up to 128 for the next aligned slot, so y is >= 64 past x
    CHECK(reinterpret_cast<char*>(y) - reinterpret_cast<char*>(x) >= 64);
}

TEST(arena_reset_reuses_memory_and_tracks_peak) {
    Arena a(1u << 20, 64);
    void* first = a.allocate(256);
    CHECK(a.used() == 256);
    a.reset();
    CHECK(a.used() == 0);
    void* again = a.allocate(256);
    CHECK(first == again);                 // same address after reset
    CHECK(a.high_water() >= 256);          // peak remembered across the reset
}

// add to tests/test_tensor.cpp

TEST(reshape_zero_copy_preserves_data) {
    Tensor t = Tensor::empty({2, 6}, DType::F32);
    for (int i = 0; i < 12; ++i) t.data_ptr<float>()[i] = float(i);
    Tensor r = t.reshape({3, 4});
    CHECK(r.storage_id() == t.storage_id());        // SAME storage -> no copy
    CHECK(r.is_contiguous());
    CHECK_CLOSE(r.at<float>({2, 3}), 11.0f, 1e-6);  // last element preserved
    r.at<float>({0, 0}) = 99.0f;                    // mutate the view...
    CHECK_CLOSE(t.at<float>({0, 0}), 99.0f, 1e-6);  // ...base sees it (same bytes)
}

TEST(transpose_swaps_strides_zero_copy) {
    Tensor t = Tensor::empty({2, 3}, DType::F32);
    for (int i = 0; i < 6; ++i) t.data_ptr<float>()[i] = float(i);
    Tensor tt = t.transpose(0, 1);                  // shape {3,2}
    CHECK(tt.storage_id() == t.storage_id());
    CHECK(tt.shape()[0] == 3 && tt.shape()[1] == 2);
    CHECK(!tt.is_contiguous());
    for (int i = 0; i < 2; ++i)                      // A[i][j] == A^T[j][i]
        for (int j = 0; j < 3; ++j)
            CHECK_CLOSE(t.at<float>({i, j}), tt.at<float>({j, i}), 1e-6);
}

TEST(slice_views_subregion) {
    Tensor t = Tensor::empty({4, 4}, DType::F32);
    for (int i = 0; i < 16; ++i) t.data_ptr<float>()[i] = float(i);
    Tensor rows = t.slice(0, 1, 3);                 // rows 1..2 -> shape {2,4}
    CHECK(rows.storage_id() == t.storage_id());
    CHECK(rows.shape()[0] == 2 && rows.shape()[1] == 4);
    CHECK_CLOSE(rows.at<float>({0, 0}),  4.0f, 1e-6);  // == t[1][0]
    CHECK_CLOSE(rows.at<float>({1, 3}), 11.0f, 1e-6);  // == t[2][3]
}

TEST(contiguous_materializes_a_transpose) {
    Tensor t = Tensor::empty({2, 3}, DType::F32);
    for (int i = 0; i < 6; ++i) t.data_ptr<float>()[i] = float(i);
    Tensor c = t.transpose(0, 1).contiguous();      // {3,2}, contiguous, real copy
    CHECK(c.is_contiguous());
    CHECK(c.storage_id() != t.storage_id());        // different storage -> a copy
    const float* p = c.data_ptr<float>();           // memory order now 0,3,1,4,2,5
    CHECK_CLOSE(p[0], 0.0f, 1e-6);
    CHECK_CLOSE(p[1], 3.0f, 1e-6);
    CHECK_CLOSE(p[2], 1.0f, 1e-6);
}

TEST(reshape_requires_contiguous) {
    Tensor t  = Tensor::empty({2, 3}, DType::F32);
    Tensor tt = t.transpose(0, 1);                  // non-contiguous
    Tensor ok = tt.contiguous().reshape({6});       // the supported path
    CHECK(ok.numel() == 6);
    CHECK(ok.is_contiguous());
}

// add to tests/test_tensor.cpp

TEST(boss_zero_copy_discipline) {
    const uint64_t before = Storage::alloc_count();

    Tensor base = Tensor::empty({1024, 1024}, DType::F32);   // (1 allocation)
    CHECK(Storage::alloc_count() == before + 1);

    // A pipeline of pure views: none may allocate data or copy bytes.
    Tensor v = base.slice(0, 0, 512)     // {512,1024}
                   .transpose(0, 1)      // {1024,512}
                   .slice(0, 0, 256);    // {256,512}
    CHECK(Storage::alloc_count() == before + 1);     // still just the base
    CHECK(v.storage_id() == base.storage_id());      // same bytes

    Tensor r = base.reshape({1024 * 1024});          // contiguous reshape: zero-copy
    CHECK(Storage::alloc_count() == before + 1);
    CHECK(r.storage_id() == base.storage_id());

    Tensor t = base;                                 // 1000 views allocate no data
    for (int i = 0; i < 1000; ++i) t = t.slice(1, 0, t.shape()[1]);
    CHECK(Storage::alloc_count() == before + 1);
    CHECK(t.storage_id() == base.storage_id());

    Tensor a = Tensor::empty({64, 48}, DType::F32);  // (+1 allocation)
    for (int i = 0; i < 64 * 48; ++i) a.data_ptr<float>()[i] = float(i);
    const uint64_t after_a = Storage::alloc_count();
    Tensor at = a.transpose(0, 1);                   // {48,64}
    CHECK(Storage::alloc_count() == after_a);        // transpose copied nothing
    CHECK(at.storage_id() == a.storage_id());

    bool all_match = true;
    for (int i = 0; i < 64 && all_match; ++i)
        for (int j = 0; j < 48; ++j)
            if (a.at<float>({i, j}) != at.at<float>({j, i})) { all_match = false; break; }
    CHECK(all_match);
}
