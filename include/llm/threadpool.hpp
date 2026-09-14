// include/llm/threadpool.hpp
#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <vector>
#include "llm/tensor.hpp"

namespace llm {

class ThreadPool {
public:
    explicit ThreadPool(int nthreads) : stop_(false) {
        for (int i = 0; i < nthreads; ++i)
            workers_.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front()); tasks_.pop();
                    }
                    task();
                }
            });
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int size() const { return (int)workers_.size(); }

    // run fn(begin,end) over [0,n) split across the pool and the caller; block until done
    void parallel_for(int n, const std::function<void(int, int)>& fn) {
        const int P = size() + 1;                       // workers + calling thread
        if (P <= 1 || n <= 1) { fn(0, n); return; }
        const int chunk = (n + P - 1) / P;
        std::atomic<int> done(0);
        int launched = 0;
        for (int b = chunk; b < n; b += chunk) {        // workers take chunks 1..P-1
            const int e = std::min(n, b + chunk);
            { std::unique_lock<std::mutex> lk(mu_);
              tasks_.push([&fn, b, e, &done] { fn(b, e); done.fetch_add(1); }); }
            ++launched;
        }
        cv_.notify_all();
        fn(0, std::min(n, chunk));                       // caller runs chunk 0
        while (done.load() < launched) std::this_thread::yield();
    }
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_;
};

inline ThreadPool*& global_pool() { static ThreadPool* p = nullptr; return p; }
inline void set_num_threads(int n) {
    delete global_pool();
    global_pool() = (n > 1) ? new ThreadPool(n - 1) : nullptr;   // n-1 workers + caller = n
}

// threaded linear: y[M,out] = x[M,in] * w[out,in]^T, parallel over output features
inline void linear_mt(Tensor& y, const Tensor& x, const Tensor& w) {
    const int M = (int)x.shape()[0], in = (int)x.shape()[1], out = (int)w.shape()[0];
    const float* X = x.data_ptr<float>();
    const float* W = w.data_ptr<float>();
    float* Y = y.data_ptr<float>();
    auto block = [&](int o0, int o1) {
        for (int o = o0; o < o1; ++o) {
            const float* wr = W + (int64_t)o * in;
            for (int m = 0; m < M; ++m) {
                const float* xr = X + (int64_t)m * in;
                float s = 0.0f; for (int k = 0; k < in; ++k) s += xr[k] * wr[k];
                Y[(int64_t)m * out + o] = s;
            }
        }
    };
    if (global_pool()) global_pool()->parallel_for(out, block);
    else block(0, out);
}

}  // namespace llm
