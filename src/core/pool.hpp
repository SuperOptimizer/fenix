// core/pool.hpp — WorkerPool: a tiny persistent task pool for ASYNC work (viewer renders,
// blocking IO like remote fetch/transcode). Distinct from parallel_for (fork-join data
// parallelism over a range): tasks are queued FIFO onto N long-lived threads and the pool
// can stop — dropping queued tasks, joining running ones — independently of completion,
// which is exactly the shutdown story a GUI needs (stop the pools BEFORE the widgets the
// tasks reference are destroyed). No load-bearing global instances: callers own their pools.
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fenix {

class WorkerPool {
  public:
    explicit WorkerPool(int threads) {
        threads = std::clamp(threads, 1, 64);
        workers_.reserve(static_cast<usize>(threads));
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { run_(); });
    }
    ~WorkerPool() { stop(); }
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // Queue a task. Silently dropped if the pool is stopping (shutdown races are expected).
    void submit(std::function<void()> fn) {
        {
            std::lock_guard lk(m_);
            if (stopping_) return;
            q_.push_back(std::move(fn));
        }
        cv_.notify_one();
    }

    // Drop every queued task, let running ones finish, join. Idempotent. Call BEFORE
    // destroying anything submitted tasks reference.
    void stop() {
        {
            std::lock_guard lk(m_);
            stopping_ = true;
            q_.clear();
        }
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
        workers_.clear();
    }

    [[nodiscard]] usize queued() const {
        std::lock_guard lk(m_);
        return q_.size();
    }

  private:
    void run_() {
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [&] { return stopping_ || !q_.empty(); });
                if (stopping_) return;
                fn = std::move(q_.front());
                q_.pop_front();
            }
            fn();
        }
    }

    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> q_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace fenix
