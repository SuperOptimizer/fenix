// tests/test_pool.cpp — the WorkerPool async task pool (core/pool.hpp): tasks run,
// stop() drops queued work but finishes the running task, submit-after-stop is a no-op.
#define FENIX_TEST_MAIN
#include "core/pool.hpp"
#include "core/test.hpp"

#include <atomic>
#include <chrono>
#include <thread>

using namespace fenix;

TEST(pool_runs_all_tasks) {
    std::atomic<int> ran{0};
    {
        WorkerPool p(4);
        for (int i = 0; i < 100; ++i) p.submit([&] { ran.fetch_add(1); });
        p.stop();  // stop after the queue drains naturally or mid-way — then re-check below
    }
    const int after_stop = ran.load();
    CHECK(after_stop <= 100);
    // Run again without stop racing the queue: destructor joins, all tasks queued before
    // any chance to observe must complete when we wait for them explicitly.
    std::atomic<int> ran2{0};
    {
        WorkerPool p(4);
        for (int i = 0; i < 100; ++i) p.submit([&] { ran2.fetch_add(1); });
        for (int spin = 0; spin < 2000 && ran2.load() < 100; ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(ran2.load() == 100);
    }
}

TEST(pool_stop_drops_queued_and_finishes_running) {
    std::atomic<int> ran{0};
    std::atomic<bool> started{false}, release{false};
    WorkerPool p(1);
    p.submit([&] {  // occupies the single worker until released
        started.store(true);
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ran.fetch_add(1);
    });
    while (!started.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (int i = 0; i < 50; ++i) p.submit([&] { ran.fetch_add(1); });
    // stop() clears the queue while the worker is still pinned, then joins once released.
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release.store(true);
    });
    p.stop();
    releaser.join();
    CHECK(ran.load() == 1);              // the running task finished; all 50 queued dropped
    p.submit([&] { ran.fetch_add(1); });  // after stop: silently dropped
    CHECK(ran.load() == 1);
}
