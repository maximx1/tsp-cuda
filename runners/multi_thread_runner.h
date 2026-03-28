#pragma once

#include "common.h"
#include <algorithm>
#include <future>
#include <vector>
#include <thread>

template<int N, int M>
void multi_thread_stream(int n, int (&routes)[N][M]) {
    if (n <= 1) {
        std::cout << "Shortest path: 0 via [ 0 ]" << std::endl;
        return;
    }

    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 8;

    const int tail_size = n - 1;
    const long long total_perms = factorial(tail_size);

    const int num_threads = static_cast<int>(std::min(static_cast<long long>(hw_threads), total_perms));
    const long long chunk_size = total_perms / num_threads;
    const long long remainder = total_perms % num_threads;

    if (global_log_progress.load()) {
        std::cout << "Launching " << num_threads << " threads for " << total_perms << " permutations" << std::endl;
    }

    ProgressReporter reporter(total_perms);
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        long long start_idx = t * chunk_size + std::min(static_cast<long long>(t), remainder);
        long long count = chunk_size + (t < remainder ? 1 : 0);

        futures.push_back(std::async(std::launch::async, [n, tail_size, start_idx, count, &routes]() {
            std::vector<int> tail = ith_permutation(tail_size, start_idx);

            for (int& v : tail) v += 1;

            std::vector<int> path(n);
            std::vector<int> weighted_paths(n);

            for (long long p = 0; p < count; ++p) {
                path[0] = 0;
                std::copy(tail.begin(), tail.end(), path.begin() + 1);

                for (int i = 0; i < n; i++) {
                    int j = (i + 1) % n;
                    weighted_paths[i] = routes[path[i]][path[j]];
                }

                update_global_best(path, weighted_paths);

                global_permutation_count.fetch_add(1, std::memory_order_relaxed);

                if (p + 1 < count) {
                    std::next_permutation(tail.begin(), tail.end());
                }
            }
        }));
    }

    reporter.start();
    for (auto& f : futures) f.get();
    reporter.finish();

    std::cout << "Shortest path: " << global_lowest_cost.load() << " via " << path_to_string(global_shortest_path) << std::endl;
}