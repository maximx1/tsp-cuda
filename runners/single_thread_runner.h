#pragma once

#include "common.h"
#include <algorithm>
#include <vector>

template<int N, int M>
void single_thread_stream(int n, int (&routes)[N][M]) {
    int current_lowest_cost = INT_MAX;
    std::vector<int> current_shortest_path;
    std::vector<int> tail;
    tail.reserve(std::max(0, n - 1));
    for (int i = 1; i < n; ++i) {
        tail.push_back(i);
    }

    std::vector<int> path(n);
    std::vector<int> weighted_paths(n);

    long long total_perms = factorial(n - 1);
    ProgressReporter reporter(total_perms);
    reporter.start();

    do {
        path[0] = 0;
        std::copy(tail.begin(), tail.end(), path.begin() + 1);

        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            weighted_paths[i] = routes[path[i]][path[j]];
        }

        int cost = calculate_total_cost(current_lowest_cost, weighted_paths);
        if (cost < current_lowest_cost) {
            current_lowest_cost = cost;
            current_shortest_path = path;
        }
        global_permutation_count.fetch_add(1, std::memory_order_relaxed);
    } while (std::next_permutation(tail.begin(), tail.end()));

    reporter.finish();

    std::cout << "Shortest path: " << current_lowest_cost << " via " << path_to_string(current_shortest_path) << std::endl;
}
