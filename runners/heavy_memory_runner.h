#pragma once

#include "common.h"
#include <algorithm>
#include <vector>

static std::vector<std::vector<int>> permutations_starting_at_zero(int n) {
    std::vector<std::vector<int>> result;
    if (n <= 0) return result;

    std::vector<int> tail;
    tail.reserve(std::max(0, n - 1));
    for (int i = 1; i < n; ++i) {
        tail.push_back(i);
    }

    do {
        std::vector<int> perm;
        perm.reserve(n);
        perm.push_back(0);
        perm.insert(perm.end(), tail.begin(), tail.end());
        result.push_back(std::move(perm));
    } while (std::next_permutation(tail.begin(), tail.end()));

    return result;
}

template<int N, int M>
void heavy_memory_version(int n, int (&routes)[N][M]) {
    std::vector<std::vector<int>> path_permutations = permutations_starting_at_zero(n);

    std::cout << "Permutations generated starting export" << std::endl;

    int current_lowest_cost = INT_MAX;
    std::vector<int> current_shortest_path;

    long long total_perms = static_cast<long long>(path_permutations.size());
    ProgressReporter reporter(total_perms);
    reporter.start();

    for (const auto& path : path_permutations) {
        size_t path_size = path.size();
        std::vector<int> weighted_paths;
        weighted_paths.reserve(path_size);

        for (int i = 0; i < path_size; i++) {
            int j = (i + 1) % path_size;
            weighted_paths.push_back(routes[path[i]][path[j]]);
        }

        int cost = calculate_total_cost(current_lowest_cost, weighted_paths);
        if (cost < current_lowest_cost) {
            current_lowest_cost = cost;
            current_shortest_path = path;
        }
        global_permutation_count.fetch_add(1, std::memory_order_relaxed);
    }

    reporter.finish();

    std::cout << "Shortest path: " << current_lowest_cost << " via " << path_to_string(current_shortest_path) << std::endl;
}
