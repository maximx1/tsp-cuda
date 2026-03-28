#pragma once

#include <atomic>
#include <chrono>
#include <climits>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern std::atomic<int> global_lowest_cost;
extern std::mutex global_best_path_mutex;
extern std::vector<int> global_shortest_path;
extern std::atomic<long long> global_permutation_count;
extern std::atomic<bool> global_log_progress;
extern std::atomic<bool> global_timer;

int calculate_total_cost(int current_record, const std::vector<int>& weighted_paths);
void update_global_best(const std::vector<int>& path, const std::vector<int>& weighted_paths);
std::string path_to_string(const std::vector<int>& path);

// Compute n! (use for permutation index math). Returns long long to handle up to ~20!.
long long factorial(int n);

// Compute the i-th lexicographic permutation of elements {0, 1, ..., n-1}.
// Useful for partitioning permutation space across threads/GPU blocks.
std::vector<int> ith_permutation(int n, long long i);

class ProgressReporter {
public:
    explicit ProgressReporter(long long total, std::function<long long()> getter = nullptr);
    ~ProgressReporter();
    void start();
    void finish();

private:
    static constexpr int kBarWidth = 30;
    static constexpr int kUpdateIntervalMs = 500;

    void run();
    void print_bar(long long current);

    long long total_;
    std::function<long long()> getter_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    bool started_{false};
};


