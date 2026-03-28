#include "common.h"
#include <iomanip>
#include <cmath>

std::atomic<int> global_lowest_cost{INT_MAX};
std::mutex global_best_path_mutex;
std::vector<int> global_shortest_path;
std::atomic<long long> global_permutation_count{0};
std::atomic<bool> global_log_progress{false};
std::atomic<bool> global_timer{false};

int calculate_total_cost(int current_record, const std::vector<int>& weighted_paths) {
    int sum = 0;

    for (int path_item : weighted_paths) {
        sum += path_item;
        if (sum > current_record) {
            return current_record;
        }
    }

    return sum;
}

void update_global_best(const std::vector<int>& path, const std::vector<int>& weighted_paths) {
    int current_best = global_lowest_cost.load(std::memory_order_relaxed);
    int cost = calculate_total_cost(current_best, weighted_paths);

    int expected = current_best;
    while (cost < expected && !global_lowest_cost.compare_exchange_weak(expected, cost, std::memory_order_relaxed)) {
    }

    if (cost < expected) {
        std::lock_guard<std::mutex> guard(global_best_path_mutex);
        global_shortest_path = path;
    }
}

std::string path_to_string(const std::vector<int>& path) {
    std::ostringstream oss;
    oss << "[ ";
    for (size_t i = 0; i < path.size(); ++i) {
        oss << path[i];
        if (i + 1 < path.size()) oss << ", ";
    }
    oss << " ]";
    return oss.str();
}

long long factorial(int n) {
    long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

std::vector<int> ith_permutation(int n, long long idx) {
    std::vector<int> perm(n);
    std::vector<long long> fact(n);

    // Precompute factorials
    fact[0] = 1;
    for (int k = 1; k < n; ++k) {
        fact[k] = fact[k - 1] * k;
    }

    // Compute factorial code (Lehmer code)
    std::vector<int> code(n);
    for (int k = 0; k < n; ++k) {
        code[k] = static_cast<int>(idx / fact[n - 1 - k]);
        idx = idx % fact[n - 1 - k];
    }

    // Convert Lehmer code to permutation
    // Adjust values: for each position from the end, bump up if a preceding value <= it
    for (int k = n - 1; k > 0; --k) {
        for (int j = k - 1; j >= 0; --j) {
            if (code[j] <= code[k]) {
                code[k]++;
            }
        }
    }

    for (int k = 0; k < n; ++k) {
        perm[k] = code[k];
    }

    return perm;
}

ProgressReporter::ProgressReporter(long long total, std::function<long long()> getter)
    : total_(total), getter_(std::move(getter)) {}

ProgressReporter::~ProgressReporter() {
    if (running_.load()) {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
    }
}

void ProgressReporter::start() {
    if (!global_log_progress.load()) return;
    started_ = true;
    print_bar(0);
    running_.store(true);
    thread_ = std::thread(&ProgressReporter::run, this);
}

void ProgressReporter::finish() {
    if (!started_) return;
    running_.store(false);
    if (thread_.joinable()) thread_.join();
    print_bar(total_);
    std::cout << std::endl;
}

void ProgressReporter::run() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kUpdateIntervalMs));
        if (!running_.load()) break;
        long long current = getter_ ? getter_() : global_permutation_count.load(std::memory_order_relaxed);
        print_bar(current);
    }
}

void ProgressReporter::print_bar(long long current) {
    auto fmt = [](long long v) -> std::string {
        const long long Q = 1'000'000'000'000'000LL;
        const long long T = 1'000'000'000'000LL;
        const long long B = 1'000'000'000LL;
        const long long M = 1'000'000LL;
        const long long K = 1'000LL;

        auto fmt_double = [](double x) -> std::string {
            std::ostringstream ss;
            if (std::fabs(x - std::round(x)) < 0.05) {
                ss << static_cast<long long>(std::llround(x));
            } else {
                ss << std::fixed << std::setprecision(1) << x;
            }
            return ss.str();
        };

        if (v >= Q) {
            long double t = static_cast<long double>(v) / static_cast<long double>(T);
            if (t >= 1'000'000.0L) return std::to_string(static_cast<long long>(t / 1'000'000.0L)) + "e6T";
            if (t >= 1'000.0L)     return std::to_string(static_cast<long long>(t / 1'000.0L))     + "e3T";
            return fmt_double(static_cast<double>(t)) + "T";
        }
        if (v >= T) return fmt_double(static_cast<double>(v) / static_cast<double>(T)) + "T";
        if (v >= B) return fmt_double(static_cast<double>(v) / static_cast<double>(B)) + "B";
        if (v >= M) return fmt_double(static_cast<double>(v) / static_cast<double>(M)) + "M";
        if (v >= K) return fmt_double(static_cast<double>(v) / static_cast<double>(K)) + "K";
        return std::to_string(v);
    };

    if (current > total_) current = total_;
    int filled = total_ > 0 ? static_cast<int>((current * kBarWidth) / total_) : kBarWidth;
    int empty = kBarWidth - filled;

    std::cout << "\r[";
    for (int i = 0; i < filled; i++) std::cout << '#';
    for (int i = 0; i < empty; i++) std::cout << ' ';
    std::cout << "]  " << fmt(current) << "/" << fmt(total_) << "   " << std::flush;
}
