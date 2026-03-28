#include "cuda_runner.h"
#include <cuda_runtime.h>
#include <climits>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>

#define TSP_MAX_CITIES 30

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        const char* msg = cudaGetErrorString(err); \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - code=" << static_cast<int>(err) << " msg='" << (msg ? msg : "<null>") << "'" << std::endl; \
        return; \
    } \
} while(0)

__device__ void ith_perm_d(int n, long long idx, int* out) {
    long long fact[TSP_MAX_CITIES];
    fact[0] = 1;
    for (int k = 1; k < n; k++) fact[k] = fact[k - 1] * k;

    int code[TSP_MAX_CITIES];
    for (int k = 0; k < n; k++) {
        code[k] = (int)(idx / fact[n - 1 - k]);
        idx = idx % fact[n - 1 - k];
    }

    for (int k = n - 1; k > 0; k--) {
        for (int j = k - 1; j >= 0; j--) {
            if (code[j] <= code[k]) code[k]++;
        }
    }

    for (int k = 0; k < n; k++) out[k] = code[k];
}

__device__ void next_perm_d(int* arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] >= arr[i + 1]) i--;
    if (i < 0) return;
    int j = n - 1;
    while (arr[j] <= arr[i]) j--;
    int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    int left = i + 1, right = n - 1;
    while (left < right) {
        tmp = arr[left]; arr[left] = arr[right]; arr[right] = tmp;
        left++; right--;
    }
}

__global__ void tsp_kernel(
    int n,
    const int* routes,
    int route_dim,
    int num_threads,
    long long total_perms,
    int* out_costs,
    int* out_paths,
    unsigned long long* progress_counter,
    int log_interval
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_threads) return;

    const int tail_size = n - 1;
    const long long chunk_size = total_perms / num_threads;
    const long long remainder = total_perms % num_threads;
    const long long start_idx = (long long)t * chunk_size + min((long long)t, remainder);
    const long long count = chunk_size + (t < remainder ? 1 : 0);

    int tail[TSP_MAX_CITIES];
    ith_perm_d(tail_size, start_idx, tail);
    for (int i = 0; i < tail_size; i++) tail[i] += 1;

    int path[TSP_MAX_CITIES];
    int local_best = INT_MAX;
    int local_best_path[TSP_MAX_CITIES];
    int local_count = 0;

    for (long long p = 0; p < count; p++) {
        path[0] = 0;
        for (int i = 0; i < tail_size; i++) path[i + 1] = tail[i];

        int cost = 0;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            cost += routes[path[i] * route_dim + path[j]];
            if (cost >= local_best) {
                cost = local_best;
                break;
            }
        }

        if (cost < local_best) {
            local_best = cost;
            for (int i = 0; i < n; i++) local_best_path[i] = path[i];
        }

        local_count++;
        if (progress_counter && local_count >= log_interval) {
            atomicAdd(progress_counter, (unsigned long long)local_count);
            local_count = 0;
        }

        if (p + 1 < count) next_perm_d(tail, tail_size);
    }

    if (progress_counter && local_count > 0) {
        atomicAdd(progress_counter, (unsigned long long)local_count);
    }

    out_costs[t] = local_best;
    for (int i = 0; i < n; i++) out_paths[t * TSP_MAX_CITIES + i] = local_best_path[i];
}

void cuda_stream_impl(int n, const int* routes_flat, int route_dim) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return;
    }

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    const int tail_size = n - 1;
    long long total_perms = 1;
    for (int i = 2; i <= tail_size; i++) total_perms *= i;

    const int sm_count = props.multiProcessorCount;
    const long long desired = (long long)sm_count * 1536;
    const int num_threads = (int)std::min(desired, total_perms);

    std::cout << "CUDA device: " << props.name << " (" << sm_count << " SMs)" << std::endl;
    std::cout << "Launching " << num_threads << " CUDA threads for " << total_perms << " permutations" << std::endl;

    int* d_routes;
    CUDA_CHECK(cudaMalloc(&d_routes, route_dim * route_dim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_routes, routes_flat, route_dim * route_dim * sizeof(int), cudaMemcpyHostToDevice));

    int* d_costs;
    CUDA_CHECK(cudaMalloc(&d_costs, num_threads * sizeof(int)));

    int* d_paths;
    CUDA_CHECK(cudaMalloc(&d_paths, num_threads * TSP_MAX_CITIES * sizeof(int)));

    bool log_progress = global_log_progress.load();
    unsigned long long* h_progress = nullptr;
    unsigned long long* d_progress = nullptr;
    const int log_interval = 10'000'000;

    if (log_progress) {
        CUDA_CHECK(cudaHostAlloc(&h_progress, sizeof(unsigned long long), cudaHostAllocMapped));
        *h_progress = 0;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_progress, h_progress, 0));
    }

    const int block_size = 256;
    const int grid_size = (num_threads + block_size - 1) / block_size;

    tsp_kernel<<<grid_size, block_size>>>(n, d_routes, route_dim, num_threads, total_perms,
        d_costs, d_paths, d_progress, log_interval);
    CUDA_CHECK(cudaGetLastError());

    ProgressReporter reporter(total_perms, [h_progress]() -> long long {
        return h_progress ? static_cast<long long>(*h_progress) : 0;
    });
    reporter.start();
    CUDA_CHECK(cudaDeviceSynchronize());
    reporter.finish();

    if (h_progress) {
        cudaFreeHost(h_progress);
    }

    std::vector<int> h_costs(num_threads);
    std::vector<int> h_paths(num_threads * TSP_MAX_CITIES);
    cudaMemcpy(h_costs.data(), d_costs, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_paths.data(), d_paths, num_threads * TSP_MAX_CITIES * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_routes);
    cudaFree(d_costs);
    cudaFree(d_paths);

    int best_cost = INT_MAX;
    int best_t = 0;
    for (int t = 0; t < num_threads; t++) {
        if (h_costs[t] < best_cost) {
            best_cost = h_costs[t];
            best_t = t;
        }
    }

    std::vector<int> best_path(n);
    for (int i = 0; i < n; i++) best_path[i] = h_paths[best_t * TSP_MAX_CITIES + i];

    global_lowest_cost.store(best_cost);
    {
        std::lock_guard<std::mutex> guard(global_best_path_mutex);
        global_shortest_path = best_path;
    }
    global_permutation_count.store(total_perms);

    std::cout << "Shortest path: " << best_cost << " via " << path_to_string(best_path) << std::endl;
}
