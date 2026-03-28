#pragma once

#include "common.h"

void cuda_stream_impl(int n, const int* routes_flat, int route_dim);

template<int N, int M>
void cuda_stream(int n, int (&routes)[N][M]) {
    if (n <= 1) {
        std::cout << "Shortest path: 0 via [ 0 ]" << std::endl;
        return;
    }
    cuda_stream_impl(n, &routes[0][0], M);
}
