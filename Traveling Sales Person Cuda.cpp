#include "Traveling Sales Person Cuda.h"
#include "runners/common.h"
#include "runners/single_thread_runner.h"
#include "runners/heavy_memory_runner.h"
#include "runners/multi_thread_runner.h"
#include "runners/cuda_runner.h"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>

void signal_handler(int) {
	_exit(1);
}

int main(int argc, char* argv[])
{
	std::signal(SIGINT, signal_handler);

	int n = 5;
	std::string mode = "multi";

	auto usage = [&]() {
		std::cout << "Usage: tsp.exe [node_count] [mode] [flags...]\n";
		std::cout << "  node_count: number of nodes (int, 0..30)\n";
		std::cout << "  mode: mem | single | multi | cuda\n";
		std::cout << "  flags: --progress --timer\n";
		std::cout << "Example: tsp.exe 5 multi --progress\n";
	};

	if (argc > 1) {
		std::string arg1 = argv[1];
		if (arg1 == "-h" || arg1 == "--help") {
			usage();
			return 0;
		}
		try {
			n = std::stoi(arg1);
		} catch (...) {
			std::cerr << "Invalid node count: " << arg1 << "\n";
			usage();
			return 1;
		}
	}

	if (argc > 2) {
		mode = argv[2];
		if (mode != "mem" && mode != "single" && mode != "multi" && mode != "cuda") {
			std::cerr << "Invalid mode: " << mode << "\n";
			usage();
			return 1;
		}
	}

	for (int i = 3; i < argc; ++i) {
		std::string flag = argv[i];
		if (flag == "--progress") {
			global_log_progress.store(true);
		} else if (flag == "--timer") {
			global_timer.store(true);
		} else {
			std::cerr << "Unknown flag: " << flag << "\n";
			usage();
			return 1;
		}
	}

	if (n == 0) {
		std::cout << "Shortest path: 0 via [ ]" << std::endl;
		return 0;
	}

	int routes[30][30] = {
	{ -1, 38, 32, 38,  9, 64, 48,  7, 65, 99, 92, 49, 55, 43, 87, 88, 43, 56, 61, 35, 73, 98, 27, 40, 10, 45,  3, 44, 75, 98 },
	{ 38, -1, 70, 66, 79, 22,  1, 88, 55, 83, 51, 21, 49, 57, 98, 68, 85, 58, 95, 49, 85, 28, 17, 85, 35, 56, 46, 55, 80,  4 },
	{ 32, 70, -1, 81, 34, 26, 69, 11, 11, 33, 22, 45, 35, 62, 22, 16, 14, 84, 55, 84, 99, 91, 82, 86, 80, 65, 21, 36, 19, 17 },
	{ 38, 66, 81, -1, 81,  8, 89, 69, 19, 83, 30, 26, 74,  4, 62, 95, 57, 43, 80, 95, 99, 91, 63, 86, 14, 73, 51, 69, 13, 80 },
	{  9, 79, 34, 81, -1, 59, 41, 10, 95, 79, 52, 11, 78, 97, 90, 25, 90,  2, 18, 13, 88, 58, 96, 66, 81, 92, 75, 90, 11,  5 },
	{ 64, 22, 26,  8, 59, -1, 64, 32, 63,  4, 95, 86, 85, 68, 33, 20, 97, 90,  4, 81, 98, 29, 31, 64, 10, 17, 76, 61, 93, 76 },
	{ 48,  1, 69, 89, 41, 64, -1, 74, 26, 35, 50, 73, 99, 49, 63, 33, 62, 58, 78, 20,  5, 62, 95, 48, 16, 46, 63, 76, 37, 67 },
	{  7, 88, 11, 69, 10, 32, 74, -1, 91, 24, 14, 70, 67, 66, 96, 38, 14, 53, 84, 38, 98, 43, 62, 79, 92, 83, 80, 70, 58, 30 },
	{ 65, 55, 11, 19, 95, 63, 26, 91, -1, 83, 69, 49, 43, 83, 90, 80,  4, 24, 56, 84, 18, 31, 22, 73, 34, 15, 98, 56, 37, 60 },
	{ 99, 83, 33, 83, 79,  4, 35, 24, 83, -1, 15, 86, 83, 51, 20, 89, 90, 82, 71, 64, 50, 95, 16, 32, 59, 93, 56, 94, 24, 52 },
	{ 92, 51, 22, 30, 52, 95, 50, 14, 69, 15, -1, 40, 19, 32, 31,  4, 95, 26, 98, 51, 37, 21, 78, 72, 85, 53, 52, 28, 55, 40 },
	{ 49, 21, 45, 26, 11, 86, 73, 70, 49, 86, 40, -1, 28, 18, 49, 45, 44, 18, 17, 47, 83, 72, 18, 16, 75, 43, 38, 80, 63, 80 },
	{ 55, 49, 35, 74, 78, 85, 99, 67, 43, 83, 19, 28, -1, 13, 15, 82, 58, 68, 35, 99, 67, 41, 27, 62, 72, 54, 36, 70, 61, 17 },
	{ 43, 57, 62,  4, 97, 68, 49, 66, 83, 51, 32, 18, 13, -1, 35,  9, 96,  4, 28, 34, 58, 76, 90, 50, 50, 99, 37, 11, 21, 45 },
	{ 87, 98, 22, 62, 90, 33, 63, 96, 90, 20, 31, 49, 15, 35, -1, 11, 90, 73, 15, 56, 98, 21, 25, 81, 18,  9, 88, 77, 72, 94 },
	{ 88, 68, 16, 95, 25, 20, 33, 38, 80, 89,  4, 45, 82,  9, 11, -1, 97,  6, 84,  1, 89, 53, 50, 20,  2, 38, 56, 23, 24, 80 },
	{ 43, 85, 14, 57, 90, 97, 62, 14,  4, 90, 95, 44, 58, 96, 90, 97, -1, 89, 62, 77, 85, 57, 92, 73, 96, 32, 15, 33, 17, 22 },
	{ 56, 58, 84, 43,  2, 90, 58, 53, 24, 82, 26, 18, 68,  4, 73,  6, 89, -1, 66, 69, 29, 20, 43, 38, 46, 91, 19, 54, 35, 56 },
	{ 61, 95, 55, 80, 18,  4, 78, 84, 56, 71, 98, 17, 35, 28, 15, 84, 62, 66, -1, 28, 79,  7, 82, 89, 87, 47, 98, 23, 91, 46 },
	{ 35, 49, 84, 95, 13, 81, 20, 38, 84, 64, 51, 47, 99, 34, 56,  1, 77, 69, 28, -1, 35, 45, 42, 50, 66, 63, 44, 55, 61, 61 },
	{ 73, 85, 99, 99, 88, 98,  5, 98, 18, 50, 37, 83, 67, 58, 98, 89, 85, 29, 79, 35, -1, 47, 53, 68, 49, 87, 60, 25, 84, 56 },
	{ 98, 28, 91, 91, 58, 29, 62, 43, 31, 95, 21, 72, 41, 76, 21, 53, 57, 20,  7, 45, 47, -1, 62, 57, 41, 17, 11, 63, 15, 63 },
	{ 27, 17, 82, 63, 96, 31, 95, 62, 22, 16, 78, 18, 27, 90, 25, 50, 92, 43, 82, 42, 53, 62, -1, 77,  8, 44, 31, 49, 40,  6 },
	{ 40, 85, 86, 86, 66, 64, 48, 79, 73, 32, 72, 16, 62, 50, 81, 20, 73, 38, 89, 50, 68, 57, 77, -1, 40, 44, 22, 49, 41, 94 },
	{ 10, 35, 80, 14, 81, 10, 16, 92, 34, 59, 85, 75, 72, 50, 18,  2, 96, 46, 87, 66, 49, 41,  8, 40, -1, 80, 93, 58, 40, 82 },
	{ 45, 56, 65, 73, 92, 17, 46, 83, 15, 93, 53, 43, 54, 99,  9, 38, 32, 91, 47, 63, 87, 17, 44, 44, 80, -1, 21, 95, 78, 27 },
	{  3, 46, 21, 51, 75, 76, 63, 80, 98, 56, 52, 38, 36, 37, 88, 56, 15, 19, 98, 44, 60, 11, 31, 22, 93, 21, -1, 89, 15, 62 },
	{ 44, 55, 36, 69, 90, 61, 76, 70, 56, 94, 28, 80, 70, 11, 77, 23, 33, 54, 23, 55, 25, 63, 49, 49, 58, 95, 89, -1,  9, 43 },
	{ 75, 80, 19, 13, 11, 93, 37, 58, 37, 24, 55, 63, 61, 21, 72, 24, 17, 35, 91, 61, 84, 15, 40, 41, 40, 78, 15,  9, -1, 85 },
	{ 98,  4, 17, 80,  5, 76, 67, 30, 60, 52, 40, 80, 17, 45, 94, 80, 22, 56, 46, 61, 56, 63,  6, 94, 82, 27, 62, 43, 85, -1 }
	};

	auto start_time = std::chrono::high_resolution_clock::now();

	if(mode == "mem") {
		heavy_memory_version(n, routes);
	} else if (mode == "single") {
		single_thread_stream(n, routes);
	} else if (mode == "multi") {
		if (n <= 10) {
			single_thread_stream(n, routes);
		} else {
			multi_thread_stream(n, routes);
		}
	} else if (mode == "cuda") {
		cuda_stream(n, routes);
	} else {
		std::cerr << "Unknown mode: " << mode << "\n";
		return 1;
	}

	if (global_timer.load()) {
		auto end_time = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration<double>(end_time - start_time).count();
		int days = static_cast<int>(elapsed / 86400);
		int hours = static_cast<int>((elapsed - days * 86400) / 3600);
		int minutes = static_cast<int>((elapsed - days * 86400 - hours * 3600) / 60);
		double seconds = elapsed - days * 86400 - hours * 3600 - minutes * 60;
		std::cout << "Elapsed: ";
		if (days > 0) std::cout << days << " days ";
		if (hours > 0) std::cout << hours << " hours ";
		if (minutes > 0) std::cout << minutes << " minutes ";
		std::cout << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;
	}

	return 0;
}
