# Traveling Salesman Problem — CUDA Accelerated

Brute-force solver for the [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) (TSP).
Given a set of cities and the distances between every pair, it finds the shortest round-trip route that visits each city exactly once and returns to the starting city.

Because the brute-force approach evaluates every possible permutation, the number of routes grows as **(n−1)!** — making this computationally expensive very quickly (e.g. 17 cities = ~20.9 trillion permutations). This project provides four execution modes, including a CUDA GPU-accelerated mode that distributes the work across thousands of GPU threads.

## Requirements

- **Windows** (tested on Windows 10/11)
- **MSVC** (Visual Studio 2022+ with C++ Desktop workload)
- **CMake** 3.10+
- **Ninja** build system (ships with Visual Studio)
- **NVIDIA CUDA Toolkit** (v12.6+ recommended — must match your GPU driver version)
- **NVIDIA GPU** with compute capability 5.0+ (tested on RTX 4070 Ti)

> **Driver compatibility:** Your installed GPU driver must support the CUDA toolkit version.
> Run `nvidia-smi` to check your driver's max supported CUDA version.
> If you get CUDA error 35 (insufficient driver), update your GPU driver or install an older CUDA toolkit.

## Execution Modes

| Mode     | Flag     | Description |
|----------|----------|-------------|
| `single` | `single` | Single-threaded CPU. Good for small n (≤10). |
| `multi`  | `multi`  | Multi-threaded CPU using all hardware threads. Auto-falls back to single-thread for n≤10. |
| `mem`    | `mem`    | Pre-generates all permutations in memory, then evaluates. High memory usage. |
| `cuda`   | `cuda`   | GPU-accelerated via CUDA. Distributes permutations across thousands of GPU threads. Best for large n. |

## Usage

```
tsp.exe [node_count] [mode] [flags...]
```

**Arguments:**
- `node_count` — number of cities (0–30, default: 5)
- `mode` — `single`, `multi`, `mem`, or `cuda` (default: `multi`)

**Flags:**
- `--progress` — show a live progress bar
- `--timer` — print elapsed time after completion
- `-h` / `--help` — show usage

**Examples:**
```
tsp.exe 12 multi --progress --timer
tsp.exe 15 cuda --progress --timer
tsp.exe 5
```

## Building in VS Code

This project uses CMake presets configured in `CMakePresets.json`.

1. Open the project folder in VS Code
2. Install the **CMake Tools** extension if you haven't already
3. Press **Ctrl+Shift+P** and run:
   - `CMake: Select Configure Preset` → choose **x64 Release**
   - `CMake: Configure` (or it auto-configures)
   - `CMake: Build` (or press **F7**)
4. The output binary is at `out/build/x64-release/tsp.exe`

## Building from the Command Line

Make sure MSVC and CUDA are on your PATH. The easiest way is to open a **Developer Command Prompt for VS** or **x64 Native Tools Command Prompt**.

```powershell
# Configure (from the project root)
cmake --preset x64-release

# Build
cmake --build out/build/x64-release --config Release

# Run
.\out\build\x64-release\tsp.exe 12 cuda --progress --timer
```

If you don't want to use presets, you can configure manually:

```powershell
mkdir build
cd build
cmake .. -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl.exe ^
  -DCMAKE_CXX_COMPILER=cl.exe ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
cmake --build .
```

## Project Structure

```
├── CMakeLists.txt                    # Build config (C++20, CUDA 17, links cudart)
├── CMakePresets.json                 # VS Code / CLI configure presets
├── Traveling Sales Person Cuda.cpp   # Entry point, CLI parsing, route matrix
├── Traveling Sales Person Cuda.h     # Main header
├── gridGen.js                        # Node.js helper to generate route matrices
└── runners/
    ├── common.h                      # Shared types, globals, ProgressReporter
    ├── common.cpp                    # Implementations (cost calc, permutation math)
    ├── single_thread_runner.h        # Single-threaded brute-force
    ├── heavy_memory_runner.h         # Memory-heavy pre-generated permutations
    ├── multi_thread_runner.h         # Multi-threaded with ith_permutation chunking
    ├── cuda_runner.h                 # CUDA template wrapper (compiled by MSVC)
    └── cuda_runner.cu                # CUDA kernel + host impl (compiled by nvcc)
```