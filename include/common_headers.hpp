#pragma once

// Common system headers used throughout the project
// This file ensures all necessary standard headers are included

// C++ Standard Library headers
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// Ensure we're using at least C++11 for std::atomic
#if __cplusplus < 201103L
    #error "This project requires C++11 or later for std::atomic support"
#endif