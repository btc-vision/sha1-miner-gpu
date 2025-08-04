#pragma once

#include <cstdint>
#include <vector>

// Test functions
bool verify_sha1_implementation();
void run_benchmark(int gpu_id);

// Utility functions
std::vector<uint8_t> generate_secure_random_message();
