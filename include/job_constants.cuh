#pragma once
#include <stdint.h>

// Legacy constants for compatibility
// These are now part of the MiningJob structure in device constant memory
extern __device__ __constant__ uint8_t g_job_msg[32];
extern __device__ __constant__ uint32_t g_target[5];