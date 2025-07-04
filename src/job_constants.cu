// job_constants.cu - Device constant definitions
#include "job_constants.cuh"

// Define the device constants
__device__ __constant__ uint8_t g_job_msg[32];
__device__ __constant__ uint32_t g_target[5];