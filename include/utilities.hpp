#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <atomic>

// Global shutdown flag
extern std::atomic<bool> g_shutdown;

// Platform-specific utilities
#ifdef USE_HIP
inline const char* getGPUPlatformName() { return "HIP/AMD"; }
inline size_t getMemoryAlignment() { return 256; }  // AMD GPUs prefer 256-byte alignment
#else
inline const char *getGPUPlatformName() { return "CUDA/NVIDIA"; }
inline size_t getMemoryAlignment() { return 128; } // NVIDIA GPUs typically use 128-byte alignment
#endif

#endif // UTILITIES_HPP
