#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <atomic>
#include <string>
#include <sstream>

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

// Helper function to convert bytes to human-readable format
inline std::string formatBytes(size_t bytes) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

#endif // UTILITIES_HPP
