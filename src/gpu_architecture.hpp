#ifndef GPU_ARCHITECTURE_HPP
#define GPU_ARCHITECTURE_HPP

#include "gpu_platform.hpp"
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

#ifdef USE_HIP

// AMD GPU Architecture enumeration
enum class AMDArchitecture {
    UNKNOWN,
    GCN3,      // Fiji, Tonga (gfx8)
    GCN4,      // Polaris (gfx8)
    GCN5,      // Vega10, Vega20 (gfx9)
    RDNA1,     // Navi10, Navi14 (gfx10.1)
    RDNA2,     // Navi21, Navi22, Navi23 (gfx10.3)
    RDNA3,     // Navi31, Navi32, Navi33 (gfx11)
    RDNA4,     // Navi44, Navi48 (gfx12) - RX 9070 XT/9070
    CDNA1,     // Arcturus (gfx908)
    CDNA2,     // Aldebaran (gfx90a)
    CDNA3      // Aqua Vanjaram (gfx940)
};

// Architecture-specific parameters
struct AMDArchParams {
    int waves_per_cu;      // Calculated based on architecture
    int wave_size;         // From device properties
    int lds_size_per_cu;   // From device properties
    int max_workgroup_size; // From device properties
    int max_waves_per_eu;   // Max concurrent waves per execution unit

    // Get parameters from actual device properties
    static AMDArchParams getFromDevice(const hipDeviceProp_t& props, AMDArchitecture arch) {
        AMDArchParams params;

        // 1. Wave size - directly from device
        params.wave_size = props.warpSize;

        // 2. LDS (Local Data Share) size - from device
        params.lds_size_per_cu = props.sharedMemPerBlock;

        // 3. Max workgroup size - from device
        params.max_workgroup_size = props.maxThreadsPerBlock;

        // 4. Calculate waves per CU based on architecture and device limits
        // This is based on register and LDS limits
        int max_threads_per_cu = props.maxThreadsPerMultiProcessor;
        params.waves_per_cu = max_threads_per_cu / params.wave_size;

        // 5. Architecture-specific adjustments
        // Some architectures have execution unit limitations
        switch (arch) {
            case AMDArchitecture::RDNA4:
            case AMDArchitecture::RDNA3:
            case AMDArchitecture::RDNA2:
                // RDNA has 2 SIMDs per CU, each can handle multiple waves
                // But practical limit is often lower due to register pressure
                params.max_waves_per_eu = 8; // 16 waves total per CU (8 per SIMD)
                break;
            case AMDArchitecture::RDNA1:
                params.max_waves_per_eu = 10; // 20 waves total per CU
                break;
            case AMDArchitecture::CDNA3:
            case AMDArchitecture::CDNA2:
            case AMDArchitecture::CDNA1:
            case AMDArchitecture::GCN5:
            case AMDArchitecture::GCN4:
                // GCN/CDNA has 4 SIMDs per CU
                params.max_waves_per_eu = 10; // 40 waves total per CU
                break;
            default:
                params.max_waves_per_eu = 8;
                break;
        }

        // Adjust waves_per_cu if it exceeds architectural limits
        int arch_limit = params.max_waves_per_eu * (arch >= AMDArchitecture::RDNA1 ? 2 : 4);
        if (params.waves_per_cu > arch_limit) {
            params.waves_per_cu = arch_limit;
        }

        return params;
    }
};

class AMDGPUDetector {
public:
    static AMDArchitecture detectArchitecture(const hipDeviceProp_t& props) {
        // Use gcnArchName for precise detection
        std::string arch_name = props.gcnArchName ? props.gcnArchName : "";

        // Parse gfxXXX format
        if (arch_name.find("gfx") == 0 && arch_name.length() >= 6) {
            int arch_num = 0;
            try {
                // Handle both numeric and hex formats
                std::string num_str = arch_name.substr(3, 3);

                // Check if it's hex (contains 'a-f')
                if (num_str.find_first_of("abcdef") != std::string::npos) {
                    arch_num = std::stoi(num_str, nullptr, 16);
                } else {
                    arch_num = std::stoi(num_str);
                }
            } catch (...) {
                return AMDArchitecture::UNKNOWN;
            }

            // Map to architecture
            if (arch_num >= 1200 && arch_num < 1300) return AMDArchitecture::RDNA4;
            if (arch_num >= 1100 && arch_num < 1200) return AMDArchitecture::RDNA3;
            if (arch_num >= 1030 && arch_num < 1100) return AMDArchitecture::RDNA2;
            if (arch_num >= 1010 && arch_num < 1030) return AMDArchitecture::RDNA1;
            if (arch_num >= 900 && arch_num < 910) return AMDArchitecture::GCN5;
            if (arch_num >= 800 && arch_num < 900) return AMDArchitecture::GCN4;

            // CDNA architectures
            if (arch_num == 908) return AMDArchitecture::CDNA1;
            if (arch_num == 0x90a || arch_num == 910) return AMDArchitecture::CDNA2;
            if (arch_num == 940) return AMDArchitecture::CDNA3;
        }

        // Fallback: detect by device name
        std::string device_name = props.name;
        if (device_name.find("gfx12") != std::string::npos) return AMDArchitecture::RDNA4;
        if (device_name.find("gfx11") != std::string::npos) return AMDArchitecture::RDNA3;
        if (device_name.find("gfx103") != std::string::npos) return AMDArchitecture::RDNA2;
        if (device_name.find("gfx101") != std::string::npos) return AMDArchitecture::RDNA1;
        if (device_name.find("Vega") != std::string::npos) return AMDArchitecture::GCN5;
        if (device_name.find("RX 9") != std::string::npos) return AMDArchitecture::RDNA4;
        if (device_name.find("RX 7") != std::string::npos) return AMDArchitecture::RDNA3;
        if (device_name.find("RX 6") != std::string::npos) return AMDArchitecture::RDNA2;
        if (device_name.find("RX 5") != std::string::npos) return AMDArchitecture::RDNA1;

        return AMDArchitecture::UNKNOWN;
    }

    static std::string getArchitectureName(AMDArchitecture arch) {
        switch (arch) {
            case AMDArchitecture::RDNA4: return "RDNA4";
            case AMDArchitecture::RDNA3: return "RDNA3";
            case AMDArchitecture::RDNA2: return "RDNA2";
            case AMDArchitecture::RDNA1: return "RDNA1";
            case AMDArchitecture::CDNA3: return "CDNA3";
            case AMDArchitecture::CDNA2: return "CDNA2";
            case AMDArchitecture::CDNA1: return "CDNA1";
            case AMDArchitecture::GCN5: return "GCN5 (Vega)";
            case AMDArchitecture::GCN4: return "GCN4 (Polaris)";
            case AMDArchitecture::GCN3: return "GCN3";
            default: return "Unknown";
        }
    }

    // Get architecture parameters - DO NOT USE the hardcoded map!
    static AMDArchParams getArchitectureParams(AMDArchitecture arch, const hipDeviceProp_t& props) {
        return AMDArchParams::getFromDevice(props, arch);
    }

    // Template version to avoid circular dependency
    template<typename ConfigType>
    static void configureForArchitecture(ConfigType& config,
                                    const hipDeviceProp_t& props,
                                    AMDArchitecture arch) {
        // Get actual parameters from device
        AMDArchParams params = AMDArchParams::getFromDevice(props, arch);
        int actual_cus = props.multiProcessorCount;

        std::cout << "\nDevice Capabilities (auto-detected):\n";
        std::cout << "  Architecture: " << getArchitectureName(arch) << "\n";
        std::cout << "  Compute Units: " << actual_cus << "\n";
        std::cout << "  Wave Size: " << params.wave_size << "\n";
        std::cout << "  Max Waves per CU: " << params.waves_per_cu << "\n";
        std::cout << "  LDS Size per Block: " << params.lds_size_per_cu / 1024 << " KB\n";
        std::cout << "  Max Workgroup Size: " << params.max_workgroup_size << "\n";
        std::cout << "  Max Threads per CU: " << props.maxThreadsPerMultiProcessor << "\n";

        // Calculate optimal blocks per CU based on occupancy goals
        // Goal: Achieve high occupancy without oversubscribing
        int target_waves_per_cu = params.waves_per_cu * 0.75; // Target 75% occupancy
        int waves_per_block = config.threads_per_block / params.wave_size;
        if (waves_per_block < 1) waves_per_block = 1;
        int blocks_per_cu = target_waves_per_cu / waves_per_block;

        // Ensure at least some minimum blocks per CU
        if (blocks_per_cu < 4) blocks_per_cu = 4;

        // Architecture-specific tuning
        switch (arch) {
            case AMDArchitecture::RDNA4:
                // RDNA4 can handle MUCH more work
                blocks_per_cu = 32;  // Maximum blocks per CU
                config.threads_per_block = 256;
                config.num_streams = 16;  // More streams for better overlap
                config.result_buffer_size = 1024;  // Larger buffer for more results

                // For 64 CUs: 64 * 32 = 2048 blocks
                config.blocks_per_stream = actual_cus * blocks_per_cu;

                // Don't cap at 2048 for RDNA4 - it can handle more
                if (config.blocks_per_stream > 4096) {
                    config.blocks_per_stream = 4096;
                }
                break;

            case AMDArchitecture::RDNA3:
                blocks_per_cu = 24;
                config.threads_per_block = 256;
                config.num_streams = 8;
                config.result_buffer_size = 256;
                config.blocks_per_stream = actual_cus * blocks_per_cu;
                if (config.blocks_per_stream > 3072) {
                    config.blocks_per_stream = 3072;
                }
                break;

            case AMDArchitecture::RDNA2:
                blocks_per_cu = 16;
                config.threads_per_block = 256;
                config.num_streams = 6;
                config.result_buffer_size = 128;
                config.blocks_per_stream = actual_cus * blocks_per_cu;
                if (config.blocks_per_stream > 2048) {
                    config.blocks_per_stream = 2048;
                }
                break;

            case AMDArchitecture::RDNA1:
                blocks_per_cu = 40;  // Increase from 32
                config.threads_per_block = 128;  // Reduce from 256 to allow more blocks
                config.num_streams = 12;  // Increase from 8
                config.result_buffer_size = 2048;  // Increase from 1024
                config.blocks_per_stream = actual_cus * blocks_per_cu;
                if (config.blocks_per_stream > 3200) {  // Increase limit from 2560
                    config.blocks_per_stream = 3200;
                }
                break;

            default:
                // Conservative for older architectures
                blocks_per_cu = 8;
                config.threads_per_block = 256;
                config.num_streams = 4;
                config.result_buffer_size = 128;
                config.blocks_per_stream = actual_cus * blocks_per_cu;
                if (config.blocks_per_stream > 1024) {
                    config.blocks_per_stream = 1024;
                }
                break;
        }

        // Verify configuration doesn't exceed device limits
        int total_threads = config.blocks_per_stream * config.threads_per_block;
        int total_waves = total_threads / params.wave_size;
        int waves_per_cu_actual = total_waves / actual_cus;

        std::cout << "\nCalculated Configuration:\n";
        std::cout << "  Blocks per CU: " << blocks_per_cu << "\n";
        std::cout << "  Total blocks per stream: " << config.blocks_per_stream << "\n";
        std::cout << "  Threads per block: " << config.threads_per_block << "\n";
        std::cout << "  Waves per CU (actual): " << waves_per_cu_actual << "\n";
        std::cout << "  Occupancy: " << (100.0 * waves_per_cu_actual / params.waves_per_cu) << "%\n";

        // Check if we need to adjust for memory or other limits
        size_t required_lds = config.threads_per_block * 64; // Rough estimate
        if (required_lds > static_cast<size_t>(params.lds_size_per_cu)) {
            std::cout << "  WARNING: May need to reduce threads per block due to LDS limits\n";
        }
    }

    static void displayDeviceLimits(const hipDeviceProp_t& props) {
        std::cout << "\nComplete Device Limits:\n";
        std::cout << "  Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
        std::cout << "  Max Threads per CU: " << props.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Max Blocks per CU: " << props.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Max Grid Size: [" << props.maxGridSize[0] << ", "
                  << props.maxGridSize[1] << ", " << props.maxGridSize[2] << "]\n";
        std::cout << "  Warp/Wave Size: " << props.warpSize << "\n";
        std::cout << "  Registers per Block: " << props.regsPerBlock << "\n";
        std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "  Total Constant Memory: " << props.totalConstMem / 1024 << " KB\n";
        std::cout << "  Memory Clock Rate: " << props.memoryClockRate / 1000 << " MHz\n";
        std::cout << "  Memory Bus Width: " << props.memoryBusWidth << " bits\n";
        std::cout << "  L2 Cache Size: " << props.l2CacheSize / 1024 << " KB\n";
        std::cout << "  Compute Capability: " << props.major << "." << props.minor << "\n";
    }

    // Check if GPU is known to have issues
    static bool hasKnownIssues(AMDArchitecture arch, const std::string& device_name) {
        // RDNA4 early drivers might have issues
        if (arch == AMDArchitecture::RDNA4) {
            // Check ROCm version
            int version;
            if (hipRuntimeGetVersion(&version) == hipSuccess) {
                // RDNA4 likely requires ROCm 6.2+ or later
                if (version < 60200000) {
                    return true;
                }
            }
        }

        // RDNA3 early drivers had issues with certain workloads
        if (arch == AMDArchitecture::RDNA3) {
            // Check ROCm version
            int version;
            if (hipRuntimeGetVersion(&version) == hipSuccess) {
                // ROCm versions before 5.7 had RDNA3 issues
                if (version < 50700000) {
                    return true;
                }
            }
        }

        // Add other known problematic configurations here

        return false;
    }
};

// Helper function to print architecture info
inline void printAMDArchitectureInfo(const hipDeviceProp_t& props) {
    AMDArchitecture arch = AMDGPUDetector::detectArchitecture(props);
    AMDArchParams params = AMDArchParams::getFromDevice(props, arch);

    std::cout << "\n=== AMD GPU Architecture Information ===\n";
    std::cout << "GPU Name: " << props.name << "\n";
    std::cout << "Architecture: " << AMDGPUDetector::getArchitectureName(arch) << "\n";
    if (props.gcnArchName) {
        std::cout << "GCN Arch Name: " << props.gcnArchName << "\n";
    }

    // Display all detected parameters
    AMDGPUDetector::displayDeviceLimits(props);

    // Calculate some derived metrics
    double memory_bandwidth_gb = (props.memoryClockRate / 1000.0) * (props.memoryBusWidth / 8.0) * 2.0 / 1000.0;
    double compute_tflops = (props.clockRate / 1000.0) * props.multiProcessorCount * 128 * 2 / 1000.0; // Rough estimate

    std::cout << "\nPerformance Metrics:\n";
    std::cout << "  Theoretical Memory Bandwidth: " << std::fixed << std::setprecision(1)
              << memory_bandwidth_gb << " GB/s\n";
    std::cout << "  Estimated FP32 Performance: " << std::setprecision(1)
              << compute_tflops << " TFLOPS\n";
    std::cout << "  Compute to Memory Ratio: " << std::setprecision(2)
              << (compute_tflops * 1000.0 / memory_bandwidth_gb) << " FLOP/byte\n";
}

#endif // USE_HIP

#endif // GPU_ARCHITECTURE_HPP