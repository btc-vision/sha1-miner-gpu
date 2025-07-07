// gpu_architecture.hpp - Enhanced AMD GPU architecture detection and handling
#ifndef GPU_ARCHITECTURE_HPP
#define GPU_ARCHITECTURE_HPP

#include "gpu_platform.hpp"
#include <string>
#include <map>

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
    int wavefront_size;
    int max_workgroup_size;
    int preferred_workgroup_multiple;
    int local_memory_size;
    int max_registers_per_cu;
    bool has_packed_math;
    bool has_matrix_instructions;
    int blocks_per_cu;
    std::string arch_name;
};

class AMDGPUDetector {
public:
    static AMDArchitecture detectArchitecture(const hipDeviceProp_t& props) {
        // Use gcnArchName for precise detection
        std::string arch_name = props.gcnArchName;

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

    static AMDArchParams getArchitectureParams(AMDArchitecture arch) {
        static const std::map<AMDArchitecture, AMDArchParams> arch_params = {
            {AMDArchitecture::GCN3, {64, 256, 64, 65536, 24576, false, false, 4, "GCN3"}},
            {AMDArchitecture::GCN4, {64, 256, 64, 65536, 24576, false, false, 4, "GCN4"}},
            {AMDArchitecture::GCN5, {64, 256, 64, 65536, 24576, true, false, 8, "GCN5/Vega"}},
            {AMDArchitecture::RDNA1, {32, 256, 32, 65536, 24576, true, false, 16, "RDNA1"}},
            {AMDArchitecture::RDNA2, {32, 256, 32, 65536, 32768, true, false, 16, "RDNA2"}},
            {AMDArchitecture::RDNA3, {32, 256, 32, 65536, 32768, true, true, 20, "RDNA3"}},
            {AMDArchitecture::RDNA4, {32, 256, 32, 65536, 32768, true, true, 24, "RDNA4"}},
            {AMDArchitecture::CDNA1, {64, 256, 64, 65536, 32768, true, true, 8, "CDNA1"}},
            {AMDArchitecture::CDNA2, {64, 256, 64, 65536, 32768, true, true, 16, "CDNA2"}},
            {AMDArchitecture::CDNA3, {64, 256, 64, 65536, 32768, true, true, 32, "CDNA3"}},
            {AMDArchitecture::UNKNOWN, {32, 256, 32, 65536, 24576, false, false, 8, "Unknown"}}
        };

        auto it = arch_params.find(arch);
        if (it != arch_params.end()) {
            return it->second;
        }
        return arch_params.at(AMDArchitecture::UNKNOWN);
    }

    // Get actual CU count (handle RDNA WGP reporting)
    static int getActualCUCount(const hipDeviceProp_t& props, AMDArchitecture arch) {
        int reported_count = props.multiProcessorCount;

        // RDNA architectures may report WGPs instead of CUs
        if (arch == AMDArchitecture::RDNA1 ||
            arch == AMDArchitecture::RDNA2 ||
            arch == AMDArchitecture::RDNA3 ||
            arch == AMDArchitecture::RDNA4) {
            // Check if this is reporting WGPs (each WGP = 2 CUs)
            // This is a heuristic based on known GPU configurations
            if (reported_count < 40) {  // Likely WGPs
                return reported_count * 2;
            }
        }

        return reported_count;
    }

    // Get architecture-specific kernel configuration
    static void configureForArchitecture(MiningSystem::Config& config,
                                          const hipDeviceProp_t& props,
                                          AMDArchitecture arch) {
        AMDArchParams params = getArchitectureParams(arch);
        int actual_cus = getActualCUCount(props, arch);

        // Base configuration
        config.threads_per_block = 256;  // Works well across all architectures
 // Architecture-specific tuning
        switch (arch) {
            case AMDArchitecture::RDNA4:
                // RDNA4 has improved dual-issue and better efficiency
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 12;
                // RDNA4 has better occupancy characteristics
                if (config.blocks_per_stream > 2560) {
                    config.blocks_per_stream = 2560;
                }
                break;

            case AMDArchitecture::RDNA3:
                // RDNA3 has dual-issue capability and better occupancy
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 8;
                // Consider using wave32 mode for better occupancy
                if (config.blocks_per_stream > 2048) {
                    config.blocks_per_stream = 2048;
                }
                break;

            case AMDArchitecture::RDNA2:
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 8;
                if (config.blocks_per_stream > 1536) {
                    config.blocks_per_stream = 1536;
                }
                break;

            case AMDArchitecture::RDNA1:
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 4;
                if (config.blocks_per_stream > 1024) {
                    config.blocks_per_stream = 1024;
                }
                break;

            case AMDArchitecture::GCN5:
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 4;
                break;

            case AMDArchitecture::GCN4:
            case AMDArchitecture::GCN3:
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 2;
                break;

            case AMDArchitecture::CDNA1:
            case AMDArchitecture::CDNA2:
            case AMDArchitecture::CDNA3:
                // Data center GPUs can handle more concurrent work
                config.blocks_per_stream = actual_cus * params.blocks_per_cu;
                config.num_streams = 16;
                config.threads_per_block = 256;
                break;

            default:
                // Conservative defaults
                config.blocks_per_stream = actual_cus * 8;
                config.num_streams = 4;
                break;
        }

        // Memory-based limits
        size_t free_mem, total_mem;
        GPU_IGNORE_RESULT(hipMemGetInfo(&free_mem, &total_mem));

        // Adjust based on available memory
        size_t mem_per_stream = sizeof(MiningResult) * config.result_buffer_size +
                                (config.blocks_per_stream * config.threads_per_block * 256);
        int max_streams_by_memory = free_mem / (mem_per_stream * 2);

        if (config.num_streams > max_streams_by_memory && max_streams_by_memory > 0) {
            config.num_streams = max_streams_by_memory;
        }

        // Ensure at least 1 stream
        if (config.num_streams < 1) {
            config.num_streams = 1;
        }
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
    AMDArchParams params = AMDGPUDetector::getArchitectureParams(arch);
    int actual_cus = AMDGPUDetector::getActualCUCount(props, arch);

    std::cout << "AMD GPU Architecture Details:\n";
    std::cout << "  Architecture: " << params.arch_name << "\n";
    std::cout << "  GCN Arch Name: " << props.gcnArchName << "\n";
    std::cout << "  Wavefront Size: " << params.wavefront_size << "\n";
    std::cout << "  Reported SMs/CUs: " << props.multiProcessorCount << "\n";
    std::cout << "  Actual CUs: " << actual_cus << "\n";
    std::cout << "  Max Workgroup Size: " << params.max_workgroup_size << "\n";
    std::cout << "  Local Memory: " << params.local_memory_size << " bytes\n";
    std::cout << "  Has Packed Math: " << (params.has_packed_math ? "Yes" : "No") << "\n";

    if (AMDGPUDetector::hasKnownIssues(arch, props.name)) {
        std::cout << "  WARNING: This GPU/driver combination may have known issues\n";
    }
}

#endif // USE_HIP

#endif // GPU_ARCHITECTURE_HPP
