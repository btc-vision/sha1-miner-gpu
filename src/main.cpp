#include <iomanip>
#include <iostream>

#include "sha1_miner.cuh"

#include "../logging/logger.hpp"
#include "configs/config.hpp"
#include "core/pool_mining.hpp"
#include "core/solo_mining.hpp"
#include "utils/platform_utils.hpp"
#include "utils/test_utils.hpp"

void show_help_menu(const char *program_name)
{
    // Show program header
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "+------------------------------------------+", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "|            SHA-1 OP_NET Miner            |", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "+------------------------------------------+", Color::RESET);
    std::cout << "\nSHA-1 OP_NET Miner\n\n";
    // Create and display options
    const auto desc = create_options_description();
    std::cout << desc << "\n";
    print_usage_examples(program_name);
    std::cout << "\n";
    LOG_INFO("MAIN", Color::BRIGHT_YELLOW, "Please connect to a pool by passing command line arguments.", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_GREEN, "For example: ", program_name,
             " --pool ws://pool.example.com:3333 --wallet YOUR_WALLET", Color::RESET);
}

void print_gpu_info(const std::vector<int> &gpu_ids_to_use)
{
    LOG_INFO("MAIN", "\nGPU Information:");
    LOG_INFO("MAIN", "=====================================");
    for (int id : gpu_ids_to_use) {
        gpuDeviceProp props;
        gpuGetDeviceProperties(&props, id);
        LOG_INFO("MAIN", "  GPU ", id, ": ", Color::BRIGHT_CYAN, props.name, Color::RESET);
        LOG_INFO("MAIN", "    Compute capability: ", props.major, ".", props.minor);
        LOG_INFO("MAIN", "    Memory: ", std::fixed, std::setprecision(2),
                 props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), " GB");
        LOG_INFO("MAIN", "    SMs/CUs: ", props.multiProcessorCount);
    }
    LOG_INFO("MAIN", "");
}

std::vector<int> determine_gpu_ids(const MiningConfig &config, const int device_count)
{
    std::vector<int> gpu_ids_to_use;

    if (config.use_all_gpus) {
        for (int i = 0; i < device_count; i++) {
            gpu_ids_to_use.push_back(i);
        }
        LOG_INFO("MAIN", "Using all ", device_count, " available GPUs");
    } else if (!config.gpu_ids.empty()) {
        gpu_ids_to_use = config.gpu_ids;
        // Validate GPU IDs
        for (const int id : gpu_ids_to_use) {
            if (id >= device_count || id < 0) {
                LOG_ERROR("MAIN", "Invalid GPU ID: ", id, ". Available GPUs: 0-", device_count - 1);
                std::exit(1);
            }
        }
        LOG_INFO("MAIN", "Using ", gpu_ids_to_use.size(), " specified GPU(s)");
    } else {
        // Default to single GPU
        gpu_ids_to_use.push_back(config.gpu_id);
        if (config.gpu_id >= device_count) {
            LOG_ERROR("MAIN", "Invalid GPU ID. Available GPUs: 0-", device_count - 1);
            std::exit(1);
        }
    }

    return gpu_ids_to_use;
}

// Main program
int main(const int argc, char *argv[])
{
    // Set up UTF-8 encoding for console output
    setup_console_encoding();

    // Set up signal handlers
    setup_signal_handlers();

    // Check if no arguments were provided
    if (argc == 1) {
        show_help_menu(argv[0]);
        return 0;
    }

    // Parse command line
    MiningConfig config = parse_args(argc, argv);

    // Configure logging based on command line options
    Logger::set_level(static_cast<LogLevel>(config.debug_level));
    Logger::enable_colors(!config.no_colors);

    // If debug mode is explicitly enabled, set to DEBUG level
    if (config.debug_mode) {
        Logger::set_level(LogLevel::DEBUG);
    }

    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "+------------------------------------------+", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "|            SHA-1 OP_NET Miner            |", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "+------------------------------------------+", Color::RESET);
    LOG_INFO("MAIN", "Debug level: ", config.debug_level, " (", config.debug_mode ? "DEBUG MODE" : "NORMAL", ")");

    // Handle test modes
    if (config.test_sha1) {
        LOG_INFO("MAIN", "Running SHA-1 tests...");
        if (!verify_sha1_implementation()) {
            LOG_ERROR("MAIN", "SHA-1 implementation verification failed!");
            return 1;
        }
        return 0;
    }

    // Verify SHA-1 implementation
    if (!verify_sha1_implementation()) {
        LOG_ERROR("MAIN", "SHA-1 implementation verification failed!");
        return 1;
    }
    LOG_INFO("MAIN", Color::GREEN, "SHA-1 implementation verified.", Color::RESET);

    // Check GPU availability
    int device_count;
    gpuGetDeviceCount(&device_count);

    if (device_count == 0) {
#ifdef USE_HIP
        LOG_ERROR("MAIN", "No AMD/HIP devices found!");
#else
        LOG_ERROR("MAIN", "No CUDA devices found!");
#endif
        return 1;
    }

    // Determine which GPUs to use
    const std::vector<int> gpu_ids_to_use = determine_gpu_ids(config, device_count);

    // Print GPU information
    print_gpu_info(gpu_ids_to_use);

    // Check if pool mining is requested
    if (config.use_pool) {
        return run_pool_mining(config);
    }

    // Run benchmark if requested
    if (config.benchmark) {
        if (gpu_ids_to_use.size() > 1) {
            LOG_WARN("MAIN", "Multi-GPU benchmark not yet implemented. Using GPU 0 only.");
            run_benchmark(gpu_ids_to_use[0]);
        } else {
            run_benchmark(gpu_ids_to_use[0]);
        }
        return 0;
    }

    // Default to solo mining
    return 0;
    // return run_solo_mining(config, gpu_ids_to_use);
}
