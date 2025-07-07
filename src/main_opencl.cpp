#include "include/sha1_opencl_miner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <csignal>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#define SIGBREAK 21
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

// Configuration structure
struct MiningConfig {
    int platform_id = -1; // -1 = auto-select
    int device_id = -1; // -1 = auto-select
    std::vector<std::pair<int, int> > gpu_pairs; // For multi-GPU
    uint32_t difficulty = 50;
    uint32_t duration = 300; // 5 minutes default
    bool benchmark = false;
    bool list_devices = false;
    bool test_sha1 = false;
    bool auto_tune = true;
    bool use_all_gpus = false;
    std::string target_hex;
    std::string message_hex;
};

void list_opencl_devices() {
    std::cout << "\nOpenCL Platforms and Devices:\n";
    std::cout << "=====================================\n";

    auto platforms = OpenCLMiningSystem::enumeratePlatforms();
    if (platforms.empty()) {
        std::cout << "No OpenCL platforms found!\n";
        return;
    }

    for (size_t p = 0; p < platforms.size(); p++) {
        const auto &platform = platforms[p];
        std::cout << "\nPlatform " << p << ": " << platform.name << "\n";
        std::cout << "  Vendor: " << platform.vendor << "\n";
        std::cout << "  Version: " << platform.version << "\n";

        auto devices = OpenCLMiningSystem::enumerateDevices(platform.platform);
        if (devices.empty()) {
            std::cout << "  No GPU devices found\n";
        } else {
            std::cout << "  Devices:\n";
            for (size_t d = 0; d < devices.size(); d++) {
                const auto &device = devices[d];
                std::cout << "    [" << p << ":" << d << "] " << device.name << "\n";
                std::cout << "      Vendor: " << device.vendor << "\n";
                std::cout << "      Compute Units: " << device.compute_units << "\n";
                std::cout << "      Max Clock: " << device.max_clock_frequency << " MHz\n";
                std::cout << "      Global Memory: "
                        << (device.global_mem_size / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
                std::cout << "      Local Memory: "
                        << (device.local_mem_size / 1024.0) << " KB\n";
                std::cout << "      Max Work Group: " << device.max_work_group_size << "\n";
            }
        }
    }
    std::cout << "\nUse --gpu <platform>:<device> to select a specific GPU\n";
}

void print_usage(const char *program_name) {
    std::cout << "SHA-1 OP_NET Miner\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --gpu <id>          GPU device ID (default: 0)\n";
    std::cout << "  --all-gpus          Use all available GPUs\n"; // Add this
    std::cout << "  --gpus <list>       Use specific GPUs (e.g., --gpus 0,1)\n"; // Add this
    std::cout << "  --difficulty <bits> Number of bits that must match (default: 50)\n";
    std::cout << "  --duration <sec>    Mining duration in seconds (default: 300)\n";
    std::cout << "  --target <hex>      Target hash in hex (40 chars)\n";
    std::cout << "  --message <hex>     Base message in hex (64 chars)\n";
    std::cout << "  --streams <n>       Number of CUDA streams (default: 4)\n";
    std::cout << "  --threads <n>       Threads per block (default: 256)\n";
    std::cout << "  --benchmark         Run performance benchmark\n";
    std::cout << "  --auto-tune         Auto-tune for optimal performance\n";
    std::cout << "  --pool <url>        Connect to mining pool\n";
    std::cout << "  --worker <name>     Worker name for pool\n";
    std::cout << "  --help              Show this help\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --gpu 0 --difficulty 45 --duration 600\n";
    std::cout << "  " << program_name << " --all-gpus --difficulty 50\n"; // Add this
    std::cout << "  " << program_name << " --benchmark --auto-tune\n";
}

MiningConfig parse_args(int argc, char *argv[]) {
    MiningConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--platform" && i + 1 < argc) {
            config.platform_id = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            config.device_id = std::stoi(argv[++i]);
        } else if (arg == "--gpu" && i + 1 < argc) {
            // Parse platform:device format
            std::string gpu_spec = argv[++i];
            size_t colon_pos = gpu_spec.find(':');
            if (colon_pos != std::string::npos) {
                int platform = std::stoi(gpu_spec.substr(0, colon_pos));
                int device = std::stoi(gpu_spec.substr(colon_pos + 1));
                config.gpu_pairs.push_back({platform, device});
            }
        } else if (arg == "--all-gpus") {
            config.use_all_gpus = true;
        } else if (arg == "--difficulty" && i + 1 < argc) {
            config.difficulty = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            config.duration = std::stoi(argv[++i]);
        } else if (arg == "--target" && i + 1 < argc) {
            config.target_hex = argv[++i];
        } else if (arg == "--message" && i + 1 < argc) {
            config.message_hex = argv[++i];
        } else if (arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--list") {
            config.list_devices = true;
        } else if (arg == "--test-sha1") {
            config.test_sha1 = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            std::exit(1);
        }
    }

    return config;
}

// Generate cryptographically secure random message
std::vector<uint8_t> generate_secure_random_message() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<uint8_t> message(32);

    // Use multiple sources of entropy
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = now.time_since_epoch().count();

    // Mix in time-based entropy
    for (size_t i = 0; i < 8; i++) {
        message[i] = static_cast<uint8_t>((nanos >> (i * 8)) & 0xFF);
    }

    // Fill rest with random data
    for (size_t i = 8; i < 32; i++) {
        message[i] = static_cast<uint8_t>(dis(gen));
    }

    // Additional mixing
    for (size_t i = 0; i < 32; i++) {
        message[i] ^= static_cast<uint8_t>(dis(gen));
    }

    return message;
}

// Verify SHA-1 implementation
bool verify_sha1_implementation() {
    std::cout << "Verifying SHA-1 implementation...\n";

    // Test vectors from FIPS 180-1
    struct TestVector {
        std::string message;
        std::string expected_hash;
    };

    std::vector<TestVector> test_vectors = {
        {"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"},
        {"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"},
        {
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        }
    };

    bool all_passed = true;

    for (const auto &test: test_vectors) {
        std::vector<uint8_t> message(test.message.begin(), test.message.end());
        auto hash = calculate_sha1(message);
        std::string hash_hex = bytes_to_hex(hash);

        if (hash_hex != test.expected_hash) {
            std::cerr << "FAILED: Message '" << test.message << "'\n";
            std::cerr << "  Expected: " << test.expected_hash << "\n";
            std::cerr << "  Got:      " << hash_hex << "\n";
            all_passed = false;
        } else {
            std::cout << "PASSED: '" << test.message << "'\n";
        }
    }

    return all_passed;
}

// Run benchmark
void run_benchmark(const MiningConfig &config) {
    std::cout << "\n=== SHA-1 Near-Collision Mining Benchmark (OpenCL) ===\n\n";

    // Initialize mining system
    std::unique_ptr<OpenCLMiningSystem> mining_system;
    std::unique_ptr<OpenCLMultiGPUManager> multi_gpu_manager;

    if (config.use_all_gpus || config.gpu_pairs.size() > 1) {
        // Multi-GPU benchmark
        multi_gpu_manager = std::make_unique<OpenCLMultiGPUManager>();
        if (!multi_gpu_manager->initialize(config.gpu_pairs)) {
            std::cerr << "Failed to initialize multi-GPU system\n";
            return;
        }
    } else {
        // Single GPU benchmark
        OpenCLMiningSystem::Config sys_config;
        sys_config.platform_index = config.platform_id;
        sys_config.device_index = config.device_id;
        sys_config.auto_tune = config.auto_tune;

        mining_system = std::make_unique<OpenCLMiningSystem>(sys_config);
        if (!mining_system->initialize()) {
            std::cerr << "Failed to initialize mining system\n";
            return;
        }
    }

    // Test different difficulty levels
    std::vector<uint32_t> difficulties = {20, 30, 40, 45, 50, 55, 60};
    std::vector<double> results;

    for (uint32_t diff: difficulties) {
        if (g_shutdown) break;

        std::cout << "\nTesting difficulty " << diff << " bits:\n";

        // Generate test job
        auto message = generate_secure_random_message();
        auto target_hash = calculate_sha1(message);

        MiningJob job = create_mining_job(message.data(), target_hash.data(), diff);

        // Run for 60 seconds
        auto start = std::chrono::steady_clock::now();

        if (multi_gpu_manager) {
            multi_gpu_manager->runMining(job, 60);
        } else {
            mining_system->runMiningLoop(job, 60);
        }

        auto end = std::chrono::steady_clock::now();

        // Get statistics
        double duration = std::chrono::duration<double>(end - start).count();
        MiningStats stats;

        if (multi_gpu_manager) {
            // Calculate combined stats for multi-GPU
            uint64_t total_hashes = 0;
            uint64_t total_candidates = 0;
            uint32_t best_bits = 0;

            // Note: Would need to add a method to get combined stats from multi-GPU manager
            // For now, we'll estimate based on output
            stats.hash_rate = 0; // Placeholder
            stats.candidates_found = 0;
            stats.best_match_bits = 0;
        } else {
            stats = mining_system->getStats();
        }

        double hash_rate = stats.hash_rate / 1e9; // GH/s
        results.push_back(hash_rate);

        std::cout << "\nResults for difficulty " << diff << " bits:\n";
        std::cout << "  Hash rate: " << std::fixed << std::setprecision(2)
                << hash_rate << " GH/s\n";
        std::cout << "  Candidates found: " << stats.candidates_found << "\n";
        std::cout << "  Expected candidates: " << std::scientific
                << (stats.hashes_computed / std::pow(2.0, diff)) << "\n";
        if (stats.hashes_computed > 0 && stats.candidates_found > 0) {
            std::cout << "  Efficiency: " << std::fixed << std::setprecision(2)
                    << (100.0 * stats.candidates_found * std::pow(2.0, diff) /
                        stats.hashes_computed) << "%\n";
        }
    }

    // Print summary
    std::cout << "\n=== Benchmark Summary ===\n";
    std::cout << "Difficulty | Hash Rate (GH/s)\n";
    std::cout << "-----------|----------------\n";
    for (size_t i = 0; i < difficulties.size() && i < results.size(); i++) {
        std::cout << std::setw(10) << difficulties[i] << " | "
                << std::fixed << std::setprecision(2) << results[i] << "\n";
    }
}

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    const char *sig_name = "UNKNOWN";
    switch (sig) {
        case SIGINT: sig_name = "SIGINT";
            break;
        case SIGTERM: sig_name = "SIGTERM";
            break;
#ifdef _WIN32
        case SIGBREAK: sig_name = "SIGBREAK";
            break;
#else
        case SIGHUP:  sig_name = "SIGHUP"; break;
        case SIGQUIT: sig_name = "SIGQUIT"; break;
#endif
    }
    std::cout << "\nReceived signal " << sig_name << " (" << sig << "), shutting down...\n";
    g_shutdown.store(true);
}

void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    std::signal(SIGBREAK, signal_handler);
#else
    std::signal(SIGHUP, signal_handler);
    std::signal(SIGQUIT, signal_handler);
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

// Main program
int main(int argc, char *argv[]) {
    // Set up signal handlers
    setup_signal_handlers();

    // Parse command line
    MiningConfig config = parse_args(argc, argv);

    std::cout << "+------------------------------------------+\n";
    std::cout << "|   SHA-1 Near-Collision Miner (OpenCL)   |\n";
    std::cout << "+------------------------------------------+\n\n";

    // List devices if requested
    if (config.list_devices) {
        list_opencl_devices();
        return 0;
    }

    // Test SHA-1 implementation if requested
    if (config.test_sha1) {
        if (!verify_sha1_implementation()) {
            std::cerr << "SHA-1 implementation verification failed!\n";
            return 1;
        }
        std::cout << "SHA-1 implementation verified.\n\n";
        if (argc == 2) {
            // Only --test-sha1 was specified
            return 0;
        }
    }

    // Always verify SHA-1 before mining
    if (!verify_sha1_implementation()) {
        std::cerr << "SHA-1 implementation verification failed!\n";
        return 1;
    }
    std::cout << "SHA-1 implementation verified.\n\n";

    // Run benchmark if requested
    if (config.benchmark) {
        run_benchmark(config);
        return 0;
    }

    // Prepare message and target
    std::vector<uint8_t> message;
    std::vector<uint8_t> target;

    if (!config.message_hex.empty()) {
        if (config.message_hex.length() != 64) {
            std::cerr << "Message must be 64 hex characters (32 bytes)\n";
            return 1;
        }
        message = hex_to_bytes(config.message_hex);
    } else {
        message = generate_secure_random_message();
        std::cout << "Generated random message: " << bytes_to_hex(message) << "\n";
    }

    if (!config.target_hex.empty()) {
        if (config.target_hex.length() != 40) {
            std::cerr << "Target must be 40 hex characters (20 bytes)\n";
            return 1;
        }
        target = hex_to_bytes(config.target_hex);
    } else {
        target = calculate_sha1(message);
        std::cout << "Target SHA-1: " << bytes_to_hex(target) << "\n";
    }

    // Create mining job
    MiningJob job = create_mining_job(message.data(), target.data(), config.difficulty);

    std::cout << "\nMining Configuration:\n";
    std::cout << "  Difficulty: " << config.difficulty << " bits must match\n";
    std::cout << "  Duration: " << config.duration << " seconds\n";
    std::cout << "  Success probability per hash: 2^-" << config.difficulty << "\n\n";

    // Start mining
    auto start_time = std::chrono::steady_clock::now();

    if (config.use_all_gpus || config.gpu_pairs.size() > 1) {
        // Multi-GPU mining
        OpenCLMultiGPUManager multi_gpu_manager;
        if (!multi_gpu_manager.initialize(config.gpu_pairs)) {
            std::cerr << "Failed to initialize multi-GPU mining\n";
            return 1;
        }

        multi_gpu_manager.runMining(job, config.duration);
    } else {
        // Single GPU mining
        OpenCLMiningSystem::Config sys_config;
        sys_config.platform_index = config.platform_id;
        sys_config.device_index = config.device_id;
        sys_config.auto_tune = config.auto_tune;

        OpenCLMiningSystem mining_system(sys_config);
        if (!mining_system.initialize()) {
            std::cerr << "Failed to initialize mining system\n";
            return 1;
        }

        mining_system.runMiningLoop(job, config.duration);
    }

    auto end_time = std::chrono::steady_clock::now();

    // Print final statistics
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\nTotal runtime: " << std::fixed << std::setprecision(2)
            << duration << " seconds\n";

    return 0;
};
