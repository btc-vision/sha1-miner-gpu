#include "sha1_miner.cuh"
#include "mining_system.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <csignal>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#define SIGBREAK 21
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

// Advanced configuration for production mining
struct MiningConfig {
    int gpu_id = 0;
    uint32_t difficulty = 50; // Default to 50 bits (reasonable for testing)
    uint32_t duration = 300; // 5 minutes default
    bool benchmark = false;
    bool use_pool = false;
    bool test_sha1 = false; // Add this
    bool test_bits = false; // Add this
    bool debug_mode = false; // Add this
    std::string pool_url;
    std::string worker_name;
    std::string target_hex;
    std::string message_hex;
    int num_streams = 4;
    int threads_per_block = 256;
    bool auto_tune = true;
};

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

void print_usage(const char *program_name) {
    std::cout << "SHA-1 OP_NET Miner\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --gpu <id>          GPU device ID (default: 0)\n";
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
    std::cout << "  " << program_name << " --benchmark --auto-tune\n";
}

MiningConfig parse_args(int argc, char *argv[]) {
    MiningConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--difficulty" && i + 1 < argc) {
            config.difficulty = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            config.duration = std::stoi(argv[++i]);
        } else if (arg == "--test-sha1") {
            config.test_sha1 = true;
        } else if (arg == "--test-bits") {
            config.test_bits = true;
        } else if (arg == "--debug") {
            config.debug_mode = true;
        } else if (arg == "--target" && i + 1 < argc) {
            config.target_hex = argv[++i];
        } else if (arg == "--message" && i + 1 < argc) {
            config.message_hex = argv[++i];
        } else if (arg == "--streams" && i + 1 < argc) {
            config.num_streams = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads_per_block = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--auto-tune") {
            config.auto_tune = true;
        } else if (arg == "--pool" && i + 1 < argc) {
            config.use_pool = true;
            config.pool_url = argv[++i];
        } else if (arg == "--worker" && i + 1 < argc) {
            config.worker_name = argv[++i];
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

// Auto-tune mining parameters
void auto_tune_parameters(MiningSystem::Config &config, int device_id) {
    std::cout << "Auto-tuning mining parameters...\n";

    gpuDeviceProp props;
    gpuGetDeviceProperties(&props, device_id);

    // Calculate optimal blocks based on architecture and SM count
    int blocks_per_sm;
    int optimal_threads;
    if (props.major >= 8) {
        // Ampere and newer (RTX 30xx, 40xx, A100, etc.)
        // These have 128 CUDA cores per SM
        blocks_per_sm = 16; // Can handle many blocks
        optimal_threads = 256; // 256 or 512 work well
    } else if (props.major == 7) {
        if (props.minor >= 5) {
            // Turing (RTX 20xx, T4)
            blocks_per_sm = 8;
            optimal_threads = 256;
        } else {
            // Volta (V100, Titan V)
            blocks_per_sm = 8;
            optimal_threads = 256;
        }
    } else if (props.major == 6) {
        // Pascal (GTX 10xx, P100)
        if (props.minor >= 1) {
            // GP102/GP104 (GTX 1080 Ti, 1080, 1070)
            blocks_per_sm = 8;
            optimal_threads = 256;
        } else {
            // GP100 (P100)
            blocks_per_sm = 4;
            optimal_threads = 256;
        }
    } else if (props.major == 5) {
        // Maxwell (GTX 9xx, GTX 750)
        blocks_per_sm = 4;
        optimal_threads = 128;
    } else {
        // Kepler and older
        blocks_per_sm = 2;
        optimal_threads = 128;
    }

    // Adjust based on register and shared memory limits
    int max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    int max_blocks_per_sm = max_threads_per_sm / optimal_threads;
    if (blocks_per_sm > max_blocks_per_sm) {
        blocks_per_sm = max_blocks_per_sm;
    }

    // Set configuration
    config.blocks_per_stream = props.multiProcessorCount * blocks_per_sm;
    config.threads_per_block = optimal_threads;

    // For very large GPUs, limit total blocks to avoid scheduling overhead
    int max_total_blocks = 2048; // Empirically determined
    if (config.blocks_per_stream > max_total_blocks) {
        config.blocks_per_stream = max_total_blocks;
    }

    // Number of streams based on GPU class
    if (props.multiProcessorCount >= 80) {
        // High-end GPUs (RTX 4090, A100, etc.)
        config.num_streams = 16;
    } else if (props.multiProcessorCount >= 40) {
        // Mid-high GPUs (RTX 3070, 2080, etc.)
        config.num_streams = 8;
    } else if (props.multiProcessorCount >= 20) {
        // Mid-range GPUs
        config.num_streams = 4;
    } else {
        // Entry-level GPUs
        config.num_streams = 2;
    }

    // Adjust streams based on available memory
    size_t free_mem, total_mem;
    gpuMemGetInfo(&free_mem, &total_mem);
    size_t mem_per_stream = sizeof(MiningResult) * config.result_buffer_size + (
                                config.blocks_per_stream * config.threads_per_block * sizeof(uint32_t) * 5);
    int max_streams_by_memory = free_mem / (mem_per_stream * 2); // Use at most 50% of free memory
    if (config.num_streams > max_streams_by_memory && max_streams_by_memory > 0) {
        config.num_streams = max_streams_by_memory;
    }

    // Result buffer size - balance between PCIe transfers and memory usage
    config.result_buffer_size = 128;

    // Special optimizations for specific GPUs
    std::string gpu_name = props.name;
    if (gpu_name.find("4090") != std::string::npos || gpu_name.find("4080") != std::string::npos) {
        // RTX 4090/4080 specific
        config.threads_per_block = 512; // These GPUs love high thread counts
        blocks_per_sm = 16;
        config.blocks_per_stream = props.multiProcessorCount * blocks_per_sm;
    } else if (gpu_name.find("A100") != std::string::npos || gpu_name.find("H100") != std::string::npos) {
        // Data center GPUs
        config.threads_per_block = 512;
        blocks_per_sm = 32; // These can handle extreme occupancy
        config.blocks_per_stream = props.multiProcessorCount * blocks_per_sm;
        config.num_streams = 32;
    }

    // Ensure we don't exceed device limits
    if (config.threads_per_block > props.maxThreadsPerBlock) {
        config.threads_per_block = props.maxThreadsPerBlock;
    }

    std::cout << "Auto-tuned configuration for " << props.name << ":\n";
    std::cout << "  Compute Capability: " << props.major << "." << props.minor << "\n";
    std::cout << "  SMs: " << props.multiProcessorCount << "\n";
    std::cout << "  Blocks per SM: " << blocks_per_sm << "\n";
    std::cout << "  Blocks per stream: " << config.blocks_per_stream << "\n";
    std::cout << "  Threads per block: " << config.threads_per_block << "\n";
    std::cout << "  Number of streams: " << config.num_streams << "\n";
    std::cout << "  Total concurrent threads: " <<
            (config.blocks_per_stream * config.threads_per_block * config.num_streams) << "\n\n";
}

// Run comprehensive benchmark
void run_benchmark(int gpu_id, bool auto_tune) {
    std::cout << "\n=== SHA-1 Near-Collision Mining Benchmark ===\n\n";

    // Initialize mining system with auto-tuned parameters
    MiningSystem::Config sys_config;
    sys_config.device_id = gpu_id;

    if (auto_tune) {
        auto_tune_parameters(sys_config, gpu_id);
    }

    g_mining_system = std::make_unique<MiningSystem>(sys_config);
    if (!g_mining_system->initialize()) {
        std::cerr << "Failed to initialize mining system\n";
        return;
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
        g_mining_system->runMiningLoop(job, 60);
        auto end = std::chrono::steady_clock::now();

        // Get statistics
        auto stats = g_mining_system->getStats();
        double duration = std::chrono::duration<double>(end - start).count();
        double hash_rate = stats.hash_rate / 1e9; // GH/s

        results.push_back(hash_rate);

        std::cout << "\nResults for difficulty " << diff << " bits:\n";
        std::cout << "  Hash rate: " << std::fixed << std::setprecision(2)
                << hash_rate << " GH/s\n";
        std::cout << "  Candidates found: " << stats.candidates_found << "\n";
        std::cout << "  Expected candidates: " << std::scientific
                << (stats.hashes_computed / std::pow(2.0, diff)) << "\n";
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(2)
                << (100.0 * stats.candidates_found * std::pow(2.0, diff) /
                    stats.hashes_computed) << "%\n";
    }

    // Print summary
    std::cout << "\n=== Benchmark Summary ===\n";
    std::cout << "Difficulty | Hash Rate (GH/s)\n";
    std::cout << "-----------|----------------\n";
    for (size_t i = 0; i < difficulties.size() && i < results.size(); i++) {
        std::cout << std::setw(10) << difficulties[i] << " | "
                << std::fixed << std::setprecision(2) << results[i] << "\n";
    }

    cleanup_mining_system();
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

// Main program
int main(int argc, char *argv[]) {
    // Set up signal handlers
    setup_signal_handlers();

    // Parse command line
    MiningConfig config = parse_args(argc, argv);

    std::cout << "+------------------------------------------+\n";
    std::cout << "|            SHA-1 OP_NET Miner            |\n";
    std::cout << "+------------------------------------------+\n\n";

    if (config.test_sha1) {
        std::cout << "Running SHA-1 tests...\n";
        if (!verify_sha1_implementation()) {
            std::cerr << "SHA-1 implementation verification failed!\n";
            return 1;
        }
    }
    // Verify SHA-1 implementation
    if (!verify_sha1_implementation()) {
        std::cerr << "SHA-1 implementation verification failed!\n";
        return 1;
    }
    std::cout << "SHA-1 implementation verified.\n\n";

    // Check CUDA availability
    int device_count;
    gpuGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    if (config.gpu_id >= device_count) {
        std::cerr << "Invalid GPU ID. Available GPUs: 0-" << (device_count - 1) << "\n";
        return 1;
    }

    // Print GPU information
    gpuDeviceProp props;
    gpuGetDeviceProperties(&props, config.gpu_id);
    std::cout << "Using GPU " << config.gpu_id << ": " << props.name << "\n";
    std::cout << "  Compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "  Memory: " << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
            << " GB\n";
    std::cout << "  SMs: " << props.multiProcessorCount << "\n\n";

    // Run benchmark if requested
    if (config.benchmark) {
        run_benchmark(config.gpu_id, config.auto_tune);
        return 0;
    }

    // Initialize mining system
    MiningSystem::Config sys_config;
    sys_config.device_id = config.gpu_id;
    sys_config.num_streams = config.num_streams;
    sys_config.threads_per_block = config.threads_per_block;

    if (config.auto_tune) {
        auto_tune_parameters(sys_config, config.gpu_id);
    }

    g_mining_system = std::make_unique<MiningSystem>(sys_config);
    if (!g_mining_system->initialize()) {
        std::cerr << "Failed to initialize mining system\n";
        return 1;
    }

    // Prepare message and target
    std::vector<uint8_t> message;
    std::vector<uint8_t> target;

    if (!config.message_hex.empty()) {
        if (config.message_hex.length() != 64) {
            std::cerr << "Message must be 64 hex characters (32 bytes)\n";
            cleanup_mining_system();
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
            cleanup_mining_system();
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
    std::cout << "  Success probability per hash: 2^-" << config.difficulty << "\n";
    std::cout << "  Expected time to find: " << std::scientific
            << (std::pow(2.0, config.difficulty) / (sys_config.blocks_per_stream *
                                                    sys_config.threads_per_block * NONCES_PER_THREAD * 1e9))
            << " seconds @ 1 GH/s\n\n";

    // Start mining
    auto start_time = std::chrono::steady_clock::now();
    g_mining_system->runMiningLoop(job, config.duration);
    auto end_time = std::chrono::steady_clock::now();

    // Print final statistics
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\nTotal runtime: " << std::fixed << std::setprecision(2)
            << duration << " seconds\n";

    // Cleanup
    cleanup_mining_system();

    return 0;
}
