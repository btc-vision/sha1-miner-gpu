#include "sha1_miner.cuh"
#include "mining_system.hpp"
#include "multi_gpu_manager.hpp"
#include "../net/pool_integration.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <csignal>
#include <chrono>
#include <thread>
#include <sstream>
#include <boost/program_options.hpp>

#ifdef _WIN32
#include <windows.h>
#define SIGBREAK 21
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#include <fcntl.h>

void setup_console_encoding() {
    // Set console code page to UTF-8
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);

    // Enable UTF-8 for C++ streams
    std::locale::global(std::locale(""));
}
#else
void setup_console_encoding() {
    // Unix systems usually handle UTF-8 properly by default
    std::locale::global(std::locale(""));
}
#endif


namespace po = boost::program_options;

// Advanced configuration for production mining
struct MiningConfig {
    // GPU configuration
    int gpu_id = 0;
    std::vector<int> gpu_ids;
    bool use_all_gpus = false;
    // Solo mining configuration
    uint32_t difficulty = 50;
    uint32_t duration = 300;
    std::string target_hex;
    std::string message_hex;

    // Performance configuration
    int num_streams = 4;
    int threads_per_block = 256;
    bool auto_tune = true;

    // Pool mining configuration
    bool use_pool = false;
    std::string pool_url;
    std::string pool_wallet;
    std::string worker_name;
    std::string pool_password = "x";
    std::vector<std::string> backup_pools;
    bool enable_pool_failover = true;

    // Operating modes
    bool benchmark = false;
    bool test_sha1 = false;
    bool test_bits = false;
    bool debug_mode = false;
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
        case SIGHUP: sig_name = "SIGHUP"; break;
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

MiningConfig parse_args(int argc, char *argv[]) {
    MiningConfig config;

    // Set default worker name to hostname
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        config.worker_name = hostname;
    } else {
        config.worker_name = "default_worker";
    }

    po::options_description desc("SHA-1 OP_NET Miner Options");
    desc.add_options()
            ("help,h", "Show this help message")
            // GPU options
            ("gpu", po::value<int>(&config.gpu_id)->default_value(0), "GPU device ID")
            ("all-gpus", po::bool_switch(&config.use_all_gpus), "Use all available GPUs")
            ("gpus", po::value<std::string>(), "Use specific GPUs (e.g., 0,1,2)")
            // Solo mining options
            ("difficulty", po::value<uint32_t>(&config.difficulty)->default_value(50),
             "Number of bits that must match")
            ("duration", po::value<uint32_t>(&config.duration)->default_value(300),
             "Mining duration in seconds")
            ("target", po::value<std::string>(&config.target_hex), "Target hash in hex (40 chars)")
            ("message", po::value<std::string>(&config.message_hex), "Base message in hex (64 chars)")
            // Pool mining options
            ("pool", po::value<std::string>(&config.pool_url),
             "Pool URL (ws://host:port or wss://host:port)")
            ("wallet", po::value<std::string>(&config.pool_wallet),
             "Wallet address for pool mining")
            ("worker", po::value<std::string>(&config.worker_name),
             "Worker name (default: hostname)")
            ("pool-pass", po::value<std::string>(&config.pool_password)->default_value("x"),
             "Pool password")
            ("backup-pool", po::value<std::vector<std::string> >(&config.backup_pools)->multitoken(),
             "Backup pool URLs for failover")
            ("no-failover", po::bool_switch(), "Disable automatic pool failover")
            // Performance options
            ("streams", po::value<int>(&config.num_streams)->default_value(4),
             "Number of CUDA streams")
            ("threads", po::value<int>(&config.threads_per_block)->default_value(256),
             "Threads per block")
            ("auto-tune", po::bool_switch(&config.auto_tune),
             "Auto-tune for optimal performance")
            // Other options
            ("benchmark", po::bool_switch(&config.benchmark), "Run performance benchmark")
            ("test-sha1", po::bool_switch(&config.test_sha1), "Test SHA-1 implementation")
            ("test-bits", po::bool_switch(&config.test_bits), "Test bit matching")
            ("debug", po::bool_switch(&config.debug_mode), "Enable debug mode");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        std::exit(1);
    }

    if (vm.count("help")) {
        std::cout << "SHA-1 OP_NET Miner\n\n";
        std::cout << desc << "\n";
        std::cout << "Examples:\n";
        std::cout << "  Solo mining: " << argv[0] << " --gpu 0 --difficulty 45 --duration 600\n";
        std::cout << "  Pool mining: " << argv[0] <<
                " --pool ws://pool.example.com:3333 --wallet YOUR_WALLET --worker rig1\n";
        std::cout << "  Multi-GPU:   " << argv[0] <<
                " --all-gpus --pool wss://secure.pool.com:443 --wallet YOUR_WALLET\n";
        std::cout << "  Benchmark:   " << argv[0] << " --benchmark --auto-tune\n";
        std::exit(0);
    }

    // Parse GPU list
    if (vm.count("gpus")) {
        std::string gpu_list = vm["gpus"].as<std::string>();
        std::stringstream ss(gpu_list);
        std::string token;
        while (std::getline(ss, token, ',')) {
            config.gpu_ids.push_back(std::stoi(token));
        }
    }

    // Disable failover if requested
    if (vm.count("no-failover")) {
        config.enable_pool_failover = false;
    }

    // Enable pool mode if pool URL is specified
    if (!config.pool_url.empty()) {
        config.use_pool = true;
    }

    // Validate pool configuration
    if (config.use_pool) {
        if (config.pool_wallet.empty()) {
            std::cerr << "Error: --wallet is required for pool mining\n";
            std::exit(1);
        }
        if (config.pool_url.empty()) {
            std::cerr << "Error: --pool URL is required for pool mining\n";
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
        blocks_per_sm = 16;
        optimal_threads = 256;
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
            blocks_per_sm = 8;
            optimal_threads = 256;
        } else {
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
    int max_total_blocks = 2048;
    if (config.blocks_per_stream > max_total_blocks) {
        config.blocks_per_stream = max_total_blocks;
    }

    // Number of streams based on GPU class
    if (props.multiProcessorCount >= 80) {
        config.num_streams = 16;
    } else if (props.multiProcessorCount >= 40) {
        config.num_streams = 8;
    } else if (props.multiProcessorCount >= 20) {
        config.num_streams = 4;
    } else {
        config.num_streams = 2;
    }

    // Adjust streams based on available memory
    size_t free_mem, total_mem;
    gpuMemGetInfo(&free_mem, &total_mem);
    size_t mem_per_stream = sizeof(MiningResult) * config.result_buffer_size +
                            (config.blocks_per_stream * config.threads_per_block * sizeof(uint32_t) * 5);
    int max_streams_by_memory = free_mem / (mem_per_stream * 2);
    if (config.num_streams > max_streams_by_memory && max_streams_by_memory > 0) {
        config.num_streams = max_streams_by_memory;
    }

    config.result_buffer_size = 128;

    // Special optimizations for specific GPUs
    std::string gpu_name = props.name;
    if (gpu_name.find("4090") != std::string::npos || gpu_name.find("4080") != std::string::npos) {
        config.threads_per_block = 512;
        blocks_per_sm = 16;
        config.blocks_per_stream = props.multiProcessorCount * blocks_per_sm;
    } else if (gpu_name.find("A100") != std::string::npos || gpu_name.find("H100") != std::string::npos) {
        config.threads_per_block = 512;
        blocks_per_sm = 32;
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

// Pool mining status display
void display_pool_stats(const MiningPool::PoolMiningSystem::PoolMiningStats &stats) {
    std::cout << "\r";
    std::cout << "[POOL] ";

    if (!stats.connected) {
        std::cout << "Disconnected - Attempting to reconnect..." << std::endl;
    } else if (!stats.authenticated) {
        std::cout << "Connected - Authentication pending..." << std::endl;
    } else {
        std::cout << "Worker: " << stats.worker_id << " | ";
        std::cout << "Diff: " << stats.current_difficulty << " | ";
        std::cout << "Hash: " << std::fixed << std::setprecision(2)
                << stats.hashrate / 1e9 << " GH/s | ";
        std::cout << "Shares: " << stats.shares_accepted << "/"
                << stats.shares_submitted;

        if (stats.shares_submitted > 0) {
            std::cout << " (" << std::setprecision(1)
                    << stats.share_success_rate * 100 << "%)";
        }

        std::cout << " | Up: " << stats.uptime.count() << "s     ";
    }

    std::cout << std::flush;
}

// Run pool mining
int run_pool_mining(const MiningConfig &config) {
    std::cout << "\n=== SHA-1 Pool Mining Mode ===\n";
    std::cout << "Pool: " << config.pool_url << "\n";
    std::cout << "Wallet: " << config.pool_wallet << "\n";
    std::cout << "Worker: " << config.worker_name << "\n\n";

    // Create pool configuration
    MiningPool::PoolConfig pool_config;
    pool_config.url = config.pool_url;
    pool_config.username = config.pool_wallet;
    pool_config.worker_name = config.worker_name;
    pool_config.password = config.pool_password;
    pool_config.auth_method = MiningPool::AuthMethod::WORKER_PASS;

    // Auto-detect TLS from URL
    pool_config.use_tls = (config.pool_url.find("wss://") == 0);
    pool_config.verify_server_cert = true;

    // Connection settings
    pool_config.keepalive_interval_s = 30;
    pool_config.response_timeout_ms = 10000;
    pool_config.reconnect_delay_ms = 5000;
    pool_config.max_reconnect_delay_ms = 60000;
    pool_config.reconnect_attempts = -1; // Infinite retries

    // Create mining configuration
    MiningPool::PoolMiningSystem::Config mining_config;
    mining_config.pool_config = pool_config;
    mining_config.dev_fee_percent = 0.0; // No dev fee

    // Set GPU configuration
    if (config.use_all_gpus) {
        int device_count;
        gpuGetDeviceCount(&device_count);
        for (int i = 0; i < device_count; i++) {
            mining_config.gpu_ids.push_back(i);
        }
        mining_config.use_all_gpus = true;
    } else if (!config.gpu_ids.empty()) {
        mining_config.gpu_ids = config.gpu_ids;
    } else {
        mining_config.gpu_ids.push_back(config.gpu_id);
    }

    // Handle multiple pools with failover
    if (!config.backup_pools.empty() && config.enable_pool_failover) {
        auto multi_pool_manager = std::make_unique<MiningPool::MultiPoolManager>();

        // Add primary pool
        multi_pool_manager->add_pool("primary", pool_config, 0);

        // Add backup pools
        int priority = 1;
        for (const auto &backup_url: config.backup_pools) {
            auto backup_config = pool_config;
            backup_config.url = backup_url;
            backup_config.use_tls = (backup_url.find("wss://") == 0);
            multi_pool_manager->add_pool("backup_" + std::to_string(priority),
                                         backup_config, priority);
            priority++;
        }

        // Start mining with failover
        if (!multi_pool_manager->start_mining(mining_config)) {
            std::cerr << "Failed to start pool mining\n";
            return 1;
        }

        std::cout << "Mining started with " << (priority) << " pool(s)\n";
        std::cout << "Press Ctrl+C to stop mining\n\n";

        // Monitor and display stats
        while (!g_shutdown) {
            auto all_stats = multi_pool_manager->get_all_stats();
            auto active_pool = multi_pool_manager->get_active_pool();

            if (all_stats.count(active_pool) > 0) {
                display_pool_stats(all_stats[active_pool]);
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        multi_pool_manager->stop_mining();
    } else {
        // Single pool mining
        auto pool_mining = std::make_unique<MiningPool::PoolMiningSystem>(mining_config);

        if (!pool_mining->start()) {
            std::cerr << "Failed to start pool mining\n";
            return 1;
        }

        std::cout << "Mining started\n";
        std::cout << "Press Ctrl+C to stop mining\n\n";

        // Monitor and display stats
        auto last_stats_time = std::chrono::steady_clock::now();
        while (!g_shutdown) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_time >= std::chrono::seconds(1)) {
                auto stats = pool_mining->get_stats();
                display_pool_stats(stats);
                last_stats_time = now;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\n\nStopping mining...\n";
        pool_mining->stop();
    }

    std::cout << "\nPool mining stopped.\n";
    return 0;
}

// Main program
int main(int argc, char *argv[]) {
    // Set up UTF-8 encoding for console output
    setup_console_encoding();

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
        return 0;
    }

    // Verify SHA-1 implementation
    if (!verify_sha1_implementation()) {
        std::cerr << "SHA-1 implementation verification failed!\n";
        return 1;
    }
    std::cout << "SHA-1 implementation verified.\n\n";

    // Check GPU availability
    int device_count;
    gpuGetDeviceCount(&device_count);

    if (device_count == 0) {
#ifdef USE_HIP
        std::cerr << "No AMD/HIP devices found!\n";
#else
        std::cerr << "No CUDA devices found!\n";
#endif
        return 1;
    }

    // Determine which GPUs to use
    std::vector<int> gpu_ids_to_use;
    if (config.use_all_gpus) {
        for (int i = 0; i < device_count; i++) {
            gpu_ids_to_use.push_back(i);
        }
        std::cout << "Using all " << device_count << " available GPUs\n";
    } else if (!config.gpu_ids.empty()) {
        gpu_ids_to_use = config.gpu_ids;
        // Validate GPU IDs
        for (int id: gpu_ids_to_use) {
            if (id >= device_count || id < 0) {
                std::cerr << "Invalid GPU ID: " << id << ". Available GPUs: 0-"
                        << (device_count - 1) << "\n";
                return 1;
            }
        }
        std::cout << "Using " << gpu_ids_to_use.size() << " specified GPU(s)\n";
    } else {
        // Default to single GPU
        gpu_ids_to_use.push_back(config.gpu_id);
        if (config.gpu_id >= device_count) {
            std::cerr << "Invalid GPU ID. Available GPUs: 0-" << (device_count - 1) << "\n";
            return 1;
        }
    }

    // Print GPU information
    std::cout << "\nGPU Information:\n";
    std::cout << "=====================================\n";
    for (int id: gpu_ids_to_use) {
        gpuDeviceProp props;
        gpuGetDeviceProperties(&props, id);
        std::cout << "  GPU " << id << ": " << props.name << "\n";
        std::cout << "    Compute capability: " << props.major << "." << props.minor << "\n";
        std::cout << "    Memory: " << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
                << " GB\n";
        std::cout << "    SMs/CUs: " << props.multiProcessorCount << "\n";
    }
    std::cout << "\n";

    // Check if pool mining is requested
    if (config.use_pool) {
        return run_pool_mining(config);
    }

    // Run benchmark if requested
    if (config.benchmark) {
        if (gpu_ids_to_use.size() > 1) {
            std::cout << "Multi-GPU benchmark not yet implemented. Using GPU 0 only.\n";
            run_benchmark(gpu_ids_to_use[0], config.auto_tune);
        } else {
            run_benchmark(gpu_ids_to_use[0], config.auto_tune);
        }
        return 0;
    }

    // Solo mining mode
    std::cout << "Running in solo mining mode\n";

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

    // Use multi-GPU manager if multiple GPUs selected
    if (gpu_ids_to_use.size() > 1) {
        MultiGPUManager multi_gpu_manager;
        if (!multi_gpu_manager.initialize(gpu_ids_to_use)) {
            std::cerr << "Failed to initialize multi-GPU manager\n";
            return 1;
        }

        // Run multi-GPU mining
        multi_gpu_manager.runMining(job, config.duration);
    } else {
        // Single GPU path
        MiningSystem::Config sys_config;
        sys_config.device_id = gpu_ids_to_use[0];
        sys_config.num_streams = config.num_streams;
        sys_config.threads_per_block = config.threads_per_block;

        if (config.auto_tune) {
            auto_tune_parameters(sys_config, gpu_ids_to_use[0]);
        }

        g_mining_system = std::make_unique<MiningSystem>(sys_config);
        if (!g_mining_system->initialize()) {
            std::cerr << "Failed to initialize mining system\n";
            return 1;
        }

        std::cout << "  Expected time to find: " << std::scientific
                << (std::pow(2.0, config.difficulty) / (sys_config.blocks_per_stream *
                                                        sys_config.threads_per_block * NONCES_PER_THREAD * 1e9))
                << " seconds @ 1 GH/s\n\n";

        g_mining_system->runMiningLoop(job, config.duration);
        cleanup_mining_system();
    }

    auto end_time = std::chrono::steady_clock::now();

    // Print final statistics
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\nTotal runtime: " << std::fixed << std::setprecision(2)
            << duration << " seconds\n";

    return 0;
}
