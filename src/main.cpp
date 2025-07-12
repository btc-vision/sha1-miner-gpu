#include "sha1_miner.cuh"
#include "mining_system.hpp"
#include "multi_gpu_manager.hpp"
#include "../net/pool_integration.hpp"
#include "../logging/logger.hpp"
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
    int num_streams = 8;
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
    int debug_level = 2; // Default to INFO level
    bool no_colors = false; // Option to disable colors
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
    LOG_INFO("MAIN", Color::YELLOW, "Received signal ", sig_name, " (", sig, "), shutting down...", Color::RESET);
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
            ("debug", po::bool_switch(&config.debug_mode), "Enable debug mode")
            ("debug-level", po::value<int>(&config.debug_level)->default_value(2),
             "Debug level (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)")
            ("no-colors", po::bool_switch(&config.no_colors), "Disable colored output");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error &e) {
        LOG_ERROR("MAIN", "Error: ", e.what());
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
        std::cout << "  Debug mode:  " << argv[0] << " --pool ws://localhost:3333 --wallet YOUR_WALLET --debug\n";
        std::cout << "  Custom log:  " << argv[0] <<
                " --pool ws://localhost:3333 --wallet YOUR_WALLET --debug-level 3\n";
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
            LOG_ERROR("MAIN", "Error: --wallet is required for pool mining");
            std::exit(1);
        }
        if (config.pool_url.empty()) {
            LOG_ERROR("MAIN", "Error: --pool URL is required for pool mining");
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
    LOG_INFO("TUNE", "Auto-tuning mining parameters...");

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
            blocks_per_sm = 8;
            optimal_threads = 256;
        }
    } else if (props.major == 5) {
        // Maxwell (GTX 9xx, GTX 750)
        blocks_per_sm = 8;
        optimal_threads = 128;
    } else {
        // Kepler and older
        blocks_per_sm = 4;
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
        config.num_streams = 4;
    }

    // Adjust streams based on available memory
    size_t free_mem, total_mem;
    (void) gpuMemGetInfo(&free_mem, &total_mem);
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

    LOG_INFO("TUNE", "Auto-tuned configuration for ", Color::BRIGHT_CYAN, props.name, Color::RESET, ":");
    LOG_INFO("TUNE", "  Compute Capability: ", props.major, ".", props.minor);
    LOG_INFO("TUNE", "  SMs: ", props.multiProcessorCount);
    LOG_INFO("TUNE", "  Blocks per SM: ", blocks_per_sm);
    LOG_INFO("TUNE", "  Blocks per stream: ", config.blocks_per_stream);
    LOG_INFO("TUNE", "  Threads per block: ", config.threads_per_block);
    LOG_INFO("TUNE", "  Number of streams: ", config.num_streams);
    LOG_INFO("TUNE", "  Total concurrent threads: ",
             (config.blocks_per_stream * config.threads_per_block * config.num_streams));
}

// Run comprehensive benchmark
void run_benchmark(int gpu_id, bool auto_tune) {
    LOG_INFO("BENCH", Color::BRIGHT_YELLOW, "=== SHA-1 Near-Collision Mining Benchmark ===", Color::RESET);

    // Initialize mining system with auto-tuned parameters
    MiningSystem::Config sys_config;
    sys_config.device_id = gpu_id;

    if (auto_tune) {
        auto_tune_parameters(sys_config, gpu_id);
    }

    g_mining_system = std::make_unique<MiningSystem>(sys_config);
    if (!g_mining_system->initialize()) {
        LOG_ERROR("BENCH", "Failed to initialize mining system");
        return;
    }

    // Test different difficulty levels
    std::vector<uint32_t> difficulties = {60};
    std::vector<double> results;

    for (uint32_t diff: difficulties) {
        if (g_shutdown) break;

        LOG_INFO("BENCH", "Testing difficulty ", diff, " bits:");

        // Generate test job
        auto message = generate_secure_random_message();
        auto target_hash = calculate_sha1(message);

        MiningJob job = create_mining_job(message.data(), target_hash.data(), diff);

        auto start = std::chrono::steady_clock::now();
        g_mining_system->runMiningLoop(job);
        auto end = std::chrono::steady_clock::now();

        // Get statistics
        auto stats = g_mining_system->getStats();
        double duration = std::chrono::duration<double>(end - start).count();
        double hash_rate = stats.hash_rate / 1e9; // GH/s

        results.push_back(hash_rate);

        LOG_INFO("BENCH", "Results for difficulty ", diff, " bits:");
        LOG_INFO("BENCH", "  Hash rate: ", std::fixed, std::setprecision(2),
                 hash_rate, " GH/s");
        LOG_INFO("BENCH", "  Candidates found: ", stats.candidates_found);
        LOG_INFO("BENCH", "  Expected candidates: ", std::scientific,
                 (stats.hashes_computed / std::pow(2.0, diff)));
        LOG_INFO("BENCH", "  Efficiency: ", std::fixed, std::setprecision(2),
                 (100.0 * stats.candidates_found * std::pow(2.0, diff) /
                     stats.hashes_computed), "%");
    }

    // Print summary
    LOG_INFO("BENCH", Color::BRIGHT_GREEN, "=== Benchmark Summary ===", Color::RESET);
    LOG_INFO("BENCH", "Difficulty | Hash Rate (GH/s)");
    LOG_INFO("BENCH", "-----------|----------------");
    for (size_t i = 0; i < difficulties.size() && i < results.size(); i++) {
        LOG_INFO("BENCH", std::setw(10), difficulties[i], " | ",
                 std::fixed, std::setprecision(2), results[i]);
    }

    cleanup_mining_system();
}

// Verify SHA-1 implementation
bool verify_sha1_implementation() {
    LOG_INFO("TEST", "Verifying SHA-1 implementation...");

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
            LOG_ERROR("TEST", "FAILED: Message '", test.message, "'");
            LOG_ERROR("TEST", "  Expected: ", test.expected_hash);
            LOG_ERROR("TEST", "  Got:      ", hash_hex);
            all_passed = false;
        } else {
            LOG_INFO("TEST", Color::GREEN, "PASSED: '", test.message, "'", Color::RESET);
        }
    }

    return all_passed;
}

// Pool mining status display
void display_pool_stats(const MiningPool::PoolMiningSystem::PoolMiningStats &stats) {
    // Clear line with carriage return
    std::cout << "\r";

    if (!stats.connected) {
        std::cout << Color::YELLOW << "[POOL] " << Color::RED
                << "Disconnected - Attempting to reconnect...\n"
                << Color::RESET << std::flush;
    } else if (!stats.authenticated) {
        std::cout << Color::YELLOW << "[POOL] " << Color::YELLOW
                << "Connected - Authentication pending...\n"
                << Color::RESET << std::flush;
    } else {
        std::cout << Color::YELLOW << "[POOL] " << Color::RESET
                << "Worker: " << Color::BRIGHT_CYAN << stats.worker_id << Color::RESET << " | "
                << "Diff: " << Color::BRIGHT_MAGENTA << stats.current_difficulty << Color::RESET << " | "
                << "Hash: " << Color::BRIGHT_GREEN << std::fixed << std::setprecision(2)
                << stats.hashrate / 1e9 << " GH/s" << Color::RESET << " | "
                << "Shares: " << Color::BRIGHT_BLUE << stats.shares_accepted << "/"
                << stats.shares_submitted << Color::RESET;

        if (stats.shares_submitted > 0) {
            double success_rate = stats.share_success_rate * 100;
            std::string color = success_rate >= 95
                                    ? Color::BRIGHT_GREEN
                                    : success_rate >= 80
                                          ? Color::BRIGHT_YELLOW
                                          : Color::BRIGHT_RED;
            std::cout << " (" << color << std::setprecision(1)
                    << success_rate << "%" << Color::RESET << ")";
        }

        std::cout << " | Up: " << Color::DIM << stats.uptime.count() << "s\n"
                << Color::RESET;
    }

    std::cout << std::flush;
}

// Run pool mining
int run_pool_mining(const MiningConfig &config) {
    LOG_INFO("POOL", Color::BRIGHT_GREEN, "=== SHA-1 Pool Mining Mode ===", Color::RESET);
    LOG_INFO("POOL", "Pool: ", Color::BRIGHT_CYAN, config.pool_url, Color::RESET);
    LOG_INFO("POOL", "Wallet: ", Color::BRIGHT_YELLOW, config.pool_wallet, Color::RESET);
    LOG_INFO("POOL", "Worker: ", Color::BRIGHT_MAGENTA, config.worker_name, Color::RESET);

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
            LOG_ERROR("POOL", "Failed to start pool mining");
            return 1;
        }

        LOG_INFO("POOL", "Mining started with ", priority, " pool(s)");
        LOG_INFO("POOL", "Press Ctrl+C to stop mining");

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
            LOG_ERROR("POOL", "Failed to start pool mining");
            return 1;
        }

        LOG_INFO("POOL", "Mining started");
        LOG_INFO("POOL", "Press Ctrl+C to stop mining");

        // Monitor and display stats
        auto last_stats_time = std::chrono::steady_clock::now();
        while (!g_shutdown) {
            if (g_shutdown) {
                LOG_ERROR("MAIN", "g_shutdown was set! Breaking main loop");
                // Add stack trace or more info here
                break;
            }

            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_time >= std::chrono::seconds(1)) {
                auto stats = pool_mining->get_stats();
                display_pool_stats(stats);
                last_stats_time = now;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        LOG_INFO("MAIN", "Exited main loop, g_shutdown=", g_shutdown.load());

        pool_mining->stop();
    }

    LOG_INFO("POOL", "Pool mining stopped.");
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

    // Configure logging based on command line options
    MiningPool::Logger::set_level(static_cast<MiningPool::LogLevel>(config.debug_level));
    MiningPool::Logger::enable_colors(!config.no_colors);

    // If debug mode is explicitly enabled, set to DEBUG level
    if (config.debug_mode) {
        MiningPool::Logger::set_level(MiningPool::LogLevel::DEBUG);
    }

    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "+------------------------------------------+", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "|            SHA-1 OP_NET Miner            |", Color::RESET);
    LOG_INFO("MAIN", Color::BRIGHT_CYAN, "+------------------------------------------+", Color::RESET);
    LOG_INFO("MAIN", "Debug level: ", config.debug_level,
             " (", config.debug_mode ? "DEBUG MODE" : "NORMAL", ")");

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
    std::vector<int> gpu_ids_to_use;
    if (config.use_all_gpus) {
        for (int i = 0; i < device_count; i++) {
            gpu_ids_to_use.push_back(i);
        }
        LOG_INFO("MAIN", "Using all ", device_count, " available GPUs");
    } else if (!config.gpu_ids.empty()) {
        gpu_ids_to_use = config.gpu_ids;
        // Validate GPU IDs
        for (int id: gpu_ids_to_use) {
            if (id >= device_count || id < 0) {
                LOG_ERROR("MAIN", "Invalid GPU ID: ", id, ". Available GPUs: 0-",
                          (device_count - 1));
                return 1;
            }
        }
        LOG_INFO("MAIN", "Using ", gpu_ids_to_use.size(), " specified GPU(s)");
    } else {
        // Default to single GPU
        gpu_ids_to_use.push_back(config.gpu_id);
        if (config.gpu_id >= device_count) {
            LOG_ERROR("MAIN", "Invalid GPU ID. Available GPUs: 0-", (device_count - 1));
            return 1;
        }
    }

    // Print GPU information
    LOG_INFO("MAIN", "\nGPU Information:");
    LOG_INFO("MAIN", "=====================================");
    for (int id: gpu_ids_to_use) {
        gpuDeviceProp props;
        gpuGetDeviceProperties(&props, id);
        LOG_INFO("MAIN", "  GPU ", id, ": ", Color::BRIGHT_CYAN, props.name, Color::RESET);
        LOG_INFO("MAIN", "    Compute capability: ", props.major, ".", props.minor);
        LOG_INFO("MAIN", "    Memory: ", std::fixed, std::setprecision(2),
                 (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)), " GB");
        LOG_INFO("MAIN", "    SMs/CUs: ", props.multiProcessorCount);
    }
    LOG_INFO("MAIN", "");

    // Check if pool mining is requested
    if (config.use_pool) {
        return run_pool_mining(config);
    }

    // Run benchmark if requested
    if (config.benchmark) {
        if (gpu_ids_to_use.size() > 1) {
            LOG_WARN("MAIN", "Multi-GPU benchmark not yet implemented. Using GPU 0 only.");
            run_benchmark(gpu_ids_to_use[0], config.auto_tune);
        } else {
            run_benchmark(gpu_ids_to_use[0], config.auto_tune);
        }
        return 0;
    }

    // Solo mining mode
    LOG_INFO("MAIN", "Running in solo mining mode");

    // Prepare message and target
    std::vector<uint8_t> message;
    std::vector<uint8_t> target;

    if (!config.message_hex.empty()) {
        if (config.message_hex.length() != 64) {
            LOG_ERROR("MAIN", "Message must be 64 hex characters (32 bytes)");
            return 1;
        }
        message = hex_to_bytes(config.message_hex);
    } else {
        message = generate_secure_random_message();
        LOG_INFO("MAIN", "Generated random message: ", bytes_to_hex(message));
    }

    if (!config.target_hex.empty()) {
        if (config.target_hex.length() != 40) {
            LOG_ERROR("MAIN", "Target must be 40 hex characters (20 bytes)");
            return 1;
        }
        target = hex_to_bytes(config.target_hex);
    } else {
        target = calculate_sha1(message);
        LOG_INFO("MAIN", "Target SHA-1: ", bytes_to_hex(target));
    }

    // Create mining job
    MiningJob job = create_mining_job(message.data(), target.data(), config.difficulty);

    LOG_INFO("MAIN", "\nMining Configuration:");
    LOG_INFO("MAIN", "  Difficulty: ", config.difficulty, " bits must match");
    LOG_INFO("MAIN", "  Duration: ", config.duration, " seconds");
    LOG_INFO("MAIN", "  Success probability per hash: 2^-", config.difficulty);

    // Start mining
    auto start_time = std::chrono::steady_clock::now();

    // Use multi-GPU manager if multiple GPUs selected
    if (gpu_ids_to_use.size() > 1) {
        MultiGPUManager multi_gpu_manager;
        if (!multi_gpu_manager.initialize(gpu_ids_to_use)) {
            LOG_ERROR("MAIN", "Failed to initialize multi-GPU manager");
            return 1;
        }

        // Run multi-GPU mining infinitely
        LOG_INFO("MAIN", "Starting infinite multi-GPU mining. Press Ctrl+C to stop.");
        multi_gpu_manager.runMining(job);
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
            LOG_ERROR("MAIN", "Failed to initialize mining system");
            return 1;
        }

        LOG_INFO("MAIN", "  Expected time to find: ", std::scientific,
                 (std::pow(2.0, config.difficulty) / (sys_config.blocks_per_stream *
                     sys_config.threads_per_block * NONCES_PER_THREAD * 1e9)),
                 " seconds @ 1 GH/s");

        LOG_INFO("MAIN", "Starting infinite mining. Press Ctrl+C to stop.");
        g_mining_system->runMiningLoop(job);
        cleanup_mining_system();
    }

    auto end_time = std::chrono::steady_clock::now();

    // Print final statistics
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    LOG_INFO("MAIN", "\nTotal runtime: ", std::fixed, std::setprecision(2),
             duration, " seconds");

    return 0;
}
