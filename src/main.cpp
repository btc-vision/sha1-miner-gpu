#include "sha1_miner.cuh"
#include "cxxsha1.hpp"
#include "globals.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#define SIGBREAK 21  // Windows-specific signal
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

// Forward declarations
void run_mining_loop(MiningJob job, uint32_t duration_seconds);

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

// Set up signal handlers
void setup_signal_handlers() {
    // Common signals
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    // Windows-specific signals
    std::signal(SIGBREAK, signal_handler);
#else
    // Unix-specific signals
    std::signal(SIGHUP, signal_handler);
    std::signal(SIGQUIT, signal_handler);

    // Ignore SIGPIPE (broken pipe)
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

// Parse command line arguments
struct Config {
    int gpu_id = 0;
    uint32_t difficulty = 120; // Default: 120 bits must match
    uint32_t duration = 60; // Default: 60 seconds
    bool benchmark = false;
    std::string target_hex;
    std::string message_hex;
};

void print_usage(const char *program_name) {
    std::cout << "SHA-1 Near-Collision Miner v2.0\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --gpu <id>          GPU device ID (default: 0)\n";
    std::cout << "  --difficulty <bits> Number of bits that must match (default: 120)\n";
    std::cout << "  --duration <sec>    Mining duration in seconds (default: 60)\n";
    std::cout << "  --target <hex>      Target hash in hex (40 chars)\n";
    std::cout << "  --message <hex>     Base message in hex (64 chars)\n";
    std::cout << "  --benchmark         Run performance benchmark\n";
    std::cout << "  --help              Show this help\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --gpu 0 --difficulty 100 --duration 300\n";
}

Config parse_args(int argc, char *argv[]) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_id = std::stoi(argv[++i]);
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

// Platform-independent high-resolution timer
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }
};


// Convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    std::vector<uint8_t> bytes;

    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        bytes.push_back(static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16)));
    }

    return bytes;
}

// Generate random message for testing
std::vector<uint8_t> generate_random_message() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<uint8_t> message(32);
    for (auto &byte: message) {
        byte = static_cast<uint8_t>(dis(gen));
    }

    return message;
}

// Calculate SHA-1 hash
std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &message) {
    SHA1 sha1;
    std::string msg_str(message.begin(), message.end());
    sha1.update(msg_str);
    std::string hex_result = sha1.final();

    // Convert hex string to binary
    std::vector<uint8_t> hash(20);
    for (int i = 0; i < 20; i++) {
        std::string byte = hex_result.substr(i * 2, 2);
        hash[i] = static_cast<uint8_t>(std::stoi(byte, nullptr, 16));
    }

    return hash;
}

// Print system information
void print_system_info() {
    std::cout << "System Information:\n";

#ifdef _WIN32
    std::cout << "  Platform: Windows\n";
#elif __linux__
    std::cout << "  Platform: Linux\n";
#elif __APPLE__
    std::cout << "  Platform: macOS\n";
#else
    std::cout << "  Platform: Unknown Unix\n";
#endif

    std::cout << "  CPU Threads: " << std::thread::hardware_concurrency() << "\n";

    // CUDA information
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "  CUDA Devices: " << device_count << "\n";

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << "    GPU " << i << ": " << props.name
                << " (SM " << props.major << "." << props.minor << ")\n";
    }
    std::cout << "\n";
}

// Run benchmark mode
void run_benchmark(int gpu_id) {
    std::cout << "\n=== SHA-1 Near-Collision Mining Benchmark ===\n\n";

    if (!init_mining_system(gpu_id)) {
        std::cerr << "Failed to initialize mining system\n";
        return;
    }

    // Test different difficulty levels
    std::vector<uint32_t> difficulties = {80, 90, 100, 110, 120, 130};

    for (uint32_t diff: difficulties) {
        if (g_shutdown) break;

        std::cout << "\nTesting difficulty " << diff << " bits:\n";

        // Create test job
        auto message = generate_random_message();
        auto target = calculate_sha1(message);

        MiningJob job = create_mining_job(message.data(), target.data(), diff);

        // Run for 30 seconds
        run_mining_loop(job, 30);
    }

    cleanup_mining_system();
}

// Main program
int main(int argc, char *argv[]) {
    // Set up signal handlers
    setup_signal_handlers();

    // Parse command line
    Config config = parse_args(argc, argv);

    std::cout << "+------------------------------------------+\n";
    std::cout << "|    SHA-1 Near-Collision Miner v2.0       |\n";
    std::cout << "+------------------------------------------+\n\n";

    // Print system information
    print_system_info();

    // Run benchmark if requested
    if (config.benchmark) {
        run_benchmark(config.gpu_id);
        return 0;
    }

    // Initialize mining system
    if (!init_mining_system(config.gpu_id)) {
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
        message = generate_random_message();
        std::cout << "Generated random message: ";
        for (uint8_t b: message) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(b);
        }
        std::cout << "\n";
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
        std::cout << "Target SHA-1: ";
        for (uint8_t b: target) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(b);
        }
        std::cout << "\n";
    }

    // Create mining job
    MiningJob job = create_mining_job(message.data(), target.data(), config.difficulty);

    std::cout << "\nMining Configuration:\n";
    std::cout << "  Difficulty: " << config.difficulty << " bits must match\n";
    std::cout << "  Duration: " << config.duration << " seconds\n";
    std::cout << "  Success probability per hash: 2^-" << config.difficulty << "\n\n";

    // Start timing
    Timer timer;
    timer.start();

    // Run mining
    run_mining_loop(job, config.duration);

    // Print final timing
    std::cout << "\nTotal runtime: " << std::fixed << std::setprecision(2)
            << timer.elapsed_seconds() << " seconds\n";

    // Cleanup
    cleanup_mining_system();

    return 0;
}
