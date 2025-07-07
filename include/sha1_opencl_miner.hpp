// sha1_opencl_miner.hpp - OpenCL version of SHA-1 near-collision miner
#ifndef SHA1_OPENCL_MINER_HPP
#define SHA1_OPENCL_MINER_HPP

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>

// Mining configuration
#define MAX_CANDIDATES_PER_BATCH 1024
#define NONCES_PER_THREAD 256
#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_BLOCKS_PER_STREAM 256

// Result structure for found candidates
struct MiningResult {
    uint64_t nonce;
    uint32_t hash[5];
    uint32_t matching_bits;
    uint32_t difficulty_score;
};

// Mining job structure
struct MiningJob {
    uint8_t base_message[32];
    uint32_t target_hash[5];
    uint32_t difficulty;
    uint64_t nonce_offset;
};

// OpenCL-specific structures
struct OpenCLMiningContext {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    // Device memory
    cl_mem d_base_message;
    cl_mem d_target_hash;
    cl_mem d_results;
    cl_mem d_result_count;
    cl_mem d_nonces_processed;

    // Host memory
    MiningResult *h_results;
    uint32_t *h_result_count;
    uint64_t *h_nonces_processed;

    // Configuration
    size_t local_work_size;
    size_t global_work_size;
    uint32_t result_capacity;

    OpenCLMiningContext() : context(nullptr), queue(nullptr), program(nullptr),
                            kernel(nullptr), device(nullptr), d_base_message(nullptr),
                            d_target_hash(nullptr), d_results(nullptr),
                            d_result_count(nullptr), d_nonces_processed(nullptr),
                            h_results(nullptr), h_result_count(nullptr),
                            h_nonces_processed(nullptr), local_work_size(256),
                            global_work_size(65536), result_capacity(MAX_CANDIDATES_PER_BATCH) {
    }
};

// Platform information
struct OpenCLPlatformInfo {
    cl_platform_id platform;
    std::string name;
    std::string vendor;
    std::string version;
    std::vector<cl_device_id> devices;
};

// Device information
struct OpenCLDeviceInfo {
    cl_device_id device;
    std::string name;
    std::string vendor;
    cl_device_type type;
    cl_uint compute_units;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    size_t max_work_group_size;
    cl_uint max_clock_frequency;
};

// Mining statistics
struct MiningStats {
    uint64_t hashes_computed;
    uint64_t candidates_found;
    uint32_t best_match_bits;
    double hash_rate;
};

// Best result tracker
class BestResultTracker {
public:
    BestResultTracker() : best_bits_(0) {
    }

    bool isNewBest(uint32_t matching_bits) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (matching_bits > best_bits_) {
            best_bits_ = matching_bits;
            return true;
        }
        return false;
    }

    uint32_t getBestBits() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return best_bits_;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        best_bits_ = 0;
    }

private:
    mutable std::mutex mutex_;
    uint32_t best_bits_;
};

// OpenCL Mining System
class OpenCLMiningSystem {
public:
    struct Config {
        int device_index;
        int platform_index;
        size_t local_work_size;
        size_t global_work_size;
        size_t result_buffer_size;
        bool auto_tune;

        Config() : device_index(0), platform_index(-1),
                   local_work_size(DEFAULT_THREADS_PER_BLOCK),
                   global_work_size(DEFAULT_BLOCKS_PER_STREAM * DEFAULT_THREADS_PER_BLOCK),
                   result_buffer_size(MAX_CANDIDATES_PER_BATCH),
                   auto_tune(true) {
        }
    };

    explicit OpenCLMiningSystem(const Config &config = Config());

    ~OpenCLMiningSystem();

    bool initialize();

    void runMiningLoop(const MiningJob &job, uint32_t duration_seconds);

    MiningStats getStats() const;

    // For multi-GPU support
    uint64_t runSingleBatch(const MiningJob &job);

    std::vector<MiningResult> getLastResults();

    const Config &getConfig() const { return config_; }

    void resetState();

    // Platform and device enumeration
    static std::vector<OpenCLPlatformInfo> enumeratePlatforms();

    static std::vector<OpenCLDeviceInfo> enumerateDevices(cl_platform_id platform);

private:
    Config config_;
    OpenCLMiningContext context_;
    OpenCLDeviceInfo device_info_;

    // Performance tracking
    std::atomic<uint64_t> total_hashes_{0};
    std::atomic<uint64_t> total_candidates_{0};
    std::chrono::steady_clock::time_point start_time_;
    BestResultTracker best_tracker_;

    // Private methods
    bool initializeOpenCL();

    bool buildKernel();

    bool allocateBuffers();

    void cleanup();

    void processResults();

    void autoTuneParameters();

    std::string getKernelSource() const;

    // Error handling
    static const char *getErrorString(cl_int error);

    void checkError(cl_int error, const std::string &operation);
};

// Multi-GPU Manager for OpenCL
class OpenCLMultiGPUManager {
public:
    struct GPUWorker {
        int platform_index;
        int device_index;
        std::unique_ptr<OpenCLMiningSystem> mining_system;
        std::unique_ptr<std::thread> worker_thread;
        std::atomic<uint64_t> hashes_computed{0};
        std::atomic<uint64_t> candidates_found{0};
        std::atomic<uint32_t> best_match_bits{0};
    };

    OpenCLMultiGPUManager();

    ~OpenCLMultiGPUManager();

    bool initialize(const std::vector<std::pair<int, int> > &platform_device_pairs);

    void runMining(const MiningJob &job, uint32_t duration_seconds);

    void printCombinedStats();

    size_t getNumGPUs() const { return workers_.size(); }

private:
    std::vector<std::unique_ptr<GPUWorker> > workers_;
    std::mutex stats_mutex_;
    std::atomic<bool> shutdown_{false};
    std::chrono::steady_clock::time_point start_time_;
    BestResultTracker global_best_tracker_;
    std::atomic<uint64_t> global_nonce_counter_{1};
    static constexpr uint64_t NONCE_BATCH_SIZE = 1ULL << 40;
    uint32_t current_difficulty_ = 0;

    void workerThread(GPUWorker *worker, const MiningJob &job, uint32_t duration_seconds);

    uint64_t getNextNonceBatch();
};

// Utility functions
std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &data);

std::string bytes_to_hex(const std::vector<uint8_t> &bytes);

std::vector<uint8_t> hex_to_bytes(const std::string &hex);

MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

// Global shutdown flag
extern std::atomic<bool> g_shutdown;

#endif // SHA1_OPENCL_MINER_HPP
