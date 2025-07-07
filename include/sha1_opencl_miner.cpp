#include "sha1_opencl_miner.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <random>

// Include the SHA-1 header for CPU-side hashing
#include "cxxsha1.hpp"

// Global shutdown flag
std::atomic<bool> g_shutdown(false);

// Define MiningResultPacked structure for the kernel
struct MiningResultPacked {
    cl_ulong nonce;
    cl_uint hash[5];
    cl_uint matching_bits;
    cl_uint difficulty_score;
};

// Utility functions implementation
std::vector<uint8_t> calculate_sha1(const std::vector<uint8_t> &data) {
    SHA1 sha1;
    sha1.update(std::string(reinterpret_cast<const char *>(data.data()), data.size()));
    std::string hex = sha1.final();

    std::vector<uint8_t> result(20);
    for (int i = 0; i < 20; i++) {
        result[i] = static_cast<uint8_t>(std::stoi(hex.substr(i * 2, 2), nullptr, 16));
    }
    return result;
}

std::string bytes_to_hex(const std::vector<uint8_t> &bytes) {
    std::ostringstream oss;
    for (uint8_t b: bytes) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return oss.str();
}

std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        result.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return result;
}

MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty) {
    MiningJob job{};

    // Copy message (32 bytes)
    std::memcpy(job.base_message, message, 32);

    // Convert target hash to uint32_t array (big-endian)
    for (int i = 0; i < 5; i++) {
        job.target_hash[i] = (static_cast<uint32_t>(target_hash[i * 4]) << 24) |
                             (static_cast<uint32_t>(target_hash[i * 4 + 1]) << 16) |
                             (static_cast<uint32_t>(target_hash[i * 4 + 2]) << 8) |
                             static_cast<uint32_t>(target_hash[i * 4 + 3]);
    }

    job.difficulty = difficulty;
    job.nonce_offset = 1;

    return job;
}

// OpenCL error string conversion
const char *OpenCLMiningSystem::getErrorString(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        default: return "Unknown error";
    }
}

void OpenCLMiningSystem::checkError(cl_int error, const std::string &operation) {
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL Error in " << operation << ": "
                << getErrorString(error) << " (" << error << ")" << std::endl;
        throw std::runtime_error(operation + " failed");
    }
}

// Get OpenCL kernel source
std::string OpenCLMiningSystem::getKernelSource() const {
    return R"CLC(
// SHA-1 constants
__constant uint K[4] = {
    0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xCA62C1D6U
};

__constant uint H0[5] = {
    0x67452301U, 0xEFCDAB89U, 0x98BADCFEU, 0x10325476U, 0xC3D2E1F0U
};

// Result structure packed into single buffer
typedef struct {
    ulong nonce;
    uint hash[5];
    uint matching_bits;
    uint difficulty_score;
} MiningResultPacked;

// Rotate left function
inline uint rotl32(uint x, uint n) {
    return (x << n) | (x >> (32 - n));
}

// Count leading zeros in XOR distance
inline uint count_leading_zeros_160bit(const uint hash[5], __constant uint* target) {
    uint total_bits = 0;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        uint xor_val = hash[i] ^ target[i];
        if (xor_val == 0) {
            total_bits += 32;
        } else {
            total_bits += clz(xor_val);
            break;
        }
    }
    return total_bits;
}

// SHA-1 mining kernel
__kernel void sha1_mining_kernel(
    __constant uchar* base_message,      // 32 bytes
    __constant uint* target_hash,        // 5 uints
    uint difficulty,
    __global MiningResultPacked* results, // Packed results
    __global uint* result_count,         // Counter
    uint result_capacity,
    ulong nonce_base,
    uint nonces_per_thread,
    __global ulong* nonces_processed    // Track actual work done
) {
    // Get global thread ID
    size_t tid = get_global_id(0);
    ulong thread_nonce_base = nonce_base + ((ulong)tid * nonces_per_thread);

    // Track nonces processed by this thread
    uint thread_nonces_processed = 0;

    // Load base message into private memory
    uchar base_msg[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        base_msg[i] = base_message[i];
    }

    // Process nonces for this thread
    for (uint i = 0; i < nonces_per_thread; i++) {
        ulong nonce = thread_nonce_base + i;
        if (nonce == 0) continue;

        thread_nonces_processed++;

        // Create message copy
        uchar msg_bytes[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            msg_bytes[j] = base_msg[j];
        }

        // Apply nonce to last 8 bytes (big-endian XOR)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            msg_bytes[24 + j] ^= (nonce >> (56 - j * 8)) & 0xFF;
        }

        // Convert to big-endian words
        uint W[16];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            W[j] = ((uint)msg_bytes[j * 4] << 24) |
                   ((uint)msg_bytes[j * 4 + 1] << 16) |
                   ((uint)msg_bytes[j * 4 + 2] << 8) |
                   (uint)msg_bytes[j * 4 + 3];
        }

        // Apply SHA-1 padding
        W[8] = 0x80000000U;
        #pragma unroll
        for (int j = 9; j < 15; j++) {
            W[j] = 0;
        }
        W[15] = 0x00000100U; // Message length: 256 bits

        // Initialize hash values
        uint a = H0[0];
        uint b = H0[1];
        uint c = H0[2];
        uint d = H0[3];
        uint e = H0[4];

        // SHA-1 rounds 0-19
        #pragma unroll
        for (int t = 0; t < 20; t++) {
            if (t >= 16) {
                uint temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                           W[(t - 14) & 15] ^ W[(t - 16) & 15];
                W[t & 15] = rotl32(temp, 1);
            }
            uint f = (b & c) | (~b & d);
            uint temp = rotl32(a, 5) + f + e + K[0] + W[t & 15];
            e = d;
            d = c;
            c = rotl32(b, 30);
            b = a;
            a = temp;
        }

        // Rounds 20-39
        #pragma unroll
        for (int t = 20; t < 40; t++) {
            uint temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                       W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = rotl32(temp, 1);
            uint f = b ^ c ^ d;
            uint temp2 = rotl32(a, 5) + f + e + K[1] + W[t & 15];
            e = d;
            d = c;
            c = rotl32(b, 30);
            b = a;
            a = temp2;
        }

        // Rounds 40-59
        #pragma unroll
        for (int t = 40; t < 60; t++) {
            uint temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                       W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = rotl32(temp, 1);
            uint f = (b & c) | (b & d) | (c & d);
            uint temp2 = rotl32(a, 5) + f + e + K[2] + W[t & 15];
            e = d;
            d = c;
            c = rotl32(b, 30);
            b = a;
            a = temp2;
        }

        // Rounds 60-79
        #pragma unroll
        for (int t = 60; t < 80; t++) {
            uint temp = W[(t - 3) & 15] ^ W[(t - 8) & 15] ^
                       W[(t - 14) & 15] ^ W[(t - 16) & 15];
            W[t & 15] = rotl32(temp, 1);
            uint f = b ^ c ^ d;
            uint temp2 = rotl32(a, 5) + f + e + K[3] + W[t & 15];
            e = d;
            d = c;
            c = rotl32(b, 30);
            b = a;
            a = temp2;
        }

        // Add initial hash values
        uint hash[5];
        hash[0] = a + H0[0];
        hash[1] = b + H0[1];
        hash[2] = c + H0[2];
        hash[3] = d + H0[3];
        hash[4] = e + H0[4];

        // Count matching bits
        uint matching_bits = count_leading_zeros_160bit(hash, target_hash);

        // If this meets difficulty, save it
        if (matching_bits >= difficulty) {
            uint idx = atomic_inc(result_count);
            if (idx < result_capacity) {
                results[idx].nonce = nonce;
                results[idx].matching_bits = matching_bits;
                results[idx].difficulty_score = matching_bits;

                #pragma unroll
                for (int j = 0; j < 5; j++) {
                    results[idx].hash[j] = hash[j];
                }
            } else {
                // Decrement if we exceeded capacity
                atomic_dec(result_count);
            }
        }
    }

    // Atomically add nonces processed by this thread
    atomic_add(nonces_processed, (ulong)thread_nonces_processed);
}
)CLC";
}

// Platform enumeration
std::vector<OpenCLPlatformInfo> OpenCLMiningSystem::enumeratePlatforms() {
    std::vector<OpenCLPlatformInfo> platforms;

    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return platforms;
    }

    std::vector<cl_platform_id> platform_ids(num_platforms);
    err = clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr);
    if (err != CL_SUCCESS) {
        return platforms;
    }

    for (cl_platform_id pid: platform_ids) {
        OpenCLPlatformInfo info;
        info.platform = pid;

        char buffer[1024];
        clGetPlatformInfo(pid, CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        info.name = buffer;

        clGetPlatformInfo(pid, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        info.vendor = buffer;

        clGetPlatformInfo(pid, CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        info.version = buffer;

        // Get devices for this platform
        cl_uint num_devices;
        err = clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            info.devices.resize(num_devices);
            clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, num_devices, info.devices.data(), nullptr);
        }

        platforms.push_back(info);
    }

    return platforms;
}

// Device enumeration
std::vector<OpenCLDeviceInfo> OpenCLMiningSystem::enumerateDevices(cl_platform_id platform) {
    std::vector<OpenCLDeviceInfo> devices;

    cl_uint num_devices;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        return devices;
    }

    std::vector<cl_device_id> device_ids(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device_ids.data(), nullptr);
    if (err != CL_SUCCESS) {
        return devices;
    }

    for (cl_device_id did: device_ids) {
        OpenCLDeviceInfo info;
        info.device = did;

        char buffer[1024];
        clGetDeviceInfo(did, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
        info.name = buffer;

        clGetDeviceInfo(did, CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr);
        info.vendor = buffer;

        clGetDeviceInfo(did, CL_DEVICE_TYPE, sizeof(cl_device_type), &info.type, nullptr);
        clGetDeviceInfo(did, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info.compute_units, nullptr);
        clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &info.global_mem_size, nullptr);
        clGetDeviceInfo(did, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &info.local_mem_size, nullptr);
        clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &info.max_work_group_size, nullptr);
        clGetDeviceInfo(did, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &info.max_clock_frequency, nullptr);

        devices.push_back(info);
    }

    return devices;
}

// Constructor
OpenCLMiningSystem::OpenCLMiningSystem(const Config &config)
    : config_(config), context_(), device_info_() {
    start_time_ = std::chrono::steady_clock::now();
}

// Destructor
OpenCLMiningSystem::~OpenCLMiningSystem() {
    cleanup();
}

// Initialize OpenCL
bool OpenCLMiningSystem::initializeOpenCL() {
    cl_int err;

    // Get platforms
    auto platforms = enumeratePlatforms();
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return false;
    }

    // Select platform and device
    cl_platform_id selected_platform = nullptr;
    cl_device_id selected_device = nullptr;

    if (config_.platform_index >= 0 && config_.platform_index < platforms.size()) {
        // Use specified platform
        selected_platform = platforms[config_.platform_index].platform;
        auto devices = enumerateDevices(selected_platform);
        if (config_.device_index >= 0 && config_.device_index < devices.size()) {
            selected_device = devices[config_.device_index].device;
            device_info_ = devices[config_.device_index];
        }
    } else {
        // Auto-select first available GPU
        for (const auto &platform: platforms) {
            auto devices = enumerateDevices(platform.platform);
            if (!devices.empty()) {
                selected_platform = platform.platform;
                selected_device = devices[0].device;
                device_info_ = devices[0];
                break;
            }
        }
    }

    if (!selected_device) {
        std::cerr << "No suitable GPU device found" << std::endl;
        return false;
    }

    context_.device = selected_device;

    // Create context
    context_.context = clCreateContext(nullptr, 1, &selected_device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    // Create command queue
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    context_.queue = clCreateCommandQueueWithProperties(context_.context, selected_device, props, &err);
#else
    context_.queue = clCreateCommandQueue(context_.context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    checkError(err, "clCreateCommandQueue");

    return true;
}

// Build kernel
bool OpenCLMiningSystem::buildKernel() {
    cl_int err;

    // Get kernel source
    std::string kernel_source = getKernelSource();
    const char *source_ptr = kernel_source.c_str();
    size_t source_size = kernel_source.length();

    // Create program
    context_.program = clCreateProgramWithSource(context_.context, 1, &source_ptr, &source_size, &err);
    checkError(err, "clCreateProgramWithSource");

    // Build program with optimizations
    const char *build_options = "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(context_.program, 1, &context_.device, build_options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(context_.program, context_.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(context_.program, context_.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        checkError(err, "clBuildProgram");
    }

    // Create kernel
    context_.kernel = clCreateKernel(context_.program, "sha1_mining_kernel", &err);
    checkError(err, "clCreateKernel");

    return true;
}

// Allocate buffers
bool OpenCLMiningSystem::allocateBuffers() {
    cl_int err;

    // Device buffers
    context_.d_base_message = clCreateBuffer(context_.context, CL_MEM_READ_ONLY, 32, nullptr, &err);
    checkError(err, "clCreateBuffer base_message");

    context_.d_target_hash = clCreateBuffer(context_.context, CL_MEM_READ_ONLY, 5 * sizeof(cl_uint), nullptr, &err);
    checkError(err, "clCreateBuffer target_hash");

    // Calculate size for packed results
    size_t result_size = context_.result_capacity * sizeof(MiningResultPacked);
    context_.d_results = clCreateBuffer(context_.context, CL_MEM_WRITE_ONLY, result_size, nullptr, &err);
    checkError(err, "clCreateBuffer results");

    context_.d_result_count = clCreateBuffer(context_.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    checkError(err, "clCreateBuffer result_count");

    context_.d_nonces_processed = clCreateBuffer(context_.context, CL_MEM_READ_WRITE, sizeof(cl_ulong), nullptr, &err);
    checkError(err, "clCreateBuffer nonces_processed");

    // Host buffers
    context_.h_results = new MiningResult[context_.result_capacity];
    context_.h_result_count = new uint32_t;
    context_.h_nonces_processed = new uint64_t;

    return true;
}

// Cleanup
void OpenCLMiningSystem::cleanup() {
    if (context_.kernel) clReleaseKernel(context_.kernel);
    if (context_.program) clReleaseProgram(context_.program);
    if (context_.d_base_message) clReleaseMemObject(context_.d_base_message);
    if (context_.d_target_hash) clReleaseMemObject(context_.d_target_hash);
    if (context_.d_results) clReleaseMemObject(context_.d_results);
    if (context_.d_result_count) clReleaseMemObject(context_.d_result_count);
    if (context_.d_nonces_processed) clReleaseMemObject(context_.d_nonces_processed);
    if (context_.queue) clReleaseCommandQueue(context_.queue);
    if (context_.context) clReleaseContext(context_.context);

    delete[] context_.h_results;
    delete context_.h_result_count;
    delete context_.h_nonces_processed;
}

// Auto-tune parameters
void OpenCLMiningSystem::autoTuneParameters() {
    std::cout << "Auto-tuning OpenCL parameters..." << std::endl;

    // Get device limits
    size_t max_work_group_size = device_info_.max_work_group_size;
    cl_uint compute_units = device_info_.compute_units;

    // Set work group size (local work size)
    config_.local_work_size = 256;
    if (config_.local_work_size > max_work_group_size) {
        config_.local_work_size = max_work_group_size;
    }

    // Set global work size based on compute units
    // Aim for high occupancy
    size_t work_groups_per_cu = 8; // Typical good occupancy
    size_t total_work_groups = compute_units * work_groups_per_cu;
    config_.global_work_size = total_work_groups * config_.local_work_size;

    // Adjust based on device type and vendor
    std::string vendor = device_info_.vendor;
    std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);

    if (vendor.find("nvidia") != std::string::npos) {
        // NVIDIA GPUs
        work_groups_per_cu = 16;
        config_.global_work_size = compute_units * work_groups_per_cu * config_.local_work_size;
    } else if (vendor.find("amd") != std::string::npos || vendor.find("advanced micro") != std::string::npos) {
        // AMD GPUs
        work_groups_per_cu = 8;
        config_.global_work_size = compute_units * work_groups_per_cu * config_.local_work_size;
    } else if (vendor.find("intel") != std::string::npos) {
        // Intel GPUs
        work_groups_per_cu = 4;
        config_.global_work_size = compute_units * work_groups_per_cu * config_.local_work_size;
    }

    // Limit maximum global work size for stability
    size_t max_global_work_size = 1024 * 1024; // 1M threads
    if (config_.global_work_size > max_global_work_size) {
        config_.global_work_size = max_global_work_size;
    }

    std::cout << "Auto-tuned configuration:" << std::endl;
    std::cout << "  Device: " << device_info_.name << std::endl;
    std::cout << "  Vendor: " << device_info_.vendor << std::endl;
    std::cout << "  Compute Units: " << compute_units << std::endl;
    std::cout << "  Local Work Size: " << config_.local_work_size << std::endl;
    std::cout << "  Global Work Size: " << config_.global_work_size << std::endl;
    std::cout << "  Work Groups: " << (config_.global_work_size / config_.local_work_size) << std::endl;
}

// Initialize
bool OpenCLMiningSystem::initialize() {
    std::cout << "Initializing OpenCL Mining System..." << std::endl;

    if (!initializeOpenCL()) {
        return false;
    }

    std::cout << "Selected device: " << device_info_.name << std::endl;
    std::cout << "Vendor: " << device_info_.vendor << std::endl;
    std::cout << "Compute Units: " << device_info_.compute_units << std::endl;
    std::cout << "Global Memory: " << (device_info_.global_mem_size / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    if (!buildKernel()) {
        return false;
    }

    if (config_.auto_tune) {
        autoTuneParameters();
    }

    context_.result_capacity = config_.result_buffer_size;

    if (!allocateBuffers()) {
        return false;
    }

    std::cout << "OpenCL Mining System initialized successfully" << std::endl;
    std::cout << "=====================================\n" << std::endl;

    return true;
}

// Process results
void OpenCLMiningSystem::processResults() {
    cl_int err;

    // Read result count
    err = clEnqueueReadBuffer(context_.queue, context_.d_result_count, CL_TRUE, 0,
                              sizeof(uint32_t), context_.h_result_count, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return;
    }

    uint32_t count = *context_.h_result_count;
    if (count == 0) return;

    // Limit to capacity
    count = std::min(count, context_.result_capacity);

    // Read packed results
    size_t result_size = count * sizeof(MiningResultPacked);
    std::vector<MiningResultPacked> packed_results(count);
    err = clEnqueueReadBuffer(context_.queue, context_.d_results, CL_TRUE, 0,
                              result_size, packed_results.data(), 0, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        return;
    }

    // Process each result
    for (uint32_t i = 0; i < count; i++) {
        MiningResult &result = context_.h_results[i];
        const MiningResultPacked &packed = packed_results[i];

        result.nonce = packed.nonce;
        result.matching_bits = packed.matching_bits;
        result.difficulty_score = packed.difficulty_score;

        // Copy hash
        for (int j = 0; j < 5; j++) {
            result.hash[j] = packed.hash[j];
        }

        // Check if this is a new best
        if (best_tracker_.isNewBest(result.matching_bits)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_
            );

            std::cout << "\n[NEW BEST!] Time: " << elapsed.count() << "s\n";
            std::cout << "  Platform: OpenCL\n";
            std::cout << "  Device: " << device_info_.name << "\n";
            std::cout << "  Nonce: 0x" << std::hex << result.nonce << std::dec << "\n";
            std::cout << "  Matching bits: " << result.matching_bits << "\n";
            std::cout << "  Hash: ";
            for (int j = 0; j < 5; j++) {
                std::cout << std::hex << std::setw(8) << std::setfill('0') << result.hash[j];
                if (j < 4) std::cout << " ";
            }
            std::cout << std::dec << "\n\n";
        }

        total_candidates_++;
    }
}

// Run single batch
uint64_t OpenCLMiningSystem::runSingleBatch(const MiningJob &job) {
    cl_int err;

    // Write job data to device
    err = clEnqueueWriteBuffer(context_.queue, context_.d_base_message, CL_FALSE, 0,
                               32, job.base_message, 0, nullptr, nullptr);
    checkError(err, "Write base_message");

    err = clEnqueueWriteBuffer(context_.queue, context_.d_target_hash, CL_FALSE, 0,
                               5 * sizeof(uint32_t), job.target_hash, 0, nullptr, nullptr);
    checkError(err, "Write target_hash");

    // Reset counters
    uint32_t zero_count = 0;
    uint64_t zero_nonces = 0;
    err = clEnqueueWriteBuffer(context_.queue, context_.d_result_count, CL_FALSE, 0,
                               sizeof(uint32_t), &zero_count, 0, nullptr, nullptr);
    err = clEnqueueWriteBuffer(context_.queue, context_.d_nonces_processed, CL_FALSE, 0,
                               sizeof(uint64_t), &zero_nonces, 0, nullptr, nullptr);

    // Set kernel arguments
    cl_uint difficulty = job.difficulty;
    cl_uint result_capacity = context_.result_capacity;
    cl_ulong nonce_base = job.nonce_offset;
    cl_uint nonces_per_thread = NONCES_PER_THREAD;

    err = clSetKernelArg(context_.kernel, 0, sizeof(cl_mem), &context_.d_base_message);
    err |= clSetKernelArg(context_.kernel, 1, sizeof(cl_mem), &context_.d_target_hash);
    err |= clSetKernelArg(context_.kernel, 2, sizeof(cl_uint), &difficulty);
    err |= clSetKernelArg(context_.kernel, 3, sizeof(cl_mem), &context_.d_results);
    err |= clSetKernelArg(context_.kernel, 4, sizeof(cl_mem), &context_.d_result_count);
    err |= clSetKernelArg(context_.kernel, 5, sizeof(cl_uint), &result_capacity);
    err |= clSetKernelArg(context_.kernel, 6, sizeof(cl_ulong), &nonce_base);
    err |= clSetKernelArg(context_.kernel, 7, sizeof(cl_uint), &nonces_per_thread);
    err |= clSetKernelArg(context_.kernel, 8, sizeof(cl_mem), &context_.d_nonces_processed);
    checkError(err, "Set kernel arguments");

    // Launch kernel
    size_t global_work_size = config_.global_work_size;
    size_t local_work_size = config_.local_work_size;

    err = clEnqueueNDRangeKernel(context_.queue, context_.kernel, 1, nullptr,
                                 &global_work_size, &local_work_size, 0, nullptr, nullptr);
    checkError(err, "Launch kernel");

    // Wait for completion
    clFinish(context_.queue);

    // Read actual nonces processed
    uint64_t actual_nonces = 0;
    err = clEnqueueReadBuffer(context_.queue, context_.d_nonces_processed, CL_TRUE, 0,
                              sizeof(uint64_t), &actual_nonces, 0, nullptr, nullptr);

    // Process results
    processResults();

    // Update total hashes
    total_hashes_ += actual_nonces;

    return actual_nonces;
}

// Get last results
std::vector<MiningResult> OpenCLMiningSystem::getLastResults() {
    std::vector<MiningResult> results;
    uint32_t count = *context_.h_result_count;
    if (count > 0 && count <= context_.result_capacity) {
        results.assign(context_.h_results, context_.h_results + count);
    }
    return results;
}

// Reset state
void OpenCLMiningSystem::resetState() {
    best_tracker_.reset();
    total_hashes_ = 0;
    total_candidates_ = 0;
    start_time_ = std::chrono::steady_clock::now();
}

// Get stats
MiningStats OpenCLMiningSystem::getStats() const {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );

    MiningStats stats;
    stats.hashes_computed = total_hashes_.load();
    stats.candidates_found = total_candidates_.load();
    stats.best_match_bits = best_tracker_.getBestBits();
    stats.hash_rate = static_cast<double>(stats.hashes_computed) /
                      static_cast<double>(elapsed.count());

    return stats;
}

// Run mining loop
void OpenCLMiningSystem::runMiningLoop(const MiningJob &job, uint32_t duration_seconds) {
    std::cout << "Starting OpenCL mining for " << duration_seconds << " seconds...\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Target hash: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << job.target_hash[i] << " ";
    }
    std::cout << "\n" << std::dec;
    std::cout << "Only new best matches will be reported.\n";
    std::cout << "=====================================\n\n";

    resetState();
    g_shutdown = false;

    auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);

    uint64_t nonce_offset = 1;
    uint64_t kernels_launched = 0;
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_hashes = 0;

    while (std::chrono::steady_clock::now() < end_time && !g_shutdown) {
        // Create job with current nonce offset
        MiningJob batch_job = job;
        batch_job.nonce_offset = nonce_offset;

        // Run single batch
        uint64_t hashes_this_batch = runSingleBatch(batch_job);
        kernels_launched++;

        // Update nonce offset
        nonce_offset += config_.global_work_size * NONCES_PER_THREAD;

        // Performance monitoring
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        if (elapsed.count() >= 5) {
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
            uint64_t current_hashes = total_hashes_.load();
            uint64_t hash_diff = current_hashes - last_hashes;

            double instant_rate = static_cast<double>(hash_diff) /
                                  static_cast<double>(elapsed.count()) / 1e9;
            double average_rate = static_cast<double>(current_hashes) /
                                  static_cast<double>(total_elapsed.count()) / 1e9;

            std::cout << "\r[" << total_elapsed.count() << "s] "
                    << "Rate: " << std::fixed << std::setprecision(2)
                    << instant_rate << " GH/s"
                    << " (avg: " << average_rate << " GH/s) | "
                    << "Best: " << best_tracker_.getBestBits() << " bits | "
                    << "Total: " << static_cast<double>(current_hashes) / 1e12
                    << " TH" << std::flush;

            last_update = now;
            last_hashes = current_hashes;
        }

        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Final statistics
    auto final_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );

    std::cout << "\n\nMining completed. Kernels launched: " << kernels_launched << "\n";
    std::cout << "=====================================\n";
    std::cout << "Final Statistics:\n";
    std::cout << "  Platform: OpenCL\n";
    std::cout << "  Device: " << device_info_.name << "\n";
    std::cout << "  Total Time: " << final_elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " << std::fixed << std::setprecision(3)
            << static_cast<double>(total_hashes_.load()) / 1e12 << " TH\n";
    std::cout << "  Average Rate: " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_hashes_.load()) / final_elapsed.count() / 1e9 << " GH/s\n";
    std::cout << "  Best Match: " << best_tracker_.getBestBits() << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates_.load() << "\n";
}

// Multi-GPU Manager Implementation
OpenCLMultiGPUManager::OpenCLMultiGPUManager() {
    start_time_ = std::chrono::steady_clock::now();
}

OpenCLMultiGPUManager::~OpenCLMultiGPUManager() {
    shutdown_ = true;
    for (auto &worker: workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
    }
}

uint64_t OpenCLMultiGPUManager::getNextNonceBatch() {
    return global_nonce_counter_.fetch_add(NONCE_BATCH_SIZE);
}

void OpenCLMultiGPUManager::workerThread(GPUWorker *worker, const MiningJob &job, uint32_t duration_seconds) {
    std::cout << "[Platform " << worker->platform_index << " Device " << worker->device_index
            << "] Worker thread started\n";

    auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);

    // Get initial nonce batch
    uint64_t current_nonce_base = getNextNonceBatch();
    uint64_t nonces_used_in_batch = 0;

    while (!shutdown_ && std::chrono::steady_clock::now() < end_time) {
        try {
            // Create job with current nonce offset
            MiningJob worker_job = job;
            worker_job.nonce_offset = current_nonce_base + nonces_used_in_batch;

            // Run a single kernel batch
            uint64_t hashes_this_round = worker->mining_system->runSingleBatch(worker_job);

            // Update worker stats
            worker->hashes_computed += hashes_this_round;
            nonces_used_in_batch += hashes_this_round;

            // Check if we need a new nonce batch
            if (nonces_used_in_batch >= NONCE_BATCH_SIZE * 0.9) {
                current_nonce_base = getNextNonceBatch();
                nonces_used_in_batch = 0;
            }

            // Get any results from this batch
            auto results = worker->mining_system->getLastResults();
            worker->candidates_found += results.size();

            // Check for new best
            for (const auto &result: results) {
                if (result.matching_bits > worker->best_match_bits) {
                    worker->best_match_bits = result.matching_bits;

                    // Check if this is a global best
                    if (global_best_tracker_.isNewBest(result.matching_bits)) {
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - start_time_
                        );
                        std::cout << "\n[Platform " << worker->platform_index
                                << " Device " << worker->device_index
                                << " - NEW BEST!] Time: " << elapsed.count() << "s\n";
                        std::cout << "  Nonce: 0x" << std::hex << result.nonce << std::dec << "\n";
                        std::cout << "  Matching bits: " << result.matching_bits << "\n";
                        std::cout << "  Hash: ";
                        for (int j = 0; j < 5; j++) {
                            std::cout << std::hex << std::setw(8) << std::setfill('0')
                                    << result.hash[j];
                            if (j < 4) std::cout << " ";
                        }
                        std::cout << std::dec << "\n\n";
                    }
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "[Platform " << worker->platform_index
                    << " Device " << worker->device_index
                    << "] Error: " << e.what() << "\n";
            // Wait a bit before retrying
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Small delay to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "[Platform " << worker->platform_index
            << " Device " << worker->device_index
            << "] Worker thread finished\n";
}

bool OpenCLMultiGPUManager::initialize(const std::vector<std::pair<int, int> > &platform_device_pairs) {
    std::cout << "\nInitializing OpenCL Multi-GPU Mining System\n";
    std::cout << "=====================================\n";

    auto platforms = OpenCLMiningSystem::enumeratePlatforms();

    // If no specific pairs provided, use all available GPUs
    std::vector<std::pair<int, int> > pairs_to_use = platform_device_pairs;
    if (pairs_to_use.empty()) {
        for (int p = 0; p < platforms.size(); p++) {
            auto devices = OpenCLMiningSystem::enumerateDevices(platforms[p].platform);
            for (int d = 0; d < devices.size(); d++) {
                pairs_to_use.push_back({p, d});
            }
        }
    }

    for (const auto &pair: pairs_to_use) {
        int platform_idx = pair.first;
        int device_idx = pair.second;

        if (platform_idx >= platforms.size()) {
            std::cerr << "Invalid platform index: " << platform_idx << std::endl;
            continue;
        }

        auto devices = OpenCLMiningSystem::enumerateDevices(platforms[platform_idx].platform);
        if (device_idx >= devices.size()) {
            std::cerr << "Invalid device index: " << device_idx << " for platform " << platform_idx << std::endl;
            continue;
        }

        auto worker = std::make_unique<GPUWorker>();
        worker->platform_index = platform_idx;
        worker->device_index = device_idx;

        OpenCLMiningSystem::Config config;
        config.platform_index = platform_idx;
        config.device_index = device_idx;
        config.auto_tune = true;

        worker->mining_system = std::make_unique<OpenCLMiningSystem>(config);

        try {
            if (!worker->mining_system->initialize()) {
                std::cerr << "Failed to initialize GPU " << device_idx
                        << " on platform " << platform_idx << std::endl;
                continue;
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception initializing GPU " << device_idx
                    << " on platform " << platform_idx << ": " << e.what() << std::endl;
            continue;
        }

        workers_.push_back(std::move(worker));
        std::cout << "Successfully initialized GPU " << device_idx
                << " on platform " << platform_idx << std::endl;
    }

    if (workers_.empty()) {
        std::cerr << "No GPUs were successfully initialized" << std::endl;
        return false;
    }

    std::cout << "\nSuccessfully initialized " << workers_.size() << " GPU(s) for mining\n";
    std::cout << "=====================================\n\n";

    return true;
}

void OpenCLMultiGPUManager::runMining(const MiningJob &job, uint32_t duration_seconds) {
    std::cout << "\nStarting multi-GPU mining on " << workers_.size() << " device(s)\n";
    std::cout << "Target difficulty: " << job.difficulty << " bits\n";
    std::cout << "Duration: " << duration_seconds << " seconds\n";
    std::cout << "=====================================\n\n";

    // Store difficulty for stats calculation
    current_difficulty_ = job.difficulty;

    shutdown_ = false;
    start_time_ = std::chrono::steady_clock::now();
    global_best_tracker_.reset();

    // Reset all worker stats
    for (auto &worker: workers_) {
        worker->hashes_computed = 0;
        worker->candidates_found = 0;
        worker->best_match_bits = 0;
    }

    // Start worker threads
    for (auto &worker: workers_) {
        worker->worker_thread = std::make_unique<std::thread>(
            &OpenCLMultiGPUManager::workerThread, this, worker.get(), job, duration_seconds
        );
    }

    // Monitor progress
    auto last_update = std::chrono::steady_clock::now();
    uint64_t last_total_hashes = 0;

    while (!shutdown_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);

        if (elapsed.count() >= duration_seconds) {
            shutdown_ = true;
            break;
        }

        // Calculate combined stats
        uint64_t total_hashes = 0;
        uint64_t total_candidates = 0;
        uint32_t best_bits = global_best_tracker_.getBestBits();

        std::vector<double> gpu_rates;
        for (const auto &worker: workers_) {
            uint64_t gpu_hash_count = worker->hashes_computed.load();
            total_hashes += gpu_hash_count;
            total_candidates += worker->candidates_found.load();

            // Calculate per-GPU rate
            double gpu_rate = 0.0;
            if (elapsed.count() > 0) {
                gpu_rate = static_cast<double>(gpu_hash_count) / elapsed.count() / 1e9;
            }
            gpu_rates.push_back(gpu_rate);
        }

        // Calculate rates
        uint64_t hash_diff = total_hashes - last_total_hashes;
        auto interval = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
        double instant_rate = 0.0;
        double average_rate = 0.0;
        if (interval.count() > 0) {
            instant_rate = static_cast<double>(hash_diff) / interval.count() / 1e9;
        }
        if (elapsed.count() > 0) {
            average_rate = static_cast<double>(total_hashes) / elapsed.count() / 1e9;
        }

        // Print status line
        std::cout << "\r[" << elapsed.count() << "s] "
                << "Rate: " << std::fixed << std::setprecision(2)
                << instant_rate << " GH/s"
                << " (avg: " << average_rate << " GH/s) | "
                << "Best: " << best_bits << " bits | "
                << "GPUs: ";

        // Show per-GPU rates
        for (size_t i = 0; i < gpu_rates.size(); i++) {
            if (i > 0) std::cout << "+";
            std::cout << std::fixed << std::setprecision(1) << gpu_rates[i];
        }

        std::cout << " | Total: " << std::fixed << std::setprecision(3)
                << static_cast<double>(total_hashes) / 1e12
                << " TH" << std::flush;

        last_update = now;
        last_total_hashes = total_hashes;
    }

    // Signal shutdown and wait for all workers
    shutdown_ = true;
    std::cout << "\n\nShutting down workers...\n";

    for (auto &worker: workers_) {
        if (worker->worker_thread && worker->worker_thread->joinable()) {
            worker->worker_thread->join();
        }
    }

    printCombinedStats();
}

void OpenCLMultiGPUManager::printCombinedStats() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_
    );

    uint64_t total_hashes = 0;
    uint64_t total_candidates = 0;
    uint32_t best_bits = global_best_tracker_.getBestBits();

    std::cout << "\n=== Multi-GPU Mining Results ===\n";
    std::cout << "=====================================\n";

    // Per-GPU stats
    for (size_t i = 0; i < workers_.size(); i++) {
        const auto &worker = workers_[i];
        uint64_t gpu_hashes = worker->hashes_computed.load();
        uint64_t gpu_candidates = worker->candidates_found.load();
        uint32_t gpu_best = worker->best_match_bits.load();
        double gpu_rate = 0.0;
        if (elapsed.count() > 0) {
            gpu_rate = static_cast<double>(gpu_hashes) / elapsed.count() / 1e9;
        }

        std::cout << "GPU [Platform " << worker->platform_index
                << " Device " << worker->device_index << "]:\n";
        std::cout << "  Total Hashes: " << std::fixed << std::setprecision(3)
                << static_cast<double>(gpu_hashes) / 1e9 << " GH\n";
        std::cout << "  Hash Rate: " << std::fixed << std::setprecision(2)
                << gpu_rate << " GH/s\n";
        std::cout << "  Best Match: " << gpu_best << " bits\n";
        std::cout << "  Candidates: " << gpu_candidates << "\n";
        if (gpu_hashes > 0 && gpu_candidates > 0) {
            double efficiency = 100.0 * gpu_candidates * std::pow(2.0, current_difficulty_) / gpu_hashes;
            std::cout << "  Efficiency: " << std::fixed << std::setprecision(4)
                    << efficiency << "%\n";
        }
        std::cout << "\n";

        total_hashes += gpu_hashes;
        total_candidates += gpu_candidates;
    }

    std::cout << "=====================================\n";
    std::cout << "Combined Statistics:\n";
    std::cout << "  Platform: OpenCL\n";
    std::cout << "  Total GPUs: " << workers_.size() << "\n";
    std::cout << "  Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "  Total Hashes: " << std::fixed << std::setprecision(3)
            << static_cast<double>(total_hashes) / 1e12 << " TH\n";
    if (elapsed.count() > 0) {
        std::cout << "  Combined Rate: " << std::fixed << std::setprecision(2)
                << static_cast<double>(total_hashes) / elapsed.count() / 1e9 << " GH/s\n";
    }

    std::cout << "  Best Match: " << best_bits << " bits\n";
    std::cout << "  Total Candidates: " << total_candidates << "\n";

    if (total_hashes > 0 && total_candidates > 0) {
        double global_efficiency = 100.0 * total_candidates * std::pow(2.0, current_difficulty_) / total_hashes;
        std::cout << "  Global Efficiency: " << std::scientific << std::setprecision(2)
                << global_efficiency << "%\n";
    }
    std::cout << "=====================================\n";
}
