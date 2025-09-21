#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <cstdio>
#include <cstring>
#include "sha1_miner_sycl.hpp"

using namespace sycl;

// Global SYCL queue and context for Intel GPU mining
static queue *g_sycl_queue = nullptr;
static context *g_sycl_context = nullptr;
static device *g_intel_device = nullptr;

// Global constant memory for base message (SYCL equivalent)
static uint32_t *d_base_message_sycl = nullptr;
static uint32_t *d_pre_swapped_base_sycl = nullptr;

// SHA-1 constants
constexpr uint32_t K0 = 0x5A827999;
constexpr uint32_t K1 = 0x6ED9EBA1;
constexpr uint32_t K2 = 0x8F1BBCDC;
constexpr uint32_t K3 = 0xCA62C1D6;

constexpr uint32_t H0_0 = 0x67452301;
constexpr uint32_t H0_1 = 0xEFCDAB89;
constexpr uint32_t H0_2 = 0x98BADCFE;
constexpr uint32_t H0_3 = 0x10325476;
constexpr uint32_t H0_4 = 0xC3D2E1F0;

// Intel GPU optimized byte swap using SYCL
inline uint32_t intel_bswap32(uint32_t x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8) |
           ((x & 0x0000FF00) << 8) |
           ((x & 0x000000FF) << 24);
}

// Intel GPU optimized rotation
inline uint32_t intel_rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// Intel GPU optimized count leading zeros
inline uint32_t intel_clz(uint32_t x) {
    if (x == 0) return 32;
    uint32_t count = 0;
    if ((x & 0xFFFF0000) == 0) { count += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { count += 8; x <<= 8; }
    if ((x & 0xF0000000) == 0) { count += 4; x <<= 4; }
    if ((x & 0xC0000000) == 0) { count += 2; x <<= 2; }
    if ((x & 0x80000000) == 0) { count += 1; }
    return count;
}

// Count leading zeros for 160-bit comparison - optimized for Intel GPU
inline uint32_t count_leading_zeros_160bit_intel(const uint32_t hash[5], const uint32_t target[5]) {
    uint32_t xor_val;
    uint32_t clz;

    xor_val = hash[0] ^ target[0];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return clz;
    }

    xor_val = hash[1] ^ target[1];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 32 + clz;
    }

    xor_val = hash[2] ^ target[2];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 64 + clz;
    }

    xor_val = hash[3] ^ target[3];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 96 + clz;
    }

    xor_val = hash[4] ^ target[4];
    if (xor_val != 0) {
        clz = intel_clz(xor_val);
        return 128 + clz;
    }

    return 160;
}

// SYCL-optimized SHA-1 round macros for Intel GPU
#define SHA1_ROUND_0_19_INTEL(a, b, c, d, e, W_val) \
    do { \
        uint32_t f = (b & c) | (~b & d); \
        uint32_t temp = intel_rotl32(a, 5) + f + e + K0 + W_val; \
        e = d; \
        d = c; \
        c = intel_rotl32(b, 30); \
        b = a; \
        a = temp; \
    } while (0)

#define SHA1_ROUND_20_39_INTEL(a, b, c, d, e, W_val) \
    do { \
        uint32_t f = b ^ c ^ d; \
        uint32_t temp = intel_rotl32(a, 5) + f + e + K1 + W_val; \
        e = d; \
        d = c; \
        c = intel_rotl32(b, 30); \
        b = a; \
        a = temp; \
    } while (0)

#define SHA1_ROUND_40_59_INTEL(a, b, c, d, e, W_val) \
    do { \
        uint32_t f = (b & c) | (d & (b ^ c)); \
        uint32_t temp = intel_rotl32(a, 5) + f + e + K2 + W_val; \
        e = d; \
        d = c; \
        c = intel_rotl32(b, 30); \
        b = a; \
        a = temp; \
    } while (0)

#define SHA1_ROUND_60_79_INTEL(a, b, c, d, e, W_val) \
    do { \
        uint32_t f = b ^ c ^ d; \
        uint32_t temp = intel_rotl32(a, 5) + f + e + K3 + W_val; \
        e = d; \
        d = c; \
        c = intel_rotl32(b, 30); \
        b = a; \
        a = temp; \
    } while (0)

// Intel GPU SHA-1 kernel using SYCL with dual hash processing for maximum throughput
void sha1_mining_kernel_intel(
    queue& q,
    const uint32_t* target_hash,
    const uint32_t* pre_swapped_base,
    uint32_t difficulty,
    MiningResult* results,
    uint32_t* result_count,
    uint32_t result_capacity,
    uint64_t nonce_base,
    uint32_t nonces_per_thread,
    uint64_t job_version,
    int total_threads
) {
    // Reset result count
    q.memset(result_count, 0, sizeof(uint32_t)).wait();

    // Launch the SYCL kernel
    auto event = q.parallel_for(range<1>(total_threads), [=](id<1> idx) {
        const uint32_t tid = idx[0];
        const uint64_t thread_nonce_base = nonce_base + (static_cast<uint64_t>(tid) * nonces_per_thread);

        // Load target hash into private memory for better performance
        uint32_t target[5];
        for (int i = 0; i < 5; i++) {
            target[i] = target_hash[i];
        }

        // Process nonces in pairs for better utilization (like NVIDIA dual implementation)
        for (uint32_t i = 0; i < nonces_per_thread; i += 2) {
            uint64_t nonce1 = thread_nonce_base + i;
            uint64_t nonce2 = thread_nonce_base + i + 1;

            // Skip if either nonce is 0
            if (nonce1 == 0) nonce1 = thread_nonce_base + nonces_per_thread;
            if (nonce2 == 0) nonce2 = thread_nonce_base + nonces_per_thread + 1;

            // Process both hashes in parallel for better performance
            uint32_t W1[16], W2[16];

            // Set fixed pre-swapped parts for both hashes (0-5)
            for (int j = 0; j < 6; j++) {
                W1[j] = pre_swapped_base[j];
                W2[j] = pre_swapped_base[j];
            }

            // Set varying parts for 6-7 using pre-swapped base and nonce XOR
            uint32_t nonce1_high = static_cast<uint32_t>(nonce1 >> 32);
            uint32_t nonce1_low = static_cast<uint32_t>(nonce1 & 0xFFFFFFFF);
            W1[6] = pre_swapped_base[6] ^ nonce1_high;
            W1[7] = pre_swapped_base[7] ^ nonce1_low;

            uint32_t nonce2_high = static_cast<uint32_t>(nonce2 >> 32);
            uint32_t nonce2_low = static_cast<uint32_t>(nonce2 & 0xFFFFFFFF);
            W2[6] = pre_swapped_base[6] ^ nonce2_high;
            W2[7] = pre_swapped_base[7] ^ nonce2_low;

            // Apply padding to both
            W1[8] = 0x80000000; W2[8] = 0x80000000;
            W1[9] = 0; W2[9] = 0;
            W1[10] = 0; W2[10] = 0;
            W1[11] = 0; W2[11] = 0;
            W1[12] = 0; W2[12] = 0;
            W1[13] = 0; W2[13] = 0;
            W1[14] = 0; W2[14] = 0;
            W1[15] = 0x00000100; W2[15] = 0x00000100;

            // Process hash 1
            uint32_t a1 = H0_0, b1 = H0_1, c1 = H0_2, d1 = H0_3, e1 = H0_4;

            // Rounds 0-15 (unrolled for performance)
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[0]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[1]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[2]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[3]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[4]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[5]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[6]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[7]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[8]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[9]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[10]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[11]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[12]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[13]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[14]);
            SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[15]);

            // Message schedule and remaining rounds (16-79)
            for (int t = 16; t < 80; t++) {
                uint32_t idx = t & 15;
                W1[idx] = intel_rotl32(W1[(t-3) & 15] ^ W1[(t-8) & 15] ^ W1[(t-14) & 15] ^ W1[(t-16) & 15], 1);

                if (t < 20) {
                    SHA1_ROUND_0_19_INTEL(a1, b1, c1, d1, e1, W1[idx]);
                } else if (t < 40) {
                    SHA1_ROUND_20_39_INTEL(a1, b1, c1, d1, e1, W1[idx]);
                } else if (t < 60) {
                    SHA1_ROUND_40_59_INTEL(a1, b1, c1, d1, e1, W1[idx]);
                } else {
                    SHA1_ROUND_60_79_INTEL(a1, b1, c1, d1, e1, W1[idx]);
                }
            }

            // Final hash 1
            uint32_t hash1[5];
            hash1[0] = a1 + H0_0;
            hash1[1] = b1 + H0_1;
            hash1[2] = c1 + H0_2;
            hash1[3] = d1 + H0_3;
            hash1[4] = e1 + H0_4;

            // Check difficulty for hash 1
            uint32_t matching_bits1 = count_leading_zeros_160bit_intel(hash1, target);
            if (matching_bits1 >= difficulty) {
                uint32_t idx = sycl::atomic_ref<uint32_t, memory_order::relaxed, memory_scope::device>(*result_count).fetch_add(1);
                if (idx < result_capacity) {
                    results[idx].nonce = nonce1;
                    results[idx].matching_bits = matching_bits1;
                    results[idx].job_version = job_version;
                    for (int j = 0; j < 5; j++) {
                        results[idx].hash[j] = hash1[j];
                    }
                }
            }

            // Process hash 2 (if within bounds)
            if (i + 1 < nonces_per_thread) {
                uint32_t a2 = H0_0, b2 = H0_1, c2 = H0_2, d2 = H0_3, e2 = H0_4;

                // Rounds 0-15 for hash 2
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[0]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[1]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[2]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[3]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[4]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[5]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[6]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[7]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[8]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[9]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[10]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[11]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[12]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[13]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[14]);
                SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[15]);

                // Message schedule and remaining rounds for hash 2
                for (int t = 16; t < 80; t++) {
                    uint32_t idx = t & 15;
                    W2[idx] = intel_rotl32(W2[(t-3) & 15] ^ W2[(t-8) & 15] ^ W2[(t-14) & 15] ^ W2[(t-16) & 15], 1);

                    if (t < 20) {
                        SHA1_ROUND_0_19_INTEL(a2, b2, c2, d2, e2, W2[idx]);
                    } else if (t < 40) {
                        SHA1_ROUND_20_39_INTEL(a2, b2, c2, d2, e2, W2[idx]);
                    } else if (t < 60) {
                        SHA1_ROUND_40_59_INTEL(a2, b2, c2, d2, e2, W2[idx]);
                    } else {
                        SHA1_ROUND_60_79_INTEL(a2, b2, c2, d2, e2, W2[idx]);
                    }
                }

                // Final hash 2
                uint32_t hash2[5];
                hash2[0] = a2 + H0_0;
                hash2[1] = b2 + H0_1;
                hash2[2] = c2 + H0_2;
                hash2[3] = d2 + H0_3;
                hash2[4] = e2 + H0_4;

                // Check difficulty for hash 2
                uint32_t matching_bits2 = count_leading_zeros_160bit_intel(hash2, target);
                if (matching_bits2 >= difficulty) {
                    uint32_t idx = sycl::atomic_ref<uint32_t, memory_order::relaxed, memory_scope::device>(*result_count).fetch_add(1);
                    if (idx < result_capacity) {
                        results[idx].nonce = nonce2;
                        results[idx].matching_bits = matching_bits2;
                        results[idx].job_version = job_version;
                        for (int j = 0; j < 5; j++) {
                            results[idx].hash[j] = hash2[j];
                        }
                    }
                }
            }
        }
    });

    event.wait();
}

// CPU-side byte swap function for initialization
inline uint32_t bswap32_cpu(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    #include <stdlib.h>
    return _byteswap_ulong(x);
#else
    return ((x & 0xFF000000) >> 24) | ((x & 0x00FF0000) >> 8) |
           ((x & 0x0000FF00) << 8) | ((x & 0x000000FF) << 24);
#endif
}

// Initialize SYCL runtime for Intel GPU
extern "C" bool initialize_sycl_runtime() {
    try {
        // Find Intel GPU device
        auto platforms = platform::get_platforms();
        device selected_device;
        bool found_intel_gpu = false;

        for (const auto& platform : platforms) {
            auto devices = platform.get_devices();
            for (const auto& device : devices) {
                if (device.is_gpu()) {
                    auto vendor = device.get_info<info::device::vendor>();
                    auto name = device.get_info<info::device::name>();

                    // Check for Intel GPU
                    if (vendor.find("Intel") != std::string::npos ||
                        name.find("Intel") != std::string::npos ||
                        name.find("Arc") != std::string::npos ||
                        name.find("Iris") != std::string::npos) {
                        selected_device = device;
                        found_intel_gpu = true;
                        printf("Found Intel GPU: %s (Vendor: %s)\n", name.c_str(), vendor.c_str());
                        break;
                    }
                }
            }
            if (found_intel_gpu) break;
        }

        if (!found_intel_gpu) {
            printf("No Intel GPU found, falling back to any available GPU\n");
            auto devices = device::get_devices(info::device_type::gpu);
            if (!devices.empty()) {
                selected_device = devices[0];
                auto name = selected_device.get_info<info::device::name>();
                printf("Using GPU: %s\n", name.c_str());
            } else {
                printf("No GPU devices found\n");
                return false;
            }
        }

        // Create SYCL context and queue
        g_intel_device = new device(selected_device);
        g_sycl_context = new context(*g_intel_device);
        g_sycl_queue = new queue(*g_sycl_context, *g_intel_device);

        // Allocate constant memory using USM
        d_base_message_sycl = malloc_device<uint32_t>(8, *g_sycl_queue);
        d_pre_swapped_base_sycl = malloc_device<uint32_t>(8, *g_sycl_queue);

        if (!d_base_message_sycl || !d_pre_swapped_base_sycl) {
            printf("Failed to allocate device memory for constant data\n");
            return false;
        }

        printf("SYCL runtime initialized successfully for Intel GPU\n");
        return true;

    } catch (const sycl::exception& e) {
        printf("SYCL exception during initialization: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        printf("Standard exception during SYCL initialization: %s\n", e.what());
        return false;
    }
}

// Update base message for SYCL
extern "C" void update_base_message_sycl(const uint32_t *base_msg_words) {
    if (!g_sycl_queue || !d_base_message_sycl || !d_pre_swapped_base_sycl) {
        printf("SYCL not initialized\n");
        return;
    }

    try {
        // Copy base message to device
        g_sycl_queue->memcpy(d_base_message_sycl, base_msg_words, 8 * sizeof(uint32_t)).wait();

        // Prepare pre-swapped version
        uint32_t pre_swapped[8];
        for (int j = 0; j < 8; j++) {
            pre_swapped[j] = bswap32_cpu(base_msg_words[j]);
        }
        g_sycl_queue->memcpy(d_pre_swapped_base_sycl, pre_swapped, 8 * sizeof(uint32_t)).wait();

    } catch (const sycl::exception& e) {
        printf("SYCL exception in update_base_message_sycl: %s\n", e.what());
    }
}

// Launch the Intel GPU mining kernel
extern "C" void launch_mining_kernel_intel(
    const DeviceMiningJob &device_job,
    uint32_t difficulty,
    uint64_t nonce_offset,
    const ResultPool &pool,
    const KernelConfig &config,
    uint64_t job_version
) {
    if (!g_sycl_queue) {
        printf("SYCL queue not initialized\n");
        return;
    }

    try {
        int total_threads = config.blocks * config.threads_per_block;

        sha1_mining_kernel_intel(
            *g_sycl_queue,
            device_job.target_hash,
            d_pre_swapped_base_sycl,
            difficulty,
            pool.results,
            pool.count,
            pool.capacity,
            nonce_offset,
            NONCES_PER_THREAD,
            job_version,
            total_threads
        );

    } catch (const sycl::exception& e) {
        printf("SYCL exception in kernel launch: %s\n", e.what());
    } catch (const std::exception& e) {
        printf("Standard exception in kernel launch: %s\n", e.what());
    }
}

// Cleanup SYCL resources
extern "C" void cleanup_sycl_runtime() {
    if (d_base_message_sycl) {
        free(d_base_message_sycl, *g_sycl_queue);
        d_base_message_sycl = nullptr;
    }
    if (d_pre_swapped_base_sycl) {
        free(d_pre_swapped_base_sycl, *g_sycl_queue);
        d_pre_swapped_base_sycl = nullptr;
    }

    delete g_sycl_queue;
    delete g_sycl_context;
    delete g_intel_device;

    g_sycl_queue = nullptr;
    g_sycl_context = nullptr;
    g_intel_device = nullptr;
}