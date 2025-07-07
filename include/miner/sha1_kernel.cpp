// SHA-1 constants
__constant uint K[4] = {
    0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xCA62C1D6U
};

__constant uint H0[5] = {
    0x67452301U, 0xEFCDAB89U, 0x98BADCFEU, 0x10325476U, 0xC3D2E1F0U
};

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
    __global uint* results_hash,         // Flattened array of hashes
    __global ulong* results_nonce,       // Array of nonces
    __global uint* results_bits,         // Array of matching bits
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
                results_nonce[idx] = nonce;
                results_bits[idx] = matching_bits;

                // Store hash (flattened)
                #pragma unroll
                for (int j = 0; j < 5; j++) {
                    results_hash[idx * 5 + j] = hash[j];
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