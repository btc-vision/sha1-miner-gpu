#pragma once

// Alternative implementations for CUDA device functions
// Use this if the intrinsics are not available or for compatibility

#ifdef __CUDACC__

// Alternative rotation implementation if __funnelshift_l is not available
__device__ __forceinline__ uint32_t rotl32_alt(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// Alternative byte swap if __byte_perm is not available
__device__ __forceinline__ uint32_t swap_endian_alt(uint32_t x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
}

// Alternative popcount if __popc is not available
__device__ __forceinline__ uint32_t popcount_alt(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0f0f0f0f;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3f;
}

// Wrapper functions that try to use intrinsics if available
__device__ __forceinline__ uint32_t device_rotl32(uint32_t x, uint32_t n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __funnelshift_l(x, x, n);
#else
    return rotl32_alt(x, n);
#endif
}

__device__ __forceinline__ uint32_t device_swap_endian(uint32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return __byte_perm(x, 0, 0x0123);
#else
    return swap_endian_alt(x);
#endif
}

__device__ __forceinline__ uint32_t device_popcount(uint32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    return __popc(x);
#else
    return popcount_alt(x);
#endif
}

// Use these in your code instead of direct intrinsics
#define rotl32(x, n) device_rotl32(x, n)
#define swap_endian(x) device_swap_endian(x)
#define count_matching_bits(a, b) (32 - device_popcount((a) ^ (b)))

#endif // __CUDACC__
