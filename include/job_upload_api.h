#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    // Legacy API for compatibility - redirects to new system
    void upload_new_job(const uint8_t msg32[32], const uint32_t digest[5]);

#ifdef __cplusplus
}
#endif