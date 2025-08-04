#pragma once

#include "../core/mining_system.hpp"
#include "config.hpp"

// Utility namespace to handle configuration application
namespace ConfigUtils {
    // Apply MiningConfig to MiningSystem::Config
    inline void applyMiningConfig(const MiningConfig &mining_config, MiningSystem::Config &sys_config)
    {
        // Apply device ID
        sys_config.device_id = mining_config.gpu_id;

        // Only apply values that were user-specified
        if (mining_config.user_specified.num_streams && mining_config.num_streams > 0) {
            sys_config.num_streams = mining_config.num_streams;
        }
        if (mining_config.user_specified.threads_per_block && mining_config.threads_per_block > 0) {
            sys_config.threads_per_block = mining_config.threads_per_block;
        }
        if (mining_config.user_specified.blocks_per_stream && mining_config.blocks_per_stream > 0) {
            sys_config.blocks_per_stream = mining_config.blocks_per_stream;
        }
        if (mining_config.user_specified.result_buffer_size && mining_config.result_buffer_size > 0) {
            sys_config.result_buffer_size = mining_config.result_buffer_size;
        }

        // Apply boolean flags
        sys_config.use_pinned_memory = mining_config.use_pinned_memory;
    }

    // Log user-specified configurations
    inline void logUserConfig(const MiningConfig &config)
    {
        LOG_INFO("CONFIG", "User-specified configurations:");
        if (config.user_specified.num_streams) {
            LOG_INFO("CONFIG", "  Streams: ", config.num_streams);
        }
        if (config.user_specified.threads_per_block) {
            LOG_INFO("CONFIG", "  Threads per block: ", config.threads_per_block);
        }
        if (config.user_specified.blocks_per_stream) {
            LOG_INFO("CONFIG", "  Blocks per stream: ", config.blocks_per_stream);
        }
        if (config.user_specified.result_buffer_size) {
            LOG_INFO("CONFIG", "  Result buffer size: ", config.result_buffer_size);
        }
        if (!config.use_pinned_memory) {
            LOG_INFO("CONFIG", "  Pinned memory disabled");
        }
    }
}  // namespace ConfigUtils
