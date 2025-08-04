#include "solo_mining.hpp"

#include <cmath>
#include <iostream>

#include "sha1_miner.cuh"

#include "../../logging/logger.hpp"
#include "../configs/config_utils.hpp"
#include "../multi_gpu_manager.hpp"
#include "../utils/test_utils.hpp"
#include "mining_system.hpp"

int run_solo_mining(const MiningConfig &config, const std::vector<int> &gpu_ids_to_use)
{
    LOG_INFO("MAIN", "Running in solo mining mode");

    // For solo mining, we generate a random message and calculate its SHA-1 as the target
    std::vector<uint8_t> message = generate_secure_random_message();
    LOG_INFO("MAIN", "Generated random message: ", bytes_to_hex(message));

    std::vector<uint8_t> target = calculate_sha1(message);
    LOG_INFO("MAIN", "Target SHA-1: ", bytes_to_hex(target));

    // Create mining job
    MiningJob job = create_mining_job(message.data(), target.data(), config.difficulty);

    LOG_INFO("MAIN", "\nMining Configuration:");
    LOG_INFO("MAIN", "  Difficulty: ", config.difficulty, " bits must match");
    LOG_INFO("MAIN", "  Success probability per hash: 2^-", config.difficulty);

    // Use multi-GPU manager if multiple GPUs selected
    if (gpu_ids_to_use.size() > 1) {
        MultiGPUManager multi_gpu_manager;

        // Pass user config to multi-GPU manager
        multi_gpu_manager.setUserConfig(&config);

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

        // Apply user-specified configurations using ConfigUtils
        ConfigUtils::applyMiningConfig(config, sys_config);

        // Log user configurations if any were specified
        if (config.user_specified.num_streams || config.user_specified.threads_per_block ||
            config.user_specified.blocks_per_stream || config.user_specified.result_buffer_size) {
            ConfigUtils::logUserConfig(config);
        }

        // Create mining system with config pointer for auto-tune to respect
        g_mining_system = std::make_unique<MiningSystem>(sys_config);
        // Pass user config for auto-tune to check
        g_mining_system->setUserConfig(&config);
        if (!g_mining_system->initialize()) {
            LOG_ERROR("MAIN", "Failed to initialize mining system");
            return 1;
        }

        // Calculate expected time based on final configuration
        auto final_config = g_mining_system->getConfig();
        double hashes_per_kernel =
            static_cast<double>(final_config.blocks_per_stream) * final_config.threads_per_block * NONCES_PER_THREAD;
        double expected_time = std::pow(2.0, config.difficulty) / (hashes_per_kernel * 1e9);

        LOG_INFO("MAIN", "  Expected time to find: ", std::scientific, expected_time, " seconds @ 1 GH/s");

        LOG_INFO("MAIN", "Starting infinite mining. Press Ctrl+C to stop.");
        g_mining_system->runMiningLoop(job);
        cleanup_mining_system();
    }

    return 0;
}
