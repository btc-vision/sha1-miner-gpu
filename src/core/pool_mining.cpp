#include "pool_mining.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>

#include "sha1_miner.cuh"

#include "../../logging/logger.hpp"

// Pool mining status display
void display_pool_stats(const MiningPool::PoolMiningSystem::PoolMiningStats &stats)
{
    // Clear line with carriage return
    std::cout << "\r";

    if (!stats.connected) {
        std::cout << Color::YELLOW << "[POOL] " << Color::RED << "Disconnected - Attempting to reconnect...\n"
                  << Color::RESET << std::flush;
    } else if (!stats.authenticated) {
        std::cout << Color::YELLOW << "[POOL] " << Color::YELLOW << "Connected - Authentication pending...\n"
                  << Color::RESET << std::flush;
    } else {
        std::cout << Color::YELLOW << "[POOL] " << Color::RESET << "Worker: " << Color::BRIGHT_CYAN << stats.worker_id
                  << Color::RESET << " | "
                  << "Diff: " << Color::BRIGHT_MAGENTA << stats.current_difficulty << Color::RESET << " | "
                  << "Hash: " << Color::BRIGHT_GREEN << std::fixed << std::setprecision(2) << stats.hashrate / 1e9
                  << " GH/s" << Color::RESET << " | "
                  << "Shares: " << Color::BRIGHT_BLUE << stats.shares_accepted << "/" << stats.shares_submitted
                  << Color::RESET;

        if (stats.shares_submitted > 0) {
            double success_rate = stats.share_success_rate * 100;
            std::string color   = success_rate >= 95   ? Color::BRIGHT_GREEN
                                  : success_rate >= 80 ? Color::BRIGHT_YELLOW
                                                       : Color::BRIGHT_RED;
            std::cout << " (" << color << std::setprecision(1) << success_rate << "%" << Color::RESET << ")";
        }

        std::cout << " | Up: " << Color::DIM << stats.uptime.count() << "s\n" << Color::RESET;
    }

    std::cout << std::flush;
}

// Run pool mining
int run_pool_mining(const MiningConfig &config)
{
    LOG_INFO("POOL", Color::BRIGHT_GREEN, "=== SHA-1 Pool Mining Mode ===", Color::RESET);
    LOG_INFO("POOL", "Pool: ", Color::BRIGHT_CYAN, config.pool_url, Color::RESET);
    LOG_INFO("POOL", "Wallet: ", Color::BRIGHT_YELLOW, config.pool_wallet, Color::RESET);
    LOG_INFO("POOL", "Worker: ", Color::BRIGHT_MAGENTA, config.worker_name, Color::RESET);

    // Create pool configuration
    MiningPool::PoolConfig pool_config;
    pool_config.url         = config.pool_url;
    pool_config.username    = config.pool_wallet;
    pool_config.worker_name = config.worker_name;
    pool_config.password    = config.pool_password;
    pool_config.auth_method = MiningPool::AuthMethod::WORKER_PASS;

    // Auto-detect TLS from URL
    pool_config.use_tls            = config.pool_url.find("wss://") == 0;
    pool_config.verify_server_cert = true;

    // Connection settings
    pool_config.keepalive_interval_s   = 30;
    pool_config.response_timeout_ms    = 10000;
    pool_config.reconnect_delay_ms     = 5000;
    pool_config.max_reconnect_delay_ms = 60000;
    pool_config.reconnect_attempts     = -1;  // Infinite retries

    // Create pool mining system configuration
    MiningPool::PoolMiningSystem::Config pool_mining_config;
    pool_mining_config.pool_config = pool_config;

    // Pass user's mining config
    pool_mining_config.mining_config = &config;

    // Set GPU configuration
    if (config.use_all_gpus) {
        int device_count;
        gpuGetDeviceCount(&device_count);
        for (int i = 0; i < device_count; i++) {
            pool_mining_config.gpu_ids.push_back(i);
        }
        pool_mining_config.use_all_gpus = true;
    } else if (!config.gpu_ids.empty()) {
        pool_mining_config.gpu_ids = config.gpu_ids;
    } else {
        pool_mining_config.gpu_ids.push_back(config.gpu_id);
    }

    // Handle multiple pools with failover
    if (!config.backup_pools.empty() && config.enable_pool_failover) {
        const auto multi_pool_manager = std::make_unique<MiningPool::MultiPoolManager>();

        // Add primary pool
        multi_pool_manager->add_pool("primary", pool_config, 0);

        // Add backup pools
        int priority = 1;
        for (const auto &backup_url : config.backup_pools) {
            auto backup_config    = pool_config;
            backup_config.url     = backup_url;
            backup_config.use_tls = backup_url.find("wss://") == 0;
            multi_pool_manager->add_pool("backup_" + std::to_string(priority), backup_config, priority);
            priority++;
        }

        // Start mining with failover
        if (!multi_pool_manager->start_mining(pool_mining_config)) {
            LOG_ERROR("POOL", "Failed to start pool mining");
            return 1;
        }

        LOG_INFO("POOL", "Mining started with ", priority, " pool(s)");
        LOG_INFO("POOL", "Press Ctrl+C to stop mining");

        // Monitor and display stats
        while (!g_shutdown) {
            auto all_stats = multi_pool_manager->get_all_stats();

            if (auto active_pool = multi_pool_manager->get_active_pool(); all_stats.count(active_pool) > 0) {
                display_pool_stats(all_stats[active_pool]);
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        multi_pool_manager->stop_mining();
    } else {
        // Single pool mining
        const auto pool_mining = std::make_unique<MiningPool::PoolMiningSystem>(pool_mining_config);

        if (!pool_mining->start()) {
            LOG_ERROR("POOL", "Failed to start pool mining");
            return 1;
        }

        LOG_INFO("POOL", "Mining started");
        LOG_INFO("POOL", "Press Ctrl+C to stop mining");

        // Monitor and display stats
        auto last_stats_time = std::chrono::steady_clock::now();
        while (!g_shutdown) {
            if (auto now = std::chrono::steady_clock::now(); now - last_stats_time >= std::chrono::seconds(1)) {
                auto stats = pool_mining->get_stats();
                display_pool_stats(stats);
                last_stats_time = now;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        pool_mining->stop();
    }

    LOG_INFO("POOL", "Pool mining stopped.");
    return 0;
}
