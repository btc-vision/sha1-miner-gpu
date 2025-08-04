#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>

namespace po = boost::program_options;

// Mining configuration with only used options
struct MiningConfig
{
    // GPU configuration
    int gpu_id = 0;
    std::vector<int> gpu_ids;
    bool use_all_gpus = false;

    // Solo mining configuration
    uint32_t difficulty = 38;

    // Core performance configuration (actually used)
    int num_streams       = 0;  // 0 = auto
    int threads_per_block = 0;  // 0 = auto
    int blocks_per_stream = 0;  // 0 = auto
    bool auto_tune        = true;

    // Memory configuration (actually used)
    size_t result_buffer_size = 0;  // 0 = auto
    bool use_pinned_memory    = true;

    // Pool mining configuration
    bool use_pool = false;
    std::string pool_url;
    std::string pool_wallet;
    std::string worker_name   = "default_worker";
    std::string pool_password = "x";
    std::vector<std::string> backup_pools;
    bool enable_pool_failover = true;

    // Operating modes
    bool benchmark  = false;
    bool test_sha1  = false;
    bool debug_mode = false;
    int debug_level = 2;
    bool no_colors  = false;

    // Tracking which values were user-specified
    struct UserSpecified
    {
        bool num_streams        = false;
        bool threads_per_block  = false;
        bool blocks_per_stream  = false;
        bool result_buffer_size = false;
    } user_specified;

    // Helper method to check if a value was user-specified
    bool wasUserSpecified(const std::string &param) const;
};

// Configuration parsing functions
po::options_description create_options_description();
void print_usage_examples(const char *program_name);
MiningConfig parse_args(int argc, char *argv[]);
