#include "config.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "../logging/logger.hpp"

po::options_description create_options_description()
{
    po::options_description desc("SHA-1 OP_NET Miner Options");
    desc.add_options()("help,h", "Show this help message")

        // GPU options
        ("gpu", po::value<int>()->default_value(0), "GPU device ID")("all-gpus", "Use all available GPUs")(
            "gpus", po::value<std::string>(), "Use specific GPUs (e.g., 0,1,2)")

        // Solo mining options
        ("difficulty", po::value<uint32_t>()->default_value(50), "Number of bits that must match")

        // Core performance options
        ("streams", po::value<int>()->default_value(0), "Number of CUDA/HIP streams (0=auto)")(
            "threads", po::value<int>()->default_value(0), "Threads per block (0=auto)")(
            "blocks", po::value<int>()->default_value(0), "Blocks per stream (0=auto)")(
            "auto-tune", po::bool_switch()->default_value(true), "Auto-tune for optimal performance")

        // Memory configuration
        ("result-buffer", po::value<size_t>()->default_value(0), "Result buffer size per stream (0=auto)")(
            "no-pinned-memory", "Disable pinned memory usage")

        // Pool mining options
        ("pool", po::value<std::string>(), "Pool URL (ws://host:port or wss://host:port)")(
            "wallet", po::value<std::string>(), "Wallet address for pool mining")("worker", po::value<std::string>(),
                                                                                  "Worker name (default: hostname)")(
            "pool-pass", po::value<std::string>()->default_value("x"), "Pool password")(
            "backup-pool", po::value<std::vector<std::string>>()->multitoken(), "Backup pool URLs for failover")(
            "no-failover", "Disable automatic pool failover")

        // Other options
        ("benchmark", "Run performance benchmark")("test-sha1", "Test SHA-1 implementation")(
            "debug", "Enable debug mode")("debug-level", po::value<int>()->default_value(2),
                                          "Debug level (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)")(
            "no-colors", "Disable colored output");

    return desc;
}

void print_usage_examples(const char *program_name)
{
    std::cout << "Examples:\n";
    std::cout << "  Basic solo mining:\n";
    std::cout << "    " << program_name << " --gpu 0 --difficulty 45\n\n";

    std::cout << "  Performance tuning:\n";
    std::cout << "    " << program_name << " --gpu 0 --streams 4 --threads 256 --blocks 128\n\n";

    std::cout << "  Multi-GPU:\n";
    std::cout << "    " << program_name << " --all-gpus --threads 512 --blocks 256\n\n";

    std::cout << "  Pool mining:\n";
    std::cout << "    " << program_name << " --pool ws://pool.example.com:3333 --wallet YOUR_WALLET \\\n";
    std::cout << "                    --worker rig1 --streams 8 --threads 256\n\n";

    std::cout << "  Benchmark mode:\n";
    std::cout << "    " << program_name << " --benchmark --gpu 0 --threads 512\n";
}

bool MiningConfig::wasUserSpecified(const std::string &param) const
{
    if (param == "num_streams")
        return user_specified.num_streams;
    if (param == "threads_per_block")
        return user_specified.threads_per_block;
    if (param == "blocks_per_stream")
        return user_specified.blocks_per_stream;
    if (param == "result_buffer_size")
        return user_specified.result_buffer_size;
    return false;
}

MiningConfig parse_args(int argc, char *argv[])
{
    MiningConfig config;

    auto desc = create_options_description();

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error &e) {
        LOG_ERROR("MAIN", "Error: ", e.what());
        std::cerr << desc << "\n";
        std::exit(1);
    }

    if (vm.count("help")) {
        std::cout << "SHA-1 OP_NET Miner\n\n";
        std::cout << desc << "\n";
        print_usage_examples(argv[0]);
        std::exit(0);
    }

    // Parse GPU configuration
    config.gpu_id       = vm["gpu"].as<int>();
    config.use_all_gpus = vm.count("all-gpus") > 0;

    if (vm.count("gpus")) {
        auto gpu_list = vm["gpus"].as<std::string>();
        std::stringstream ss(gpu_list);
        std::string token;
        while (std::getline(ss, token, ',')) {
            config.gpu_ids.push_back(std::stoi(token));
        }
    }

    // Parse solo mining options
    config.difficulty = vm["difficulty"].as<uint32_t>();

    // Parse performance options with tracking
    if (vm.count("streams") && vm["streams"].as<int>() != 0) {
        config.num_streams                = vm["streams"].as<int>();
        config.user_specified.num_streams = true;
    }

    if (vm.count("threads") && vm["threads"].as<int>() != 0) {
        config.threads_per_block                = vm["threads"].as<int>();
        config.user_specified.threads_per_block = true;
    }

    if (vm.count("blocks") && vm["blocks"].as<int>() != 0) {
        config.blocks_per_stream                = vm["blocks"].as<int>();
        config.user_specified.blocks_per_stream = true;
    }

    config.auto_tune = vm["auto-tune"].as<bool>();

    // Parse memory configuration
    if (vm.count("result-buffer") && vm["result-buffer"].as<size_t>() != 0) {
        config.result_buffer_size                = vm["result-buffer"].as<size_t>();
        config.user_specified.result_buffer_size = true;
    }

    config.use_pinned_memory = vm.count("no-pinned-memory") == 0;

    // Parse pool options
    if (vm.count("pool")) {
        config.pool_url = vm["pool"].as<std::string>();
        config.use_pool = true;
    }

    if (vm.count("wallet"))
        config.pool_wallet = vm["wallet"].as<std::string>();
    if (vm.count("worker"))
        config.worker_name = vm["worker"].as<std::string>();
    config.pool_password = vm["pool-pass"].as<std::string>();

    if (vm.count("backup-pool")) {
        config.backup_pools = vm["backup-pool"].as<std::vector<std::string>>();
    }

    config.enable_pool_failover = vm.count("no-failover") == 0;

    // Parse other options
    config.benchmark   = vm.count("benchmark") > 0;
    config.test_sha1   = vm.count("test-sha1") > 0;
    config.debug_mode  = vm.count("debug") > 0;
    config.debug_level = vm["debug-level"].as<int>();
    config.no_colors   = vm.count("no-colors") > 0;

    // Validate pool configuration
    if (config.use_pool) {
        if (config.pool_wallet.empty()) {
            LOG_ERROR("MAIN", "Error: --wallet is required for pool mining");
            std::exit(1);
        }
    }

    // Validate some configurations
    if (config.threads_per_block > 0 && (config.threads_per_block & config.threads_per_block - 1) != 0) {
        LOG_WARN("MAIN", "Threads per block should be a power of 2 for optimal performance");
    }

    // If auto-tune is disabled, log which parameters will be manually configured
    if (!config.auto_tune) {
        LOG_INFO("MAIN", "Auto-tune disabled. Using manual configuration for:");
        if (config.user_specified.num_streams)
            LOG_INFO("MAIN", "  - Streams: ", config.num_streams);
        if (config.user_specified.threads_per_block)
            LOG_INFO("MAIN", "  - Threads per block: ", config.threads_per_block);
        if (config.user_specified.blocks_per_stream)
            LOG_INFO("MAIN", "  - Blocks per stream: ", config.blocks_per_stream);
    }

    return config;
}
