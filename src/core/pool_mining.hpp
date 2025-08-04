#pragma once

#include "../../net/pool_integration.hpp"
#include "../configs/config.hpp"

// Pool mining functions
void display_pool_stats(const MiningPool::PoolMiningSystem::PoolMiningStats &stats);

int run_pool_mining(const MiningConfig &config);
