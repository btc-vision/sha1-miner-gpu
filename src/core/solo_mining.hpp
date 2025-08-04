#pragma once

#include <vector>
#include "../configs/config.hpp"

// Solo mining functions
int run_solo_mining(const MiningConfig &config, const std::vector<int> &gpu_ids_to_use);
