#ifndef SHA1_MINER_FUNCTIONS_H
#define SHA1_MINER_FUNCTIONS_H

#include "sha1_miner.cuh"

void cleanup_mining_system();

// Mining job creation
MiningJob create_mining_job(const uint8_t *message, const uint8_t *target_hash, uint32_t difficulty);

// Main mining loop
void run_mining_loop(MiningJob job);

#endif // SHA1_MINER_FUNCTIONS_H
