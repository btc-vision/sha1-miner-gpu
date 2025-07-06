# SHA-1 Miner

A highly optimized CUDA-based miner for finding SHA-1 near-collisions. This implementation uses advanced GPU optimization techniques to achieve maximum performance while minimizing PCIe bandwidth usage.

## Key Features

- **Near-Collision Detection**: Configurable difficulty levels (number of matching bits)
- **Early Exit Optimization**: Stops computation early when a hash cannot meet difficulty requirements
- **Minimal PCIe Bandwidth**: Only transfers candidates that meet difficulty threshold
- **Multi-Stream Processing**: Overlaps computation and memory transfers
- **Warp-Level Optimization**: Uses warp shuffle operations for efficient reduction
- **Advanced Bit Manipulation**: Fast bit counting and matching algorithms

## Architecture Overview

### 1. **Kernel Design**
The core mining kernel (`sha1_near_collision_kernel`) implements:
- Early exit checks every 20 rounds
- Warp-level collaborative processing
- Shared memory for result aggregation
- Multiple nonces per thread (configurable via `NONCES_PER_THREAD`)

### 2. **Difficulty System**
- **Matching Bits**: Primary metric - counts total matching bits across 160-bit hash
- **Difficulty Score**: Secondary metric - counts consecutive matching bits from MSB
- **Configurable Threshold**: Only results meeting the difficulty are returned

### 3. **Performance Optimizations**
- **PTX Instructions**: Direct use of PTX for rotations and byte swapping
- **Instruction-Level Parallelism**: Unrolled loops with optimal scheduling
- **Memory Coalescing**: Aligned memory accesses for maximum bandwidth
- **Constant Memory**: Job parameters stored in fast constant memory
- **L2 Cache Persistence**: Leverages persistent L2 cache on Ampere+ GPUs

## Building

### Requirements
- CUDA Toolkit 11.0 or newer
- C++17 compatible compiler
- CMake 3.18 or newer
- NVIDIA GPU with Compute Capability 7.5+ (RTX 20xx or newer recommended)

### Build Instructions
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Usage

### Basic Mining
```bash
./sha1_miner --gpu 0 --difficulty 100 --duration 300
```

### Mining with Specific Target
```bash
./sha1_miner --gpu 0 --difficulty 120 \
    --message "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" \
    --target "da39a3ee5e6b4b0d3255bfef95601890afd80709"
```

### Benchmark Mode
```bash
./sha1_miner --benchmark
```

### Command Line Options
- `--gpu <id>`: GPU device ID (default: 0)
- `--difficulty <bits>`: Number of bits that must match (default: 120)
- `--duration <seconds>`: Mining duration (default: 60)
- `--target <hex>`: Target hash in hex (40 characters)
- `--message <hex>`: Base message in hex (64 characters)
- `--benchmark`: Run performance benchmark
- `--help`: Show help message

## Performance Tuning

### Kernel Parameters
Edit `sha1_miner.cuh` to adjust:
- `NONCES_PER_THREAD`: Hashes computed per thread (default: 8)
- `MAX_CANDIDATES_PER_BATCH`: Result buffer size (default: 1024)

### Launch Configuration
The system automatically configures based on GPU capabilities:
- Blocks per stream: `SM_count * 4`
- Threads per block: 256
- Number of streams: 4

### Memory Usage
Approximate GPU memory usage:
- Result buffers: `num_streams * MAX_CANDIDATES_PER_BATCH * sizeof(MiningResult)`
- Constant memory: ~100 bytes for job parameters
- Shared memory: Minimal (used for warp-level aggregation)

## Algorithm Details

### Early Exit Strategy
The kernel implements progressive checking:
1. After 40 rounds: Check if partial state is on track
2. After 60 rounds: More aggressive check with tighter bounds
3. Estimates maximum possible matching bits based on rounds remaining

### Near-Collision Detection
Two metrics are used:
1. **Total Matching Bits**: Popcount of XOR between computed and target hash
2. **Consecutive Matching Bits**: Leading zeros in XOR result

### Nonce Distribution
- Each thread processes `NONCES_PER_THREAD` consecutive nonces
- Nonces are distributed across blocks and threads for optimal coverage
- 64-bit nonce space allows for extended mining sessions

## Expected Performance

On modern GPUs (RTX 30xx/40xx):
- Hash rate: 50-200 GH/s depending on GPU and early exit effectiveness
- PCIe bandwidth: <1 MB/s for typical difficulty levels
- Power efficiency: ~300-500 MH/J

## Monitoring

The miner displays real-time statistics:
- Instantaneous and average hash rates
- Number of candidates found
- Total hashes computed

For detailed profiling:
```bash
# System-wide profiling
make profile

# Kernel-specific profiling
make ncu-profile
```

## Troubleshooting

### Low Hash Rate
- Ensure GPU is not thermal throttling
- Check that GPU boost clocks are active
- Verify PCIe link speed (should be Gen3 x16 or better)

### No Candidates Found
- Lower the difficulty setting
- Increase mining duration
- Verify target hash is correct

### Build Errors
- Update CUDA toolkit to latest version
- Ensure compute capability matches your GPU
- Check CMake version requirements

## Future Improvements

Potential optimizations to explore:
1. **Adaptive Difficulty**: Dynamically adjust based on success rate
2. **Multi-GPU Support**: Distribute work across multiple GPUs
3. **CPU Verification**: Offload candidate verification to CPU
4. **Network Pool Support**: Connect to mining pools
5. **Advanced Collision Techniques**: Implement differential path construction

## References

- [SHAttered Attack](https://shattered.io/) - First practical SHA-1 collision
- [SHA-1 is a Shambles](https://sha-mbles.github.io/) - Chosen-prefix collisions
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/) - NVIDIA documentation

## License

This software is provided for educational and research purposes only. Use responsibly and in accordance with applicable laws and regulations.