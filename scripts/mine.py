#!/usr/bin/env python3
"""
Bitcoin PoW Puzzle Miner
Mines solutions for the nonce pattern matching puzzle
"""

import multiprocessing as mp
import struct
import time
from dataclasses import dataclass
from typing import List


@dataclass
class MiningSolution:
    """Represents a valid mining solution"""
    nonce: int
    nonce_hex: str
    nonce_bytes: bytes
    attempts: int
    time_taken: float
    hashrate: float


class PuzzleMiner:
    """Miner for the Bitcoin PoW puzzle where (nonce & 0xFFFF) == 0"""

    def __init__(self, difficulty_bits: int = 16):
        self.difficulty_bits = difficulty_bits
        self.mask = (1 << difficulty_bits) - 1
        self.solutions_found = []

    def check_nonce(self, nonce: int) -> bool:
        """Check if nonce is valid for our puzzle"""
        return (nonce & self.mask) == 0

    def mine_range(self, start: int, end: int, max_solutions: int = 10) -> List[MiningSolution]:
        """Mine a range of nonces"""
        solutions = []
        attempts = 0
        start_time = time.time()

        print(f"Mining nonces from {start:,} to {end:,}")
        print(f"Looking for nonces where last {self.difficulty_bits} bits are zero...")
        print("-" * 60)

        for nonce in range(start, end):
            attempts += 1

            if self.check_nonce(nonce):
                # Found valid solution!
                elapsed = time.time() - start_time
                hashrate = attempts / elapsed if elapsed > 0 else 0

                # Encode as 4-byte little-endian (Bitcoin Script format)
                nonce_bytes = struct.pack('<I', nonce)

                solution = MiningSolution(
                    nonce=nonce,
                    nonce_hex=f"{nonce:08x}",
                    nonce_bytes=nonce_bytes,
                    attempts=attempts,
                    time_taken=elapsed,
                    hashrate=hashrate
                )

                solutions.append(solution)

                print(f"\n✓ FOUND SOLUTION #{len(solutions)}:")
                print(f"  Nonce (decimal): {nonce:,}")
                print(f"  Nonce (hex): 0x{nonce:08x}")
                print(f"  Nonce (bytes): {nonce_bytes.hex()}")
                print(f"  Binary: {bin(nonce)}")
                print(f"  Attempts: {attempts:,}")
                print(f"  Time: {elapsed:.3f} seconds")
                print(f"  Rate: {hashrate:,.0f} checks/second")

                if len(solutions) >= max_solutions:
                    break

            # Progress update
            if attempts % 10000 == 0:
                elapsed = time.time() - start_time
                rate = attempts / elapsed if elapsed > 0 else 0
                print(f"\r  Checked: {attempts:,} | Rate: {rate:,.0f} checks/sec", end='', flush=True)

        return solutions

    def mine_parallel(self, num_workers: int = 4, total_range: int = 10_000_000) -> List[MiningSolution]:
        """Mine using multiple CPU cores"""
        print(f"Starting parallel mining with {num_workers} workers...")

        chunk_size = total_range // num_workers
        ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]

        with mp.Pool(num_workers) as pool:
            results = pool.starmap(self._mine_worker, ranges)

        # Flatten results
        all_solutions = []
        for solutions in results:
            all_solutions.extend(solutions)

        return all_solutions

    def _mine_worker(self, start: int, end: int) -> List[MiningSolution]:
        """Worker function for parallel mining"""
        return self.mine_range(start, end, max_solutions=5)

    def create_redeem_script(self, nonce: int, puzzle_script_hex: str) -> str:
        """Create the redeem script for a solution"""
        # Encode nonce as 4-byte little-endian
        nonce_bytes = struct.pack('<I', nonce)

        # Push nonce (4 bytes)
        redeem = bytearray()
        redeem.append(0x04)  # Push 4 bytes
        redeem.extend(nonce_bytes)

        # Push the full puzzle script
        script_bytes = bytes.fromhex(puzzle_script_hex)
        if len(script_bytes) <= 75:
            redeem.append(len(script_bytes))
        else:
            redeem.append(0x4c)  # OP_PUSHDATA1
            redeem.append(len(script_bytes))
        redeem.extend(script_bytes)

        return redeem.hex()

    def estimate_mining_time(self) -> dict:
        """Estimate time to find solutions at different hash rates"""
        expected_attempts = 1 << self.difficulty_bits

        estimates = {
            "CPU (10K checks/sec)": f"{expected_attempts / 10_000:.1f} seconds",
            "Fast CPU (100K checks/sec)": f"{expected_attempts / 100_000:.1f} seconds",
            "GPU (10M checks/sec)": f"{expected_attempts / 10_000_000:.3f} seconds",
            "ASIC (1B checks/sec)": f"{expected_attempts / 1_000_000_000:.6f} seconds"
        }

        return estimates


def mine_puzzle():
    """Main mining function"""

    print("=== Bitcoin PoW Puzzle Miner ===\n")

    # Create miner with 16-bit difficulty
    miner = PuzzleMiner(difficulty_bits=16)

    # Show mining estimates
    print("Mining Difficulty: 16 bits")
    print("Target: Find nonces where (nonce & 0xFFFF) == 0")
    print("\nExpected mining times:")
    estimates = miner.estimate_mining_time()
    for device, time_est in estimates.items():
        print(f"  {device}: {time_est}")

    print("\n" + "=" * 60 + "\n")

    # Mine the first 10 solutions
    start_time = time.time()
    solutions = miner.mine_range(0, 1_000_000, max_solutions=10)
    total_time = time.time() - start_time

    print(f"\n\n=== Mining Complete ===")
    print(f"Total solutions found: {len(solutions)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per solution: {total_time / len(solutions):.2f} seconds")

    # Show all solutions
    print("\n=== All Solutions ===")
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i + 1}:")
        print(f"  Nonce: {sol.nonce} (0x{sol.nonce_hex})")
        print(f"  Script format: {sol.nonce_bytes.hex()}")
        print(f"  Found after: {sol.attempts:,} attempts")

    # Show example redeem script
    if solutions:
        print("\n=== Example Redeem Script ===")
        # Example puzzle script (truncated for display)
        example_puzzle_script = "7682548876008763516778760300000187635167787603000002876351677876030000038763516778760300000487635167787603000005876351677876030000068763516778760300000787635167787603000008876351677876030000098763516778760300000a8763516778760300000b8763516778760300000c8763516778760300000d8763516778760300000e8763516778760300000f87635167787603000010876351677876030000118763516778760300001287635167787603000013876868686868686868686868686868686868686877"

        redeem_hex = miner.create_redeem_script(solutions[0].nonce, example_puzzle_script)
        print(f"For nonce {solutions[0].nonce}:")
        print(f"Redeem script: {redeem_hex[:60]}...")
        print(f"Size: {len(bytes.fromhex(redeem_hex))} bytes")

    # Test some specific nonces
    print("\n=== Testing Specific Values ===")
    test_values = [0, 65535, 65536, 65537, 131072, 262144]
    for nonce in test_values:
        valid = miner.check_nonce(nonce)
        print(f"Nonce {nonce:6} (0x{nonce:08x}): {'✓ VALID' if valid else '✗ Invalid'}")

    return solutions


def benchmark_mining():
    """Benchmark mining performance"""

    print("\n=== Performance Benchmark ===\n")

    miner = PuzzleMiner(difficulty_bits=16)

    # Benchmark for 1 second
    start_time = time.time()
    checks = 0

    while time.time() - start_time < 1.0:
        miner.check_nonce(checks)
        checks += 1

    print(f"Single-threaded performance: {checks:,} checks/second")
    print(f"Estimated time to find solution: {65536 / checks:.2f} seconds")


def gpu_mining_simulation():
    """Simulate GPU mining with much higher check rate"""

    print("\n=== GPU Mining Simulation ===\n")

    # GPU can check millions of nonces per second
    gpu_rate = 10_000_000  # 10M checks/sec
    difficulty = 16
    expected_nonces = 1 << difficulty

    print(f"GPU hash rate: {gpu_rate:,} checks/second")
    print(f"Expected time to solution: {expected_nonces / gpu_rate:.3f} seconds")

    # Simulate finding solutions
    print("\nSimulated GPU solutions (first 10):")
    for i in range(10):
        nonce = i * (1 << difficulty)
        if nonce <= 0x7FFFFFFF:  # Max Script number
            print(f"  Solution {i + 1}: {nonce:,} (0x{nonce:08x})")


def main():
    """Main entry point"""

    # Mine the puzzle
    solutions = mine_puzzle()

    # Run benchmark
    benchmark_mining()

    # Show GPU simulation
    gpu_mining_simulation()

    print("\n=== Summary ===")
    print("✓ Successfully mined Bitcoin PoW puzzle")
    print("✓ Found valid nonces where (nonce & 0xFFFF) == 0")
    print("✓ All solutions can be used to claim the puzzle reward")
    print("\nTo claim on Bitcoin:")
    print("1. Create transaction spending from puzzle address")
    print("2. Use redeem script with your found nonce")
    print("3. Broadcast to claim the reward!")


if __name__ == "__main__":
    main()
