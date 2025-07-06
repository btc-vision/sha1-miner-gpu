#!/bin/bash

# SHA-1 Near-Collision Miner - Comprehensive Test Script
# This script runs various tests to validate the miner implementation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0
SKIPPED=0

# Helper functions
print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

print_section() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

# Check if executables exist
check_executables() {
    print_section "Checking Executables"

    if [ -f "./sha1_miner" ] || [ -f "./build/sha1_miner" ]; then
        print_pass "SHA-1 miner found"
        if [ -f "./build/sha1_miner" ]; then
            MINER="./build/sha1_miner"
        else
            MINER="./sha1_miner"
        fi
    else
        print_fail "SHA-1 miner not found"
        echo "Please build the project first: ./build.sh"
        exit 1
    fi

    if [ -f "./verify_sha1" ] || [ -f "./build/verify_sha1" ]; then
        print_pass "Verification tool found"
        if [ -f "./build/verify_sha1" ]; then
            VERIFY="./build/verify_sha1"
        else
            VERIFY="./verify_sha1"
        fi
    else
        print_fail "Verification tool not found"
        exit 1
    fi
}

# Run basic tests
run_basic_tests() {
    print_section "Running Basic Tests"

    # Test 1: Help output
    print_test "Testing help output"
    if $MINER --help &>/dev/null; then
        print_pass "Help command works"
    else
        print_fail "Help command failed"
    fi

    # Test 2: SHA-1 verification
    print_test "Running SHA-1 verification tests"
    if $VERIFY > verify_output.log 2>&1; then
        if grep -q "All tests completed successfully" verify_output.log; then
            print_pass "SHA-1 implementation verified"
        else
            print_fail "SHA-1 verification failed"
            cat verify_output.log
        fi
    else
        print_fail "Verification tool crashed"
    fi
    rm -f verify_output.log
}

# Test different difficulty levels
test_difficulty_levels() {
    print_section "Testing Difficulty Levels"

    DIFFICULTIES=(60 80 100 120)
    DURATION=10

    for DIFF in "${DIFFICULTIES[@]}"; do
        print_test "Testing difficulty $DIFF bits"

        if timeout 15 $MINER --gpu 0 --difficulty $DIFF --duration $DURATION > test_$DIFF.log 2>&1; then
            # Check if any candidates were found
            if grep -q "CANDIDATE" test_$DIFF.log; then
                COUNT=$(grep -c "CANDIDATE" test_$DIFF.log)
                print_pass "Difficulty $DIFF: Found $COUNT candidates"
            else
                if [ $DIFF -le 80 ]; then
                    print_fail "Difficulty $DIFF: No candidates found (expected some)"
                else
                    print_pass "Difficulty $DIFF: No candidates found (expected for high difficulty)"
                fi
            fi

            # Check hash rate
            if RATE=$(grep -oP "Rate: \K[0-9.]+" test_$DIFF.log | tail -1); then
                echo "  Hash rate: $RATE GH/s"
            fi
        else
            print_fail "Difficulty $DIFF test failed or timed out"
        fi

        rm -f test_$DIFF.log
    done
}

# Test error handling
test_error_handling() {
    print_section "Testing Error Handling"

    # Test 1: Invalid GPU ID
    print_test "Testing invalid GPU ID"
    if $MINER --gpu 999 --duration 1 &>/dev/null; then
        print_fail "Should have failed with invalid GPU ID"
    else
        print_pass "Correctly rejected invalid GPU ID"
    fi

    # Test 2: Invalid difficulty
    print_test "Testing invalid difficulty"
    if $MINER --difficulty 200 --duration 1 &>/dev/null; then
        print_fail "Should have failed with difficulty > 160"
    else
        print_pass "Correctly rejected invalid difficulty"
    fi

    # Test 3: Invalid hex input
    print_test "Testing invalid hex input"
    if $MINER --target "ZZZZ" --duration 1 &>/dev/null; then
        print_fail "Should have failed with invalid hex"
    else
        print_pass "Correctly rejected invalid hex input"
    fi
}

# Performance benchmark
run_performance_benchmark() {
    print_section "Running Performance Benchmark"

    print_test "Running 30-second benchmark"

    if timeout 35 $MINER --gpu 0 --difficulty 100 --duration 30 > benchmark.log 2>&1; then
        # Extract performance metrics
        if FINAL_RATE=$(grep -oP "Average Rate: \K[0-9.]+" benchmark.log | tail -1); then
            print_pass "Benchmark completed: $FINAL_RATE GH/s average"

            # Check if performance is reasonable
            if (( $(echo "$FINAL_RATE > 10" | bc -l) )); then
                echo -e "  ${GREEN}Performance looks good!${NC}"
            else
                echo -e "  ${YELLOW}Performance seems low. Check GPU utilization.${NC}"
            fi
        else
            print_fail "Could not extract performance metrics"
        fi

        # Check for any errors
        if grep -q "Error\|error\|CUDA Error" benchmark.log; then
            print_fail "Errors detected during benchmark"
            grep -i error benchmark.log
        fi
    else
        print_fail "Benchmark failed or timed out"
    fi

    rm -f benchmark.log
}

# Memory leak test
test_memory_leak() {
    print_section "Testing Memory Stability"

    print_test "Running memory leak test (60 seconds)"

    # Get initial GPU memory
    INIT_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "  Initial GPU memory: ${INIT_MEM} MB"

    # Run miner
    if timeout 65 $MINER --gpu 0 --difficulty 110 --duration 60 > memtest.log 2>&1; then
        # Get final GPU memory
        sleep 2  # Let GPU memory settle
        FINAL_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "  Final GPU memory: ${FINAL_MEM} MB"

        # Check for significant memory increase
        MEM_DIFF=$((FINAL_MEM - INIT_MEM))
        if [ $MEM_DIFF -lt 100 ]; then
            print_pass "No significant memory leak detected (diff: ${MEM_DIFF} MB)"
        else
            print_fail "Possible memory leak detected (diff: ${MEM_DIFF} MB)"
        fi
    else
        print_fail "Memory test failed"
    fi

    rm -f memtest.log
}

# GPU utilization test
test_gpu_utilization() {
    print_section "Testing GPU Utilization"

    print_test "Monitoring GPU utilization during mining"

    # Start miner in background
    $MINER --gpu 0 --difficulty 100 --duration 20 > utilization.log 2>&1 &
    MINER_PID=$!

    sleep 5  # Let it warm up

    # Monitor GPU utilization
    UTIL_SUM=0
    UTIL_COUNT=0

    for i in {1..10}; do
        if UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); then
            UTIL_SUM=$((UTIL_SUM + UTIL))
            UTIL_COUNT=$((UTIL_COUNT + 1))
            echo -n "."
        fi
        sleep 1
    done
    echo ""

    # Wait for miner to finish
    wait $MINER_PID

    if [ $UTIL_COUNT -gt 0 ]; then
        AVG_UTIL=$((UTIL_SUM / UTIL_COUNT))
        echo "  Average GPU utilization: ${AVG_UTIL}%"

        if [ $AVG_UTIL -gt 80 ]; then
            print_pass "Good GPU utilization (${AVG_UTIL}%)"
        elif [ $AVG_UTIL -gt 50 ]; then
            print_skip "Moderate GPU utilization (${AVG_UTIL}%)"
        else
            print_fail "Low GPU utilization (${AVG_UTIL}%)"
        fi
    else
        print_skip "Could not monitor GPU utilization"
    fi

    rm -f utilization.log
}

# Main test execution
main() {
    echo ""
    echo "SHA-1 Near-Collision Miner Test Suite"
    echo "====================================="
    echo ""

    # Check for NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found. NVIDIA GPU required.${NC}"
        exit 1
    fi

    # Display GPU info
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,compute_cap --format=csv
    echo ""

    # Run all tests
    check_executables
    run_basic_tests
    test_error_handling
    test_difficulty_levels
    run_performance_benchmark
    test_memory_leak
    test_gpu_utilization

    # Summary
    print_section "Test Summary"
    echo -e "Tests passed:  ${GREEN}$PASSED${NC}"
    echo -e "Tests failed:  ${RED}$FAILED${NC}"
    echo -e "Tests skipped: ${YELLOW}$SKIPPED${NC}"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed successfully!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed. Please check the output above.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"