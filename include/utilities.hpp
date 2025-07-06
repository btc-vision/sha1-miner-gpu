#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <atomic>

// Global shutdown flag
extern std::atomic<bool> g_shutdown;

// The calculate_sha1 and hex_to_bytes functions are already defined in sha1_miner.cuh
// We'll just include that header when needed instead of redefining them

// Additional utility functions that don't conflict

// Print a byte array as hex
inline void print_hex(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(data[i]);
    }
    std::cout << std::dec;
}

// Print a vector of bytes as hex
inline void print_hex(const std::vector<uint8_t> &data) {
    print_hex(data.data(), data.size());
}

// Convert uint32_t array to byte array (big-endian)
inline std::vector<uint8_t> uint32_array_to_bytes(const uint32_t *arr, size_t count) {
    std::vector<uint8_t> result(count * 4);
    for (size_t i = 0; i < count; i++) {
        result[i * 4 + 0] = (arr[i] >> 24) & 0xFF;
        result[i * 4 + 1] = (arr[i] >> 16) & 0xFF;
        result[i * 4 + 2] = (arr[i] >> 8) & 0xFF;
        result[i * 4 + 3] = arr[i] & 0xFF;
    }
    return result;
}

// Convert byte array to uint32_t array (big-endian)
inline std::vector<uint32_t> bytes_to_uint32_array(const uint8_t *bytes, size_t len) {
    std::vector<uint32_t> result(len / 4);
    for (size_t i = 0; i < len / 4; i++) {
        result[i] = (static_cast<uint32_t>(bytes[i * 4]) << 24) |
                    (static_cast<uint32_t>(bytes[i * 4 + 1]) << 16) |
                    (static_cast<uint32_t>(bytes[i * 4 + 2]) << 8) |
                    static_cast<uint32_t>(bytes[i * 4 + 3]);
    }
    return result;
}

// Generate a timestamp string
inline std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Format hash rate for display
inline std::string format_hashrate(double hashrate) {
    std::stringstream ss;
    if (hashrate >= 1e12) {
        ss << std::fixed << std::setprecision(2) << (hashrate / 1e12) << " TH/s";
    } else if (hashrate >= 1e9) {
        ss << std::fixed << std::setprecision(2) << (hashrate / 1e9) << " GH/s";
    } else if (hashrate >= 1e6) {
        ss << std::fixed << std::setprecision(2) << (hashrate / 1e6) << " MH/s";
    } else if (hashrate >= 1e3) {
        ss << std::fixed << std::setprecision(2) << (hashrate / 1e3) << " KH/s";
    } else {
        ss << std::fixed << std::setprecision(2) << hashrate << " H/s";
    }
    return ss.str();
}

// Format large numbers with commas
inline std::string format_number(uint64_t n) {
    std::string num = std::to_string(n);
    int insertPosition = num.length() - 3;
    while (insertPosition > 0) {
        num.insert(insertPosition, ",");
        insertPosition -= 3;
    }
    return num;
}

// Calculate expected time to find a solution
inline double expected_time_seconds(uint32_t difficulty, double hashrate) {
    return std::pow(2.0, difficulty) / hashrate;
}

// Format time duration
inline std::string format_duration(double seconds) {
    std::stringstream ss;
    if (seconds < 60) {
        ss << std::fixed << std::setprecision(1) << seconds << " seconds";
    } else if (seconds < 3600) {
        ss << std::fixed << std::setprecision(1) << (seconds / 60) << " minutes";
    } else if (seconds < 86400) {
        ss << std::fixed << std::setprecision(1) << (seconds / 3600) << " hours";
    } else {
        ss << std::fixed << std::setprecision(1) << (seconds / 86400) << " days";
    }
    return ss.str();
}

#endif // UTILITIES_HPP
