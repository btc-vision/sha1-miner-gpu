#pragma once

// This wrapper provides additional convenience functions for the SHA1 class
// to make it easier to work with binary data

#include "cxxsha1.hpp"
#include <vector>
#include <cstdint>

// Helper function to compute SHA-1 of binary data and return binary result
inline std::vector<uint8_t> sha1_binary(const uint8_t* data, size_t len) {
    SHA1 sha1;
    sha1.update(std::string(reinterpret_cast<const char*>(data), len));
    std::string hex = sha1.final();

    std::vector<uint8_t> result(20);
    for (int i = 0; i < 20; i++) {
        result[i] = static_cast<uint8_t>(std::stoi(hex.substr(i * 2, 2), nullptr, 16));
    }

    return result;
}

// Helper function to compute SHA-1 of a vector
inline std::vector<uint8_t> sha1_binary(const std::vector<uint8_t>& data) {
    return sha1_binary(data.data(), data.size());
}

// Helper to convert binary hash to hex string
inline std::string sha1_hex(const uint8_t* hash) {
    std::ostringstream oss;
    for (int i = 0; i < 20; i++) {
        oss << std::hex << std::setfill('0') << std::setw(2)
            << static_cast<int>(hash[i]);
    }
    return oss.str();
}

// Helper to convert hex string to binary
inline std::vector<uint8_t> hex_to_binary(const std::string& hex) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        result.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return result;
}