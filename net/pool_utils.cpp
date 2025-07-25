#include "pool_protocol.hpp"
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <atomic>

namespace MiningPool {
    static std::atomic<uint64_t> message_counter{1};

    namespace Utils {
        uint64_t generate_message_id() {
            auto now = std::chrono::high_resolution_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()).count();

            // Combine timestamp with counter for uniqueness
            uint64_t counter = message_counter.fetch_add(1);
            return (timestamp << 16) | (counter & 0xFFFF);
        }

        uint64_t current_timestamp_ms() {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        }

        std::string bytes_to_hex(const uint8_t *data, size_t len) {
            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (size_t i = 0; i < len; ++i) {
                ss << std::setw(2) << static_cast<int>(data[i]);
            }
            return ss.str();
        }

        std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
            std::vector<uint8_t> bytes;
            for (size_t i = 0; i < hex.length(); i += 2) {
                std::string byteString = hex.substr(i, 2);
                uint8_t byte = static_cast<uint8_t>(std::stoi(byteString, nullptr, 16));
                bytes.push_back(byte);
            }
            return bytes;
        }
    } // namespace Utils
} // namespace MiningPool
