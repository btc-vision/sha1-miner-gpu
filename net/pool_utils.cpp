#include "pool_protocol.hpp"
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <atomic>

namespace MiningPool {
    namespace Utils {
        uint64_t generate_message_id() {
            // Use a combination of timestamp and counter for uniqueness
            static std::atomic<uint32_t> counter{0};

            auto now = std::chrono::steady_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()
            ).count();

            uint32_t count = counter.fetch_add(1, std::memory_order_relaxed);

            // Combine timestamp (high 32 bits) and counter (low 32 bits)
            return (static_cast<uint64_t>(timestamp & 0xFFFFFFFF) << 32) | count;
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
