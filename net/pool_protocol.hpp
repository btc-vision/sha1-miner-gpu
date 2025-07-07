#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace MiningPool {
    // Protocol version
    constexpr uint32_t PROTOCOL_VERSION = 1;

    // Message types
    enum class MessageType {
        // Client -> Server
        HELLO = 0x01,
        AUTH = 0x02,
        SUBMIT_SHARE = 0x03,
        KEEPALIVE = 0x04,
        GET_JOB = 0x05,
        HASHRATE_REPORT = 0x06,

        // Server -> Client
        WELCOME = 0x11,
        AUTH_RESPONSE = 0x12,
        NEW_JOB = 0x13,
        SHARE_RESULT = 0x14,
        DIFFICULTY_ADJUST = 0x15,
        POOL_STATUS = 0x16,
        ERROR = 0x17,
        RECONNECT = 0x18
    };

    // Error codes
    enum class ErrorCode {
        NONE = 0,
        INVALID_MESSAGE = 1001,
        AUTH_FAILED = 1002,
        INVALID_SHARE = 1003,
        DUPLICATE_SHARE = 1004,
        JOB_NOT_FOUND = 1005,
        RATE_LIMITED = 1006,
        PROTOCOL_ERROR = 1007,
        INTERNAL_ERROR = 1008,
        BANNED = 1009
    };

    // Share difficulty validation result
    enum class ShareStatus {
        ACCEPTED,
        REJECTED_LOW_DIFFICULTY,
        REJECTED_INVALID,
        REJECTED_STALE,
        REJECTED_DUPLICATE
    };

    // Authentication methods
    enum class AuthMethod {
        WORKER_PASS, // Traditional username.worker/password
        API_KEY, // API key based
        CERTIFICATE // TLS client certificate
    };

    // Base message structure
    struct Message {
        MessageType type;
        uint64_t id; // Message ID for request/response matching
        uint64_t timestamp; // Unix timestamp in milliseconds
        json payload; // Message-specific payload

        std::string serialize() const;

        static std::optional<Message> deserialize(const std::string &data);
    };

    // Client messages
    struct HelloMessage {
        uint32_t protocol_version;
        std::string client_version;
        std::vector<std::string> capabilities; // e.g., ["gpu", "cpu", "multi-gpu"]

        json to_json() const;

        static HelloMessage from_json(const json &j);
    };

    struct AuthMessage {
        AuthMethod method;
        std::string username; // Format: "wallet_address.worker_name" or "username.worker"
        std::string password; // Optional password or API key
        std::string session_id; // For reconnection

        json to_json() const;

        static AuthMessage from_json(const json &j);
    };

    struct SubmitShareMessage {
        std::string job_id;
        uint64_t nonce;
        std::vector<uint8_t> hash; // 20 bytes SHA-1 hash
        uint32_t matching_bits;
        std::string worker_name; // Optional, for multi-worker setups

        json to_json() const;

        static SubmitShareMessage from_json(const json &j);
    };

    struct HashrateReportMessage {
        double hashrate; // Hashes per second
        uint32_t gpu_count;
        std::vector<double> gpu_hashrates; // Per-GPU hashrates
        uint64_t shares_found;
        uint64_t uptime_seconds;

        json to_json() const;

        static HashrateReportMessage from_json(const json &j);
    };

    // Server messages
    struct WelcomeMessage {
        uint32_t protocol_version;
        std::string pool_name;
        std::string pool_version;
        std::vector<std::string> supported_algorithms; // ["sha1-collision"]
        uint32_t min_difficulty;
        uint32_t max_difficulty;

        json to_json() const;

        static WelcomeMessage from_json(const json &j);
    };

    struct AuthResponseMessage {
        bool success;
        std::string session_id;
        std::string worker_id;
        ErrorCode error_code;
        std::string error_message;

        json to_json() const;

        static AuthResponseMessage from_json(const json &j);
    };

    struct JobMessage {
        std::string job_id;
        std::vector<uint8_t> base_message; // 32 bytes
        std::vector<uint8_t> target_hash; // 20 bytes
        uint32_t difficulty; // Required matching bits
        uint64_t nonce_start; // Starting nonce range
        uint64_t nonce_end; // Ending nonce range (0 = no limit)
        uint32_t expires_in_seconds; // Job expiration time
        bool clean_jobs; // Should drop all previous jobs

        json to_json() const;

        static JobMessage from_json(const json &j);
    };

    struct ShareResultMessage {
        std::string share_id;
        ShareStatus status;
        std::string job_id;
        double difficulty_credited; // Actual difficulty credited (for vardiff)
        uint64_t total_shares; // Total shares from this worker
        double total_difficulty; // Total difficulty submitted
        std::string message; // Optional message (e.g., "Block found!")

        json to_json() const;

        static ShareResultMessage from_json(const json &j);
    };

    struct DifficultyAdjustMessage {
        uint32_t new_difficulty;
        std::string reason; // "vardiff", "manual", "pool_adjust"
        uint32_t min_difficulty; // Minimum allowed
        uint32_t max_difficulty; // Maximum allowed
        double target_time; // Target seconds between shares

        json to_json() const;

        static DifficultyAdjustMessage from_json(const json &j);
    };

    struct PoolStatusMessage {
        uint64_t active_workers;
        double pool_hashrate;
        uint64_t blocks_found;
        uint64_t total_shares;
        double average_difficulty;
        std::map<std::string, std::string> network_info; // Additional network stats

        json to_json() const;

        static PoolStatusMessage from_json(const json &j);
    };

    // Connection configuration
    struct PoolConfig {
        std::string url; // ws://pool.example.com:3333 or wss://...
        bool use_tls;
        std::string tls_cert_file; // Optional client certificate
        std::string tls_key_file; // Optional client key
        bool verify_server_cert;

        // Reconnection settings
        uint32_t reconnect_delay_ms = 5000;
        uint32_t max_reconnect_delay_ms = 60000;
        int reconnect_attempts = -1; // -1 = infinite

        // Performance settings
        uint32_t keepalive_interval_s = 30;
        uint32_t response_timeout_ms = 10000;
        uint32_t share_submit_timeout_ms = 5000;

        // Worker settings
        std::string username;
        std::string password;
        std::string worker_name;
        AuthMethod auth_method = AuthMethod::WORKER_PASS;
        bool debug_mode = false;
    };

    // Worker statistics
    struct WorkerStats {
        std::string worker_id;
        std::chrono::steady_clock::time_point connected_since;
        uint64_t shares_accepted;
        uint64_t shares_rejected;
        double total_difficulty_accepted;
        double average_hashrate;
        double current_hashrate;
        uint32_t current_difficulty;
        std::chrono::steady_clock::time_point last_share_time;
        std::chrono::steady_clock::time_point last_job_time;
    };

    // Share information for submission
    struct Share {
        std::string job_id;
        uint64_t nonce;
        std::vector<uint8_t> hash;
        uint32_t matching_bits;
        std::chrono::steady_clock::time_point found_time;

        // Calculated fields
        double difficulty() const {
            return std::pow(2.0, matching_bits);
        }
    };

    // Job tracking
    struct PoolJob {
        std::string job_id;
        JobMessage job_data;
        std::chrono::steady_clock::time_point received_time;
        std::chrono::steady_clock::time_point expiry_time;
        bool is_active;

        bool is_expired() const {
            return std::chrono::steady_clock::now() > expiry_time;
        }
    };

    // Callback interfaces
    class IPoolEventHandler {
    public:
        virtual ~IPoolEventHandler() = default;

        // Connection events
        virtual void on_connected() = 0;

        virtual void on_disconnected(const std::string &reason) = 0;

        virtual void on_error(ErrorCode code, const std::string &message) = 0;

        // Authentication
        virtual void on_authenticated(const std::string &worker_id) = 0;

        virtual void on_auth_failed(ErrorCode code, const std::string &reason) = 0;

        // Job management
        virtual void on_new_job(const PoolJob &job) = 0;

        virtual void on_job_cancelled(const std::string &job_id) = 0;

        // Share results
        virtual void on_share_accepted(const ShareResultMessage &result) = 0;

        virtual void on_share_rejected(const ShareResultMessage &result) = 0;

        // Difficulty adjustment
        virtual void on_difficulty_changed(uint32_t new_difficulty) = 0;

        // Pool status
        virtual void on_pool_status(const PoolStatusMessage &status) = 0;
    };

    // Utility functions
    namespace Utils {
        // Convert binary data to hex string
        std::string binary_to_hex(const std::vector<uint8_t> &data);

        // Convert hex string to binary data
        std::vector<uint8_t> hex_to_binary(const std::string &hex);

        // Calculate share difficulty from matching bits
        double calculate_share_difficulty(uint32_t matching_bits);

        // Validate share against target difficulty
        bool validate_share_difficulty(const Share &share, uint32_t target_difficulty);

        // Generate unique message ID
        uint64_t generate_message_id();

        // Get current timestamp in milliseconds
        uint64_t current_timestamp_ms();
    }
} // namespace MiningPool
