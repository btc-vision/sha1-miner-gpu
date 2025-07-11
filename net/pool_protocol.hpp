#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <chrono>
#include <mutex>
#include <optional>
#include <nlohmann/json.hpp>

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
        ERROR_PROBLEM = 0x17,
        RECONNECT = 0x18,
        KEEP_ALIVE_RESPONSE = 0x19,
        REPORT_RECEIVED = 0x1A
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
        BANNED = 1009,
        NO_WORK = 1010,
        CONNECTION_LOST = 1011,
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
        nlohmann::json payload; // Message-specific payload

        std::string serialize() const;

        static std::optional<Message> deserialize(const std::string &data);
    };

    // Client messages
    struct HelloMessage {
        uint32_t protocol_version;
        std::string client_version;
        std::vector<std::string> capabilities;
        // e.g., ["gpu", "cpu", "multi-gpu", "vardiff", "unique-targets", "opnet-integration"]
        std::string user_agent;

        nlohmann::json to_json() const;

        static HelloMessage from_json(const nlohmann::json &j);
    };

    // Client info for better difficulty adjustment
    struct ClientInfo {
        double estimated_hashrate = 0;
        uint32_t gpu_count = 0;
        std::string miner_version;
    };

    struct AuthMessage {
        AuthMethod method;
        std::string username; // Format: "wallet_address.worker_name"
        std::string password; // Optional password or API key
        std::string session_id; // For reconnection
        std::string otp; // Optional OTP
        std::string client_nonce; // Optional nonce
        std::optional<ClientInfo> client_info; // Optional client information

        nlohmann::json to_json() const;

        static AuthMessage from_json(const nlohmann::json &j);
    };

    struct SubmitShareMessage {
        std::string job_id;
        uint64_t nonce;
        std::string hash; // Hex string of SHA-1 hash
        uint32_t matching_bits;
        std::string worker_name; // Optional, for multi-worker setups
        std::string extra_nonce; // Optional extra nonce

        nlohmann::json to_json() const;

        static SubmitShareMessage from_json(const nlohmann::json &j);
    };

    struct HashrateReportMessage {
        double hashrate; // Hashes per second
        uint64_t shares_submitted;
        uint64_t shares_accepted;
        uint64_t uptime_seconds;
        uint32_t gpu_count;
        nlohmann::json gpu_stats; // Flexible GPU statistics

        nlohmann::json to_json() const;

        static HashrateReportMessage from_json(const nlohmann::json &j);
    };

    // Server messages
    struct WelcomeMessage {
        std::string pool_name;
        std::string pool_version;
        uint32_t protocol_version;
        uint32_t min_difficulty;
        std::vector<std::string> features;
        std::string motd;

        nlohmann::json to_json() const;

        static WelcomeMessage from_json(const nlohmann::json &j);
    };

    struct AuthResponseMessage {
        bool success;
        std::string session_id;
        std::string worker_id;
        uint32_t initial_difficulty = 0;
        ErrorCode error_code;
        std::string error_message;

        nlohmann::json to_json() const;

        static AuthResponseMessage from_json(const nlohmann::json &j);
    };

    struct JobMessage {
        std::string job_id;
        uint32_t target_difficulty;
        std::string target_pattern;
        std::string prefix_data;
        std::string suffix_data;
        uint64_t nonce_start;
        uint64_t nonce_end;
        std::string algorithm;
        nlohmann::json extra_data;
        bool clean_jobs;
        uint32_t expires_in_seconds;

        nlohmann::json to_json() const;

        static JobMessage from_json(const nlohmann::json &j);
    };

    struct ShareResultMessage {
        std::string job_id;
        ShareStatus status;
        uint32_t difficulty_credited;
        uint32_t bits_matched = 0;
        std::string message;
        double share_value = 0;
        uint64_t total_shares = 0;
        nlohmann::json difficulty_info;

        nlohmann::json to_json() const;

        static ShareResultMessage from_json(const nlohmann::json &j);
    };

    struct DifficultyAdjustMessage {
        uint32_t new_difficulty;
        std::string reason;
        uint32_t effective_in_seconds;

        nlohmann::json to_json() const;

        static DifficultyAdjustMessage from_json(const nlohmann::json &j);
    };

    struct PoolStatusMessage {
        uint32_t connected_workers;
        double total_hashrate;
        double shares_per_minute;
        uint64_t epochs_completed;
        uint32_t current_epoch_number;
        uint64_t current_epoch_shares;
        double pool_fee_percent;
        double minimum_payout;
        nlohmann::json extra_info;

        nlohmann::json to_json() const;

        static PoolStatusMessage from_json(const nlohmann::json &j);
    };

    // Connection configuration
    struct PoolConfig {
        std::string url;
        bool use_tls = false;
        std::string tls_cert_file;
        std::string tls_key_file;
        bool verify_server_cert = true;

        // Reconnection settings
        uint32_t reconnect_delay_ms = 5000;
        uint32_t max_reconnect_delay_ms = 60000;
        int reconnect_attempts = -1;

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

    struct WorkerStats {
        std::string worker_id;
        std::chrono::steady_clock::time_point connected_since;
        uint64_t shares_accepted = 0;
        uint64_t shares_rejected = 0;
        double total_difficulty_accepted = 0;
        double average_hashrate = 0;
        double current_hashrate = 0;
        uint32_t current_difficulty = 0;
        std::chrono::steady_clock::time_point last_share_time;
        std::chrono::steady_clock::time_point last_job_time;
    };

    // Share information for submission
    struct Share {
        std::string job_id;
        uint64_t nonce;
        std::string hash; // Hex string
        uint32_t matching_bits;
        std::chrono::steady_clock::time_point found_time;

        // Calculated fields
        double difficulty() const {
            return std::pow(2.0, matching_bits);
        }
    };

    // Job tracking with epoch support
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
        // Generate unique message ID
        uint64_t generate_message_id();

        // Get current timestamp in milliseconds
        uint64_t current_timestamp_ms();

        // Convert bytes to hex string
        std::string bytes_to_hex(const uint8_t *data, size_t len);

        // Convert hex string to bytes
        std::vector<uint8_t> hex_to_bytes(const std::string &hex);
    }

    // Difficulty converter utility class
    class DifficultyConverter {
    public:
        // Convert bits to scaled difficulty (2^bits)
        static double bitsToScaledDifficulty(uint32_t bits) {
            return std::pow(2.0, static_cast<double>(bits));
        }

        // Format difficulty for display
        static std::string formatDifficulty(double difficulty) {
            if (difficulty < 1e6) return std::to_string(static_cast<uint64_t>(difficulty));
            if (difficulty < 1e9) return std::to_string(static_cast<uint64_t>(difficulty / 1e6)) + "M";
            if (difficulty < 1e12) return std::to_string(static_cast<uint64_t>(difficulty / 1e9)) + "G";
            return std::to_string(static_cast<uint64_t>(difficulty / 1e12)) + "T";
        }
    };
} // namespace MiningPool
