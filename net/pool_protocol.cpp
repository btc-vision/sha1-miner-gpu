#include "pool_protocol.hpp"
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>

namespace MiningPool {
    // Message serialization/deserialization
    std::string Message::serialize() const {
        json j;
        j["type"] = static_cast<int>(type);
        j["id"] = id;
        j["timestamp"] = timestamp;
        j["payload"] = payload;
        return j.dump();
    }

    std::optional<Message> Message::deserialize(const std::string &data) {
        try {
            json j = json::parse(data);
            Message msg;
            msg.type = static_cast<MessageType>(j["type"].get<int>());
            msg.id = j["id"].get<uint64_t>();
            msg.timestamp = j["timestamp"].get<uint64_t>();
            msg.payload = j["payload"];
            return msg;
        } catch (const std::exception &e) {
            return std::nullopt;
        }
    }

    // HelloMessage implementation
    json HelloMessage::to_json() const {
        json j;
        j["protocol_version"] = protocol_version;
        j["client_version"] = client_version;
        j["capabilities"] = capabilities;
        return j;
    }

    HelloMessage HelloMessage::from_json(const json &j) {
        HelloMessage msg;
        msg.protocol_version = j["protocol_version"].get<uint32_t>();
        msg.client_version = j["client_version"].get<std::string>();
        msg.capabilities = j["capabilities"].get<std::vector<std::string> >();
        return msg;
    }

    // AuthMessage implementation
    json AuthMessage::to_json() const {
        json j;
        j["method"] = static_cast<int>(method);
        j["username"] = username;
        j["password"] = password;
        j["session_id"] = session_id;
        return j;
    }

    AuthMessage AuthMessage::from_json(const json &j) {
        AuthMessage msg;
        msg.method = static_cast<AuthMethod>(j["method"].get<int>());
        msg.username = j["username"].get<std::string>();
        msg.password = j.value("password", "");
        msg.session_id = j.value("session_id", "");
        return msg;
    }

    // SubmitShareMessage implementation
    json SubmitShareMessage::to_json() const {
        json j;
        j["job_id"] = job_id;
        j["nonce"] = nonce;
        j["hash"] = Utils::binary_to_hex(hash);
        j["matching_bits"] = matching_bits;
        j["worker_name"] = worker_name;
        return j;
    }

    SubmitShareMessage SubmitShareMessage::from_json(const json &j) {
        SubmitShareMessage msg;
        msg.job_id = j["job_id"].get<std::string>();
        msg.nonce = j["nonce"].get<uint64_t>();
        msg.hash = Utils::hex_to_binary(j["hash"].get<std::string>());
        msg.matching_bits = j["matching_bits"].get<uint32_t>();
        msg.worker_name = j.value("worker_name", "");
        return msg;
    }

    // HashrateReportMessage implementation
    json HashrateReportMessage::to_json() const {
        json j;
        j["hashrate"] = hashrate;
        j["gpu_count"] = gpu_count;
        j["gpu_hashrates"] = gpu_hashrates;
        j["shares_found"] = shares_found;
        j["uptime_seconds"] = uptime_seconds;
        return j;
    }

    HashrateReportMessage HashrateReportMessage::from_json(const json &j) {
        HashrateReportMessage msg;
        msg.hashrate = j["hashrate"].get<double>();
        msg.gpu_count = j["gpu_count"].get<uint32_t>();
        msg.gpu_hashrates = j.value("gpu_hashrates", std::vector<double>());
        msg.shares_found = j["shares_found"].get<uint64_t>();
        msg.uptime_seconds = j["uptime_seconds"].get<uint64_t>();
        return msg;
    }

    // WelcomeMessage implementation
    json WelcomeMessage::to_json() const {
        json j;
        j["protocol_version"] = protocol_version;
        j["pool_name"] = pool_name;
        j["pool_version"] = pool_version;
        j["supported_algorithms"] = supported_algorithms;
        j["min_difficulty"] = min_difficulty;
        j["max_difficulty"] = max_difficulty;
        return j;
    }

    WelcomeMessage WelcomeMessage::from_json(const json &j) {
        WelcomeMessage msg;
        msg.protocol_version = j["protocol_version"].get<uint32_t>();
        msg.pool_name = j["pool_name"].get<std::string>();
        msg.pool_version = j["pool_version"].get<std::string>();
        msg.supported_algorithms = j["supported_algorithms"].get<std::vector<std::string> >();
        msg.min_difficulty = j["min_difficulty"].get<uint32_t>();
        msg.max_difficulty = j["max_difficulty"].get<uint32_t>();
        return msg;
    }

    // AuthResponseMessage implementation
    json AuthResponseMessage::to_json() const {
        json j;
        j["success"] = success;
        j["session_id"] = session_id;
        j["worker_id"] = worker_id;
        j["error_code"] = static_cast<int>(error_code);
        j["error_message"] = error_message;
        return j;
    }

    AuthResponseMessage AuthResponseMessage::from_json(const json &j) {
        AuthResponseMessage msg;
        msg.success = j["success"].get<bool>();
        msg.session_id = j.value("session_id", "");
        msg.worker_id = j.value("worker_id", "");
        msg.error_code = static_cast<ErrorCode>(j.value("error_code", 0));
        msg.error_message = j.value("error_message", "");
        return msg;
    }

    // JobMessage implementation
    json JobMessage::to_json() const {
        json j;
        j["job_id"] = job_id;
        j["base_message"] = Utils::binary_to_hex(base_message);
        j["target_hash"] = Utils::binary_to_hex(target_hash);
        j["difficulty"] = difficulty;
        j["nonce_start"] = nonce_start;
        j["nonce_end"] = nonce_end;
        j["expires_in_seconds"] = expires_in_seconds;
        j["clean_jobs"] = clean_jobs;
        return j;
    }

    JobMessage JobMessage::from_json(const json &j) {
        JobMessage msg;
        msg.job_id = j["job_id"].get<std::string>();
        msg.base_message = Utils::hex_to_binary(j["base_message"].get<std::string>());
        msg.target_hash = Utils::hex_to_binary(j["target_hash"].get<std::string>());
        msg.difficulty = j["difficulty"].get<uint32_t>();
        msg.nonce_start = j.value("nonce_start", 0ULL);
        msg.nonce_end = j.value("nonce_end", 0ULL);
        msg.expires_in_seconds = j["expires_in_seconds"].get<uint32_t>();
        msg.clean_jobs = j.value("clean_jobs", false);
        return msg;
    }

    // ShareResultMessage implementation
    json ShareResultMessage::to_json() const {
        json j;
        j["share_id"] = share_id;
        j["status"] = static_cast<int>(status);
        j["job_id"] = job_id;
        j["difficulty_credited"] = difficulty_credited;
        j["total_shares"] = total_shares;
        j["total_difficulty"] = total_difficulty;
        j["message"] = message;
        return j;
    }

    ShareResultMessage ShareResultMessage::from_json(const json &j) {
        ShareResultMessage msg;
        msg.share_id = j["share_id"].get<std::string>();
        msg.status = static_cast<ShareStatus>(j["status"].get<int>());
        msg.job_id = j["job_id"].get<std::string>();
        msg.difficulty_credited = j["difficulty_credited"].get<double>();
        msg.total_shares = j["total_shares"].get<uint64_t>();
        msg.total_difficulty = j["total_difficulty"].get<double>();
        msg.message = j.value("message", "");
        return msg;
    }

    // DifficultyAdjustMessage implementation
    json DifficultyAdjustMessage::to_json() const {
        json j;
        j["new_difficulty"] = new_difficulty;
        j["reason"] = reason;
        j["min_difficulty"] = min_difficulty;
        j["max_difficulty"] = max_difficulty;
        j["target_time"] = target_time;
        return j;
    }

    DifficultyAdjustMessage DifficultyAdjustMessage::from_json(const json &j) {
        DifficultyAdjustMessage msg;
        msg.new_difficulty = j["new_difficulty"].get<uint32_t>();
        msg.reason = j["reason"].get<std::string>();
        msg.min_difficulty = j["min_difficulty"].get<uint32_t>();
        msg.max_difficulty = j["max_difficulty"].get<uint32_t>();
        msg.target_time = j["target_time"].get<double>();
        return msg;
    }

    // PoolStatusMessage implementation
    json PoolStatusMessage::to_json() const {
        json j;
        j["active_workers"] = active_workers;
        j["pool_hashrate"] = pool_hashrate;
        j["blocks_found"] = blocks_found;
        j["total_shares"] = total_shares;
        j["average_difficulty"] = average_difficulty;
        j["network_info"] = network_info;
        return j;
    }

    PoolStatusMessage PoolStatusMessage::from_json(const json &j) {
        PoolStatusMessage msg;
        msg.active_workers = j["active_workers"].get<uint64_t>();
        msg.pool_hashrate = j["pool_hashrate"].get<double>();
        msg.blocks_found = j["blocks_found"].get<uint64_t>();
        msg.total_shares = j["total_shares"].get<uint64_t>();
        msg.average_difficulty = j["average_difficulty"].get<double>();
        msg.network_info = j.value("network_info", std::map<std::string, std::string>());
        return msg;
    }

    // Utility functions implementation
    namespace Utils {
        std::string binary_to_hex(const std::vector<uint8_t> &data) {
            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (uint8_t byte: data) {
                ss << std::setw(2) << static_cast<int>(byte);
            }
            return ss.str();
        }

        std::vector<uint8_t> hex_to_binary(const std::string &hex) {
            std::vector<uint8_t> binary;
            for (size_t i = 0; i < hex.length(); i += 2) {
                std::string byte_string = hex.substr(i, 2);
                uint8_t byte = static_cast<uint8_t>(std::stoul(byte_string, nullptr, 16));
                binary.push_back(byte);
            }
            return binary;
        }

        double calculate_share_difficulty(uint32_t matching_bits) {
            return std::pow(2.0, matching_bits);
        }

        bool validate_share_difficulty(const Share &share, uint32_t target_difficulty) {
            return share.matching_bits >= target_difficulty;
        }

        uint64_t generate_message_id() {
            static std::atomic<uint64_t> counter{0};
            static std::random_device rd;
            static std::mt19937_64 gen(rd());

            // Combine timestamp with counter and random value for uniqueness
            uint64_t timestamp = current_timestamp_ms();
            uint64_t count = counter.fetch_add(1);
            uint64_t random = gen();

            return (timestamp << 20) | (count & 0xFFFFF) | (random & 0xFFF);
        }

        uint64_t current_timestamp_ms() {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        }
    } // namespace Utils
} // namespace MiningPool
