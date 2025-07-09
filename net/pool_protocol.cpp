#include "pool_protocol.hpp"
#include "../logging/logger.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <atomic>

namespace MiningPool {
    static std::atomic<uint64_t> message_counter{1};

    // Message serialization/deserialization
    std::string Message::serialize() const {
        nlohmann::json j;
        j["type"] = type;
        j["id"] = std::to_string(id);
        j["timestamp"] = timestamp;
        j["payload"] = payload;
        return j.dump();
    }

    std::optional<Message> Message::deserialize(const std::string &data) {
        try {
            nlohmann::json j = nlohmann::json::parse(data);
            Message msg;
            // Handle type field
            msg.type = static_cast<MessageType>(j["type"].get<int>());
            // Handle id field - it might be a string or number
            if (j["id"].is_string()) {
                msg.id = std::stoull(j["id"].get<std::string>());
            } else if (j["id"].is_number()) {
                msg.id = j["id"].get<uint64_t>();
            } else {
                throw std::runtime_error("Invalid id field type");
            }
            // Handle timestamp field - it might be a string or number
            if (j["timestamp"].is_string()) {
                msg.timestamp = std::stoull(j["timestamp"].get<std::string>());
            } else if (j["timestamp"].is_number()) {
                msg.timestamp = j["timestamp"].get<uint64_t>();
            } else {
                throw std::runtime_error("Invalid timestamp field type");
            }
            // Payload is already a JSON object
            msg.payload = j["payload"];
            LOG_TRACE("MESSAGE", "Deserialized - type: ", static_cast<int>(msg.type),
                      " id: ", msg.id, " timestamp: ", msg.timestamp);

            return msg;
        } catch (const std::exception &e) {
            LOG_ERROR("MESSAGE", "Deserialize error: ", e.what());
            LOG_DEBUG("MESSAGE", "Raw data: ", data);
            return std::nullopt;
        }
    }

    // HelloMessage
    nlohmann::json HelloMessage::to_json() const {
        nlohmann::json j;
        j["protocol_version"] = protocol_version;
        j["client_version"] = client_version;
        j["capabilities"] = capabilities;
        j["user_agent"] = user_agent;
        return j;
    }

    HelloMessage HelloMessage::from_json(const nlohmann::json &j) {
        HelloMessage msg;
        msg.protocol_version = j["protocol_version"].get<uint32_t>();
        msg.client_version = j["client_version"].get<std::string>();
        msg.capabilities = j["capabilities"].get<std::vector<std::string> >();
        if (j.contains("user_agent")) {
            msg.user_agent = j["user_agent"].get<std::string>();
        }
        return msg;
    }

    // WelcomeMessage
    nlohmann::json WelcomeMessage::to_json() const {
        nlohmann::json j;
        j["pool_name"] = pool_name;
        j["pool_version"] = pool_version;
        j["protocol_version"] = protocol_version;
        j["min_difficulty"] = min_difficulty;
        j["features"] = features;
        j["motd"] = motd;
        return j;
    }

    WelcomeMessage WelcomeMessage::from_json(const nlohmann::json &j) {
        WelcomeMessage msg;
        msg.pool_name = j["pool_name"].get<std::string>();
        msg.pool_version = j["pool_version"].get<std::string>();
        msg.protocol_version = j["protocol_version"].get<uint32_t>();
        msg.min_difficulty = j["min_difficulty"].get<uint32_t>();
        msg.features = j["features"].get<std::vector<std::string> >();
        if (j.contains("motd")) {
            msg.motd = j["motd"].get<std::string>();
        }
        return msg;
    }

    // AuthMessage - Enhanced for epoch support
    nlohmann::json AuthMessage::to_json() const {
        nlohmann::json j;
        // Convert enum to string to match TypeScript
        switch (method) {
            case AuthMethod::WORKER_PASS:
                j["method"] = "worker_pass";
                break;
            case AuthMethod::API_KEY:
                j["method"] = "api_key";
                break;
            case AuthMethod::CERTIFICATE:
                j["method"] = "certificate";
                break;
        }
        j["username"] = username;
        if (!password.empty()) j["password"] = password;
        if (!session_id.empty()) j["session_id"] = session_id;
        if (!otp.empty()) j["otp"] = otp;
        if (!client_nonce.empty()) j["client_nonce"] = client_nonce;
        // Add client_info for hashrate estimation
        if (client_info.has_value()) {
            nlohmann::json info;
            if (client_info->estimated_hashrate > 0) {
                info["estimated_hashrate"] = client_info->estimated_hashrate;
            }
            if (client_info->gpu_count > 0) {
                info["gpu_count"] = client_info->gpu_count;
            }
            if (!client_info->miner_version.empty()) {
                info["miner_version"] = client_info->miner_version;
            }
            j["client_info"] = info;
        }
        return j;
    }

    AuthMessage AuthMessage::from_json(const nlohmann::json &j) {
        AuthMessage msg;

        // Parse method from string
        std::string method_str = j["method"].get<std::string>();
        if (method_str == "worker_pass") {
            msg.method = AuthMethod::WORKER_PASS;
        } else if (method_str == "api_key") {
            msg.method = AuthMethod::API_KEY;
        } else if (method_str == "certificate") {
            msg.method = AuthMethod::CERTIFICATE;
        } else {
            throw std::runtime_error("Invalid auth method: " + method_str);
        }

        msg.username = j["username"].get<std::string>();
        if (j.contains("password") && !j["password"].is_null()) {
            msg.password = j["password"].get<std::string>();
        }
        if (j.contains("session_id") && !j["session_id"].is_null()) {
            msg.session_id = j["session_id"].get<std::string>();
        }
        if (j.contains("otp") && !j["otp"].is_null()) {
            msg.otp = j["otp"].get<std::string>();
        }
        if (j.contains("client_nonce") && !j["client_nonce"].is_null()) {
            msg.client_nonce = j["client_nonce"].get<std::string>();
        }
        // Parse client_info if present
        if (j.contains("client_info") && !j["client_info"].is_null()) {
            ClientInfo info;
            auto ci = j["client_info"];
            if (ci.contains("estimated_hashrate")) {
                info.estimated_hashrate = ci["estimated_hashrate"].get<double>();
            }
            if (ci.contains("gpu_count")) {
                info.gpu_count = ci["gpu_count"].get<uint32_t>();
            }
            if (ci.contains("miner_version")) {
                info.miner_version = ci["miner_version"].get<std::string>();
            }
            msg.client_info = info;
        }
        return msg;
    }

    // AuthResponseMessage
    nlohmann::json AuthResponseMessage::to_json() const {
        nlohmann::json j;
        j["success"] = success;
        j["session_id"] = session_id;
        j["worker_id"] = worker_id;
        j["error_code"] = static_cast<int>(error_code);
        j["error_message"] = error_message;
        return j;
    }

    AuthResponseMessage AuthResponseMessage::from_json(const nlohmann::json &j) {
        AuthResponseMessage msg;
        msg.success = j["success"].get<bool>();

        if (j.contains("session_id") && !j["session_id"].is_null()) {
            msg.session_id = j["session_id"].get<std::string>();
        }
        if (j.contains("worker_id") && !j["worker_id"].is_null()) {
            msg.worker_id = j["worker_id"].get<std::string>();
        }
        if (j.contains("error_code") && !j["error_code"].is_null()) {
            msg.error_code = static_cast<ErrorCode>(j["error_code"].get<int>());
        } else {
            msg.error_code = ErrorCode::NONE;
        }
        if (j.contains("error_message") && !j["error_message"].is_null()) {
            msg.error_message = j["error_message"].get<std::string>();
        }
        return msg;
    }

    // JobMessage - Enhanced for unique salted preimages
    nlohmann::json JobMessage::to_json() const {
        nlohmann::json j;
        j["job_id"] = job_id;
        j["target_difficulty"] = target_difficulty;
        j["target_pattern"] = target_pattern;
        j["prefix_data"] = prefix_data;
        j["suffix_data"] = suffix_data;
        j["nonce_start"] = std::to_string(nonce_start); // Convert to string for JS bigint
        j["nonce_end"] = std::to_string(nonce_end); // Convert to string for JS bigint
        j["algorithm"] = algorithm;
        j["extra_data"] = extra_data;
        j["clean_jobs"] = clean_jobs;
        j["expires_in_seconds"] = expires_in_seconds;
        return j;
    }

    JobMessage JobMessage::from_json(const nlohmann::json &j) {
        JobMessage msg;
        msg.job_id = j["job_id"].get<std::string>();
        msg.target_difficulty = j["target_difficulty"].get<uint32_t>();
        msg.target_pattern = j["target_pattern"].get<std::string>();
        msg.prefix_data = j["prefix_data"].get<std::string>();
        msg.suffix_data = j["suffix_data"].get<std::string>();
        // Handle bigint fields
        if (j["nonce_start"].is_string()) {
            msg.nonce_start = std::stoull(j["nonce_start"].get<std::string>());
        } else {
            msg.nonce_start = j["nonce_start"].get<uint64_t>();
        }

        if (j["nonce_end"].is_string()) {
            msg.nonce_end = std::stoull(j["nonce_end"].get<std::string>());
        } else {
            msg.nonce_end = j["nonce_end"].get<uint64_t>();
        }

        msg.algorithm = j["algorithm"].get<std::string>();
        if (j.contains("extra_data") && !j["extra_data"].is_null()) {
            msg.extra_data = j["extra_data"];
        }
        msg.clean_jobs = j["clean_jobs"].get<bool>();
        msg.expires_in_seconds = j["expires_in_seconds"].get<uint32_t>();
        return msg;
    }

    // SubmitShareMessage
    nlohmann::json SubmitShareMessage::to_json() const {
        nlohmann::json j;
        j["job_id"] = job_id;
        j["nonce"] = std::to_string(nonce); // Convert to string for JS bigint
        j["hash"] = hash;
        j["matching_bits"] = matching_bits;
        if (!worker_name.empty()) j["worker_name"] = worker_name;
        if (!extra_nonce.empty()) j["extra_nonce"] = extra_nonce;
        return j;
    }

    SubmitShareMessage SubmitShareMessage::from_json(const nlohmann::json &j) {
        SubmitShareMessage msg;
        msg.job_id = j["job_id"].get<std::string>();

        // Handle bigint nonce - it could be string or number
        if (j["nonce"].is_string()) {
            msg.nonce = std::stoull(j["nonce"].get<std::string>());
        } else if (j["nonce"].is_number_integer()) {
            msg.nonce = j["nonce"].get<uint64_t>();
        } else if (j["nonce"].is_number_unsigned()) {
            msg.nonce = j["nonce"].get<uint64_t>();
        } else {
            throw std::runtime_error("Invalid nonce type");
        }

        msg.hash = j["hash"].get<std::string>();
        msg.matching_bits = j["matching_bits"].get<uint32_t>();

        if (j.contains("worker_name") && !j["worker_name"].is_null()) {
            msg.worker_name = j["worker_name"].get<std::string>();
        }
        if (j.contains("extra_nonce") && !j["extra_nonce"].is_null()) {
            msg.extra_nonce = j["extra_nonce"].get<std::string>();
        }
        return msg;
    }

    // ShareResultMessage
    nlohmann::json ShareResultMessage::to_json() const {
        nlohmann::json j;
        j["job_id"] = job_id;

        // Convert enum to string to match TypeScript
        switch (status) {
            case ShareStatus::ACCEPTED:
                j["status"] = "accepted";
                break;
            case ShareStatus::REJECTED_LOW_DIFFICULTY:
                j["status"] = "rejected_low_difficulty";
                break;
            case ShareStatus::REJECTED_INVALID:
                j["status"] = "rejected_invalid";
                break;
            case ShareStatus::REJECTED_STALE:
                j["status"] = "rejected_stale";
                break;
            case ShareStatus::REJECTED_DUPLICATE:
                j["status"] = "rejected_duplicate";
                break;
        }

        j["difficulty_credited"] = difficulty_credited;
        if (!message.empty()) j["message"] = message;
        if (share_value > 0) j["share_value"] = share_value;
        if (total_shares > 0) j["total_shares"] = total_shares;
        return j;
    }

    ShareResultMessage ShareResultMessage::from_json(const nlohmann::json &j) {
        ShareResultMessage msg;
        msg.job_id = j["job_id"].get<std::string>();

        // Parse status from string
        std::string status_str = j["status"].get<std::string>();
        if (status_str == "accepted") {
            msg.status = ShareStatus::ACCEPTED;
        } else if (status_str == "rejected_low_difficulty") {
            msg.status = ShareStatus::REJECTED_LOW_DIFFICULTY;
        } else if (status_str == "rejected_invalid") {
            msg.status = ShareStatus::REJECTED_INVALID;
        } else if (status_str == "rejected_stale") {
            msg.status = ShareStatus::REJECTED_STALE;
        } else if (status_str == "rejected_duplicate") {
            msg.status = ShareStatus::REJECTED_DUPLICATE;
        } else {
            throw std::runtime_error("Invalid share status: " + status_str);
        }

        msg.difficulty_credited = j["difficulty_credited"].get<uint32_t>();
        if (j.contains("message") && !j["message"].is_null()) {
            msg.message = j["message"].get<std::string>();
        }
        if (j.contains("share_value") && !j["share_value"].is_null()) {
            msg.share_value = j["share_value"].get<double>();
        }
        if (j.contains("total_shares") && !j["total_shares"].is_null()) {
            msg.total_shares = j["total_shares"].get<uint64_t>();
        }
        return msg;
    }

    // HashrateReportMessage
    nlohmann::json HashrateReportMessage::to_json() const {
        nlohmann::json j;
        j["hashrate"] = hashrate;
        j["shares_submitted"] = shares_submitted;
        j["shares_accepted"] = shares_accepted;
        j["uptime_seconds"] = uptime_seconds;
        j["gpu_count"] = gpu_count;
        j["gpu_stats"] = gpu_stats;
        return j;
    }

    HashrateReportMessage HashrateReportMessage::from_json(const nlohmann::json &j) {
        HashrateReportMessage msg;
        msg.hashrate = j["hashrate"].get<double>();
        msg.shares_submitted = j["shares_submitted"].get<uint64_t>();
        msg.shares_accepted = j["shares_accepted"].get<uint64_t>();
        msg.uptime_seconds = j["uptime_seconds"].get<uint64_t>();
        msg.gpu_count = j["gpu_count"].get<uint32_t>();
        msg.gpu_stats = j["gpu_stats"];
        return msg;
    }

    // DifficultyAdjustMessage
    nlohmann::json DifficultyAdjustMessage::to_json() const {
        nlohmann::json j;
        j["new_difficulty"] = new_difficulty;
        j["reason"] = reason;
        j["effective_in_seconds"] = effective_in_seconds;
        return j;
    }

    DifficultyAdjustMessage DifficultyAdjustMessage::from_json(const nlohmann::json &j) {
        DifficultyAdjustMessage msg;
        msg.new_difficulty = j["new_difficulty"].get<uint32_t>();
        msg.reason = j["reason"].get<std::string>();
        msg.effective_in_seconds = j["effective_in_seconds"].get<uint32_t>();
        return msg;
    }

    // PoolStatusMessage - Enhanced with epoch info
    nlohmann::json PoolStatusMessage::to_json() const {
        nlohmann::json j;
        j["connected_workers"] = connected_workers;
        j["total_hashrate"] = total_hashrate;
        j["shares_per_minute"] = shares_per_minute;
        j["epochs_completed"] = epochs_completed;
        j["current_epoch_number"] = current_epoch_number;
        j["current_epoch_shares"] = current_epoch_shares;
        j["pool_fee_percent"] = pool_fee_percent;
        j["minimum_payout"] = minimum_payout;
        j["extra_info"] = extra_info;
        return j;
    }

    PoolStatusMessage PoolStatusMessage::from_json(const nlohmann::json &j) {
        PoolStatusMessage msg;
        msg.connected_workers = j["connected_workers"].get<uint32_t>();
        msg.total_hashrate = j["total_hashrate"].get<double>();
        msg.shares_per_minute = j["shares_per_minute"].get<double>();
        // Handle both old (blocks_found) and new (epochs_completed) field names
        if (j.contains("epochs_completed")) {
            msg.epochs_completed = j["epochs_completed"].get<uint64_t>();
        } else if (j.contains("blocks_found")) {
            msg.epochs_completed = j["blocks_found"].get<uint64_t>();
        }

        // Handle epoch-specific fields
        if (j.contains("current_epoch_number")) {
            msg.current_epoch_number = j["current_epoch_number"].get<uint32_t>();
        }
        if (j.contains("current_epoch_shares")) {
            msg.current_epoch_shares = j["current_epoch_shares"].get<uint64_t>();
        } else if (j.contains("current_round_shares")) {
            msg.current_epoch_shares = j["current_round_shares"].get<uint64_t>();
        }

        msg.pool_fee_percent = j["pool_fee_percent"].get<double>();
        msg.minimum_payout = j["minimum_payout"].get<double>();
        msg.extra_info = j["extra_info"];

        return msg;
    }
} // namespace MiningPool
