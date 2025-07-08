#include "pool_protocol.hpp"
#include <sstream>
#include <iomanip>
#include <atomic>

namespace MiningPool {
    // Message serialization/deserialization
    std::string Message::serialize() const {
        nlohmann::json j;
        j["type"] = static_cast<int>(type);
        j["id"] = id;
        j["timestamp"] = timestamp;
        j["payload"] = payload;
        return j.dump();
    }

    std::optional<Message> Message::deserialize(const std::string &data) {
        try {
            nlohmann::json j = nlohmann::json::parse(data);
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

    // AuthMessage
    nlohmann::json AuthMessage::to_json() const {
        nlohmann::json j;
        j["method"] = static_cast<int>(method);
        j["username"] = username;
        j["password"] = password;
        j["session_id"] = session_id;
        j["otp"] = otp;
        j["client_nonce"] = client_nonce;
        return j;
    }

    AuthMessage AuthMessage::from_json(const nlohmann::json &j) {
        AuthMessage msg;
        msg.method = static_cast<AuthMethod>(j["method"].get<int>());
        msg.username = j["username"].get<std::string>();
        if (j.contains("password")) {
            msg.password = j["password"].get<std::string>();
        }
        if (j.contains("session_id")) {
            msg.session_id = j["session_id"].get<std::string>();
        }
        if (j.contains("otp")) {
            msg.otp = j["otp"].get<std::string>();
        }
        if (j.contains("client_nonce")) {
            msg.client_nonce = j["client_nonce"].get<std::string>();
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
        msg.session_id = j["session_id"].get<std::string>();
        msg.worker_id = j["worker_id"].get<std::string>();
        msg.error_code = static_cast<ErrorCode>(j["error_code"].get<int>());
        msg.error_message = j["error_message"].get<std::string>();
        return msg;
    }

    // JobMessage
    nlohmann::json JobMessage::to_json() const {
        nlohmann::json j;
        j["job_id"] = job_id;
        j["target_difficulty"] = target_difficulty;
        j["target_pattern"] = target_pattern;
        j["prefix_data"] = prefix_data;
        j["suffix_data"] = suffix_data;
        j["nonce_start"] = nonce_start;
        j["nonce_end"] = nonce_end;
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
        msg.nonce_start = j["nonce_start"].get<uint64_t>();
        msg.nonce_end = j["nonce_end"].get<uint64_t>();
        msg.algorithm = j["algorithm"].get<std::string>();
        msg.extra_data = j["extra_data"];
        msg.clean_jobs = j["clean_jobs"].get<bool>();
        msg.expires_in_seconds = j["expires_in_seconds"].get<uint32_t>();
        return msg;
    }

    // SubmitShareMessage
    nlohmann::json SubmitShareMessage::to_json() const {
        nlohmann::json j;
        j["job_id"] = job_id;
        j["nonce"] = nonce;
        j["hash"] = hash;
        j["matching_bits"] = matching_bits;
        j["worker_name"] = worker_name;
        j["extra_nonce"] = extra_nonce;
        return j;
    }

    SubmitShareMessage SubmitShareMessage::from_json(const nlohmann::json &j) {
        SubmitShareMessage msg;
        msg.job_id = j["job_id"].get<std::string>();
        msg.nonce = j["nonce"].get<uint64_t>();
        msg.hash = j["hash"].get<std::string>();
        msg.matching_bits = j["matching_bits"].get<uint32_t>();
        msg.worker_name = j["worker_name"].get<std::string>();
        if (j.contains("extra_nonce")) {
            msg.extra_nonce = j["extra_nonce"].get<std::string>();
        }
        return msg;
    }

    // ShareResultMessage
    nlohmann::json ShareResultMessage::to_json() const {
        nlohmann::json j;
        j["job_id"] = job_id;
        j["status"] = static_cast<int>(status);
        j["difficulty_credited"] = difficulty_credited;
        j["message"] = message;
        j["share_value"] = share_value;
        j["total_shares"] = total_shares;
        return j;
    }

    ShareResultMessage ShareResultMessage::from_json(const nlohmann::json &j) {
        ShareResultMessage msg;
        msg.job_id = j["job_id"].get<std::string>();
        msg.status = static_cast<ShareStatus>(j["status"].get<int>());
        msg.difficulty_credited = j["difficulty_credited"].get<uint32_t>();
        msg.message = j["message"].get<std::string>();
        msg.share_value = j["share_value"].get<double>();
        msg.total_shares = j["total_shares"].get<uint64_t>();
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

    // PoolStatusMessage
    nlohmann::json PoolStatusMessage::to_json() const {
        nlohmann::json j;
        j["connected_workers"] = connected_workers;
        j["total_hashrate"] = total_hashrate;
        j["shares_per_minute"] = shares_per_minute;
        j["blocks_found"] = blocks_found;
        j["current_round_shares"] = current_round_shares;
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
        msg.blocks_found = j["blocks_found"].get<uint64_t>();
        msg.current_round_shares = j["current_round_shares"].get<uint64_t>();
        msg.pool_fee_percent = j["pool_fee_percent"].get<double>();
        msg.minimum_payout = j["minimum_payout"].get<double>();
        msg.extra_info = j["extra_info"];
        return msg;
    }
} // namespace MiningPool
