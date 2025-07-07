#include "pool_client.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>

namespace MiningPool {
    // Message implementation
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
            msg.id = j["id"];
            msg.timestamp = j["timestamp"];
            msg.payload = j["payload"];
            return msg;
        } catch (const std::exception &e) {
            return std::nullopt;
        }
    }

    // Client message implementations
    json HelloMessage::to_json() const {
        return json{
            {"protocol_version", protocol_version},
            {"client_version", client_version},
            {"capabilities", capabilities}
        };
    }

    HelloMessage HelloMessage::from_json(const json &j) {
        HelloMessage msg;
        msg.protocol_version = j["protocol_version"];
        msg.client_version = j["client_version"];
        msg.capabilities = j["capabilities"].get<std::vector<std::string> >();
        return msg;
    }

    json AuthMessage::to_json() const {
        return json{
            {"method", static_cast<int>(method)},
            {"username", username},
            {"password", password},
            {"session_id", session_id}
        };
    }

    AuthMessage AuthMessage::from_json(const json &j) {
        AuthMessage msg;
        msg.method = static_cast<AuthMethod>(j["method"].get<int>());
        msg.username = j["username"];
        msg.password = j.value("password", "");
        msg.session_id = j.value("session_id", "");
        return msg;
    }

    json SubmitShareMessage::to_json() const {
        return json{
            {"job_id", job_id},
            {"nonce", nonce},
            {"hash", Utils::binary_to_hex(hash)},
            {"matching_bits", matching_bits},
            {"worker_name", worker_name}
        };
    }

    SubmitShareMessage SubmitShareMessage::from_json(const json &j) {
        SubmitShareMessage msg;
        msg.job_id = j["job_id"];
        msg.nonce = j["nonce"];
        msg.hash = Utils::hex_to_binary(j["hash"]);
        msg.matching_bits = j["matching_bits"];
        msg.worker_name = j.value("worker_name", "");
        return msg;
    }

    // Server message implementations
    json JobMessage::to_json() const {
        return json{
            {"job_id", job_id},
            {"base_message", Utils::binary_to_hex(base_message)},
            {"target_hash", Utils::binary_to_hex(target_hash)},
            {"difficulty", difficulty},
            {"nonce_start", nonce_start},
            {"nonce_end", nonce_end},
            {"expires_in_seconds", expires_in_seconds},
            {"clean_jobs", clean_jobs}
        };
    }

    JobMessage JobMessage::from_json(const json &j) {
        JobMessage msg;
        msg.job_id = j["job_id"];
        msg.base_message = Utils::hex_to_binary(j["base_message"]);
        msg.target_hash = Utils::hex_to_binary(j["target_hash"]);
        msg.difficulty = j["difficulty"];
        msg.nonce_start = j.value("nonce_start", 0ULL);
        msg.nonce_end = j.value("nonce_end", 0ULL);
        msg.expires_in_seconds = j["expires_in_seconds"];
        msg.clean_jobs = j.value("clean_jobs", false);
        return msg;
    }

    // Utility functions
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
            std::vector<uint8_t> data;
            for (size_t i = 0; i < hex.length(); i += 2) {
                std::string byte_str = hex.substr(i, 2);
                uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
                data.push_back(byte);
            }
            return data;
        }

        uint64_t generate_message_id() {
            static std::atomic<uint64_t> counter{1};
            return counter.fetch_add(1);
        }

        uint64_t current_timestamp_ms() {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
        }

        bool validate_share_difficulty(const Share &share, uint32_t target_difficulty) {
            return share.matching_bits >= target_difficulty;
        }
    } // namespace Utils

    // PoolClient implementation
    PoolClient::PoolClient(const PoolConfig &config, IPoolEventHandler *handler)
        : config_(config), event_handler_(handler) {
        worker_stats_.worker_id = config.worker_name;
        worker_stats_.connected_since = std::chrono::steady_clock::now();

        if (config.use_tls) {
            wss_client_ = std::make_unique<WssTlsClient>();
            setup_wss_handlers();
        } else {
            ws_client_ = std::make_unique<WsClient>();
            setup_ws_handlers();
        }
    }

    PoolClient::~PoolClient() {
        disconnect();
    }

    void PoolClient::setup_ws_handlers() {
        ws_client_->set_access_channels(websocketpp::log::alevel::none);
        ws_client_->set_error_channels(websocketpp::log::elevel::all);

        ws_client_->init_asio();

        ws_client_->set_open_handler(
            [this](websocketpp::connection_hdl hdl) { on_open(hdl); });
        ws_client_->set_close_handler(
            [this](websocketpp::connection_hdl hdl) { on_close(hdl); });
        ws_client_->set_message_handler(
            [this](websocketpp::connection_hdl hdl, WsMessage msg) {
                on_message(hdl, msg);
            });
        ws_client_->set_fail_handler(
            [this](websocketpp::connection_hdl hdl) { on_fail(hdl); });
    }

    void PoolClient::setup_wss_handlers() {
        wss_client_->set_access_channels(websocketpp::log::alevel::none);
        wss_client_->set_error_channels(websocketpp::log::elevel::all);

        wss_client_->init_asio();
        wss_client_->set_tls_init_handler(
            [this](websocketpp::connection_hdl hdl) {
                return on_tls_init(hdl);
            });

        wss_client_->set_open_handler(
            [this](websocketpp::connection_hdl hdl) { on_open(hdl); });
        wss_client_->set_close_handler(
            [this](websocketpp::connection_hdl hdl) { on_close(hdl); });
        wss_client_->set_message_handler(
            [this](websocketpp::connection_hdl hdl, WssMessage msg) {
                on_message_tls(hdl, msg);
            });
        wss_client_->set_fail_handler(
            [this](websocketpp::connection_hdl hdl) { on_fail(hdl); });
    }

    bool PoolClient::connect() {
        if (connected_.load()) {
            return true;
        }

        running_ = true;

        try {
            websocketpp::lib::error_code ec;

            if (config_.use_tls) {
                auto con = wss_client_->get_connection(config_.url, ec);
                if (ec) {
                    std::cerr << "Could not create connection: " << ec.message() << std::endl;
                    return false;
                }

                connection_hdl_ = con->get_handle();
                wss_client_->connect(con);
            } else {
                auto con = ws_client_->get_connection(config_.url, ec);
                if (ec) {
                    std::cerr << "Could not create connection: " << ec.message() << std::endl;
                    return false;
                }

                connection_hdl_ = con->get_handle();
                ws_client_->connect(con);
            }

            // Start threads
            io_thread_ = std::thread(&PoolClient::io_loop, this);
            message_processor_thread_ = std::thread(&PoolClient::message_processor_loop, this);
            keepalive_thread_ = std::thread(&PoolClient::keepalive_loop, this);

            // Wait for connection
            auto start = std::chrono::steady_clock::now();
            while (!connected_.load() &&
                   std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            return connected_.load();
        } catch (const std::exception &e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            return false;
        }
    }

    void PoolClient::disconnect() {
        if (!running_.load()) {
            return;
        }

        running_ = false;
        connected_ = false;
        authenticated_ = false;

        // Close connection
        try {
            if (config_.use_tls && wss_client_) {
                wss_client_->close(connection_hdl_,
                                   websocketpp::close::status::going_away, "Client disconnect");
            } else if (ws_client_) {
                ws_client_->close(connection_hdl_,
                                  websocketpp::close::status::going_away, "Client disconnect");
            }
        } catch (...) {
        }

        // Notify condition variables
        outgoing_cv_.notify_all();
        incoming_cv_.notify_all();

        // Join threads
        if (io_thread_.joinable()) io_thread_.join();
        if (message_processor_thread_.joinable()) message_processor_thread_.join();
        if (keepalive_thread_.joinable()) keepalive_thread_.join();
    }

    void PoolClient::io_loop() {
        try {
            if (config_.use_tls) {
                wss_client_->run();
            } else {
                ws_client_->run();
            }
        } catch (const std::exception &e) {
            std::cerr << "IO loop error: " << e.what() << std::endl;
        }
    }

    void PoolClient::keepalive_loop() {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(config_.keepalive_interval_s));

            if (connected_.load() && authenticated_.load()) {
                Message keepalive;
                keepalive.type = MessageType::KEEPALIVE;
                keepalive.id = Utils::generate_message_id();
                keepalive.timestamp = Utils::current_timestamp_ms();
                send_message(keepalive);
            }
        }
    }

    void PoolClient::message_processor_loop() {
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(incoming_mutex_);
            incoming_cv_.wait(lock, [this] {
                return !incoming_queue_.empty() || !running_.load();
            });

            while (!incoming_queue_.empty()) {
                Message msg = incoming_queue_.front();
                incoming_queue_.pop();
                lock.unlock();

                process_message(msg);

                lock.lock();
            }
        }
    }

    void PoolClient::on_open(websocketpp::connection_hdl hdl) {
        std::cout << "Connected to pool" << std::endl;
        connected_ = true;
        event_handler_->on_connected();

        // Send hello message
        HelloMessage hello;
        hello.protocol_version = PROTOCOL_VERSION;
        hello.client_version = "SHA1-Miner/1.0";
        hello.capabilities = {"gpu", "multi-gpu", "vardiff"};

        Message msg;
        msg.type = MessageType::HELLO;
        msg.id = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload = hello.to_json();

        send_message(msg);
    }

    void PoolClient::on_close(websocketpp::connection_hdl hdl) {
        std::cout << "Disconnected from pool" << std::endl;
        connected_ = false;
        authenticated_ = false;
        event_handler_->on_disconnected("Connection closed");
    }

    void PoolClient::on_message(websocketpp::connection_hdl hdl, WsMessage msg) {
        auto parsed = Message::deserialize(msg->get_payload());
        if (parsed) {
            std::lock_guard<std::mutex> lock(incoming_mutex_);
            incoming_queue_.push(*parsed);
            incoming_cv_.notify_one();
        }
    }

    void PoolClient::on_message_tls(websocketpp::connection_hdl hdl, WssMessage msg) {
        auto parsed = Message::deserialize(msg->get_payload());
        if (parsed) {
            std::lock_guard<std::mutex> lock(incoming_mutex_);
            incoming_queue_.push(*parsed);
            incoming_cv_.notify_one();
        }
    }

    void PoolClient::on_fail(websocketpp::connection_hdl hdl) {
        std::cerr << "Connection failed" << std::endl;
        connected_ = false;
        event_handler_->on_error(ErrorCode::PROTOCOL_ERROR, "Connection failed");
    }

    std::shared_ptr<SslContext> PoolClient::on_tls_init(websocketpp::connection_hdl hdl) {
        auto ctx = std::make_shared<SslContext>(SslContext::tlsv12_client);

        try {
            ctx->set_options(SslContext::default_workarounds |
                             SslContext::no_sslv2 |
                             SslContext::no_sslv3 |
                             SslContext::single_dh_use);

            if (config_.verify_server_cert) {
                ctx->set_verify_mode(websocketpp::lib::asio::ssl::verify_peer);
            } else {
                ctx->set_verify_mode(websocketpp::lib::asio::ssl::verify_none);
            }

            // Load client certificate if provided
            if (!config_.tls_cert_file.empty() && !config_.tls_key_file.empty()) {
                ctx->use_certificate_file(config_.tls_cert_file, SslContext::pem);
                ctx->use_private_key_file(config_.tls_key_file, SslContext::pem);
            }
        } catch (const std::exception &e) {
            std::cerr << "TLS initialization error: " << e.what() << std::endl;
        }

        return ctx;
    }

    void PoolClient::send_message(const Message &msg) {
        try {
            std::string payload = msg.serialize();

            if (config_.use_tls) {
                wss_client_->send(connection_hdl_, payload,
                                  websocketpp::frame::opcode::text);
            } else {
                ws_client_->send(connection_hdl_, payload,
                                 websocketpp::frame::opcode::text);
            }
        } catch (const std::exception &e) {
            std::cerr << "Send error: " << e.what() << std::endl;
        }
    }

    void PoolClient::process_message(const Message &msg) {
        switch (msg.type) {
            case MessageType::WELCOME:
                handle_welcome(WelcomeMessage::from_json(msg.payload));
                break;
            case MessageType::AUTH_RESPONSE:
                handle_auth_response(AuthResponseMessage::from_json(msg.payload));
                break;
            case MessageType::NEW_JOB:
                handle_new_job(JobMessage::from_json(msg.payload));
                break;
            case MessageType::SHARE_RESULT:
                handle_share_result(ShareResultMessage::from_json(msg.payload));
                break;
            case MessageType::DIFFICULTY_ADJUST:
                handle_difficulty_adjust(DifficultyAdjustMessage::from_json(msg.payload));
                break;
            case MessageType::POOL_STATUS:
                handle_pool_status(PoolStatusMessage::from_json(msg.payload));
                break;
            case MessageType::ERROR:
                handle_error(msg);
                break;
            default:
                std::cerr << "Unknown message type: " << static_cast<int>(msg.type) << std::endl;
        }
    }

    void PoolClient::handle_welcome(const WelcomeMessage &welcome) {
        std::cout << "Pool: " << welcome.pool_name << " v" << welcome.pool_version << std::endl;

        // Authenticate
        authenticate();
    }

    bool PoolClient::authenticate() {
        AuthMessage auth;
        auth.method = config_.auth_method;
        auth.username = config_.username + "." + config_.worker_name;
        auth.password = config_.password;
        auth.session_id = session_id_;

        Message msg;
        msg.type = MessageType::AUTH;
        msg.id = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload = auth.to_json();

        send_message(msg);
        return true;
    }

    void PoolClient::handle_auth_response(const AuthResponseMessage &response) {
        if (response.success) {
            authenticated_ = true;
            session_id_ = response.session_id;
            worker_id_ = response.worker_id;
            event_handler_->on_authenticated(worker_id_);

            // Request initial job
            request_job();
        } else {
            authenticated_ = false;
            event_handler_->on_auth_failed(response.error_code, response.error_message);
        }
    }

    void PoolClient::handle_new_job(const JobMessage &job) {
        PoolJob pool_job;
        pool_job.job_id = job.job_id;
        pool_job.job_data = job;
        pool_job.received_time = std::chrono::steady_clock::now();
        pool_job.expiry_time = pool_job.received_time +
                               std::chrono::seconds(job.expires_in_seconds);
        pool_job.is_active = true; {
            std::lock_guard<std::mutex> lock(jobs_mutex_);

            if (job.clean_jobs) {
                // Cancel all existing jobs
                for (auto &[id, existing_job]: active_jobs_) {
                    existing_job.is_active = false;
                    event_handler_->on_job_cancelled(id);
                }
                active_jobs_.clear();
            }

            active_jobs_[job.job_id] = pool_job;
            current_job_id_ = job.job_id;
        }

        event_handler_->on_new_job(pool_job);
    }

    void PoolClient::submit_share(const Share &share) {
        if (!authenticated_.load()) {
            return;
        }

        SubmitShareMessage submit;
        submit.job_id = share.job_id;
        submit.nonce = share.nonce;
        submit.hash = share.hash;
        submit.matching_bits = share.matching_bits;
        submit.worker_name = config_.worker_name;

        Message msg;
        msg.type = MessageType::SUBMIT_SHARE;
        msg.id = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload = submit.to_json();

        send_message(msg);
    }

    void PoolClient::handle_share_result(const ShareResultMessage &result) {
        update_stats(result);

        if (result.status == ShareStatus::ACCEPTED) {
            event_handler_->on_share_accepted(result);
        } else {
            event_handler_->on_share_rejected(result);
        }
    }

    void PoolClient::update_stats(const ShareResultMessage &result) {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        if (result.status == ShareStatus::ACCEPTED) {
            worker_stats_.shares_accepted++;
            worker_stats_.total_difficulty_accepted += result.difficulty_credited;
        } else {
            worker_stats_.shares_rejected++;
        }

        worker_stats_.last_share_time = std::chrono::steady_clock::now();
    }

    WorkerStats PoolClient::get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return worker_stats_;
    }
} // namespace MiningPool
