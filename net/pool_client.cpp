// pool_client_uws.cpp - High-performance pool client implementation
#include "pool_client.hpp"
#include <iostream>
#include <sstream>
#include <regex>

namespace MiningPool {
    // Helper to parse URL
    PoolClient::ParsedUrl PoolClient::parse_url(const std::string &url) {
        ParsedUrl result;

        std::regex url_regex(R"(^(wss?):\/\/([^:\/]+)(?::(\d+))?(\/.*)?$)");
        std::smatch matches;

        if (std::regex_match(url, matches, url_regex)) {
            result.is_secure = (matches[1].str() == "wss");
            result.host = matches[2].str();
            result.port = matches[3].length() ? std::stoi(matches[3].str()) : (result.is_secure ? 443 : 80);
            result.path = matches[4].length() ? matches[4].str() : "/";
        } else {
            throw std::invalid_argument("Invalid WebSocket URL: " + url);
        }

        return result;
    }

    PoolClient::PoolClient(const PoolConfig &config, IPoolEventHandler *handler)
        : config_(config), event_handler_(handler) {
        worker_stats_.worker_id = config.worker_name;
        worker_stats_.connected_since = std::chrono::steady_clock::now();
    }

    PoolClient::~PoolClient() {
        disconnect();
    }

    bool PoolClient::connect() {
        if (connected_.load()) {
            return true;
        }

        running_ = true;

        try {
            auto parsed = parse_url(config_.url);

            // Start IO thread with uWebSockets event loop
            io_thread_ = std::make_unique<std::thread>(&PoolClient::io_loop, this);

            // Start auxiliary threads
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

    void PoolClient::io_loop() {
        auto parsed = parse_url(config_.url);

        if (config_.use_tls) {
            // Configure TLS options
            tls_options_.cert_file_name = config_.tls_cert_file.empty() ? nullptr : config_.tls_cert_file.c_str();
            tls_options_.key_file_name = config_.tls_key_file.empty() ? nullptr : config_.tls_key_file.c_str();

            // Run secure WebSocket app
            uWS::SSLApp({
                        .cert_file_name = tls_options_.cert_file_name,
                        .key_file_name = tls_options_.key_file_name
                    }).ws<WebSocketData>("/*", {
                                             .compression = uWS::SHARED_COMPRESSOR,
                                             .maxPayloadLength = 16 * 1024 * 1024,
                                             .idleTimeout = 60,
                                             .maxBackpressure = 1 * 1024 * 1024,
                                             .closeOnBackpressureLimit = false,
                                             .resetIdleTimeoutOnSend = true,
                                             .sendPingsAutomatically = true,

                                             .upgrade = nullptr,
                                             .open = [this](auto *ws) {
                                                 ws->getUserData()->client = this;
                                                 this->ws_ = ws;
                                                 connected_ = true;
                                                 event_handler_->on_connected();

                                                 // Send hello message
                                                 HelloMessage hello;
                                                 hello.protocol_version = PROTOCOL_VERSION;
                                                 hello.client_version = "SHA1-Miner/1.0-uWS";
                                                 hello.capabilities = {"gpu", "multi-gpu", "vardiff"};

                                                 Message msg;
                                                 msg.type = MessageType::HELLO;
                                                 msg.id = Utils::generate_message_id();
                                                 msg.timestamp = Utils::current_timestamp_ms();
                                                 msg.payload = hello.to_json();

                                                 send_message(msg);
                                             },
                                             .message = [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                                                 auto parsed = Message::deserialize(std::string(message));
                                                 if (parsed) {
                                                     std::lock_guard<std::mutex> lock(incoming_mutex_);
                                                     incoming_queue_.push(*parsed);
                                                     incoming_cv_.notify_one();
                                                 }
                                             },
                                             .drain = [](auto *ws) {
                                                 // Handle backpressure
                                             },
                                             .ping = [](auto *ws, std::string_view) {
                                                 // Ping received
                                             },
                                             .pong = [](auto *ws, std::string_view) {
                                                 // Pong received
                                             },
                                             .close = [this](auto *ws, int code, std::string_view message) {
                                                 connected_ = false;
                                                 authenticated_ = false;
                                                 event_handler_->on_disconnected(std::string(message));
                                             }
                                         }).connect(parsed.host, parsed.port, parsed.path, config_.keepalive_interval_s,
                                                    {{"User-Agent", "SHA1-Miner/1.0"}})
                    .run();
        } else {
            // Run non-secure WebSocket app
            uWS::App().ws<WebSocketData>("/*", {
                                             .compression = uWS::SHARED_COMPRESSOR,
                                             .maxPayloadLength = 16 * 1024 * 1024,
                                             .idleTimeout = 60,
                                             .maxBackpressure = 1 * 1024 * 1024,
                                             .closeOnBackpressureLimit = false,
                                             .resetIdleTimeoutOnSend = true,
                                             .sendPingsAutomatically = true,

                                             .upgrade = nullptr,
                                             .open = [this](auto *ws) {
                                                 ws->getUserData()->client = this;
                                                 this->ws_ = ws;
                                                 connected_ = true;
                                                 event_handler_->on_connected();

                                                 // Send hello message
                                                 HelloMessage hello;
                                                 hello.protocol_version = PROTOCOL_VERSION;
                                                 hello.client_version = "SHA1-Miner/1.0-uWS";
                                                 hello.capabilities = {"gpu", "multi-gpu", "vardiff"};

                                                 Message msg;
                                                 msg.type = MessageType::HELLO;
                                                 msg.id = Utils::generate_message_id();
                                                 msg.timestamp = Utils::current_timestamp_ms();
                                                 msg.payload = hello.to_json();

                                                 send_message(msg);
                                             },
                                             .message = [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                                                 auto parsed = Message::deserialize(std::string(message));
                                                 if (parsed) {
                                                     std::lock_guard<std::mutex> lock(incoming_mutex_);
                                                     incoming_queue_.push(*parsed);
                                                     incoming_cv_.notify_one();
                                                 }
                                             },
                                             .drain = [](auto *ws) {
                                                 // Handle backpressure
                                             },
                                             .ping = [](auto *ws, std::string_view) {
                                                 // Ping received
                                             },
                                             .pong = [](auto *ws, std::string_view) {
                                                 // Pong received
                                             },
                                             .close = [this](auto *ws, int code, std::string_view message) {
                                                 connected_ = false;
                                                 authenticated_ = false;
                                                 event_handler_->on_disconnected(std::string(message));
                                             }
                                         }).connect(parsed.host, parsed.port, parsed.path, config_.keepalive_interval_s,
                                                    {{"User-Agent", "SHA1-Miner/1.0"}})
                    .run();
        }
    }

    void PoolClient::disconnect() {
        if (!running_.load()) {
            return;
        }

        running_ = false;
        connected_ = false;
        authenticated_ = false;

        // Close WebSocket connection
        if (ws_) {
            ws_->close();
        }

        // Stop the event loop
        if (loop_) {
            loop_->defer([this]() {
                if (ws_) {
                    ws_->close();
                }
            });
        }

        // Notify condition variables
        outgoing_cv_.notify_all();
        incoming_cv_.notify_all();

        // Join threads
        if (io_thread_ && io_thread_->joinable()) {
            io_thread_->join();
        }
        if (message_processor_thread_.joinable()) {
            message_processor_thread_.join();
        }
        if (keepalive_thread_.joinable()) {
            keepalive_thread_.join();
        }
    }

    void PoolClient::send_message(const Message &msg) {
        if (!connected_.load() || !ws_) {
            return;
        }

        std::string payload = msg.serialize();

        // uWebSockets is thread-safe for send operations
        ws_->send(payload, uWS::OpCode::TEXT);
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
            request_job();
        } else {
            authenticated_ = false;
            event_handler_->on_auth_failed(response.error_code, response.error_message);
        }
    }

    void PoolClient::request_job() {
        Message msg;
        msg.type = MessageType::GET_JOB;
        msg.id = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        send_message(msg);
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

    // Implement remaining message handlers...
    void PoolClient::handle_difficulty_adjust(const DifficultyAdjustMessage &adjust) { {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            worker_stats_.current_difficulty = adjust.new_difficulty;
        }
        event_handler_->on_difficulty_changed(adjust.new_difficulty);
    }

    void PoolClient::handle_pool_status(const PoolStatusMessage &status) {
        event_handler_->on_pool_status(status);
    }

    void PoolClient::handle_error(const Message &msg) {
        ErrorCode code = static_cast<ErrorCode>(msg.payload.value("code", 0));
        std::string message = msg.payload.value("message", "Unknown error");
        event_handler_->on_error(code, message);
    }
} // namespace MiningPool
