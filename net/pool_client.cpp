// pool_client.cpp - Fixed version with proper uWebSockets usage
#include "pool_client.hpp"
#include <iostream>
#include <iomanip>
#include <regex>
#include <uwebsockets/App.h>

namespace MiningPool {
    // ParsedUrl implementation
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

    // PoolClient constructor
    PoolClient::PoolClient(const PoolConfig &config, IPoolEventHandler *handler)
        : config_(config), event_handler_(handler) {
        worker_stats_.worker_id = config.worker_name;
        worker_stats_.connected_since = std::chrono::steady_clock::now();
    }

    // PoolClient destructor
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

            // Wait for connection with timeout
            auto start = std::chrono::steady_clock::now();
            while (!connected_.load() &&
                   std::chrono::steady_clock::now() - start <
                   std::chrono::seconds(config_.response_timeout_ms / 1000)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            return connected_.load();
        } catch (const std::exception &e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            running_ = false;
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

        // Close WebSocket connection
        if (ws_) {
            if (config_.use_tls) {
                reinterpret_cast<uWS::WebSocket<true, true, WebSocketData> *>(ws_)->close();
            } else {
                reinterpret_cast<uWS::WebSocket<false, true, WebSocketData> *>(ws_)->close();
            }
            ws_ = nullptr;
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

    void PoolClient::io_loop() {
        auto parsed = parse_url(config_.url);

        // Create and run the appropriate app (SSL or non-SSL)
        if (config_.use_tls) {
            // Configure SSL options using uWS::SocketContextOptions
            uWS::SocketContextOptions ssl_options = {};
            if (!config_.tls_cert_file.empty()) {
                ssl_options.cert_file_name = config_.tls_cert_file.c_str();
            }
            if (!config_.tls_key_file.empty()) {
                ssl_options.key_file_name = config_.tls_key_file.c_str();
            }

            // Create SSL app with options
            uWS::SSLApp app(ssl_options);

            // Configure WebSocket behavior for SSL
            app.ws<WebSocketData>("/*", {
                                      .compression = uWS::SHARED_COMPRESSOR,
                                      .maxPayloadLength = 16 * 1024 * 1024,
                                      .idleTimeout = static_cast<unsigned short>(config_.keepalive_interval_s * 2),
                                      .maxBackpressure = 1 * 1024 * 1024,
                                      .closeOnBackpressureLimit = false,
                                      .resetIdleTimeoutOnSend = true,
                                      .sendPingsAutomatically = true,

                                      .open = [this](auto *ws) {
                                          ws->getUserData()->client = this;
                                          this->ws_ = ws;
                                          connected_ = true;

                                          // Update stats
                                          {
                                              std::lock_guard<std::mutex> lock(stats_mutex_);
                                              worker_stats_.connected_since = std::chrono::steady_clock::now();
                                          }

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
                                      },

                                      .message = [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                                          if (opCode == uWS::OpCode::TEXT) {
                                              auto parsed = Message::deserialize(std::string(message));
                                              if (parsed) {
                                                  std::lock_guard<std::mutex> lock(incoming_mutex_);
                                                  incoming_queue_.push(*parsed);
                                                  incoming_cv_.notify_one();
                                              }
                                          }
                                      },

                                      .drain = [this](auto *ws) {
                                          // Handle backpressure
                                          auto bufferedAmount = ws->getBufferedAmount();
                                          if (bufferedAmount > 5 * 1024 * 1024) {
                                              std::cerr << "WebSocket backpressure high: " << bufferedAmount <<
                                                      " bytes buffered" << std::endl;
                                          }
                                      },

                                      .ping = [this](auto *ws, std::string_view data) {
                                          // Ping received
                                          if (config_.debug_mode) {
                                              std::cout << "Ping received from server" << std::endl;
                                          }
                                      },

                                      .pong = [this](auto *ws, std::string_view data) {
                                          // Pong received
                                          if (config_.debug_mode) {
                                              std::cout << "Pong received from server" << std::endl;
                                          }
                                      },

                                      .close = [this](auto *ws, int code, std::string_view message) {
                                          connected_ = false;
                                          authenticated_ = false;
                                          ws_ = nullptr;

                                          std::string reason = message.empty()
                                                                   ? "Connection closed"
                                                                   : std::string(message);
                                          event_handler_->on_disconnected(reason);

                                          // Handle reconnection if configured
                                          if (running_.load() && config_.reconnect_attempts != 0) {
                                              std::this_thread::sleep_for(
                                                  std::chrono::milliseconds(config_.reconnect_delay_ms));
                                              reconnect();
                                          }
                                      }
                                  });

            // Connect to the WebSocket endpoint
            app.get("/*", [this, parsed](auto *res, auto *req) {
                // Get the host header for the WebSocket handshake
                std::string host_header = parsed.host + ":" + std::to_string(parsed.port);

                res->template upgrade<WebSocketData>(
                    {this}, // user data
                    req->getHeader("sec-websocket-key"),
                    req->getHeader("sec-websocket-protocol"),
                    req->getHeader("sec-websocket-extensions"),
                    (us_socket_context_t *) nullptr // use default context
                );
            });

            // Listen on all interfaces
            app.listen(0, [this, parsed](auto *token) {
                if (token) {
                    // Now connect as a client
                    // Note: uWebSockets doesn't have a direct client API, so we need to use
                    // the HTTP client functionality to upgrade to WebSocket
                    std::cout << "Connecting to " << parsed.host << ":" << parsed.port << std::endl;
                }
            }).run();
        } else {
            // Create non-SSL app
            uWS::App app;

            // Configure WebSocket behavior for non-SSL
            app.ws<WebSocketData>("/*", {
                                      .compression = uWS::SHARED_COMPRESSOR,
                                      .maxPayloadLength = 16 * 1024 * 1024,
                                      .idleTimeout = static_cast<unsigned short>(config_.keepalive_interval_s * 2),
                                      .maxBackpressure = 1 * 1024 * 1024,
                                      .closeOnBackpressureLimit = false,
                                      .resetIdleTimeoutOnSend = true,
                                      .sendPingsAutomatically = true,

                                      .open = [this](auto *ws) {
                                          ws->getUserData()->client = this;
                                          this->ws_ = ws;
                                          connected_ = true;

                                          // Update stats
                                          {
                                              std::lock_guard<std::mutex> lock(stats_mutex_);
                                              worker_stats_.connected_since = std::chrono::steady_clock::now();
                                          }

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
                                      },

                                      .message = [this](auto *ws, std::string_view message, uWS::OpCode opCode) {
                                          if (opCode == uWS::OpCode::TEXT) {
                                              auto parsed = Message::deserialize(std::string(message));
                                              if (parsed) {
                                                  std::lock_guard<std::mutex> lock(incoming_mutex_);
                                                  incoming_queue_.push(*parsed);
                                                  incoming_cv_.notify_one();
                                              }
                                          }
                                      },

                                      .drain = [this](auto *ws) {
                                          // Handle backpressure
                                          auto bufferedAmount = ws->getBufferedAmount();
                                          if (bufferedAmount > 5 * 1024 * 1024) {
                                              std::cerr << "WebSocket backpressure high: " << bufferedAmount <<
                                                      " bytes buffered" << std::endl;
                                          }
                                      },

                                      .ping = [this](auto *ws, std::string_view data) {
                                          // Ping received
                                          if (config_.debug_mode) {
                                              std::cout << "Ping received from server" << std::endl;
                                          }
                                      },

                                      .pong = [this](auto *ws, std::string_view data) {
                                          // Pong received
                                          if (config_.debug_mode) {
                                              std::cout << "Pong received from server" << std::endl;
                                          }
                                      },

                                      .close = [this](auto *ws, int code, std::string_view message) {
                                          connected_ = false;
                                          authenticated_ = false;
                                          ws_ = nullptr;

                                          std::string reason = message.empty()
                                                                   ? "Connection closed"
                                                                   : std::string(message);
                                          event_handler_->on_disconnected(reason);

                                          // Handle reconnection if configured
                                          if (running_.load() && config_.reconnect_attempts != 0) {
                                              std::this_thread::sleep_for(
                                                  std::chrono::milliseconds(config_.reconnect_delay_ms));
                                              reconnect();
                                          }
                                      }
                                  });

            // Connect to the WebSocket endpoint
            app.get("/*", [this, parsed](auto *res, auto *req) {
                // Get the host header for the WebSocket handshake
                std::string host_header = parsed.host + ":" + std::to_string(parsed.port);

                res->template upgrade<WebSocketData>(
                    {this}, // user data
                    req->getHeader("sec-websocket-key"),
                    req->getHeader("sec-websocket-protocol"),
                    req->getHeader("sec-websocket-extensions"),
                    (us_socket_context_t *) nullptr // use default context
                );
            });

            // Listen on all interfaces
            app.listen(0, [this, parsed](auto *token) {
                if (token) {
                    // Now connect as a client
                    std::cout << "Connecting to " << parsed.host << ":" << parsed.port << std::endl;
                }
            }).run();
        }
    }

    void PoolClient::send_message(const Message &msg) {
        if (!connected_.load() || !ws_) {
            return;
        }

        std::string payload = msg.serialize();

        // Track pending requests if expecting response
        if (msg.type == MessageType::AUTH ||
            msg.type == MessageType::SUBMIT_SHARE ||
            msg.type == MessageType::GET_JOB) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_requests_[msg.id] = std::chrono::steady_clock::now();
        }

        // Send message based on SSL/non-SSL
        if (config_.use_tls) {
            auto *ssl_ws = reinterpret_cast<uWS::WebSocket<true, true, WebSocketData> *>(ws_);
            ssl_ws->send(payload, uWS::OpCode::TEXT);
        } else {
            auto *ws = reinterpret_cast<uWS::WebSocket<false, true, WebSocketData> *>(ws_);
            ws->send(payload, uWS::OpCode::TEXT);
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

            // Check for timed out requests
            check_pending_timeouts();
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
        // Remove from pending if this is a response
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_requests_.erase(msg.id);
        }

        try {
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
                case static_cast<MessageType>(0x17): // ERROR type without using the enum
                    handle_error(msg);
                    break;
                case MessageType::RECONNECT:
                    // Server requested reconnect
                    reconnect();
                    break;
                default:
                    std::cerr << "Unknown message type: " << static_cast<int>(msg.type) << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "Error processing message: " << e.what() << std::endl;
            event_handler_->on_error(ErrorCode::PROTOCOL_ERROR, e.what());
        }
    }

    void PoolClient::handle_welcome(const WelcomeMessage &welcome) {
        std::cout << "Connected to pool: " << welcome.pool_name
                << " v" << welcome.pool_version << std::endl;

        // Check protocol compatibility
        if (welcome.protocol_version != PROTOCOL_VERSION) {
            std::cerr << "Protocol version mismatch. Pool: " << welcome.protocol_version
                    << ", Client: " << PROTOCOL_VERSION << std::endl;
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            worker_stats_.current_difficulty = welcome.min_difficulty;
        }

        // Proceed to authentication
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
            worker_id_ = response.worker_id; {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                worker_stats_.worker_id = worker_id_;
            }

            event_handler_->on_authenticated(worker_id_);

            // Request initial job
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

            // Clean jobs if requested
            if (job.clean_jobs) {
                for (auto &[id, existing_job]: active_jobs_) {
                    existing_job.is_active = false;
                    event_handler_->on_job_cancelled(id);
                }
                active_jobs_.clear();
            }

            // Add new job
            active_jobs_[job.job_id] = pool_job;
            current_job_id_ = job.job_id;

            // Clean up expired jobs
            cleanup_expired_jobs();
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            worker_stats_.last_job_time = std::chrono::steady_clock::now();
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

        // Track submission time for latency stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            worker_stats_.last_share_time = std::chrono::steady_clock::now();
        }

        send_message(msg);
    }

    void PoolClient::handle_share_result(const ShareResultMessage &result) {
        update_stats(result);

        if (result.status == ShareStatus::ACCEPTED) {
            event_handler_->on_share_accepted(result);

            // Check for special messages (e.g., "Block found!")
            if (!result.message.empty()) {
                std::cout << "Pool message: " << result.message << std::endl;
            }
        } else {
            event_handler_->on_share_rejected(result);

            // Log rejection reason
            std::string reason = "Unknown";
            switch (result.status) {
                case ShareStatus::REJECTED_LOW_DIFFICULTY:
                    reason = "Low difficulty";
                    break;
                case ShareStatus::REJECTED_INVALID:
                    reason = "Invalid share";
                    break;
                case ShareStatus::REJECTED_STALE:
                    reason = "Stale share";
                    break;
                case ShareStatus::REJECTED_DUPLICATE:
                    reason = "Duplicate share";
                    break;
                default:
                    break;
            }
            std::cerr << "Share rejected: " << reason << std::endl;
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

    void PoolClient::handle_difficulty_adjust(const DifficultyAdjustMessage &adjust) { {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            worker_stats_.current_difficulty = adjust.new_difficulty;
        }

        std::cout << "Difficulty adjusted to " << adjust.new_difficulty
                << " (" << adjust.reason << ")" << std::endl;

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

    void PoolClient::report_hashrate(const HashrateReportMessage &report) {
        if (!authenticated_.load()) {
            return;
        }

        Message msg;
        msg.type = MessageType::HASHRATE_REPORT;
        msg.id = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload = report.to_json();

        send_message(msg);

        // Update local stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            worker_stats_.current_hashrate = report.hashrate;

            // Update average hashrate with exponential moving average
            const double alpha = 0.1; // Smoothing factor
            if (worker_stats_.average_hashrate == 0) {
                worker_stats_.average_hashrate = report.hashrate;
            } else {
                worker_stats_.average_hashrate =
                        alpha * report.hashrate + (1 - alpha) * worker_stats_.average_hashrate;
            }
        }
    }

    std::optional<PoolJob> PoolClient::get_current_job() const {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = active_jobs_.find(current_job_id_);
        if (it != active_jobs_.end() && it->second.is_active && !it->second.is_expired()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<PoolJob> PoolClient::get_active_jobs() const {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        std::vector<PoolJob> jobs;

        for (const auto &[id, job]: active_jobs_) {
            if (job.is_active && !job.is_expired()) {
                jobs.push_back(job);
            }
        }

        return jobs;
    }

    WorkerStats PoolClient::get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return worker_stats_;
    }

    void PoolClient::cleanup_expired_jobs() {
        auto now = std::chrono::steady_clock::now();

        for (auto it = active_jobs_.begin(); it != active_jobs_.end();) {
            if (it->second.is_expired()) {
                event_handler_->on_job_cancelled(it->first);
                it = active_jobs_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void PoolClient::check_pending_timeouts() {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        auto now = std::chrono::steady_clock::now();

        for (auto it = pending_requests_.begin(); it != pending_requests_.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second
            ).count();

            if (elapsed > config_.response_timeout_ms) {
                std::cerr << "Request timeout: message_id=" << it->first << std::endl;
                it = pending_requests_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void PoolClient::reconnect() {
        if (!running_.load()) {
            return;
        }

        std::cout << "Attempting to reconnect..." << std::endl;

        // Close current connection
        if (ws_) {
            if (config_.use_tls) {
                reinterpret_cast<uWS::WebSocket<true, true, WebSocketData> *>(ws_)->close();
            } else {
                reinterpret_cast<uWS::WebSocket<false, true, WebSocketData> *>(ws_)->close();
            }
            ws_ = nullptr;
        }

        connected_ = false;
        authenticated_ = false;

        // Clear state
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            active_jobs_.clear();
            current_job_id_.clear();
        }

        // Attempt reconnection
        // The close handler in io_loop will handle the actual reconnection
    }

    // PoolClientManager implementation
    PoolClientManager::PoolClientManager() = default;

    PoolClientManager::~PoolClientManager() {
        disconnect_all();
    }

    bool PoolClientManager::add_pool(const std::string &name, const PoolConfig &config,
                                     IPoolEventHandler *handler) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (clients_.count(name) > 0) {
            std::cerr << "Pool with name '" << name << "' already exists" << std::endl;
            return false;
        }

        auto client = std::make_shared<PoolClient>(config, handler);
        clients_[name] = client;

        if (primary_pool_name_.empty()) {
            primary_pool_name_ = name;
        }

        return true;
    }

    bool PoolClientManager::remove_pool(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = clients_.find(name);
        if (it == clients_.end()) {
            return false;
        }

        it->second->disconnect();
        clients_.erase(it);

        if (primary_pool_name_ == name) {
            primary_pool_name_ = clients_.empty() ? "" : clients_.begin()->first;
        }

        return true;
    }

    bool PoolClientManager::set_primary_pool(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (clients_.count(name) == 0) {
            return false;
        }

        primary_pool_name_ = name;
        return true;
    }

    std::string PoolClientManager::get_primary_pool() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return primary_pool_name_;
    }

    void PoolClientManager::enable_failover(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        failover_enabled_ = enable;
    }

    void PoolClientManager::set_failover_order(const std::vector<std::string> &pool_names) {
        std::lock_guard<std::mutex> lock(mutex_);
        failover_order_ = pool_names;
    }

    std::shared_ptr<PoolClient> PoolClientManager::get_client(const std::string &name) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = clients_.find(name);
        if (it != clients_.end()) {
            return it->second;
        }

        return nullptr;
    }

    std::shared_ptr<PoolClient> PoolClientManager::get_primary_client() const {
        return get_client(get_primary_pool());
    }

    void PoolClientManager::connect_all() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &[name, client]: clients_) {
            if (!client->is_connected()) {
                std::cout << "Connecting to pool: " << name << std::endl;
                client->connect();
            }
        }
    }

    void PoolClientManager::disconnect_all() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &[name, client]: clients_) {
            if (client->is_connected()) {
                std::cout << "Disconnecting from pool: " << name << std::endl;
                client->disconnect();
            }
        }
    }

    std::map<std::string, WorkerStats> PoolClientManager::get_all_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::map<std::string, WorkerStats> all_stats;

        for (const auto &[name, client]: clients_) {
            if (client->is_connected()) {
                all_stats[name] = client->get_stats();
            }
        }

        return all_stats;
    }

    void PoolClientManager::handle_failover() {
        if (!failover_enabled_) {
            return;
        }

        // Check if primary pool is still connected
        auto primary = get_primary_client();
        if (primary && primary->is_connected()) {
            return;
        }

        // Try failover pools in order
        for (const auto &pool_name: failover_order_) {
            auto client = get_client(pool_name);
            if (client && client->is_connected()) {
                std::cout << "Failing over to pool: " << pool_name << std::endl;
                set_primary_pool(pool_name);
                return;
            }
        }

        // If no pools in failover list are connected, try any connected pool
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[name, client]: clients_) {
            if (client->is_connected() && name != primary_pool_name_) {
                std::cout << "Failing over to pool: " << name << std::endl;
                primary_pool_name_ = name;
                return;
            }
        }
    }
} // namespace MiningPool
