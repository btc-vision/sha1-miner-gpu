#include "pool_client.hpp"
#include "../logging/logger.hpp"
#include <regex>

#include "gpu_platform.hpp"

namespace MiningPool {
    // ParsedUrl implementation
    PoolClient::ParsedUrl PoolClient::parse_url(const std::string &url) {
        ParsedUrl result;

        // Updated regex that properly handles localhost and other hostnames
        const std::regex url_regex(R"(^(wss?):\/\/([^:\/]+)(?::(\d+))?(\/.*)?$)");
        std::smatch matches;

        if (std::regex_match(url, matches, url_regex)) {
            result.is_secure = (matches[1].str() == "wss");
            result.host = matches[2].str();
            // Extract port with proper defaults
            if (matches[3].length()) {
                result.port = matches[3].str();
            } else {
                result.port = result.is_secure ? "443" : "80";
            }
            result.path = matches[4].length() ? matches[4].str() : "/";
            // Debug output
            LOG_DEBUG("CLIENT", "Parsed URL - Host: ", result.host, ", Port: ", result.port,
                      ", Path: ", result.path, ", Secure: ", result.is_secure);
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

        // Initialize work guard to keep io_context running
        work_guard_ = std::make_unique<net::executor_work_guard<net::io_context::executor_type> >(ioc_.get_executor());
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

            // IMPORTANT: Start the IO thread FIRST before any other operations
            io_thread_ = std::make_unique<std::thread>(&PoolClient::io_loop, this);
            // Give IO thread time to start
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Then start auxiliary threads
            message_processor_thread_ = std::thread(&PoolClient::message_processor_loop, this);
            keepalive_thread_ = std::thread(&PoolClient::keepalive_loop, this);

            // Add DNS resolution debugging
            LOG_INFO("CLIENT", "Resolving host: ", parsed.host, ":", parsed.port);

            // Create a promise/future to synchronize the connection
            std::promise<bool> connection_promise;
            auto connection_future = connection_promise.get_future();

            // Post the connection work to the io_context
            net::post(ioc_, [this, parsed, &connection_promise]() {
                try {
                    // Resolve host
                    tcp::resolver resolver{ioc_};
                    boost::system::error_code ec;
                    auto const results = resolver.resolve(parsed.host, parsed.port, ec);

                    if (ec) {
                        throw std::runtime_error("Failed to resolve host: " + parsed.host + " - " + ec.message());
                    }

                    if (results.empty()) {
                        throw std::runtime_error("Failed to resolve host: " + parsed.host);
                    }

                    // Print resolved addresses
                    for (auto const &endpoint: results) {
                        LOG_DEBUG("CLIENT", "Resolved to: ", endpoint.endpoint().address().to_string(),
                                  ":", endpoint.endpoint().port());
                    }

                    if (config_.use_tls) {
                        // Initialize SSL context
                        ssl_ctx_.set_verify_mode(config_.verify_server_cert ? ssl::verify_peer : ssl::verify_none);
                        if (!config_.tls_cert_file.empty() && !config_.tls_key_file.empty()) {
                            ssl_ctx_.use_certificate_file(config_.tls_cert_file, ssl::context::pem);
                            ssl_ctx_.use_private_key_file(config_.tls_key_file, ssl::context::pem);
                        }

                        // Create SSL stream
                        wss_ = std::make_unique<websocket::stream<ssl::stream<tcp::socket> > >(ioc_, ssl_ctx_);

                        // Connect TCP socket
                        beast::get_lowest_layer(*wss_).connect(results.begin()->endpoint());

                        // Perform SSL handshake
                        wss_->next_layer().handshake(ssl::stream_base::client);

                        // Set WebSocket options
                        wss_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
                        wss_->set_option(websocket::stream_base::decorator(
                            [](websocket::request_type &req) {
                                req.set(http::field::user_agent, "SHA1-Miner/1.0 Boost.Beast");
                            }));

                        // Perform WebSocket handshake
                        wss_->handshake(parsed.host, parsed.path);
                    } else {
                        // Create plain WebSocket stream
                        ws_ = std::make_unique<websocket::stream<tcp::socket> >(ioc_);

                        // Connect TCP socket
                        beast::get_lowest_layer(*ws_).connect(results.begin()->endpoint());

                        // Set WebSocket options
                        ws_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
                        ws_->set_option(websocket::stream_base::decorator(
                            [](websocket::request_type &req) {
                                req.set(http::field::user_agent, "SHA1-Miner/1.0 Boost.Beast");
                            }));

                        // Perform WebSocket handshake with host header
                        ws_->handshake(parsed.host + ":" + parsed.port, parsed.path);
                    }

                    connected_ = true;
                    worker_stats_.connected_since = std::chrono::steady_clock::now();
                    // Start reading immediately in the IO thread
                    LOG_DEBUG("CLIENT", "Starting async read loop");
                    do_read();
                    connection_promise.set_value(true);
                } catch (const std::exception &e) {
                    LOG_ERROR("CLIENT", "Connection error in IO thread: ", e.what());
                    connection_promise.set_value(false);
                }
            });

            // Wait for connection to complete
            if (!connection_future.get()) {
                running_ = false;
                connected_ = false;
                return false;
            }

            // Connection successful, notify handler
            event_handler_->on_connected();

            // Send hello message after connection is established
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            HelloMessage hello;
            hello.protocol_version = PROTOCOL_VERSION;
            hello.client_version = "SHA1-Miner/1.0";
            hello.capabilities = {"gpu", "multi-gpu", "vardiff"};
            hello.user_agent = "SHA1-Miner";

            Message msg;
            msg.type = MessageType::HELLO;
            msg.id = Utils::generate_message_id();
            msg.timestamp = Utils::current_timestamp_ms();
            msg.payload = hello.to_json();

            LOG_INFO("CLIENT", "Sending HELLO message");
            send_message(msg);

            return true;
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Connection error: ", e.what());
            running_ = false;
            connected_ = false;
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
        try {
            if (config_.use_tls && wss_) {
                if (wss_->is_open()) {
                    wss_->close(websocket::close_code::normal);
                }
            } else if (ws_) {
                if (ws_->is_open()) {
                    ws_->close(websocket::close_code::normal);
                }
            }
        } catch (const std::exception &e) {
            // Ignore errors during shutdown
        }

        // Stop io_context
        if (work_guard_) {
            work_guard_.reset();
        }
        ioc_.stop();

        // Notify condition variables
        outgoing_cv_.notify_all();
        incoming_cv_.notify_all();

        // IMPORTANT: Don't join threads if we're being called from one of them
        // Check if we're being called from io_thread
        if (io_thread_ && io_thread_->joinable() && io_thread_->get_id() != std::this_thread::get_id()) {
            io_thread_->join();
        }
        // Check if we're being called from message_processor_thread
        if (message_processor_thread_.joinable() && message_processor_thread_.get_id() != std::this_thread::get_id()) {
            message_processor_thread_.join();
        }

        // Check if we're being called from keepalive_thread
        if (keepalive_thread_.joinable() &&
            keepalive_thread_.get_id() != std::this_thread::get_id()) {
            keepalive_thread_.join();
        }
    }

    void PoolClient::io_loop() {
        try {
            ioc_.run();
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "IO loop error: ", e.what());
        }
    }

    void PoolClient::send_message(const Message &msg) {
        LOG_DEBUG("CLIENT", "send_message called for type: ", static_cast<int>(msg.type), ", ID: ", msg.id);
        if (!connected_.load()) {
            LOG_ERROR("CLIENT", "Cannot send message - not connected");
            return;
        }

        try {
            std::string payload = msg.serialize();
            LOG_DEBUG("CLIENT", "Message serialized, length: ", payload.length());
            LOG_TRACE("CLIENT", "Payload: ", payload);

            // Track pending requests if expecting response
            if (msg.type == MessageType::AUTH ||
                msg.type == MessageType::SUBMIT_SHARE ||
                msg.type == MessageType::GET_JOB) {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_requests_[msg.id] = std::chrono::steady_clock::now();
                LOG_DEBUG("CLIENT", "Added message ID ", msg.id, " to pending requests");
            }

            // Queue message for sending
            {
                std::lock_guard<std::mutex> lock(outgoing_mutex_);
                outgoing_queue_.push(msg);
                LOG_DEBUG("CLIENT", "Message queued, queue size: ", outgoing_queue_.size());
                outgoing_cv_.notify_one();
            }

            // Trigger write if not already writing
            net::post(ioc_, [this]() {
                LOG_DEBUG("CLIENT", "Posted do_write to io_context");
                do_write();
            });
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Exception in send_message: ", e.what());
        }
    }

    void PoolClient::do_read() {
        if (!connected_.load()) {
            LOG_ERROR("CLIENT", "do_read called but not connected!");
            return;
        }

        LOG_TRACE("CLIENT", "Setting up async_read");
        // Clear buffer before reading
        buffer_.consume(buffer_.size());
        auto read_handler = [this](beast::error_code ec, std::size_t bytes_transferred) {
            LOG_TRACE("CLIENT", "async_read completed - ec: ", ec, ", bytes: ", bytes_transferred);
            if (!ec) {
                LOG_TRACE("CLIENT", "Read data: ", beast::buffers_to_string(buffer_.data()));
            }
            on_read(ec, bytes_transferred);
        };

        if (config_.use_tls && wss_) {
            wss_->async_read(buffer_, read_handler);
        } else if (ws_) {
            ws_->async_read(buffer_, read_handler);
        } else {
            LOG_ERROR("CLIENT", "ERROR: No websocket stream available!");
        }
    }

    void PoolClient::do_write() {
        if (!connected_.load() || write_in_progress_.load()) {
            return;
        }
        std::unique_lock<std::mutex> lock(outgoing_mutex_);
        if (outgoing_queue_.empty()) {
            return;
        }
        write_in_progress_ = true;
        Message msg = outgoing_queue_.front();
        outgoing_queue_.pop();
        lock.unlock();

        current_write_payload_ = msg.serialize();
        auto write_handler = [this](beast::error_code ec, std::size_t bytes_transferred) {
            write_in_progress_ = false;
            on_write(ec, bytes_transferred);
        };

        if (config_.use_tls && wss_) {
            wss_->async_write(net::buffer(current_write_payload_), write_handler);
        } else if (ws_) {
            ws_->async_write(net::buffer(current_write_payload_), write_handler);
        }
    }

    void PoolClient::on_write(beast::error_code ec, std::size_t bytes_transferred) {
        if (ec) {
            LOG_ERROR("CLIENT", "Write error: ", ec.message());
            return;
        }

        // Check if more messages to send
        std::lock_guard<std::mutex> lock(outgoing_mutex_);
        if (!outgoing_queue_.empty()) {
            net::post(ioc_, [this]() { do_write(); });
        }
    }

    void PoolClient::on_read(beast::error_code ec, std::size_t bytes_transferred) {
        if (ec) {
            if (ec != websocket::error::closed) {
                std::string error_msg = ec.message();
#ifdef _WIN32
                if (error_msg.find('\xd9') != std::string::npos || error_msg.find('\xda') != std::string::npos ||
                    error_msg.find('\xc6') != std::string::npos) {
                    error_msg = "Connection closed by remote host (error code: " + std::to_string(ec.value()) + ")";
                }
#endif
                LOG_ERROR("CLIENT", "Read error: ", error_msg);
            }

            connected_ = false;
            authenticated_ = false;
            event_handler_->on_disconnected(ec.message());

            if (running_.load() && config_.reconnect_attempts != 0) {
                net::post(ioc_, [this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_delay_ms));
                    reconnect();
                });
            }
            return;
        }

        // Check if we're still connected before processing
        if (!connected_.load()) {
            LOG_ERROR("CLIENT", "Received data but not connected, ignoring");
            return;
        }

        std::string data = beast::buffers_to_string(buffer_.data());
        buffer_.consume(bytes_transferred);

        LOG_DEBUG("CLIENT", "Received ", bytes_transferred, " bytes");
        auto parsed = Message::deserialize(data);
        if (parsed) {
            LOG_DEBUG("CLIENT", "Successfully parsed message type: ", static_cast<int>(parsed->type),
                      ", id: ", parsed->id);

            // Check for error messages that should stop further reading
            if (parsed->type == MessageType::ERROR_PROBLEM) {
                int error_code = parsed->payload.value("code", 0);
                if (error_code == static_cast<int>(ErrorCode::INVALID_MESSAGE) ||
                    error_code == static_cast<int>(ErrorCode::PROTOCOL_ERROR)) {
                    LOG_ERROR("CLIENT", "Received fatal error, stopping reads");
                    connected_ = false;
                }
            }

            std::lock_guard<std::mutex> lock(incoming_mutex_);
            incoming_queue_.push(*parsed);
            incoming_cv_.notify_one();
        }

        // Continue reading only if still connected
        if (connected_.load()) {
            do_read();
        }
    }

    bool PoolClient::is_authenticated() const {
        return authenticated_.load();
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
        LOG_DEBUG("CLIENT", "Message processor thread started (tid: ", std::this_thread::get_id(), ")");
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(incoming_mutex_);
            LOG_TRACE("CLIENT", "Message processor waiting for messages...");
            // Wait for messages with timeout to check running status
            bool has_message = incoming_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this] {
                bool result = !incoming_queue_.empty() || !running_.load();
                if (result) {
                    LOG_TRACE("CLIENT", "Wait condition met - queue empty: ", incoming_queue_.empty(),
                              ", running: ", running_.load());
                }
                return result;
            });

            if (!has_message) {
                LOG_TRACE("CLIENT", "Message processor timeout - no messages");
                continue;
            }

            while (!incoming_queue_.empty()) {
                Message msg = incoming_queue_.front();
                incoming_queue_.pop();
                LOG_DEBUG("CLIENT", "Dequeued message for processing - type: ", static_cast<int>(msg.type),
                          ", id: ", msg.id);
                lock.unlock();
                try {
                    process_message(msg);
                } catch (const std::exception &e) {
                    LOG_ERROR("CLIENT", "Exception processing message: ", e.what());
                }

                lock.lock();
            }
        }

        LOG_DEBUG("CLIENT", "Message processor thread stopped");
    }

    void PoolClient::process_message(const Message &msg) {
        // Remove from pending if this is a response
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_requests_.erase(msg.id);
        }

        LOG_DEBUG("CLIENT", "process_message called with type: ", static_cast<int>(msg.type),
                  " (0x", std::hex, static_cast<int>(msg.type), std::dec, ")");

        try {
            switch (msg.type) {
                case MessageType::WELCOME: // This is 0x11 = 17
                    LOG_INFO("CLIENT", "Handling WELCOME message");
                    handle_welcome(WelcomeMessage::from_json(msg.payload));
                    break;
                case MessageType::AUTH_RESPONSE:
                    LOG_INFO("CLIENT", "Handling AUTH_RESPONSE message");
                    handle_auth_response(AuthResponseMessage::from_json(msg.payload));
                    break;
                case MessageType::NEW_JOB:
                    LOG_INFO("CLIENT", "Handling NEW_JOB message");
                    handle_new_job(JobMessage::from_json(msg.payload));
                    break;
                case MessageType::SHARE_RESULT:
                    LOG_INFO("CLIENT", "Handling SHARE_RESULT message");
                    handle_share_result(ShareResultMessage::from_json(msg.payload));
                    break;
                case MessageType::DIFFICULTY_ADJUST:
                    LOG_INFO("CLIENT", "Handling DIFFICULTY_ADJUST message");
                    handle_difficulty_adjust(DifficultyAdjustMessage::from_json(msg.payload));
                    break;
                case MessageType::POOL_STATUS:
                    LOG_INFO("CLIENT", "Handling POOL_STATUS message");
                    handle_pool_status(PoolStatusMessage::from_json(msg.payload));
                    break;
                case MessageType::ERROR_PROBLEM: // This is 0x17 = 23
                    LOG_WARN("CLIENT", "Handling ERROR message");
                    handle_error(msg);
                    break;
                case MessageType::RECONNECT:
                    LOG_INFO("CLIENT", "Handling RECONNECT message");
                    // Server requested reconnect
                    reconnect();
                    break;
                default:
                    LOG_ERROR("CLIENT", "Unknown message type: ", static_cast<int>(msg.type),
                              " (hex: 0x", std::hex, static_cast<int>(msg.type), std::dec, ")");
            }
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Error processing message type ", static_cast<int>(msg.type),
                      ": ", e.what());
            event_handler_->on_error(ErrorCode::PROTOCOL_ERROR, e.what());
        }
    }

    void PoolClient::handle_welcome(const WelcomeMessage &welcome) {
        LOG_INFO("CLIENT", Color::GREEN, "Connected to pool: ", Color::RESET, welcome.pool_name,
                 " v", welcome.pool_version);

        // Check protocol compatibility
        if (welcome.protocol_version != PROTOCOL_VERSION) {
            LOG_WARN("CLIENT", "Protocol version mismatch. Pool: ", welcome.protocol_version,
                     ", Client: ", PROTOCOL_VERSION);
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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        AuthMessage auth;
        auth.method = config_.auth_method;
        auth.username = config_.username + "." + config_.worker_name;
        auth.password = config_.password;
        auth.session_id = session_id_;

        // Add client info for better server-side difficulty adjustment
        ClientInfo client_info;

        // Estimate hashrate based on GPU count
        // The actual mining system will be initialized later by PoolMiningSystem
        int gpu_count = 0;
        gpuGetDeviceCount(&gpu_count);

        if (gpu_count > 0) {
            client_info.gpu_count = gpu_count;
            // Estimate based on typical GPU performance
            // Modern GPUs typically achieve 2-4 GH/s for SHA-1
            client_info.estimated_hashrate = gpu_count * 2.5e9; // 2.5 GH/s per GPU
        } else {
            // CPU fallback (though not recommended)
            client_info.gpu_count = 0;
            client_info.estimated_hashrate = 100e6; // 100 MH/s for CPU
        }

        client_info.miner_version = "SHA1-Miner/1.0.0";
        auth.client_info = client_info;

        LOG_INFO("CLIENT", "Sending AUTH message:");
        LOG_DEBUG("CLIENT", "  Method: ", (auth.method == AuthMethod::WORKER_PASS
                      ? "worker_pass"
                      : auth.method == AuthMethod::API_KEY
                      ? "api_key"
                      : "certificate"));
        LOG_DEBUG("CLIENT", "  Username: ", auth.username);
        LOG_DEBUG("CLIENT", "  Password: ", (auth.password.empty() ? "<empty>" : "<set>"));
        LOG_DEBUG("CLIENT", "  Estimated hashrate: ", client_info.estimated_hashrate / 1e9, " GH/s");
        LOG_DEBUG("CLIENT", "  GPU count: ", client_info.gpu_count);

        Message msg;
        msg.type = MessageType::AUTH;
        msg.id = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload = auth.to_json();

        LOG_TRACE("CLIENT", "AUTH payload: ", msg.payload.dump());

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
        // Add small delay to avoid triggering rate limits
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

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
        LOG_INFO("SHARE", "PoolClient::submit_share called");
        LOG_INFO("SHARE", "  Job ID: ", share.job_id);
        LOG_INFO("SHARE", "  Nonce: ", share.nonce);
        LOG_INFO("SHARE", "  Hash: ", share.hash);
        LOG_INFO("SHARE", "  Bits: ", share.matching_bits);

        if (!authenticated_.load()) {
            LOG_ERROR("SHARE", "Not authenticated, cannot submit share");
            return;
        }

        if (!connected_.load()) {
            LOG_ERROR("SHARE", "Not connected, cannot submit share");
            return;
        }

        // Validate share data
        if (share.job_id.empty()) {
            LOG_ERROR("SHARE", "Empty job ID, cannot submit share");
            return;
        }

        if (share.hash.empty()) {
            LOG_ERROR("SHARE", "Empty hash, cannot submit share");
            return;
        }

        // Validate hash format (should be 40 hex characters)
        if (share.hash.length() != 40) {
            LOG_ERROR("SHARE", "Invalid hash length: ", share.hash.length(), " (expected 40)");
            return;
        }

        // Check if hash contains only hex characters
        for (char c: share.hash) {
            if (!std::isxdigit(c)) {
                LOG_ERROR("SHARE", "Invalid hash character: '", c, "' in hash: ", share.hash);
                return;
            }
        }

        try {
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
            LOG_DEBUG("SHARE", "Creating message payload...");
            msg.payload = submit.to_json();
            LOG_INFO("SHARE", "Sending share message with ID: ", msg.id);

            // Track submission time for latency stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                worker_stats_.last_share_time = std::chrono::steady_clock::now();
            }

            send_message(msg);
            LOG_INFO("SHARE", "Share message sent successfully");
        } catch (const std::exception &e) {
            LOG_ERROR("SHARE", "Exception in submit_share: ", e.what());
        }
    }

    void PoolClient::handle_share_result(const ShareResultMessage &result) {
        update_stats(result);

        if (result.status == ShareStatus::ACCEPTED) {
            event_handler_->on_share_accepted(result);

            // Check for special messages (e.g., "Block found!")
            if (!result.message.empty()) {
                LOG_INFO("POOL", Color::BRIGHT_GREEN, "Pool message: ", result.message, Color::RESET);
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
            LOG_ERROR("POOL", "Share rejected: ", reason);
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

        LOG_INFO("POOL", "Difficulty adjusted to ", Color::BRIGHT_MAGENTA, adjust.new_difficulty,
                 Color::RESET, " (", adjust.reason, ")");

        event_handler_->on_difficulty_changed(adjust.new_difficulty);
    }

    void PoolClient::handle_pool_status(const PoolStatusMessage &status) {
        event_handler_->on_pool_status(status);
    }

    void PoolClient::handle_error(const Message &msg) {
        int error_code_int = msg.payload.value("code", 0);
        ErrorCode code = static_cast<ErrorCode>(error_code_int);
        std::string message = msg.payload.value("message", "Unknown error");
        LOG_ERROR("CLIENT", Color::RED, "Received error from server - code: ", error_code_int,
                  ", message: ", message, Color::RESET);
        // Handle specific error codes
        switch (code) {
            case ErrorCode::INVALID_MESSAGE:
            case ErrorCode::PROTOCOL_ERROR:
                // These are fatal errors, disconnect
                LOG_ERROR("CLIENT", "Fatal protocol error, disconnecting...");
                connected_ = false;
                authenticated_ = false;
                // Close the websocket
                if (ws_ && ws_->is_open()) {
                    beast::error_code ec;
                    ws_->close(websocket::close_code::protocol_error, ec);
                }
                break;
            case ErrorCode::AUTH_FAILED:
                authenticated_ = false;
                break;
            case ErrorCode::RATE_LIMITED:
                // Back off for a bit
                std::this_thread::sleep_for(std::chrono::seconds(5));
                break;
            default:
                break;
        }

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
                LOG_ERROR("CLIENT", "Request timeout: message_id=", it->first);
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

        LOG_INFO("CLIENT", "Attempting to reconnect...");

        // Reset connection state
        connected_ = false;
        authenticated_ = false;

        // Close existing connections
        try {
            if (config_.use_tls && wss_) {
                if (wss_->is_open()) {
                    wss_->close(websocket::close_code::normal);
                }
                wss_.reset();
            } else if (ws_) {
                if (ws_->is_open()) {
                    ws_->close(websocket::close_code::normal);
                }
                ws_.reset();
            }
        } catch (...) {
            // Ignore errors
        }

        // Reset work guard and restart io_context
        work_guard_.reset();
        ioc_.restart();
        work_guard_ = std::make_unique<net::executor_work_guard<net::io_context::executor_type> >(
            ioc_.get_executor()
        );

        // Try to reconnect
        try {
            auto parsed = parse_url(config_.url);

            // Resolve host
            tcp::resolver resolver{ioc_};
            auto const results = resolver.resolve(parsed.host, parsed.port);

            if (config_.use_tls) {
                // Create new SSL stream
                wss_ = std::make_unique<websocket::stream<ssl::stream<tcp::socket> > >(ioc_, ssl_ctx_);
                beast::get_lowest_layer(*wss_).connect(results.begin()->endpoint());
                wss_->next_layer().handshake(ssl::stream_base::client);
                wss_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
                wss_->handshake(parsed.host, parsed.path);
            } else {
                // Create new plain WebSocket stream
                ws_ = std::make_unique<websocket::stream<tcp::socket> >(ioc_);
                beast::get_lowest_layer(*ws_).connect(results.begin()->endpoint());
                ws_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
                ws_->handshake(parsed.host + ":" + parsed.port, parsed.path);
            }

            connected_ = true;
            worker_stats_.connected_since = std::chrono::steady_clock::now();
            event_handler_->on_connected();

            // Send hello message
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            HelloMessage hello;
            hello.protocol_version = PROTOCOL_VERSION;
            hello.client_version = "SHA1-Miner/1.0";
            hello.capabilities = {"gpu", "multi-gpu", "vardiff"};
            hello.user_agent = "SHA1-Miner";

            Message msg;
            msg.type = MessageType::HELLO;
            msg.id = Utils::generate_message_id();
            msg.timestamp = Utils::current_timestamp_ms();
            msg.payload = hello.to_json();

            send_message(msg);

            // Start read loop
            do_read();
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Reconnection failed: ", e.what());

            // Schedule another reconnection attempt
            if (config_.reconnect_attempts != 0) {
                net::post(ioc_, [this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_delay_ms));
                    reconnect();
                });
            }
        }
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
            LOG_ERROR("POOL_MGR", "Pool with name '", name, "' already exists");
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
                LOG_INFO("POOL_MGR", "Connecting to pool: ", name);
                client->connect();
            }
        }
    }

    void PoolClientManager::disconnect_all() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &[name, client]: clients_) {
            if (client->is_connected()) {
                LOG_INFO("POOL_MGR", "Disconnecting from pool: ", name);
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
                LOG_INFO("POOL_MGR", "Failing over to pool: ", pool_name);
                set_primary_pool(pool_name);
                return;
            }
        }

        // If no pools in failover list are connected, try any connected pool
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[name, client]: clients_) {
            if (client->is_connected() && name != primary_pool_name_) {
                LOG_INFO("POOL_MGR", "Failing over to pool: ", name);
                primary_pool_name_ = name;
                return;
            }
        }
    }
} // namespace MiningPool
