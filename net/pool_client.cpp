#include "pool_client.hpp"
#include <iostream>
#include <regex>

// Make sure OpenSSL SHA1 doesn't conflict with our SHA1
#ifdef SHA1
#undef SHA1
#endif

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
            std::cout << "Parsed URL - Host: " << result.host << ", Port: " << result.port
                    << ", Path: " << result.path
                    << ", Secure: " << result.is_secure << std::endl;
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

            // Start auxiliary threads first
            message_processor_thread_ = std::thread(&PoolClient::message_processor_loop, this);
            keepalive_thread_ = std::thread(&PoolClient::keepalive_loop, this);

            // Start IO thread
            io_thread_ = std::make_unique<std::thread>(&PoolClient::io_loop, this);

            // Add DNS resolution debugging
            std::cout << "Resolving host: " << parsed.host << ":" << parsed.port << std::endl;

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
                std::cout << "Resolved to: " << endpoint.endpoint().address().to_string()
                        << ":" << endpoint.endpoint().port() << std::endl;
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
                // Use the original host from URL, not the resolved IP
                ws_->handshake(parsed.host + ":" + parsed.port, parsed.path);
            }

            connected_ = true;
            worker_stats_.connected_since = std::chrono::steady_clock::now();
            event_handler_->on_connected();

            // Send hello message after a small delay to avoid rate limiting
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

            return true;
        } catch (const std::exception &e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
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
                    // Use the numeric value directly to avoid Windows macro conflicts
                    wss_->close(static_cast<websocket::close_code>(1000)); // 1000 is normal closure
                }
            } else if (ws_) {
                if (ws_->is_open()) {
                    // Use the numeric value directly to avoid Windows macro conflicts
                    ws_->close(static_cast<websocket::close_code>(1000)); // 1000 is normal closure
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
        try {
            ioc_.run();
        } catch (const std::exception &e) {
            std::cerr << "IO loop error: " << e.what() << std::endl;
        }
    }

    void PoolClient::send_message(const Message &msg) {
        if (!connected_.load()) {
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

        // Queue message for sending
        {
            std::lock_guard<std::mutex> lock(outgoing_mutex_);
            outgoing_queue_.push(msg);
            outgoing_cv_.notify_one();
        }

        // Trigger write if not already writing
        net::post(ioc_, [this]() { do_write(); });
    }

    void PoolClient::do_read() {
        auto self = shared_from_this();

        auto read_handler = [this, self](beast::error_code ec, std::size_t bytes_transferred) {
            on_read(ec, bytes_transferred);
        };

        if (config_.use_tls && wss_) {
            wss_->async_read(buffer_, read_handler);
        } else if (ws_) {
            ws_->async_read(buffer_, read_handler);
        }
    }

    void PoolClient::do_write() {
        if (!connected_.load()) {
            return;
        }

        std::unique_lock<std::mutex> lock(outgoing_mutex_);
        if (outgoing_queue_.empty()) {
            return;
        }

        Message msg = outgoing_queue_.front();
        outgoing_queue_.pop();
        lock.unlock();

        std::string payload = msg.serialize();
        auto self = shared_from_this();

        auto write_handler = [this, self](beast::error_code ec, std::size_t bytes_transferred) {
            on_write(ec, bytes_transferred);
        };

        if (config_.use_tls && wss_) {
            wss_->async_write(net::buffer(payload), write_handler);
        } else if (ws_) {
            ws_->async_write(net::buffer(payload), write_handler);
        }
    }

    void PoolClient::on_read(beast::error_code ec, std::size_t bytes_transferred) {
        if (ec) {
            if (ec != websocket::error::closed) {
                std::cerr << "Read error: " << ec.message() << std::endl;
            }
            connected_ = false;
            authenticated_ = false;
            event_handler_->on_disconnected(ec.message());

            // Handle reconnection if configured
            if (running_.load() && config_.reconnect_attempts != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_delay_ms));
                reconnect();
            }
            return;
        }

        // Parse the message
        std::string data = beast::buffers_to_string(buffer_.data());
        buffer_.consume(bytes_transferred);

        auto parsed = Message::deserialize(data);
        if (parsed) {
            std::lock_guard<std::mutex> lock(incoming_mutex_);
            incoming_queue_.push(*parsed);
            incoming_cv_.notify_one();
        }

        // Continue reading
        do_read();
    }

    void PoolClient::on_write(beast::error_code ec, std::size_t bytes_transferred) {
        if (ec) {
            std::cerr << "Write error: " << ec.message() << std::endl;
            return;
        }

        // Check if more messages to send
        std::lock_guard<std::mutex> lock(outgoing_mutex_);
        if (!outgoing_queue_.empty()) {
            net::post(ioc_, [this]() { do_write(); });
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
                case MessageType::ERROR_PROBLEM:
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
        // Add small delay to avoid triggering rate limits
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
        int error_code_int = msg.payload.value("code", 0);
        ErrorCode code = static_cast<ErrorCode>(error_code_int);
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

        // The disconnect will trigger reconnection through on_close handler
        disconnect();
        // Try to reconnect after a delay
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_delay_ms));
        connect();
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
