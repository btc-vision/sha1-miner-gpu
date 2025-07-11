#pragma once

// Include pool_protocol first (before any OpenSSL headers)
#include "pool_protocol.hpp"

// Standard library headers
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <memory>

// Windows-specific defines
#ifdef _WIN32
#ifndef _WIN32_WINNT
        #define _WIN32_WINNT 0x0A00  // Windows 10
#endif
#ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
#endif
#endif

// Boost.Beast includes - Works with Boost 1.88
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/strand.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

namespace MiningPool {
    class PoolClient : public std::enable_shared_from_this<PoolClient> {
    public:
        PoolClient(const PoolConfig &config, IPoolEventHandler *handler);

        ~PoolClient();

        // Connection management
        bool connect();

        void disconnect();

        bool is_connected() const { return connected_.load(); }

        // Authentication
        bool authenticate();

        // Job management
        void request_job();

        std::optional<PoolJob> get_current_job() const;

        std::vector<PoolJob> get_active_jobs() const;

        // Share submission
        void submit_share(const Share &share);

        // Statistics
        void report_hashrate(const HashrateReportMessage &report);

        WorkerStats get_stats() const;

        // Message handling
        void send_message(const Message &msg);

        bool is_authenticated() const;

    private:
        std::atomic<int> reconnect_attempt_count_{0};
        std::atomic<bool> reconnecting_{false};

        std::atomic<bool> write_in_progress_{false};
        std::string current_write_payload_;

        // Configuration
        PoolConfig config_;
        IPoolEventHandler *event_handler_;

        // Connection state
        std::atomic<bool> connected_{false};
        std::atomic<bool> authenticated_{false};
        std::atomic<bool> running_{false};
        std::string session_id_;
        std::string worker_id_;

        // Boost.Beast WebSocket components
        net::io_context ioc_;

        // Plain WebSocket
        std::unique_ptr<websocket::stream<tcp::socket> > ws_;

        // SSL WebSocket - using ssl::stream from boost::asio::ssl
        std::unique_ptr<websocket::stream<ssl::stream<tcp::socket> > > wss_;
        ssl::context ssl_ctx_{ssl::context::tlsv12_client};

        std::unique_ptr<std::thread> io_thread_;
        std::unique_ptr<net::executor_work_guard<net::io_context::executor_type> > work_guard_;

        // Threading
        std::thread keepalive_thread_;
        std::thread message_processor_thread_;

        // Message queues
        std::queue<Message> outgoing_queue_;
        std::queue<Message> incoming_queue_;
        mutable std::mutex outgoing_mutex_;
        mutable std::mutex incoming_mutex_;
        std::condition_variable outgoing_cv_;
        std::condition_variable incoming_cv_;

        // Job management
        mutable std::mutex jobs_mutex_;
        std::unordered_map<std::string, PoolJob> active_jobs_;
        std::string current_job_id_;

        // Statistics
        mutable std::mutex stats_mutex_;
        WorkerStats worker_stats_;

        // Pending requests
        std::mutex pending_mutex_;

        struct PendingRequest {
            std::chrono::steady_clock::time_point timestamp;
            MessageType type;
        };
        std::map<uint64_t, PendingRequest> pending_requests_;

        // Internal methods
        void io_loop();

        void keepalive_loop();

        void message_processor_loop();

        void on_resolve(beast::error_code ec, tcp::resolver::results_type results);

        void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type ep);

        void on_ssl_handshake(beast::error_code ec);

        void on_handshake(beast::error_code ec);

        void on_write(beast::error_code ec, std::size_t bytes_transferred);

        void on_read(beast::error_code ec, std::size_t bytes_transferred);

        void do_read();

        void do_write();

        // Message processing
        void process_message(const Message &msg);

        void handle_welcome(const WelcomeMessage &welcome);

        void handle_auth_response(const AuthResponseMessage &response);

        void handle_new_job(const JobMessage &job);

        void handle_share_result(const ShareResultMessage &result);

        void handle_difficulty_adjust(const DifficultyAdjustMessage &adjust);

        void handle_pool_status(const PoolStatusMessage &status);

        void handle_error(const Message &msg);

        // Utility methods
        void cleanup_expired_jobs();

        void check_pending_timeouts();

        void reconnect();

        void update_stats(const ShareResultMessage &result);

        // Parse URL components
        struct ParsedUrl {
            std::string host;
            std::string port;
            std::string path;
            bool is_secure;
        };

        static ParsedUrl parse_url(const std::string &url);

        // Buffer for reading messages
        beast::flat_buffer buffer_;
    };

    // Thread-safe pool client wrapper (same as before)
    class PoolClientManager {
    public:
        PoolClientManager();

        ~PoolClientManager();

        bool add_pool(const std::string &name, const PoolConfig &config,
                      IPoolEventHandler *handler);

        bool remove_pool(const std::string &name);

        bool set_primary_pool(const std::string &name);

        std::string get_primary_pool() const;

        void enable_failover(bool enable);

        void set_failover_order(const std::vector<std::string> &pool_names);

        std::shared_ptr<PoolClient> get_client(const std::string &name) const;

        std::shared_ptr<PoolClient> get_primary_client() const;

        void connect_all();

        void disconnect_all();

        std::map<std::string, WorkerStats> get_all_stats() const;

    private:
        mutable std::mutex mutex_;
        std::map<std::string, std::shared_ptr<PoolClient> > clients_;
        std::string primary_pool_name_;
        bool failover_enabled_ = false;
        std::vector<std::string> failover_order_;

        void handle_failover();
    };
} // namespace MiningPool
