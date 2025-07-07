// pool_client.hpp - High-performance pool client using uWebSockets
#pragma once

#include "pool_protocol.hpp"
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <memory>

// Forward declarations for uWebSockets
namespace uWS {
    template<bool SSL, bool isServer, typename USERDATA>
    struct WebSocket;
    struct Loop;
}

namespace MiningPool {
    class PoolClient {
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

    private:
        // Configuration
        PoolConfig config_;
        IPoolEventHandler *event_handler_;

        // Connection state
        std::atomic<bool> connected_{false};
        std::atomic<bool> authenticated_{false};
        std::atomic<bool> running_{false};
        std::string session_id_;
        std::string worker_id_;

        // uWebSockets components
        struct WebSocketData {
            PoolClient *client;
        };

        std::unique_ptr<std::thread> io_thread_;
        uWS::Loop *loop_ = nullptr;
        void *ws_ = nullptr; // Changed to void* to avoid template issues

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
        std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> pending_requests_;

        // Internal methods
        void io_loop();

        void keepalive_loop();

        void message_processor_loop();

        void setup_websocket_handlers();

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
            int port;
            std::string path;
            bool is_secure;
        };

        ParsedUrl parse_url(const std::string &url);
    };

    // Thread-safe pool client wrapper
    class PoolClientManager {
    public:
        PoolClientManager();

        ~PoolClientManager();

        // Client management
        bool add_pool(const std::string &name, const PoolConfig &config,
                      IPoolEventHandler *handler);

        bool remove_pool(const std::string &name);

        // Pool switching
        bool set_primary_pool(const std::string &name);

        std::string get_primary_pool() const;

        // Failover support
        void enable_failover(bool enable);

        void set_failover_order(const std::vector<std::string> &pool_names);

        // Get client
        std::shared_ptr<PoolClient> get_client(const std::string &name) const;

        std::shared_ptr<PoolClient> get_primary_client() const;

        // Batch operations
        void connect_all();

        void disconnect_all();

        // Statistics
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
