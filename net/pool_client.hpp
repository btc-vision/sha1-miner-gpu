#pragma once

#include "pool_protocol.hpp"
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_tls_client.hpp>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

namespace MiningPool {

// WebSocket client types
using WsClient = websocketpp::client<websocketpp::config::asio_client>;
using WssTlsClient = websocketpp::client<websocketpp::config::asio_tls_client>;
using WsMessage = websocketpp::config::asio_client::message_type::ptr;
using WssMessage = websocketpp::config::asio_tls_client::message_type::ptr;
using SslContext = websocketpp::lib::asio::ssl::context;

class PoolClient {
public:
    PoolClient(const PoolConfig& config, IPoolEventHandler* handler);
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
    void submit_share(const Share& share);

    // Statistics
    void report_hashrate(const HashrateReportMessage& report);
    WorkerStats get_stats() const;

    // Message handling
    void send_message(const Message& msg);

private:
    // Configuration
    PoolConfig config_;
    IPoolEventHandler* event_handler_;

    // Connection state
    std::atomic<bool> connected_{false};
    std::atomic<bool> authenticated_{false};
    std::atomic<bool> running_{false};
    std::string session_id_;
    std::string worker_id_;

    // WebSocket clients
    std::unique_ptr<WsClient> ws_client_;
    std::unique_ptr<WssTlsClient> wss_client_;
    websocketpp::connection_hdl connection_hdl_;

    // Threading
    std::thread io_thread_;
    std::thread keepalive_thread_;
    std::thread message_processor_thread_;

    // Message queues
    std::queue<Message> outgoing_queue_;
    std::queue<Message> incoming_queue_;
    std::mutex outgoing_mutex_;
    std::mutex incoming_mutex_;
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
    void setup_ws_handlers();
    void setup_wss_handlers();
    void io_loop();
    void keepalive_loop();
    void message_processor_loop();

    // WebSocket handlers
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, WsMessage msg);
    void on_message_tls(websocketpp::connection_hdl hdl, WssMessage msg);
    void on_fail(websocketpp::connection_hdl hdl);

    // TLS configuration
    std::shared_ptr<SslContext> on_tls_init(websocketpp::connection_hdl hdl);

    // Message processing
    void process_message(const Message& msg);
    void handle_welcome(const WelcomeMessage& welcome);
    void handle_auth_response(const AuthResponseMessage& response);
    void handle_new_job(const JobMessage& job);
    void handle_share_result(const ShareResultMessage& result);
    void handle_difficulty_adjust(const DifficultyAdjustMessage& adjust);
    void handle_pool_status(const PoolStatusMessage& status);
    void handle_error(const Message& msg);

    // Utility methods
    void cleanup_expired_jobs();
    void check_pending_timeouts();
    void reconnect();
    void update_stats(const ShareResultMessage& result);
};

// Thread-safe pool client wrapper
class PoolClientManager {
public:
    PoolClientManager();
    ~PoolClientManager();

    // Client management
    bool add_pool(const std::string& name, const PoolConfig& config,
                  IPoolEventHandler* handler);
    bool remove_pool(const std::string& name);

    // Pool switching
    bool set_primary_pool(const std::string& name);
    std::string get_primary_pool() const;

    // Failover support
    void enable_failover(bool enable);
    void set_failover_order(const std::vector<std::string>& pool_names);

    // Get client
    std::shared_ptr<PoolClient> get_client(const std::string& name) const;
    std::shared_ptr<PoolClient> get_primary_client() const;

    // Batch operations
    void connect_all();
    void disconnect_all();

    // Statistics
    std::map<std::string, WorkerStats> get_all_stats() const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<PoolClient>> clients_;
    std::string primary_pool_name_;
    bool failover_enabled_ = false;
    std::vector<std::string> failover_order_;

    void handle_failover();
};

// Stratum compatibility layer
class StratumAdapter : public IPoolEventHandler {
public:
    StratumAdapter();

    // Convert Stratum messages to our protocol
    Message stratum_to_protocol(const json& stratum_msg);
    json protocol_to_stratum(const Message& msg);

    // IPoolEventHandler implementation
    void on_connected() override;
    void on_disconnected(const std::string& reason) override;
    void on_error(ErrorCode code, const std::string& message) override;
    void on_authenticated(const std::string& worker_id) override;
    void on_auth_failed(ErrorCode code, const std::string& reason) override;
    void on_new_job(const PoolJob& job) override;
    void on_job_cancelled(const std::string& job_id) override;
    void on_share_accepted(const ShareResultMessage& result) override;
    void on_share_rejected(const ShareResultMessage& result) override;
    void on_difficulty_changed(uint32_t new_difficulty) override;
    void on_pool_status(const PoolStatusMessage& status) override;

private:
    // Stratum-specific state
    uint64_t stratum_id_counter_ = 1;
    std::unordered_map<uint64_t, uint64_t> id_mapping_;  // stratum_id -> protocol_id
};

} // namespace MiningPool