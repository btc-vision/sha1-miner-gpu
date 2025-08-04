// logger.hpp - Enhanced logging system with colors and debug levels
#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace MiningPool {
    // Debug levels
    enum class LogLevel { ERR = 0, WARNING = 1, INFO = 2, DEBUG = 3, TRACE = 4 };

    // ANSI color codes
    namespace Color {
        const std::string RESET = "\033[0m";
        const std::string BLACK = "\033[30m";
        const std::string RED = "\033[31m";
        const std::string GREEN = "\033[32m";
        const std::string YELLOW = "\033[33m";
        const std::string BLUE = "\033[34m";
        const std::string MAGENTA = "\033[35m";
        const std::string CYAN = "\033[36m";
        const std::string WHITE = "\033[37m";
        const std::string BRIGHT_BLACK = "\033[90m";
        const std::string BRIGHT_RED = "\033[91m";
        const std::string BRIGHT_GREEN = "\033[92m";
        const std::string BRIGHT_YELLOW = "\033[93m";
        const std::string BRIGHT_BLUE = "\033[94m";
        const std::string BRIGHT_MAGENTA = "\033[95m";
        const std::string BRIGHT_CYAN = "\033[96m";
        const std::string BRIGHT_WHITE = "\033[97m";

        // Background colors
        const std::string BG_RED = "\033[41m";
        const std::string BG_GREEN = "\033[42m";
        const std::string BG_YELLOW = "\033[43m";
        const std::string BG_BLUE = "\033[44m";
        const std::string BG_MAGENTA = "\033[45m";
        const std::string BG_CYAN = "\033[46m";

        // Text styles
        const std::string BOLD = "\033[1m";
        const std::string DIM = "\033[2m";
        const std::string ITALIC = "\033[3m";
        const std::string UNDERLINE = "\033[4m";
    } // namespace Color

    class Logger {
    private:
        static LogLevel current_level_;
        static std::mutex mutex_;
        static bool colors_enabled_;

#ifdef _WIN32
        static HANDLE console_handle_;
        static WORD original_attributes_;
        static bool windows_console_initialized_;

        static void init_windows_console() {
            if (!windows_console_initialized_) {
                console_handle_ = GetStdHandle(STD_OUTPUT_HANDLE);
                CONSOLE_SCREEN_BUFFER_INFO csbi;
                GetConsoleScreenBufferInfo(console_handle_, &csbi);
                original_attributes_ = csbi.wAttributes;

                // Enable ANSI escape sequences on Windows 10+
                DWORD mode;
                GetConsoleMode(console_handle_, &mode);
                SetConsoleMode(console_handle_, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);

                windows_console_initialized_ = true;
            }
        }
#endif

        static std::string get_timestamp() {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

            std::stringstream ss;
            ss << std::put_time(std::localtime(&time), "%H:%M:%S");
            ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
            return ss.str();
        }

        static std::string get_level_string(LogLevel level) {
            switch (level) {
                case LogLevel::ERR:
                    return "ERROR";
                case LogLevel::WARNING:
                    return "WARN ";
                case LogLevel::INFO:
                    return "INFO ";
                case LogLevel::DEBUG:
                    return "DEBUG";
                case LogLevel::TRACE:
                    return "TRACE";
                default:
                    return "?????";
            }
        }

        static std::string get_level_color(LogLevel level) {
            if (!colors_enabled_)
                return "";

            switch (level) {
                case LogLevel::ERR:
                    return Color::BRIGHT_RED;
                case LogLevel::WARNING:
                    return Color::BRIGHT_YELLOW;
                case LogLevel::INFO:
                    return Color::BRIGHT_GREEN;
                case LogLevel::DEBUG:
                    return Color::BRIGHT_CYAN;
                case LogLevel::TRACE:
                    return Color::BRIGHT_MAGENTA;
                default:
                    return Color::RESET;
            }
        }

    public:
        static void set_level(LogLevel level) { current_level_ = level; }

        static void enable_colors(bool enable) {
            colors_enabled_ = enable;
#ifdef _WIN32
            if (enable) {
                init_windows_console();
            }
#endif
        }

        static bool should_log(LogLevel level) { return level <= current_level_; }

        template<typename... Args>
        static void log(LogLevel level, const std::string &component, Args... args) {
            if (!should_log(level))
                return;

            std::lock_guard<std::mutex> lock(mutex_);

            // Timestamp
            if (colors_enabled_)
                std::cout << Color::DIM;
            std::cout << "[" << get_timestamp() << "] ";
            if (colors_enabled_)
                std::cout << Color::RESET;

            // Level with color
            std::cout << get_level_color(level) << "[" << get_level_string(level) << "] ";
            if (colors_enabled_)
                std::cout << Color::RESET;

            // Component
            if (colors_enabled_)
                std::cout << Color::BRIGHT_BLUE;
            std::cout << "[" << component << "] ";
            if (colors_enabled_)
                std::cout << Color::RESET;

            // Message
            ((std::cout << args), ...);
            std::cout << std::endl;
        }

        // Convenience functions
        template<typename... Args>
        static void error(const std::string &component, Args... args) {
            log(LogLevel::ERR, component, args...);
        }

        template<typename... Args>
        static void warn(const std::string &component, Args... args) {
            log(LogLevel::WARNING, component, args...);
        }

        template<typename... Args>
        static void info(const std::string &component, Args... args) {
            log(LogLevel::INFO, component, args...);
        }

        template<typename... Args>
        static void debug(const std::string &component, Args... args) {
            log(LogLevel::DEBUG, component, args...);
        }

        template<typename... Args>
        static void trace(const std::string &component, Args... args) {
            log(LogLevel::TRACE, component, args...);
        }
    };

    // Static member initialization
    inline LogLevel Logger::current_level_ = LogLevel::INFO;
    inline std::mutex Logger::mutex_;
    inline bool Logger::colors_enabled_ = true;

#ifdef _WIN32
    inline HANDLE Logger::console_handle_ = nullptr;
    inline WORD Logger::original_attributes_ = 0;
    inline bool Logger::windows_console_initialized_ = false;
#endif

    // Convenience macros
#define LOG_ERROR(component, ...) MiningPool::Logger::error(component, __VA_ARGS__)
#define LOG_WARN(component, ...) MiningPool::Logger::warn(component, __VA_ARGS__)
#define LOG_INFO(component, ...) MiningPool::Logger::info(component, __VA_ARGS__)
#define LOG_DEBUG(component, ...) MiningPool::Logger::debug(component, __VA_ARGS__)
#define LOG_TRACE(component, ...) MiningPool::Logger::trace(component, __VA_ARGS__)
} // namespace MiningPool

// Global namespace convenience functions for use outside MiningPool namespace
using MiningPool::Logger;
using MiningPool::LogLevel;

// Create a global Color namespace alias for easier access
namespace Color = MiningPool::Color;
