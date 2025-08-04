#include "platform_utils.hpp"

#include <csignal>

#include <locale>
#include "../../logging/logger.hpp"
#include "sha1_miner.cuh"

#include "globals.hpp"

#ifdef _WIN32
#include <windows.h>
#define SIGBREAK 21
#else
#include <unistd.h>
#endif

#ifdef _WIN32
void setup_console_encoding() {
    // Set console code page to UTF-8
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);

    // Enable UTF-8 for C++ streams
    std::locale::global(std::locale(""));
}
#else
void setup_console_encoding() {
    // Unix systems usually handle UTF-8 properly by default
    std::locale::global(std::locale(""));
}
#endif

void signal_handler(int sig) {
    const char *sig_name;
    switch (sig) {
        case SIGINT:
            sig_name = "SIGINT";
            break;
        case SIGTERM:
            sig_name = "SIGTERM";
            break;
#ifdef _WIN32
        case SIGBREAK:
            sig_name = "SIGBREAK";
            break;
        default:
            sig_name = "UNKNOWN";
            break;
#else
        case SIGHUP:
            sig_name = "SIGHUP";
            break;
        case SIGQUIT:
            sig_name = "SIGQUIT";
            break;
#endif
    }
    LOG_INFO("MAIN", Color::YELLOW, "Received signal ", sig_name, " (", sig, "), shutting down...", Color::RESET);
    g_shutdown.store(true);
}

void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef _WIN32
    std::signal(SIGBREAK, signal_handler);
#else
    std::signal(SIGHUP, signal_handler);
    std::signal(SIGQUIT, signal_handler);
    std::signal(SIGPIPE, SIG_IGN);
#endif
}
