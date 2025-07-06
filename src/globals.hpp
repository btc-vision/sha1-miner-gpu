#pragma once
#include <atomic>

// Global shutdown flag - declare as extern for other files to use
extern std::atomic<bool> g_shutdown;