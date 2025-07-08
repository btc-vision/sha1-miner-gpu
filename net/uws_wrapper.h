// uws_wrapper.h - Cross-platform wrapper for uWebSockets includes
#pragma once

// Try different include paths based on installation method
#if defined(_WIN32) && __has_include(<uWebSockets/App.h>)
// Windows with vcpkg installation
#include <uWebSockets/App.h>
#elif __has_include(<App.h>)
    // Direct include (external/uWebSockets/src is in include path)
    #include <App.h>
#elif __has_include("App.h")
    // Local include
    #include "App.h"
#elif __has_include("uWebSockets/src/App.h")
    // Relative path from net directory
    #include "uWebSockets/src/App.h"
#elif __has_include("../external/uWebSockets/src/App.h")
    // Relative path from net directory to external
    #include "../external/uWebSockets/src/App.h"
#else
    #error "Cannot find uWebSockets App.h header. Please check your uWebSockets installation."
#endif
