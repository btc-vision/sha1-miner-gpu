@echo off
echo =====================================
echo Patching WebSocketpp for Boost 1.88
echo =====================================
echo.
echo This will patch websocketpp to work with Boost 1.88
echo.

cd external\websocketpp

echo Creating compatibility patches...

REM Create a new file: websocketpp/transport/asio/base.hpp
mkdir websocketpp\transport\asio 2>nul
(
echo #ifndef WEBSOCKETPP_TRANSPORT_ASIO_BASE_HPP
echo #define WEBSOCKETPP_TRANSPORT_ASIO_BASE_HPP
echo.
echo #include ^<boost/version.hpp^>
echo.
echo #if BOOST_VERSION ^>= 107000
echo namespace boost { namespace asio {
echo     using io_service = io_context;
echo     template^<typename E^> using work = executor_work_guard^<E^>;
echo }}
echo #endif
echo.
echo #endif // WEBSOCKETPP_TRANSPORT_ASIO_BASE_HPP
) > websocketpp\transport\asio\base.hpp

echo.
echo Patching connection.hpp...
REM Add include to connection.hpp at the beginning
powershell -Command "$content = Get-Content 'websocketpp\transport\asio\connection.hpp'; $newContent = @('#include <websocketpp/transport/asio/base.hpp>') + $content; $newContent | Set-Content 'websocketpp\transport\asio\connection.hpp'"

echo Patching endpoint.hpp...
REM Add include to endpoint.hpp at the beginning
powershell -Command "$content = Get-Content 'websocketpp\transport\asio\endpoint.hpp'; $newContent = @('#include <websocketpp/transport/asio/base.hpp>') + $content; $newContent | Set-Content 'websocketpp\transport\asio\endpoint.hpp'"

echo.
echo Creating timer compatibility fix...
REM Create timer_fix.hpp
(
echo #ifndef WEBSOCKETPP_TIMER_FIX_HPP
echo #define WEBSOCKETPP_TIMER_FIX_HPP
echo.
echo #include ^<boost/version.hpp^>
echo #include ^<boost/asio/steady_timer.hpp^>
echo.
echo #if BOOST_VERSION ^>= 107000
echo namespace boost { namespace asio {
echo template^<typename Clock, typename WaitTraits, typename Executor^>
echo std::size_t expires_from_now(basic_waitable_timer^<Clock, WaitTraits, Executor^>^& timer,
echo                              const typename Clock::duration^& duration) {
echo     return timer.expires_after(duration);
echo }
echo.
echo template^<typename Clock, typename WaitTraits, typename Executor^>
echo typename Clock::duration expires_from_now(const basic_waitable_timer^<Clock, WaitTraits, Executor^>^& timer) {
echo     return timer.expiry() - Clock::now();
echo }
echo }}
echo #endif
echo.
echo #endif
) > websocketpp\transport\asio\timer_fix.hpp

echo.
echo Adding timer fix to connection.hpp...
REM Add timer fix include after base.hpp
powershell -Command "(Get-Content 'websocketpp\transport\asio\connection.hpp') -replace '#include <websocketpp/transport/asio/base.hpp>', '#include <websocketpp/transport/asio/base.hpp>`n#include <websocketpp/transport/asio/timer_fix.hpp>' | Set-Content 'websocketpp\transport\asio\connection.hpp'"

echo.
echo Fixing resolver iterator...
REM Create a fix for resolver iterator issue
powershell -Command "(Get-Content 'websocketpp\transport\asio\endpoint.hpp') -replace 'resolver::iterator', 'resolver::results_type' | Set-Content 'websocketpp\transport\asio\endpoint.hpp'"

echo.
echo Fixing work class usage...
powershell -Command "(Get-Content 'websocketpp\transport\asio\endpoint.hpp') -replace 'lib::shared_ptr<lib::asio::io_service::work>', 'lib::shared_ptr<lib::asio::executor_work_guard<lib::asio::io_context::executor_type>>' | Set-Content 'websocketpp\transport\asio\endpoint.hpp'"

cd ..\..

echo.
echo =====================================
echo Patching Complete!
echo =====================================
echo.
echo WebSocketpp has been patched for Boost 1.88 compatibility.
echo Now try building your project again.
echo.
pause