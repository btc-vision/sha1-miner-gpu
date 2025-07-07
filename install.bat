@echo off
REM Fix vcpkg installation and install dependencies

echo Fixing vcpkg installation...
echo.

cd /d "%~dp0"

REM Check if we're in the right directory
if not exist "vcpkg" (
    echo ERROR: vcpkg directory not found!
    echo Please run this script from your sha1-miner directory
    pause
    exit /b 1
)

cd vcpkg

echo Updating vcpkg...
git pull
if errorlevel 1 (
    echo WARNING: Could not update vcpkg via git
)

echo.
echo Bootstrapping vcpkg...
call bootstrap-vcpkg.bat
if errorlevel 1 (
    echo ERROR: Failed to bootstrap vcpkg
    pause
    exit /b 1
)

echo.
echo Integrating vcpkg...
vcpkg integrate install

echo.
echo Updating ports...
vcpkg update

echo.
echo Installing packages with correct triplet...
echo This will take 10-30 minutes depending on your system...
echo.

REM Install packages one by one with explicit triplet
echo Installing boost-system...
vcpkg install boost-system:x64-windows
if errorlevel 1 goto :error

echo.
echo Installing boost-thread...
vcpkg install boost-thread:x64-windows
if errorlevel 1 goto :error

echo.
echo Installing openssl...
vcpkg install openssl:x64-windows
if errorlevel 1 goto :error

echo.
echo Installing nlohmann-json...
vcpkg install nlohmann-json:x64-windows
if errorlevel 1 goto :error

echo.
echo Installing websocketpp (header-only)...
vcpkg install websocketpp:x64-windows
if errorlevel 1 (
    echo.
    echo WARNING: websocketpp failed to install via vcpkg
    echo We'll install it manually...
    cd ..
    goto :manual_websocketpp
)

cd ..
goto :success

:manual_websocketpp
echo.
echo Installing WebSocketPP manually...
if not exist "external" mkdir external
cd external

if exist "websocketpp" (
    echo WebSocketPP already exists in external directory
) else (
    echo Cloning WebSocketPP...
    git clone https://github.com/zaphoyd/websocketpp.git
    if errorlevel 1 (
        echo ERROR: Failed to clone WebSocketPP
        echo.
        echo Please manually download from: https://github.com/zaphoyd/websocketpp
        echo And extract to: %CD%\websocketpp
        pause
        exit /b 1
    )
)
cd ..

:success
echo.
echo =====================================
echo Dependencies installed successfully!
echo =====================================
echo.
echo Next steps:
echo 1. Create a build directory: mkdir build
echo 2. Enter build directory: cd build
echo 3. Configure with CMake:
echo    cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="%CD%\vcpkg\scripts\buildsystems\vcpkg.cmake" ..
echo 4. Build:
echo    cmake --build . --config Release
echo.
pause
exit /b 0

:error
echo.
echo ERROR: Package installation failed
echo Try running vcpkg remove --outdated and then run this script again
pause
exit /b 1