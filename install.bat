@echo off
REM SHA-1 OP_NET Miner - Windows Dependencies Installer
REM This script only installs dependencies (vcpkg packages and uWebSockets)

setlocal enabledelayedexpansion

REM Set colors
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

REM Default installation directory
if "%~1"=="" (
    set "INSTALL_DIR=%CD%"
) else (
    set "INSTALL_DIR=%~1"
)

REM Header
cls
echo =====================================
echo SHA-1 Miner - Dependencies Installer
echo =====================================
echo.
echo Working directory: %INSTALL_DIR%
echo.

REM Check for NVIDIA GPU
echo %INFO% Checking for NVIDIA GPU...
wmic path win32_VideoController get name 2>nul | find /i "NVIDIA" >nul
if errorlevel 1 (
    echo %WARNING% No NVIDIA GPU detected.
    echo          AMD GPUs are not yet supported on Windows.
    echo          Continuing with dependency installation...
    echo.
)
echo.

REM Check for CUDA
echo %INFO% Checking for CUDA installation...
if "%CUDA_PATH%"=="" (
    echo %WARNING% CUDA not found. You'll need CUDA Toolkit for building:
    echo           https://developer.nvidia.com/cuda-downloads
    echo.
) else (
    echo %SUCCESS% CUDA found at: %CUDA_PATH%
)
echo.

REM Check for Git
echo %INFO% Checking for Git...
where git >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Git not found. Please install Git from:
    echo         https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)
echo %SUCCESS% Git found
echo.

REM Check for CMake
echo %INFO% Checking for CMake...
where cmake >nul 2>&1
if errorlevel 1 (
    echo %WARNING% CMake not found. You'll need CMake for building:
    echo           https://cmake.org/download/
    echo.
)
echo.

REM Check for Visual Studio
echo %INFO% Checking for Visual Studio...
set "VS_FOUND=0"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"

if "%VS_FOUND%"=="0" (
    echo %WARNING% Visual Studio 2022 not found. You'll need it for building:
    echo           https://visualstudio.microsoft.com/downloads/
    echo.
)
echo.

cd /d "%INSTALL_DIR%"

REM Setup vcpkg
echo %INFO% Setting up vcpkg for C++ dependencies...
if exist "vcpkg\vcpkg.exe" (
    echo %INFO% vcpkg already exists, updating...
    cd vcpkg
    git pull
    call bootstrap-vcpkg.bat
    cd ..
) else (
    echo %INFO% Cloning vcpkg...
    git clone https://github.com/Microsoft/vcpkg.git
    if errorlevel 1 (
        echo %ERROR% Failed to clone vcpkg
        pause
        exit /b 1
    )
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..
)
echo.

REM Install dependencies
echo %INFO% Installing C++ dependencies (this will take 10-30 minutes)...
echo.

REM Install OpenSSL
echo %INFO% [1/5] Installing OpenSSL...
vcpkg\vcpkg install openssl:x64-windows
if errorlevel 1 (
    echo %WARNING% OpenSSL installation had issues, but continuing...
)

echo.
echo %INFO% [2/5] Installing Boost libraries...
vcpkg\vcpkg install boost-system:x64-windows boost-thread:x64-windows boost-program-options:x64-windows
if errorlevel 1 (
    echo %WARNING% Boost installation had issues, but continuing...
)

echo.
echo %INFO% [3/5] Installing nlohmann-json...
vcpkg\vcpkg install nlohmann-json:x64-windows
if errorlevel 1 (
    echo %WARNING% JSON installation had issues, but continuing...
)

echo.
echo %INFO% [4/5] Installing zlib...
vcpkg\vcpkg install zlib:x64-windows
if errorlevel 1 (
    echo %WARNING% zlib installation had issues, but continuing...
)

echo.
echo %INFO% [5/5] Installing uWebSockets...
REM Try vcpkg first
vcpkg\vcpkg install uwebsockets:x64-windows
if errorlevel 1 (
    echo %INFO% Installing uWebSockets manually...
    if not exist "external" mkdir external
    cd external

    REM Clone uSockets (dependency)
    if exist "uSockets" (
        echo %INFO% uSockets already exists, updating...
        cd uSockets
        git pull origin master
        cd ..
    ) else (
        echo %INFO% Cloning uSockets...
        git clone https://github.com/uNetworking/uSockets.git
    )

    REM Clone uWebSockets
    if exist "uWebSockets" (
        echo %INFO% uWebSockets already exists, updating...
        cd uWebSockets
        git pull origin master
        cd ..
    ) else (
        echo %INFO% Cloning uWebSockets...
        git clone --recursive https://github.com/uNetworking/uWebSockets.git
    )

    cd ..
    echo %SUCCESS% uWebSockets installed manually in external/
) else (
    echo %SUCCESS% uWebSockets installed via vcpkg
)

REM Integrate vcpkg
echo.
echo %INFO% Integrating vcpkg with system...
vcpkg\vcpkg integrate install

REM Set environment variable
echo.
echo %INFO% Setting OPENSSL_ROOT_DIR environment variable...
set "OPENSSL_ROOT_DIR=%INSTALL_DIR%\vcpkg\installed\x64-windows"
setx OPENSSL_ROOT_DIR "%INSTALL_DIR%\vcpkg\installed\x64-windows" >nul 2>&1

echo.
echo =====================================
echo Dependencies Installation Complete!
echo =====================================
echo.
echo Installed packages:
echo   - OpenSSL (SSL/TLS support)
echo   - Boost (system, thread, program-options)
echo   - nlohmann-json (JSON parsing)
echo   - zlib (compression)
echo   - uWebSockets (WebSocket client)
echo.
echo vcpkg location: %INSTALL_DIR%\vcpkg
echo.
echo To use these dependencies in your project:
echo   1. Set CMAKE_TOOLCHAIN_FILE to:
echo      %INSTALL_DIR%\vcpkg\scripts\buildsystems\vcpkg.cmake
echo.
echo   2. If using CMakePresets.json, it's already configured
echo.
echo   3. For manual builds:
echo      cmake -DCMAKE_TOOLCHAIN_FILE="%INSTALL_DIR%\vcpkg\scripts\buildsystems\vcpkg.cmake" ..
echo.
pause