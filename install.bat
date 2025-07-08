@echo off
REM SHA-1 OP_NET Miner - Windows Dependencies Installer
REM This script installs dependencies with compatible versions

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

REM Clean up any existing vcpkg.json that might interfere
if exist vcpkg.json (
    echo %INFO% Removing existing vcpkg.json to avoid conflicts...
    del vcpkg.json
)

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

REM Use classic mode to install specific versions
echo %INFO% Installing C++ dependencies with compatible versions...
echo %INFO% This will take 10-30 minutes on first run...
echo.

REM Install dependencies one by one
echo %INFO% [1/4] Installing OpenSSL...
vcpkg\vcpkg install openssl:x64-windows
if errorlevel 1 (
    echo %WARNING% OpenSSL installation had issues, but continuing...
)

echo.
echo %INFO% [2/4] Installing Boost (this will take a while)...
echo %INFO% Installing Boost 1.88 with Beast support...

REM Install Boost libraries including Beast
vcpkg\vcpkg install boost:x64-windows

if errorlevel 1 (
    echo %WARNING% Boost installation had issues, trying individual components...
    vcpkg\vcpkg install boost-system:x64-windows
    vcpkg\vcpkg install boost-thread:x64-windows
    vcpkg\vcpkg install boost-program-options:x64-windows
    vcpkg\vcpkg install boost-date-time:x64-windows
    vcpkg\vcpkg install boost-regex:x64-windows
    vcpkg\vcpkg install boost-random:x64-windows
    vcpkg\vcpkg install boost-asio:x64-windows
    vcpkg\vcpkg install boost-beast:x64-windows
    vcpkg\vcpkg install boost-chrono:x64-windows
    vcpkg\vcpkg install boost-atomic:x64-windows
)

echo.
echo %INFO% [3/4] Installing nlohmann-json...
vcpkg\vcpkg install nlohmann-json:x64-windows
if errorlevel 1 (
    echo %WARNING% JSON installation had issues, but continuing...
)

echo.
echo %INFO% [4/4] Installing zlib...
vcpkg\vcpkg install zlib:x64-windows
if errorlevel 1 (
    echo %WARNING% zlib installation had issues, but continuing...
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

REM Check what Boost version we got
echo.
echo %INFO% Checking installed Boost version...
vcpkg\vcpkg list boost

echo.
echo =====================================
echo Dependencies Installation Complete!
echo =====================================
echo.
echo Installed packages:
echo   - OpenSSL (SSL/TLS support)
echo   - Boost 1.88 libraries with Beast:
echo     * boost-system
echo     * boost-thread
echo     * boost-program-options
echo     * boost-asio
echo     * boost-beast