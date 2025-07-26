@echo off
REM SHA-1 OP_NET Miner - Windows Dependencies Installer
REM This script installs dependencies with compatible versions
REM Now supports both NVIDIA and AMD GPUs

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

REM Detect GPU type
set "GPU_TYPE=NONE"
set "HAS_NVIDIA=0"
set "HAS_AMD=0"

REM Check for GPUs using PowerShell (faster and more reliable)
echo %INFO% Detecting installed GPUs...
for /f "tokens=*" %%i in ('powershell -Command "Get-WmiObject -Class Win32_VideoController | Select-Object -ExpandProperty Name" 2^>nul') do (
    echo %INFO% Found GPU: %%i
    echo %%i | findstr /i "NVIDIA" >nul
    if not errorlevel 1 (
        set "HAS_NVIDIA=1"
        set "GPU_TYPE=NVIDIA"
        echo %SUCCESS% NVIDIA GPU detected.
    )
    echo %%i | findstr /i "AMD" >nul
    if not errorlevel 1 (
        set "HAS_AMD=1"
        set "GPU_TYPE=AMD"
        echo %SUCCESS% AMD GPU detected.
    )
    echo %%i | findstr /i "Radeon" >nul
    if not errorlevel 1 (
        set "HAS_AMD=1"
        set "GPU_TYPE=AMD"
        echo %SUCCESS% AMD Radeon GPU detected.
    )
)

if "%GPU_TYPE%"=="NONE" (
    echo %WARNING% No supported GPU detected.
    echo          The miner requires either NVIDIA or AMD GPU.
    echo          Continuing with dependency installation...
)
echo.

REM GPU-specific checks
if "%HAS_NVIDIA%"=="1" (
    REM Check for CUDA
    echo %INFO% Checking for CUDA installation...
    if "%CUDA_PATH%"=="" (
        echo %WARNING% CUDA not found. You'll need CUDA Toolkit for NVIDIA GPUs:
        echo           https://developer.nvidia.com/cuda-downloads
        echo.
    ) else (
        echo %SUCCESS% CUDA found at: %CUDA_PATH%
    )
)

if "%HAS_AMD%"=="1" (
    REM Check for AMD ROCm/HIP SDK
    echo %INFO% Checking for AMD HIP SDK installation...
    set "HIP_FOUND=0"
    
    REM Check common HIP SDK installation paths
    if exist "%ProgramFiles%\AMD\ROCm" (
        set "HIP_PATH=%ProgramFiles%\AMD\ROCm"
        set "HIP_FOUND=1"
    )
    if exist "%ProgramFiles%\AMD\HIP SDK" (
        set "HIP_PATH=%ProgramFiles%\AMD\HIP SDK"
        set "HIP_FOUND=1"
    )
    if exist "C:\Program Files\AMD\ROCm" (
        set "HIP_PATH=C:\Program Files\AMD\ROCm"
        set "HIP_FOUND=1"
    )
    if exist "C:\hip" (
        set "HIP_PATH=C:\hip"
        set "HIP_FOUND=1"
    )
    
    if "%HIP_FOUND%"=="1" (
        echo %SUCCESS% AMD HIP SDK found at: !HIP_PATH!
        setx HIP_PATH "!HIP_PATH!" >nul 2>&1
        setx ROCM_PATH "!HIP_PATH!" >nul 2>&1
    ) else (
        echo %WARNING% AMD HIP SDK not found. For AMD GPU support, install:
        echo           AMD ROCm/HIP SDK: https://rocm.docs.amd.com/en/latest/deploy/windows/index.html
        echo           OR use Windows Subsystem for Linux (WSL2) with ROCm
        echo.
        echo %INFO% Note: Full AMD GPU support is better on Linux.
        echo        Consider using WSL2 with Ubuntu and ROCm for best performance.
    )
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
) else (
    echo %SUCCESS% CMake found
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
) else (
    echo %SUCCESS% Visual Studio 2022 found
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
echo     * boost-chrono
echo     * boost-atomic
echo     * boost-date-time
echo     * boost-regex
echo     * boost-random
echo   - nlohmann-json (JSON parsing)
echo   - zlib (compression)
echo.
echo GPU Support Status:
if "%HAS_NVIDIA%"=="1" (
    echo   - NVIDIA GPU: Detected
    if not "%CUDA_PATH%"=="" (
        echo   - CUDA: Installed at %CUDA_PATH%
    ) else (
        echo   - CUDA: Not installed - required for NVIDIA GPUs
    )
)
if "%HAS_AMD%"=="1" (
    echo   - AMD GPU: Detected
    if "%HIP_FOUND%"=="1" (
        echo   - HIP SDK: Installed at !HIP_PATH!
    ) else (
        echo   - HIP SDK: Not installed - required for AMD GPUs
        echo.
        echo   %INFO% For AMD GPU support on Windows:
        echo         1. Install AMD HIP SDK from AMD website
        echo         2. OR use WSL2 with Ubuntu and ROCm (recommended)
    )
)
if "%GPU_TYPE%"=="NONE" (
    echo   - No supported GPU detected
)
echo.
echo Next steps:
echo   1. Make sure all required tools are installed:
if "%VS_FOUND%"=="0" (
    echo      - Install Visual Studio 2022
)
if "%HAS_NVIDIA%"=="1" if "%CUDA_PATH%"=="" (
    echo      - Install CUDA Toolkit
)
if "%HAS_AMD%"=="1" if "%HIP_FOUND%"=="0" (
    echo      - Install AMD HIP SDK or use WSL2 with ROCm
)
echo   2. Configure the project:
if "%HAS_NVIDIA%"=="1" (
    echo      cmake --preset windows-ninja-release
)
if "%HAS_AMD%"=="1" (
    echo      cmake --preset windows-ninja-release -DUSE_HIP=ON
    echo      Note: Full AMD support may require WSL2 with Linux
)
echo   3. Build the project:
echo      cmake --build --preset windows-release
echo.
if "%HAS_AMD%"=="1" if "%HIP_FOUND%"=="0" (
    echo %WARNING% AMD GPU Windows Support Note:
    echo.
    echo AMD GPU compute support on Windows is limited. For best performance
    echo and compatibility, consider using:
    echo.
    echo Option 1: WSL2 with Ubuntu and ROCm
    echo   - Install WSL2: wsl --install
    echo   - Install Ubuntu from Microsoft Store
    echo   - Follow Linux ROCm installation guide
    echo.
    echo Option 2: Native Windows HIP SDK (experimental)
    echo   - Download from AMD ROCm website
    echo   - May have limited GPU architecture support
    echo.
)
echo Press any key to exit...
pause >nul