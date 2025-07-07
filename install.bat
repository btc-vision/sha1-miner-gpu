@echo off
REM SHA-1 OP_NET Miner - Complete Windows Installation Script
REM This script installs all dependencies including uWebSockets

setlocal enabledelayedexpansion

REM Set colors
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

REM Default installation directory
if "%~1"=="" (
    set "INSTALL_DIR=%USERPROFILE%\sha1-miner"
) else (
    set "INSTALL_DIR=%~1"
)

REM Header
cls
echo =====================================
echo SHA-1 OP_NET Miner - Windows Installer
echo =====================================
echo.
echo Installation directory: %INSTALL_DIR%
echo.

REM Check for NVIDIA GPU
echo %INFO% Checking for NVIDIA GPU...
wmic path win32_VideoController get name 2>nul | find /i "NVIDIA" >nul
if errorlevel 1 (
    echo %ERROR% No NVIDIA GPU detected.
    echo         AMD GPUs are not yet supported on Windows.
    pause
    exit /b 1
)
echo %SUCCESS% NVIDIA GPU detected
echo.

REM Check for CUDA
echo %INFO% Checking for CUDA installation...
if "%CUDA_PATH%"=="" (
    echo %ERROR% CUDA not found. Please install CUDA Toolkit first:
    echo         https://developer.nvidia.com/cuda-downloads
    echo.
    echo         After installing CUDA, run this installer again.
    pause
    exit /b 1
)
echo %SUCCESS% CUDA found at: %CUDA_PATH%
echo.

REM Create installation directory
echo %INFO% Creating installation directory...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
cd /d "%INSTALL_DIR%"
echo.

REM Check for Git
echo %INFO% Checking for Git...
where git >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Git not found. Please install Git from:
    echo         https://git-scm.com/download/win
    echo.
    echo Opening download page...
    start https://git-scm.com/download/win
    pause
    exit /b 1
)
echo %SUCCESS% Git found
echo.

REM Check for CMake
echo %INFO% Checking for CMake...
where cmake >nul 2>&1
if errorlevel 1 (
    echo %ERROR% CMake not found. Please install CMake from:
    echo         https://cmake.org/download/
    echo.
    echo Opening download page...
    start https://cmake.org/download/
    pause
    exit /b 1
)
echo %SUCCESS% CMake found
echo.

REM Check for Visual Studio
echo %INFO% Checking for Visual Studio...
set "VS_FOUND=0"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=1"

if "%VS_FOUND%"=="0" (
    echo %ERROR% Visual Studio not found. Please install Visual Studio 2019 or 2022
    echo         with "Desktop development with C++" workload.
    echo.
    echo         Download from: https://visualstudio.microsoft.com/downloads/
    echo.
    echo Opening download page...
    start https://visualstudio.microsoft.com/downloads/
    pause
    exit /b 1
)
echo %SUCCESS% Visual Studio found
echo.

REM Setup vcpkg
echo %INFO% Setting up vcpkg for C++ dependencies...
if exist "%INSTALL_DIR%\vcpkg\vcpkg.exe" (
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

REM Install OpenSSL first (it's often problematic)
echo %INFO% Installing OpenSSL...
vcpkg\vcpkg install openssl:x64-windows
if errorlevel 1 (
    echo %WARNING% OpenSSL installation had issues, but continuing...
)

echo %INFO% Installing Boost libraries...
vcpkg\vcpkg install boost-system:x64-windows boost-thread:x64-windows boost-program-options:x64-windows
if errorlevel 1 (
    echo %WARNING% Boost installation had issues, but continuing...
)

echo %INFO% Installing nlohmann-json...
vcpkg\vcpkg install nlohmann-json:x64-windows
if errorlevel 1 (
    echo %WARNING% JSON installation had issues, but continuing...
)

echo %INFO% Installing zlib...
vcpkg\vcpkg install zlib:x64-windows
if errorlevel 1 (
    echo %WARNING% zlib installation had issues, but continuing...
)

REM Check if uWebSockets is available in vcpkg
echo %INFO% Checking for uWebSockets in vcpkg...
vcpkg\vcpkg search uwebsockets | findstr "uwebsockets " >nul 2>&1
if errorlevel 1 (
    echo %INFO% uWebSockets not found in vcpkg, will install manually...
    goto :manual_uwebsockets
) else (
    echo %INFO% Installing uWebSockets from vcpkg...
    vcpkg\vcpkg install uwebsockets:x64-windows
    if errorlevel 1 (
        echo %WARNING% vcpkg installation failed, installing manually...
        goto :manual_uwebsockets
    ) else (
        echo %SUCCESS% uWebSockets installed via vcpkg
        goto :after_uwebsockets
    )
)

:manual_uwebsockets
echo.
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
    if errorlevel 1 (
        echo %ERROR% Failed to clone uSockets
        pause
        exit /b 1
    )
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
    if errorlevel 1 (
        echo %ERROR% Failed to clone uWebSockets
        pause
        exit /b 1
    )
)

cd ..
echo %SUCCESS% uWebSockets installed manually

:after_uwebsockets

REM Integrate vcpkg
vcpkg\vcpkg integrate install

REM Set OpenSSL environment variable
echo.
echo %INFO% Setting OpenSSL environment variable...
set "OPENSSL_ROOT_DIR=%INSTALL_DIR%\vcpkg\installed\x64-windows"
setx OPENSSL_ROOT_DIR "%INSTALL_DIR%\vcpkg\installed\x64-windows" >nul 2>&1

REM Create directory structure
echo.
echo %INFO% Creating project structure...
if not exist "src" mkdir src
if not exist "include" mkdir include
if not exist "include\miner" mkdir include\miner
if not exist "build" mkdir build

REM Prompt for source files
echo.
echo %WARNING% Please copy your source files to:
echo           %INSTALL_DIR%
echo.
echo Required files:
echo   - CMakeLists.txt
echo   - src\main.cpp
echo   - src\mining_system.cpp
echo   - src\multi_gpu_manager.cpp
echo   - src\globals.cpp
echo   - src\pool_client.cpp (or pool_client_uws.cpp)
echo   - src\pool_integration.cpp
echo   - include\*.hpp files
echo   - include\miner\*.cuh files
echo   - include\miner\*.cu files
echo.
set /p "READY=Have you copied all source files? (y/n): "
if /i not "%READY%"=="y" (
    echo.
    echo Please copy the files and run this installer again.
    pause
    exit /b 0
)

REM Configure and build
echo.
echo %INFO% Configuring project with CMake...
cd build

REM Determine Visual Studio version
set "CMAKE_GENERATOR=Visual Studio 17 2022"
if not exist "%ProgramFiles%\Microsoft Visual Studio\2022" (
    set "CMAKE_GENERATOR=Visual Studio 16 2019"
)

REM Configure with all paths
cmake -G "%CMAKE_GENERATOR%" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE="%INSTALL_DIR%\vcpkg\scripts\buildsystems\vcpkg.cmake" ^
    -DOPENSSL_ROOT_DIR="%INSTALL_DIR%\vcpkg\installed\x64-windows" ^
    -DUWEBSOCKETS_ROOT="%INSTALL_DIR%\external\uWebSockets" ^
    -DUSOCKETS_ROOT="%INSTALL_DIR%\external\uSockets" ^
    ..

if errorlevel 1 (
    echo %ERROR% CMake configuration failed
    echo.
    echo Common issues:
    echo - Missing source files
    echo - CMakeLists.txt not found
    echo - Dependencies not properly installed
    pause
    exit /b 1
)

echo.
echo %INFO% Building project (this may take several minutes)...
cmake --build . --config Release --parallel
if errorlevel 1 (
    echo %ERROR% Build failed
    echo.
    echo Check the error messages above for details.
    pause
    exit /b 1
)

cd ..
echo %SUCCESS% Build completed successfully!
echo.

REM Create launcher scripts
echo %INFO% Creating launcher scripts...

REM Main launcher
(
echo @echo off
echo setlocal
echo.
echo set "MINER_PATH=%%~dp0build\Release\sha1_miner.exe"
echo.
echo if not exist "%%MINER_PATH%%" ^(
echo     echo Error: Miner not found at %%MINER_PATH%%
echo     pause
echo     exit /b 1
echo ^)
echo.
echo "%%MINER_PATH%%" %%*
) > run_miner.bat

REM Create example configurations
if not exist "configs" mkdir configs

REM Solo mining config
(
echo @echo off
echo REM Solo Mining Configuration
echo cd /d "%%~dp0"
echo call run_miner.bat --gpu 0 --difficulty 45 --duration 3600 --auto-tune
echo pause
) > configs\solo_mining.bat

REM Pool mining config
(
echo @echo off
echo REM Pool Mining Configuration
echo REM Edit these values:
echo set "POOL_URL=ws://pool.example.com:3333"
echo set "WALLET=YOUR_WALLET_ADDRESS_HERE"
echo set "WORKER_NAME=%%COMPUTERNAME%%"
echo.
echo cd /d "%%~dp0"
echo call run_miner.bat --pool %%POOL_URL%% --wallet %%WALLET%% --worker %%WORKER_NAME%% --auto-tune
echo pause
) > configs\pool_mining.bat

REM Multi-GPU config
(
echo @echo off
echo REM Multi-GPU Pool Mining Configuration
echo REM Edit these values:
echo set "POOL_URL=ws://pool.example.com:3333"
echo set "WALLET=YOUR_WALLET_ADDRESS_HERE"
echo set "WORKER_NAME=%%COMPUTERNAME%%"
echo.
echo cd /d "%%~dp0"
echo call run_miner.bat --all-gpus --pool %%POOL_URL%% --wallet %%WALLET%% --worker %%WORKER_NAME%% --auto-tune
echo pause
) > configs\multi_gpu_mining.bat

REM Create desktop shortcut
echo %INFO% Creating desktop shortcut...
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\SHA1 Miner.lnk'); $SC.TargetPath = '%INSTALL_DIR%\run_miner.bat'; $SC.WorkingDirectory = '%INSTALL_DIR%'; $SC.IconLocation = 'shell32.dll,12'; $SC.Save()"

echo.
echo =====================================
echo Installation completed successfully!
echo =====================================
echo.
echo Installation directory: %INSTALL_DIR%
echo.
echo To run the miner:
echo   1. Use the desktop shortcut "SHA1 Miner"
echo   2. Or open Command Prompt and run:
echo      cd %INSTALL_DIR%
echo      run_miner.bat --help
echo.
echo Example commands:
echo   Solo mining:  run_miner.bat --gpu 0 --difficulty 45
echo   Pool mining:  run_miner.bat --pool ws://pool.com:3333 --wallet YOUR_WALLET
echo   Multi-GPU:    run_miner.bat --all-gpus --pool ws://pool.com:3333 --wallet YOUR_WALLET
echo.
echo Configuration examples in: %INSTALL_DIR%\configs
echo.
echo For CLion users:
echo   - Open the project folder in CLion
echo   - Use the CMakePresets.json for easy configuration
echo   - Or manually set CMAKE_TOOLCHAIN_FILE to: %INSTALL_DIR%\vcpkg\scripts\buildsystems\vcpkg.cmake
echo.
pause