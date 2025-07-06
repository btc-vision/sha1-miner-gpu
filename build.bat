@echo off
REM SHA-1 Near-Collision Miner - Windows Build Script
REM This script configures and builds the miner with optimal settings for Windows

setlocal enabledelayedexpansion

REM Default values
set BUILD_TYPE=Release
set GPU_ARCH=
set CLEAN_BUILD=0
set GENERATOR="Visual Studio 17 2022"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
if /i "%~1"=="--arch" (
    set GPU_ARCH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--clean" (
    set CLEAN_BUILD=1
    shift
    goto :parse_args
)
if /i "%~1"=="--vs2019" (
    set GENERATOR="Visual Studio 16 2019"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    echo SHA-1 Near-Collision Miner Build Script for Windows
    echo.
    echo Usage: %0 [options]
    echo.
    echo Options:
    echo   --debug          Build in debug mode ^(default: Release^)
    echo   --arch ^<arch^>    Specify GPU architecture ^(e.g., 86 for RTX 3080^)
    echo   --clean          Clean build directory before building
    echo   --vs2019         Use Visual Studio 2019 instead of 2022
    echo   --help           Show this help message
    echo.
    echo Examples:
    echo   %0                    # Release build
    echo   %0 --debug            # Debug build
    echo   %0 --arch 86          # Build for RTX 3080 only
    echo   %0 --clean            # Clean build
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1
:end_parse

echo SHA-1 Near-Collision Miner Build Script for Windows
echo ====================================================
echo.

REM Check for CUDA
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: nvcc not found. Please install CUDA toolkit.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)

REM Display CUDA version
echo CUDA Version:
nvcc --version | findstr "release"
echo.

REM Check for CMake
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: cmake not found. Please install CMake.
    echo Download from: https://cmake.org/download/
    exit /b 1
)

REM Detect GPU if architecture not specified
if "%GPU_ARCH%"=="" (
    echo Detecting GPU architecture...

    REM Use nvidia-smi to get GPU info
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu^=gpu_name^,compute_cap --format^=csv^,noheader 2^>nul') do (
        set GPU_NAME=%%a
        set COMPUTE_CAP=%%b
        REM Remove dots from compute capability
        set GPU_ARCH=!COMPUTE_CAP:.=!
        echo Detected: !GPU_NAME! ^(Compute Capability !COMPUTE_CAP!^)
        goto :gpu_detected
    )

    echo Warning: Could not detect GPU. Building for all architectures.
    :gpu_detected
    echo.
)

REM Set build directory
set BUILD_DIR=build
if "%BUILD_TYPE%"=="Debug" (
    set BUILD_DIR=build-debug
)

REM Clean if requested
if %CLEAN_BUILD%==1 (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
)

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM Configure with CMake
echo Configuring...
set CMAKE_ARGS=-DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if not "%GPU_ARCH%"=="" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_CUDA_ARCHITECTURES=%GPU_ARCH%
)

cmake -G %GENERATOR% %CMAKE_ARGS% ..
if %errorlevel% neq 0 (
    echo.
    echo Error: CMake configuration failed
    cd ..
    exit /b 1
)

REM Build
echo.
echo Building %BUILD_TYPE% configuration...
cmake --build . --config %BUILD_TYPE% --parallel
if %errorlevel% neq 0 (
    echo.
    echo Error: Build failed
    cd ..
    exit /b 1
)

REM Return to original directory
cd ..

echo.
echo Build complete!
echo Executables:
echo   %BUILD_DIR%\%BUILD_TYPE%\sha1_miner.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\verify_sha1.exe

REM Create batch files for convenience
echo @echo off > sha1_miner.bat
echo %BUILD_DIR%\%BUILD_TYPE%\sha1_miner.exe %%* >> sha1_miner.bat

echo @echo off > verify_sha1.bat
echo %BUILD_DIR%\%BUILD_TYPE%\verify_sha1.exe %%* >> verify_sha1.bat

echo.
echo Convenience batch files created:
echo   sha1_miner.bat
echo   verify_sha1.bat
echo.
echo Next steps:
echo 1. Run tests: verify_sha1.bat
echo 2. Run benchmark: sha1_miner.bat --benchmark
echo 3. Start mining: sha1_miner.bat --gpu 0 --difficulty 100 --duration 60
echo.

REM Optional: Run quick test
set /p RUN_TEST="Run verification test now? (Y/N): "
if /i "%RUN_TEST%"=="Y" (
    echo.
    echo Running verification test...
    %BUILD_DIR%\%BUILD_TYPE%\verify_sha1.exe
)

endlocal