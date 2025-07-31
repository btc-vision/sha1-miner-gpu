@echo off
setlocal enabledelayedexpansion

:: Build script for SHA1 Miner GPU project
:: Supports both NVIDIA/CUDA and AMD/HIP builds

:: Initialize GPU backend
if "%GPU_BACKEND%"=="" set GPU_BACKEND=NVIDIA

:: Manual ROCm path override - uncomment and modify if auto-detection fails
:: set "ROCM_PATH=C:\Program Files\AMD\ROCm\6.2"

:MAIN_MENU
cls
echo =====================================
echo   SHA1 Miner Build System
echo =====================================
echo   GPU Backend: %GPU_BACKEND%
echo =====================================
echo.
echo 1. Configure Project
echo 2. Build Project
echo 3. Test Project
echo 4. Clean Build Directory
echo 5. Setup vcpkg (if needed)
echo 6. Switch GPU Backend
echo 7. Exit
echo.
set /p choice="Select option (1-7): "

if "%choice%"=="1" goto CONFIGURE
if "%choice%"=="2" goto BUILD
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto CLEAN
if "%choice%"=="5" goto SETUP_VCPKG
if "%choice%"=="6" goto SWITCH_BACKEND
if "%choice%"=="7" goto EXIT
goto MAIN_MENU

:SWITCH_BACKEND
cls
echo =====================================
echo   Select GPU Backend
echo =====================================
echo.
echo Current backend: %GPU_BACKEND%
echo.
echo 1. NVIDIA (CUDA)
echo 2. AMD (HIP/ROCm)
echo 3. Back to Main Menu
echo.
set /p backend_choice="Select GPU backend (1-3): "

if "%backend_choice%"=="1" (
    set GPU_BACKEND=NVIDIA
    echo GPU backend set to NVIDIA/CUDA
) else if "%backend_choice%"=="2" (
    set GPU_BACKEND=AMD
    echo GPU backend set to AMD/HIP
) else if "%backend_choice%"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:CONFIGURE
cls
echo =====================================
echo   Configure Presets
echo =====================================
echo.

:: Setup Visual Studio environment for Ninja builds
call :SETUP_VS_ENV

if "%GPU_BACKEND%"=="AMD" (
    echo AMD/HIP Build Options:
    echo 1. Windows AMD Release (HIP/ROCm)
    echo 2. Windows AMD Debug (HIP/ROCm)
    echo 3. Back to Main Menu
    echo.
    set /p config_choice="Select configuration (1-3): "

    if "!config_choice!"=="1" (
        set PRESET=windows-hip-release
        set BUILD_DIR=build\hip-release
        set CMAKE_BUILD_TYPE=Release
    ) else if "!config_choice!"=="2" (
        set PRESET=windows-hip-debug
        set BUILD_DIR=build\hip-debug
        set CMAKE_BUILD_TYPE=Debug
    ) else if "!config_choice!"=="3" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto CONFIGURE
    )

    echo.
    echo Configuring with preset: %PRESET%
    echo.

    :: Check AMD dependencies
    call :CHECK_AMD_TOOLS
    if errorlevel 1 (
        echo Configuration aborted due to missing AMD/ROCm tools.
        pause
        goto MAIN_MENU
    )

    :: Check if vcpkg exists
    if not exist "vcpkg\vcpkg.exe" (
        echo WARNING: vcpkg not found! You may need to run Setup vcpkg first.
        echo.
        set /p continue_anyway="Continue anyway? (y/n): "
        if /i "!continue_anyway!" neq "y" (
            goto MAIN_MENU
        )
    )

    :: Create AMD/HIP configuration
    echo Creating AMD/HIP configuration...
    echo Build directory: %BUILD_DIR%

    :: Create build directory if it doesn't exist
    if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

    :: Set vcpkg paths
    if exist "vcpkg\scripts\buildsystems\vcpkg.cmake" (
        set VCPKG_CMAKE=-DCMAKE_TOOLCHAIN_FILE=%CD%\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static
    ) else (
        echo WARNING: vcpkg toolchain not found!
        set VCPKG_CMAKE=
    )

    :: Properly quote ROCM_PATH for CMake
    set "QUOTED_ROCM_PATH=%ROCM_PATH%"
    cmake -B %BUILD_DIR% -G "Ninja" -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_STANDARD=20 -DUSE_HIP=ON -DROCM_PATH="%QUOTED_ROCM_PATH%" -DHIP_ARCH="%HIP_ARCHITECTURES%" %VCPKG_CMAKE%

    if errorlevel 1 (
        echo.
        echo Configuration failed!
        pause
    ) else (
        echo.
        echo Configuration successful!
        pause
    )
    goto MAIN_MENU
) else (
    echo NVIDIA/CUDA Build Options:
    echo 1. Windows Release (Ninja + CUDA 12.9)
    echo 2. Windows Debug (Ninja + CUDA 12.9)
    echo 3. Windows Release (Visual Studio 2022)
    echo 4. Windows Debug (Visual Studio 2022)
    echo 5. Back to Main Menu
    echo.
    set /p config_choice="Select configuration (1-5): "

    if "!config_choice!"=="1" (
        set PRESET=windows-ninja-release
        set BUILD_DIR=build\release
    ) else if "!config_choice!"=="2" (
        set PRESET=windows-ninja-debug
        set BUILD_DIR=build\debug
    ) else if "!config_choice!"=="3" (
        set PRESET=windows-vs2022-release
        set BUILD_DIR=build\vs2022-release
    ) else if "!config_choice!"=="4" (
        set PRESET=windows-vs2022-debug
        set BUILD_DIR=build\vs2022-debug
    ) else if "!config_choice!"=="5" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto CONFIGURE
    )

    echo.
    echo Configuring with preset: %PRESET%
    echo.

    :: Check NVIDIA dependencies
    call :CHECK_NVIDIA_TOOLS

    :: Run CMake configure for NVIDIA
    cmake --preset %PRESET%
)

if errorlevel 1 (
    echo.
    echo Configuration failed!
    pause
) else (
    echo.
    echo Configuration successful!
    pause
)
goto MAIN_MENU

:BUILD
cls
echo ==============================
echo   Build Presets
echo ==============================
echo.

:: Setup Visual Studio environment for builds
call :SETUP_VS_ENV

if "%GPU_BACKEND%"=="AMD" (
    echo AMD/HIP Build Options:
    echo 1. Windows AMD Release
    echo 2. Windows AMD Debug
    echo 3. Back to Main Menu
    echo.
    set /p build_choice="Select build configuration (1-3): "

    if "!build_choice!"=="1" (
        set BUILD_DIR=build\hip-release
        set BUILD_CONFIG=Release
    ) else if "!build_choice!"=="2" (
        set BUILD_DIR=build\hip-debug
        set BUILD_CONFIG=Debug
    ) else if "!build_choice!"=="3" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto BUILD
    )

    echo.
    echo Building AMD/HIP configuration...
    echo Build directory: %BUILD_DIR%

    if not exist "%BUILD_DIR%" (
        echo.
        echo ERROR: Build directory %BUILD_DIR% does not exist!
        echo Please configure the project first - Option 1 in main menu.
        echo.
        echo Steps:
        echo 1. Go back to main menu
        echo 2. Select "1. Configure Project"
        echo 3. Choose AMD configuration - Release or Debug
        echo 4. Then come back here to build
        echo.
        pause
        goto MAIN_MENU
    )

    if not exist "%BUILD_DIR%\CMakeCache.txt" (
        echo.
        echo ERROR: Project not configured in %BUILD_DIR%
        echo Please configure the project first.
        pause
        goto MAIN_MENU
    )

    cmake --build %BUILD_DIR% --verbose -j 12

    if errorlevel 1 (
        echo.
        echo Build failed!
        pause
    ) else (
        echo.
        echo Build successful!
        pause
    )
    goto MAIN_MENU
) else (
    echo NVIDIA/CUDA Build Options:
    echo 1. Windows Release (Ninja)
    echo 2. Windows Debug (Ninja)
    echo 3. Windows Release (Visual Studio)
    echo 4. Windows Debug (Visual Studio)
    echo 5. Back to Main Menu
    echo.
    set /p build_choice="Select build configuration (1-5): "

    if "!build_choice!"=="1" (
        set BUILD_PRESET=windows-release
    ) else if "!build_choice!"=="2" (
        set BUILD_PRESET=windows-debug
    ) else if "!build_choice!"=="3" (
        set BUILD_PRESET=windows-release-vs
    ) else if "!build_choice!"=="4" (
        set BUILD_PRESET=windows-debug-vs
    ) else if "!build_choice!"=="5" (
        goto MAIN_MENU
    ) else (
        echo Invalid selection!
        pause
        goto BUILD
    )

    echo.
    echo Building with preset: %BUILD_PRESET%
    cmake --build --preset %BUILD_PRESET%
)

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
) else (
    echo.
    echo Build successful!
    pause
)
goto MAIN_MENU

:TEST
cls
echo =====================================
echo   Test Presets
echo =====================================
echo.
echo 1. Test current configuration
echo 2. Run custom test command
echo 3. Back to Main Menu
echo.
set /p test_choice="Select test option (1-3): "

if "%test_choice%"=="1" (
    echo.
    if "%GPU_BACKEND%"=="AMD" (
        echo Running tests for AMD/HIP build...
        if exist "build\hip-release\sha1_miner.exe" (
            cd build\hip-release
            sha1_miner.exe --test
            cd ..\..
        ) else if exist "build\hip-debug\sha1_miner.exe" (
            cd build\hip-debug
            sha1_miner.exe --test
            cd ..\..
        ) else (
            echo No AMD build found! Build first.
        )
    ) else (
        echo Running tests for NVIDIA/CUDA build...
        ctest --preset test-windows-release
    )
    pause
    goto MAIN_MENU
) else if "%test_choice%"=="2" (
    echo.
    set /p test_cmd="Enter test command: "
    !test_cmd!
    pause
    goto MAIN_MENU
) else if "%test_choice%"=="3" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
    pause
    goto TEST
)

:CLEAN
cls
echo =====================================
echo   Clean Build Directories
echo =====================================
echo.
echo 1. Clean Release build
echo 2. Clean Debug build
echo 3. Clean VS2022 Release build
echo 4. Clean VS2022 Debug build
echo 5. Clean AMD/HIP Release build
echo 6. Clean AMD/HIP Debug build
echo 7. Clean all builds
echo 8. Back to Main Menu
echo.
set /p clean_choice="Select directory to clean (1-8): "

if "%clean_choice%"=="1" (
    if exist "build\release" (
        echo Cleaning build\release...
        rmdir /s /q "build\release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="2" (
    if exist "build\debug" (
        echo Cleaning build\debug...
        rmdir /s /q "build\debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="3" (
    if exist "build\vs2022-release" (
        echo Cleaning build\vs2022-release...
        rmdir /s /q "build\vs2022-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="4" (
    if exist "build\vs2022-debug" (
        echo Cleaning build\vs2022-debug...
        rmdir /s /q "build\vs2022-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="5" (
    if exist "build\hip-release" (
        echo Cleaning build\hip-release...
        rmdir /s /q "build\hip-release"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="6" (
    if exist "build\hip-debug" (
        echo Cleaning build\hip-debug...
        rmdir /s /q "build\hip-debug"
        echo Done!
    ) else (
        echo Directory not found!
    )
) else if "%clean_choice%"=="7" (
    echo Cleaning all build directories...
    if exist "build" rmdir /s /q "build"
    echo Done!
) else if "%clean_choice%"=="8" (
    goto MAIN_MENU
) else (
    echo Invalid selection!
)
pause
goto MAIN_MENU

:SETUP_VCPKG
cls
echo =====================================
echo   Setup vcpkg
echo =====================================
echo.

if exist "vcpkg\vcpkg.exe" (
    echo vcpkg is already installed!
    echo.
    set /p update_choice="Do you want to update vcpkg? (y/n): "
    if /i "!update_choice!"=="y" (
        cd vcpkg
        git pull
        .\bootstrap-vcpkg.bat
        cd ..
    )
) else (
    echo vcpkg not found. Installing...
    echo.

    :: Clone vcpkg
    git clone https://github.com/Microsoft/vcpkg.git

    :: Bootstrap vcpkg
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..

    echo.
    echo vcpkg installed successfully!
)

echo.
echo Installing required packages...
echo This may take a while...
echo.

cd vcpkg
.\vcpkg install boost-beast boost-asio boost-system boost-thread boost-program-options boost-date-time boost-regex boost-random openssl curl zlib nlohmann-json --triplet x64-windows
cd ..

echo.
echo Package installation complete!
pause
goto MAIN_MENU

:CHECK_NVIDIA_TOOLS
:: Check for CUDA
if not exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" (
    echo WARNING: CUDA 12.9 not found at expected location!
    echo CUDA builds may fail.
    echo.

    :: Try to find any CUDA installation
    for /d %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%i\bin\nvcc.exe" (
            echo Found CUDA installation at: %%i
            echo Consider updating CMakePresets.json to use this version.
            echo.
        )
    )
)
exit /b 0

:CHECK_AMD_TOOLS
:: Set default ROCm path if not set
if "%ROCM_PATH%"=="" (
    call :AUTO_DETECT_ROCM
    if errorlevel 1 exit /b 1
)

:: Validate ROCm installation
if not exist "%ROCM_PATH%\bin\hipcc.bin.exe" (
    if not exist "%ROCM_PATH%\bin\hipcc.exe" (
        echo ERROR: hipcc not found in %ROCM_PATH%\bin\
        echo Attempting to re-detect ROCm installation...
        call :AUTO_DETECT_ROCM
        if errorlevel 1 exit /b 1
    )
)

:: Extract ROCm version from path if possible
for %%i in ("%ROCM_PATH%") do set ROCM_VERSION=%%~nxi
echo ROCm Version: %ROCM_VERSION%

:: Set HIP environment variables
set HIP_PATH=%ROCM_PATH%
set HSA_PATH=%ROCM_PATH%
set HIP_PLATFORM=amd
set HIP_RUNTIME=rocclr
set HIP_COMPILER=clang

:: Auto-detect GPU architecture if not set
if "%HIP_ARCHITECTURES%"=="" (
    call :AUTO_DETECT_GPU_ARCH
)

:: Check ROCm components
call :CHECK_ROCM_COMPONENTS

echo.
echo ROCm Configuration Summary:
echo   Installation: %ROCM_PATH%
echo   Version: %ROCM_VERSION%
echo   HIP Architectures: %HIP_ARCHITECTURES%
echo.
exit /b 0

:: ========================================
:: Auto-detect ROCm installation
:: ========================================
:AUTO_DETECT_ROCM
echo Searching for ROCm installation...
set ROCM_FOUND=0

:: First check if ROCM_PATH environment variable is already set
if defined ROCM_PATH (
    if exist "%ROCM_PATH%\bin\hipcc.bin.exe" (
        echo Using ROCM_PATH from environment: %ROCM_PATH%
        set ROCM_FOUND=1
        exit /b 0
    )
    if exist "%ROCM_PATH%\bin\hipcc.exe" (
        echo Using ROCM_PATH from environment: %ROCM_PATH%
        set ROCM_FOUND=1
        exit /b 0
    )
)

:: Method 1: Check Program Files for versioned installations
echo Checking Program Files for ROCm...
if exist "C:\Program Files\AMD\ROCm" (
    for /d %%v in ("C:\Program Files\AMD\ROCm\*") do (
        if exist "%%v\bin\hipcc.bin.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
        if exist "%%v\bin\hipcc.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
    )
)

:: Method 2: Check for direct installation without version folder
if exist "C:\Program Files\AMD\ROCm\bin\hipcc.bin.exe" (
    set "ROCM_PATH=C:\Program Files\AMD\ROCm"
    set ROCM_FOUND=1
    echo Found ROCm at: C:\Program Files\AMD\ROCm
    exit /b 0
)
if exist "C:\Program Files\AMD\ROCm\bin\hipcc.exe" (
    set "ROCM_PATH=C:\Program Files\AMD\ROCm"
    set ROCM_FOUND=1
    echo Found ROCm at: C:\Program Files\AMD\ROCm
    exit /b 0
)

:: Method 3: Check C:\ROCm for versioned installations
if exist "C:\ROCm" (
    for /d %%v in ("C:\ROCm\*") do (
        if exist "%%v\bin\hipcc.bin.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
        if exist "%%v\bin\hipcc.exe" (
            set "ROCM_PATH=%%v"
            set ROCM_FOUND=1
            echo Found ROCm at: %%v
            exit /b 0
        )
    )
)

:: Method 4: Check registry
echo Checking registry for ROCm installation...
for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\AMD\ROCm" /v InstallDir 2^>nul') do (
    if exist "%%b\bin\hipcc.bin.exe" (
        set "ROCM_PATH=%%b"
        set ROCM_FOUND=1
        echo Found ROCm via registry at: %%b
        exit /b 0
    )
    if exist "%%b\bin\hipcc.exe" (
        set "ROCM_PATH=%%b"
        set ROCM_FOUND=1
        echo Found ROCm via registry at: %%b
        exit /b 0
    )
)

:: Method 5: Check for HIP_PATH environment variable
if defined HIP_PATH (
    if exist "%HIP_PATH%\bin\hipcc.bin.exe" (
        set "ROCM_PATH=%HIP_PATH%"
        set ROCM_FOUND=1
        echo Found ROCm via HIP_PATH: %HIP_PATH%
        exit /b 0
    )
    if exist "%HIP_PATH%\bin\hipcc.exe" (
        set "ROCM_PATH=%HIP_PATH%"
        set ROCM_FOUND=1
        echo Found ROCm via HIP_PATH: %HIP_PATH%
        exit /b 0
    )
)

:: Not found
echo.
echo ERROR: ROCm not found!
echo.
echo Please either:
echo 1. Install ROCm from: https://www.amd.com/en/developer/rocm-software.html
echo 2. Set ROCM_PATH environment variable to your ROCm installation directory
echo    Example: set ROCM_PATH=C:\Program Files\AMD\ROCm\6.2
echo.
echo Searched in:
echo   - C:\Program Files\AMD\ROCm\[version]\
echo   - C:\Program Files\AMD\ROCm\
echo   - C:\ROCm\[version]\
echo   - Windows Registry
echo   - Environment variables (ROCM_PATH, HIP_PATH)
echo.
pause
exit /b 1

:: ========================================
:: Auto-detect GPU architecture
:: ========================================
:AUTO_DETECT_GPU_ARCH
echo Auto-detecting GPU architecture...

:: Try using rocm-smi to detect GPU
if exist "%ROCM_PATH%\bin\rocm-smi.exe" (
    :: Get GPU info
    "%ROCM_PATH%\bin\rocm-smi.exe" --showproductname >"%TEMP%\gpu_info.txt" 2>nul
    if !errorlevel!==0 (
        :: Parse GPU names and determine architectures
        set GPU_ARCHS=
        for /f "tokens=*" %%a in ('findstr /i "GPU" "%TEMP%\gpu_info.txt"') do (
            :: Detect architecture based on GPU name
            echo %%a | findstr /i "RX.6[89]00" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1030,gfx1031,
            echo %%a | findstr /i "RX.7[89]00" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1100,gfx1101,gfx1102,
            echo %%a | findstr /i "RX.7600" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1102,
            echo %%a | findstr /i "RX.7700" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1101,
            echo %%a | findstr /i "RX.7800" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1101,
            echo %%a | findstr /i "RX.7900" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx1100,
            echo %%a | findstr /i "MI200" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx90a,
            echo %%a | findstr /i "MI250" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx90a,
            echo %%a | findstr /i "MI300" >nul && set GPU_ARCHS=!GPU_ARCHS!gfx940,gfx941,gfx942,
        )
        del "%TEMP%\gpu_info.txt" >nul 2>&1

        if not "!GPU_ARCHS!"=="" (
            :: Remove trailing comma
            set HIP_ARCHITECTURES=!GPU_ARCHS:~0,-1!
            echo Detected GPU architectures: !HIP_ARCHITECTURES!
            exit /b 0
        )
    )
)

:: Try using hipinfo if available
if exist "%ROCM_PATH%\bin\hipinfo.exe" (
    "%ROCM_PATH%\bin\hipinfo.exe" >"%TEMP%\hip_info.txt" 2>nul
    if !errorlevel!==0 (
        for /f "tokens=2 delims=:" %%a in ('findstr /i "gcnArchName" "%TEMP%\hip_info.txt"') do (
            set ARCH_NAME=%%a
            set ARCH_NAME=!ARCH_NAME: =!
            if not "!ARCH_NAME!"=="" (
                set HIP_ARCHITECTURES=!ARCH_NAME!
                echo Detected GPU architecture: !HIP_ARCHITECTURES!
                del "%TEMP%\hip_info.txt" >nul 2>&1
                exit /b 0
            )
        )
        del "%TEMP%\hip_info.txt" >nul 2>&1
    )
)

:: Fallback to default architectures
echo Could not auto-detect GPU architecture. Using defaults for RDNA2/RDNA3...
set HIP_ARCHITECTURES=gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201
echo Default architectures: %HIP_ARCHITECTURES%
exit /b 0

:: ========================================
:: Check ROCm components
:: ========================================
:CHECK_ROCM_COMPONENTS
echo.
echo Checking ROCm components:

:: Check hipcc
if exist "%ROCM_PATH%\bin\hipcc.bin.exe" (
    echo [✓] hipcc found (hipcc.bin.exe)
    "%ROCM_PATH%\bin\hipcc.bin.exe" --version >nul 2>&1
    if !errorlevel!==0 (
        echo [✓] hipcc is functional
    ) else (
        echo [!] hipcc found but not functional
    )
) else if exist "%ROCM_PATH%\bin\hipcc.exe" (
    echo [✓] hipcc found (hipcc.exe)
    "%ROCM_PATH%\bin\hipcc.exe" --version >nul 2>&1
    if !errorlevel!==0 (
        echo [✓] hipcc is functional
    ) else (
        echo [!] hipcc found but not functional
    )
) else (
    echo [✗] hipcc NOT found
)

:: Check hipconfig
if exist "%ROCM_PATH%\bin\hipconfig.exe" (
    echo [✓] hipconfig found
) else (
    echo [✗] hipconfig NOT found
)

:: Check CMake modules
if exist "%ROCM_PATH%\lib\cmake\hip" (
    echo [✓] HIP CMake modules found
) else (
    echo [✗] HIP CMake modules NOT found
)

:: Check runtime libraries
if exist "%ROCM_PATH%\bin\amdhip64.dll" (
    echo [✓] HIP runtime library found
) else (
    echo [✗] HIP runtime library NOT found
)

:: Check ROCm-SMI
if exist "%ROCM_PATH%\bin\rocm-smi.exe" (
    echo [✓] rocm-smi found
) else (
    echo [!] rocm-smi NOT found (optional)
)

exit /b 0

:EXIT
echo.
echo Exiting build system...
exit /b 0

:SETUP_VS_ENV
:: Check if VS environment is already set up
if defined VSCMD_VER (
    echo Visual Studio environment already configured
    exit /b 0
)

:: Try to find and setup Visual Studio 2022 environment
echo Setting up Visual Studio environment...

:: First try VS2022 Community
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 (
        echo ERROR: Failed to initialize Visual Studio 2022 Community environment
        pause
        exit /b 1
    )
    echo Visual Studio 2022 Community environment loaded
    exit /b 0
)

:: Try VS2022 Professional
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 (
        echo ERROR: Failed to initialize Visual Studio 2022 Professional environment
        pause
        exit /b 1
    )
    echo Visual Studio 2022 Professional environment loaded
    exit /b 0
)

:: Try VS2022 Enterprise
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 (
        echo ERROR: Failed to initialize Visual Studio 2022 Enterprise environment
        pause
        exit /b 1
    )
    echo Visual Studio 2022 Enterprise environment loaded
    exit /b 0
)

:: Try VS2022 BuildTools
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 (
        echo ERROR: Failed to initialize Visual Studio 2022 Build Tools environment
        pause
        exit /b 1
    )
    echo Visual Studio 2022 Build Tools environment loaded
    exit /b 0
)

:: Try using vswhere to find VS installation
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
        call "%%i\VC\Auxiliary\Build\vcvars64.bat"
        if errorlevel 1 (
            echo ERROR: Failed to initialize Visual Studio environment from: %%i
            pause
            exit /b 1
        )
        echo Visual Studio environment loaded from: %%i
        exit /b 0
    )
)

echo ERROR: Could not find Visual Studio 2022 installation!
echo Please ensure Visual Studio 2022 with C++ development tools is installed.
echo Alternatively, run this script from a "Developer Command Prompt for VS 2022".
pause
exit /b 1

:: Check for required tools at startup
where cmake >nul 2>nul
if errorlevel 1 (
    echo ERROR: CMake not found in PATH!
    echo Please install CMake and add it to your PATH.
    pause
    exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
    echo WARNING: Ninja not found in PATH!
    echo Ninja builds will not work without it.
    echo.
    echo You can install Ninja by:
    echo 1. Download from https://github.com/ninja-build/ninja/releases
    echo 2. Or install via winget: winget install Ninja-build.Ninja
    echo 3. Or install via chocolatey: choco install ninja
    echo.
)

:: Display current backend
echo Current GPU Backend: %GPU_BACKEND%
echo.
pause

:: Start main menu
goto MAIN_MENU
