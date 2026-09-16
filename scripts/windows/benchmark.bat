@echo off
setlocal enabledelayedexpansion

:: ================================================================
:: FeatherTalk Windows benchmark script (ASCII-only)
:: Usage: benchmark.bat [--repo-url <url>] [--repeats N] [--backends list] [--help]
:: ================================================================

:: ---------- Defaults ----------
set "REPO_URL=https://github.com/znicelya/FeatherTalk-Desktop.git"
set "REPEATS=3"
set "BACKENDS=cpu,wgpu,cuda,rocm"
set "PROJECT_ROOT=target\benchmark\full-pipeline"
set "EPOCHS=1"
set "MAX_OUTPUT_FRAMES=0"
set "KEEP_PROJECTS=0"
set "FIXTURE=tools\feathertalk-benchmark\fixtures\kanghui_5s.mp4"
set "REPORT_DIR=target\benchmark\reports"
set "SKIP_FFMPEG=0"
set "SKIP_BUILD=0"

set "FFMPEG_DIR=C:\ffmpeg"
set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FFMPEG_MIN_MAJOR=5"

:: ---------- Help ----------
if "%~1"=="--help" goto :usage
if "%~1"=="-h" goto :usage

:: ---------- Parse args ----------
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--repo-url" ( set "REPO_URL=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--repeats"  ( set "REPEATS=%~2"   & shift & shift & goto :parse_args )
if /i "%~1"=="--backends" ( set "BACKENDS=%~2"  & shift & shift & goto :parse_args )
if /i "%~1"=="--project-root" ( set "PROJECT_ROOT=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--epochs" ( set "EPOCHS=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--max-output-frames" ( set "MAX_OUTPUT_FRAMES=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--keep-projects" ( set "KEEP_PROJECTS=1" & shift & goto :parse_args )
if /i "%~1"=="--skip-ffmpeg" ( set "SKIP_FFMPEG=1" & shift & goto :parse_args )
if /i "%~1"=="--skip-build"  ( set "SKIP_BUILD=1"  & shift & goto :parse_args )
echo [err] unknown argument: %~1
goto :usage
:args_done

echo [bench] FeatherTalk Windows benchmark
echo [bench] backends: %BACKENDS%  ^| repeats: %REPEATS%

:: ================================================================
:: 1. Environment detection
:: ================================================================
echo [bench] detecting environment...

where git >nul 2>&1
if errorlevel 1 (
    echo [err] git not found - install Git for Windows
    exit /b 1
)

where curl >nul 2>&1
if errorlevel 1 (
    echo [err] curl not found - bundled with Windows 10 1803+
    exit /b 1
)

:: ---- GPU detection (simplified: only existence check) ----
set "HAS_NVIDIA=0"
set "HAS_AMD=0"

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "HAS_NVIDIA=1"
    echo [bench] NVIDIA GPU detected - nvidia-smi available
) else (
    echo [bench] no NVIDIA GPU detected
)

echo [bench] AMD detection skipped on Windows

:: ================================================================
:: 2. Install FFmpeg
:: ================================================================
if "%SKIP_FFMPEG%"=="1" (
    echo [warn] --skip-ffmpeg specified, skipping FFmpeg install
    goto :ffmpeg_done
)

echo [bench] checking FFmpeg...

set "FFMPEG_OK=0"
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    set "VER_TOKEN="
    for /f "tokens=3" %%v in ('ffmpeg -version 2^>^&1 ^| findstr /b "ffmpeg version"') do (
        set "VER_TOKEN=%%v"
        goto :ffmpeg_ver_parsed
    )
    :ffmpeg_ver_parsed
    if defined VER_TOKEN (
        for /f "tokens=1 delims=." %%m in ("!VER_TOKEN!") do set "CUR_MAJOR=%%m"
        echo !CUR_MAJOR! | findstr /r "^[0-9][0-9][0-9][0-9]$" >nul
        if not errorlevel 1 (
            echo [bench] existing FFmpeg is a dev build, OK
            set "FFMPEG_OK=1"
        ) else (
            if !CUR_MAJOR! GEQ %FFMPEG_MIN_MAJOR% (
                echo [bench] existing FFmpeg major=!CUR_MAJOR! satisfies min %FFMPEG_MIN_MAJOR%
                set "FFMPEG_OK=1"
            ) else (
                echo [warn] existing FFmpeg major=!CUR_MAJOR! is too old
            )
        )
    )
)

if "%FFMPEG_OK%"=="1" goto :ffmpeg_done

echo [bench] downloading FFmpeg static build...
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
pushd "%FFMPEG_DIR%"

if not exist "ffmpeg.zip" (
    curl -L -o ffmpeg.zip "%FFMPEG_URL%"
    if errorlevel 1 (
        echo [err] FFmpeg download failed
        popd
        exit /b 1
    )
)

echo [bench] extracting FFmpeg...
if exist "ffmpeg-master-latest-win64-gpl" rmdir /s /q "ffmpeg-master-latest-win64-gpl"
tar -xf ffmpeg.zip
if errorlevel 1 (
    echo [err] extraction failed
    popd
    exit /b 1
)

if not exist "bin" mkdir bin
copy /Y "ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"  "bin\" >nul
copy /Y "ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe" "bin\" >nul

echo [bench] FFmpeg installed to %FFMPEG_DIR%\bin
popd

:ffmpeg_done

set "PATH=%FFMPEG_DIR%\bin;%PATH%"

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [err] ffmpeg not available
    exit /b 1
)

for /f "tokens=*" %%i in ('ffmpeg -version 2^>^&1 ^| findstr /b "ffmpeg version"') do (
    echo [bench] %%i
    goto :ffmpeg_ver_done
)
:ffmpeg_ver_done

:: ================================================================
:: 3. Install Rust toolchain
:: ================================================================
echo [bench] checking Rust toolchain...

where cargo >nul 2>&1
if errorlevel 1 (
    echo [bench] installing rustup...
    curl -L -o rustup-init.exe https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe
    if errorlevel 1 (
        echo [err] rustup-init.exe download failed
        exit /b 1
    )
    rustup-init.exe -y --default-toolchain none
    if errorlevel 1 (
        echo [err] rustup install failed
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

where rustup >nul 2>&1
if errorlevel 1 (
    echo [err] rustup not installed
    exit /b 1
)

for /f "tokens=*" %%i in ('rustup --version 2^>^&1') do set "RUSTUP_VER=%%i"
for /f "tokens=*" %%i in ('cargo --version 2^>^&1')  do set "CARGO_VER=%%i"
echo [bench] %RUSTUP_VER%
echo [bench] %CARGO_VER%

:: ================================================================
:: 4. Clone repo
:: ================================================================
if not exist "Cargo.toml" goto :clone_repo
if not exist "rust-toolchain.toml" goto :clone_repo
echo [bench] already in repo root: %CD%
goto :repo_ready

:clone_repo
echo [bench] cloning %REPO_URL% ...
for %%i in ("%REPO_URL%") do set "CLONE_DIR=%%~ni"
if exist "%CLONE_DIR%" (
    echo [warn] directory %CLONE_DIR% exists, entering
    cd /d "%CLONE_DIR%"
) else (
    git clone "%REPO_URL%"
    if errorlevel 1 (
        echo [err] git clone failed
        exit /b 1
    )
    cd /d "%CLONE_DIR%"
)

:repo_ready

echo [bench] resolving toolchain from rust-toolchain.toml...
rustup show active-toolchain 2>nul
if errorlevel 1 rustup show

:: ================================================================
:: 5. Env vars
:: ================================================================
echo [bench] configuring worker env vars...

if not defined FEATHERTALK_WORKER_FFMPEG (
    for /f "tokens=*" %%i in ('where ffmpeg') do set "FEATHERTALK_WORKER_FFMPEG=%%i"
)
if not defined FEATHERTALK_WORKER_FFPROBE (
    for /f "tokens=*" %%i in ('where ffprobe') do set "FEATHERTALK_WORKER_FFPROBE=%%i"
)
if not defined FEATHERTALK_WORKER_SCRFD_DIR  set "FEATHERTALK_WORKER_SCRFD_DIR=%CD%\models\scrfd_2_5g"
if not defined FEATHERTALK_WORKER_PFLD_DIR   set "FEATHERTALK_WORKER_PFLD_DIR=%CD%\models\pfld_ghost_one"
if not defined FEATHERTALK_WORKER_HUBERT_DIR set "FEATHERTALK_WORKER_HUBERT_DIR=%CD%\models\feather_hubert"
if not defined FEATHERTALK_WORKER_VGG19_DIR  set "FEATHERTALK_WORKER_VGG19_DIR=%CD%\models\vgg19"

echo [bench]   FFMPEG     = %FEATHERTALK_WORKER_FFMPEG%
echo [bench]   FFPROBE    = %FEATHERTALK_WORKER_FFPROBE%
echo [bench]   SCRFD_DIR  = %FEATHERTALK_WORKER_SCRFD_DIR%

for %%d in ("%FEATHERTALK_WORKER_SCRFD_DIR%" "%FEATHERTALK_WORKER_PFLD_DIR%" "%FEATHERTALK_WORKER_HUBERT_DIR%" "%FEATHERTALK_WORKER_VGG19_DIR%") do (
    if exist %%d (
        echo [bench]   model dir exists: %%d
    ) else (
        echo [warn] model dir missing: %%d
    )
)

if not exist "%FIXTURE%" (
    echo [err] fixture missing: %FIXTURE%
    exit /b 1
)

:: ================================================================
:: 6. Full pipeline
:: ================================================================
echo [bench] preparing full pipeline...

if not exist "target\benchmark" mkdir "target\benchmark"
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

echo [bench] input: %FIXTURE%
echo [bench] project root: %PROJECT_ROOT%
echo [bench] pipeline: probe_media -^> normalize_media -^> extract_frames -^> extract_features -^> lock_asset_package -^> train -^> render

:: ================================================================
:: 7. Build
:: ================================================================
if "%SKIP_BUILD%"=="1" (
    echo [warn] --skip-build specified
) else (
    echo [bench] building feathertalk-benchmark...
    cargo build --locked -p feathertalk-benchmark
    if errorlevel 1 (
        echo [err] build failed
        exit /b 1
    )
)

:: ================================================================
:: 8. Run per backend
:: ================================================================
echo [bench] running benchmark - repeats=%REPEATS%, backends=%BACKENDS%

for %%b in (%BACKENDS:,= %) do (
    set "CUR_BACKEND=%%b"

    if /i "%%b"=="cuda" (
        if "%HAS_NVIDIA%"=="0" (
            echo [warn] skip cuda - no NVIDIA GPU
            goto :next_backend
        )
    )
    if /i "%%b"=="rocm" (
        echo [warn] skip rocm - Windows ROCm not supported in this script
        goto :next_backend
    )

    echo [bench] ===== backend: %%b =====

    set "RUN_BACKEND=%%b"
    set "REPORT_JSON=%REPORT_DIR%\report-%%b.json"
    set "FEATHERTALK_WORKER_BACKEND=%%b"
    set "EXTRA_ARGS="
    if not "%MAX_OUTPUT_FRAMES%"=="0" set "EXTRA_ARGS=--max-output-frames %MAX_OUTPUT_FRAMES%"
    if "%KEEP_PROJECTS%"=="1" set "EXTRA_ARGS=!EXTRA_ARGS! --keep-projects"

    cargo run --locked -p feathertalk-benchmark -- ^
        --input "%FIXTURE%" ^
        --project-root "%PROJECT_ROOT%" ^
        --repeats %REPEATS% ^
        --epochs %EPOCHS% ^
        --label %%b ^
        --backend !RUN_BACKEND! ^
        --json !EXTRA_ARGS! > "!REPORT_JSON!"

    if errorlevel 1 (
        echo [warn] backend %%b failed, see !REPORT_JSON!
    ) else (
        echo [bench] report saved: !REPORT_JSON!
    )

    :next_backend
    echo.
)

:: ================================================================
:: 9. Summary
:: ================================================================
echo [bench] done. report dir: %REPORT_DIR%
dir /b "%REPORT_DIR%\*.json" 2>nul
if errorlevel 1 echo [warn] no JSON reports generated

endlocal
exit /b 0

:: ================================================================
:: Help
:: ================================================================
:usage
echo FeatherTalk Windows benchmark script
echo.
echo Usage:
echo   benchmark.bat [OPTIONS]
echo.
echo Options:
echo   --repo-url ^<URL^>     repository URL - default %REPO_URL%
echo   --repeats ^<N^>        warm repeats - default %REPEATS%
echo   --backends ^<LIST^>    comma-separated backends - default %BACKENDS%
echo                         choices: cpu,wgpu,cuda,rocm
echo   --project-root ^<PATH^> full-pipeline project root - default %PROJECT_ROOT%
echo   --epochs ^<N^>         training epochs - default %EPOCHS%
echo   --max-output-frames ^<N^> render frame cap; 0 renders the full fixture
echo   --keep-projects       keep successful run project directories
echo   --skip-ffmpeg         skip FFmpeg install
echo   --skip-build          skip cargo build
echo   -h, --help            show this help
echo.
echo Examples:
echo   benchmark.bat
echo   benchmark.bat --backends cpu,wgpu --repeats 5
echo   benchmark.bat --skip-ffmpeg
echo.
echo Output:
echo   %REPORT_DIR%\report-^<backend^>.json
exit /b 0
