@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Build and run Chapter 4 CUDA programs on Windows (NVCC)
REM Programs:
REM   - poly_44_avoid_redundancy.cu  -> bin\poly44.exe
REM   - poly_45_precision.cu         -> bin\poly45.exe
REM   - poly_46_testing.cu           -> bin\poly46.exe
REM
REM Usage:
REM   build.bat                      ^&^& builds all
REM   build.bat run44                ^&^& builds + runs poly44
REM   build.bat run45
REM   build.bat run46
REM   build.bat clean
REM
REM Options (key=value anywhere on the command line):
REM   ARCH=sm_80 | sm_86 | sm_90 ...
REM   DEBUG=0|1          (1 uses -G -g -O0 for device debug)
REM   FAST_MATH=0|1      (1 adds -use_fast_math for float)
REM   CXXSTD=c++17       (host C++ standard)
REM Examples:
REM   build.bat ARCH=sm_86
REM   build.bat run45 FAST_MATH=1
REM   build.bat DEBUG=1
REM ============================================================

REM -- Defaults (can be overridden by key=value args)
set "ARCH=sm_80"
set "DEBUG=0"
set "FAST_MATH=0"
set "CXXSTD=c++17"
set "CMD=all"

REM -- Parse command line: first non key=value token is the command
for %%A in (%*) do (
  for /f "tokens=1* delims==" %%K in ("%%~A") do (
    if not "%%L"=="" (
      set "%%K=%%L"
    ) else (
      if /I "!CMD!"=="all" set "CMD=%%K"
    )
  )
)

REM -- Check nvcc availability
where nvcc >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Install CUDA Toolkit or add nvcc to PATH.
  exit /b 1
)

REM -- Compose NVCC flags
set "NVCCFLAGS=-std=%CXXSTD% -arch=%ARCH% -lineinfo"
if "%DEBUG%"=="1" (
  set "NVCCFLAGS=-std=%CXXSTD% -arch=%ARCH% -G -g -O0 -lineinfo"
) else (
  set "NVCCFLAGS=%NVCCFLAGS% -O3 --expt-relaxed-constexpr"
)
if "%FAST_MATH%"=="1" set "NVCCFLAGS=%NVCCFLAGS% -use_fast_math"
REM MSVC host: basic exception model; quiet warnings can be adjusted
set "NVCCFLAGS=%NVCCFLAGS% -Xcompiler ^"/EHsc^""

REM -- Ensure bin directory
if not exist "bin" mkdir "bin"

if /I "%CMD%"=="all"  goto :build_all
if /I "%CMD%"=="run44" goto :run44
if /I "%CMD%"=="run45" goto :run45
if /I "%CMD%"=="run46" goto :run46
if /I "%CMD%"=="clean" goto :clean
if /I "%CMD%"=="help"  goto :help

REM Unknown command -> help
echo [INFO] Unknown command "%CMD%". Showing help.
goto :help

:build_all
call :build "poly_44_avoid_redundancy.cu" "bin\poly44.exe" || exit /b 1
call :build "poly_45_precision.cu"         "bin\poly45.exe" || exit /b 1
call :build "poly_46_testing.cu"           "bin\poly46.exe" || exit /b 1
echo.
echo [OK] Built all targets into .\bin
exit /b 0

:run44
call :build "poly_44_avoid_redundancy.cu" "bin\poly44.exe" || exit /b 1
echo.
echo [RUN] bin\poly44.exe
echo ------------------------------------------------------------
bin\poly44.exe
exit /b %errorlevel%

:run45
call :build "poly_45_precision.cu" "bin\poly45.exe" || exit /b 1
echo.
echo [RUN] bin\poly45.exe
echo ------------------------------------------------------------
bin\poly45.exe
exit /b %errorlevel%

:run46
call :build "poly_46_testing.cu" "bin\poly46.exe" || exit /b 1
echo.
echo [RUN] bin\poly46.exe
echo ------------------------------------------------------------
bin\poly46.exe
exit /b %errorlevel%

:build
REM %1 = source .cu, %2 = output .exe
set "SRC=%~1"
set "OUT=%~2"
if not exist "%SRC%" (
  echo [ERROR] Missing source: %SRC%
  exit /b 1
)
echo [NVCC] %SRC%
nvcc %NVCCFLAGS% -o "%OUT%" "%SRC%"
if errorlevel 1 (
  echo [ERROR] nvcc failed for %SRC%
  exit /b 1
)
exit /b 0

:clean
if exist "bin" (
  echo [CLEAN] Removing .\bin
  rmdir /s /q "bin"
) else (
  echo [CLEAN] Nothing to do.
)
exit /b 0

:help
echo Usage: build.bat [command] [key=value ...]
echo.
echo Commands:
echo   all       Build all programs ^(default^)
echo   run44     Build and run poly_44_avoid_redundancy
echo   run45     Build and run poly_45_precision
echo   run46     Build and run poly_46_testing
echo   clean     Remove .\bin directory
echo.
echo Options:
echo   ARCH=sm_80   GPU arch (e.g., sm_75 sm_80 sm_86 sm_90)
echo   DEBUG=0|1    1 enables device debug flags
echo   FAST_MATH=0|1 1 adds -use_fast_math
echo   CXXSTD=c++17  Host C++ standard
exit /b 0
