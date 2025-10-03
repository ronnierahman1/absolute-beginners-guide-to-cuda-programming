@echo off
REM ============================================================
REM Build & Run batch for Chapter 8 — Monte Carlo π (CUDA)
REM Usage:
REM   build_run.bat                 -> build (Release, sm_60), no run
REM   build_run.bat run             -> build then run with defaults
REM   build_run.bat run 100000000   -> run with SAMPLES=1e8, TPB default (256)
REM   build_run.bat run 100000000 512 -> run with SAMPLES=1e8, TPB=512
REM   build_run.bat clean           -> remove bin\monte_carlo_pi.exe
REM   build_run.bat debug           -> build Debug (sm_60)
REM   build_run.bat sm 86           -> build Release for sm_86
REM ============================================================

setlocal ENABLEDELAYEDEXPANSION

REM ---- Defaults ------------------------------------------------
set "NVCC=nvcc"
set "SM=60"                     REM e.g., 53 (Jetson Nano), 75 (Turing), 86 (Ampere), 90 (Ada/Lovelace)
set "BUILD_TYPE=Release"        REM Release or Debug
set "SRC=monte_carlo_pi.cu"
set "BIN=bin"
set "TARGET=%BIN%\monte_carlo_pi.exe"
set "SAMPLES=50000000"
set "TPB=256"

REM ---- Check nvcc exists --------------------------------------
where %NVCC% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Please install CUDA Toolkit or add nvcc to PATH.
  exit /b 1
)

REM ---- Parse simple commands ----------------------------------
if /I "%~1"=="clean"  goto :CLEAN
if /I "%~1"=="debug"  ( set "BUILD_TYPE=Debug" & goto :BUILD )
if /I "%~1"=="sm"     (
  if "%~2"=="" ( echo [ERROR] Missing SM number. Example: build_run.bat sm 86 & exit /b 1 )
  set "SM=%~2"
  shift & shift
)

if /I "%~1"=="run" (
  if not "%~2"=="" set "SAMPLES=%~2"
  if not "%~3"=="" set "TPB=%~3"
  goto :BUILD_AND_RUN
)

REM default: build only
goto :BUILD

:BUILD_AND_RUN
call :BUILD || exit /b 1
echo.
echo [RUN] %TARGET% %SAMPLES% %TPB%
"%TARGET%" %SAMPLES% %TPB%
exit /b %ERRORLEVEL%

:BUILD
echo [INFO] NVCC       = %NVCC%
echo [INFO] SM         = %SM%
echo [INFO] BUILD_TYPE = %BUILD_TYPE%
echo [INFO] SRC        = %SRC%
echo [INFO] TARGET     = %TARGET%

if not exist "%BIN%" mkdir "%BIN%" >nul 2>&1

set "COMMON_FLAGS=-std=c++14"
if /I "%BUILD_TYPE%"=="Debug" (
  set "NVCCFLAGS=-O0 -g -G -lineinfo %COMMON_FLAGS%"
) else (
  set "NVCCFLAGS=-O3 -lineinfo %COMMON_FLAGS%"
)

REM Link cuRAND (present on all CUDA installs)
set "LIBS=-lcurand"

echo.
echo [BUILD] %NVCC% -arch=sm_%SM% %NVCCFLAGS% -o %TARGET% %SRC% %LIBS%
%NVCC% -arch=sm_%SM% %NVCCFLAGS% -o "%TARGET%" "%SRC%" %LIBS%
if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)
echo [OK] Built %TARGET%
exit /b 0

:CLEAN
if exist "%TARGET%" (
  del /q "%TARGET%"
  echo [OK] Removed %TARGET%
) else (
  echo [INFO] Nothing to clean.
)
exit /b 0
