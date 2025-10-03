@echo off
rem ============================================================
rem build_ch6.bat — Chapter 6 (Teams and Scratchpads)
rem Builds/runs chapter6_demo (min + argmin with shared memory)
rem
rem Usage:
rem   build_ch6.bat build           - compile the demo
rem   build_ch6.bat run [N]         - run the demo with optional problem size N
rem   build_ch6.bat clean           - remove build artifacts
rem   build_ch6.bat info            - print configuration
rem   build_ch6.bat                 - default: build
rem ============================================================

setlocal EnableExtensions EnableDelayedExpansion

:: ---- Toolchain ----
set "NVCC=nvcc"

:: ---- Configuration (override by setting env vars before calling) ----
if not defined ARCH set "ARCH=sm_70"
if not defined NVFLAGS set "NVFLAGS=-O2 -std=c++14 -Xcompiler /W3"
if not defined BIN_DIR set "BIN_DIR=bin"

:: ---- Sources/Outputs ----
set "SRC=chapter6_demo.cu"
set "EXE=%BIN_DIR%\chapter6_demo.exe"

:: ---- Ensure nvcc exists ----
where %NVCC% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Install CUDA Toolkit or add nvcc to PATH.
  exit /b 1
)

:: ---- Helper: ensure bin dir ----
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%" >nul 2>&1

:: ---- Commands ----
if /i "%~1"=="clean" (
  if exist "%BIN_DIR%" rmdir /s /q "%BIN_DIR%"
  echo Cleaned.
  exit /b 0
)

if /i "%~1"=="info" (
  echo NVCC   = %NVCC%
  echo ARCH   = %ARCH%
  echo NVFLAGS= %NVFLAGS%
  echo SRC    = %SRC%
  echo EXE    = %EXE%
  exit /b 0
)

if /i "%~1"=="run" (
  if not exist "%EXE%" (
    echo [INFO] Executable not found, building first...
    goto :build
  ) else (
    goto :run
  )
)

if /i "%~1"=="build" (
  goto :build
)

:: Default action: build
:build
echo [BUILD] %SRC% -> %EXE%
"%NVCC%" %NVFLAGS% -arch=%ARCH% "%SRC%" -o "%EXE%"
if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)
echo [OK] Built "%EXE%"
if /i "%~1"=="build" exit /b 0

:: Optional fallthrough to run if user specified 'run'
:run
set "N=%~2"
if defined N (
  echo [RUN] %EXE% %N%
  "%EXE%" %N%
) else (
  echo [RUN] %EXE%
  "%EXE%"
)
exit /b %errorlevel%
