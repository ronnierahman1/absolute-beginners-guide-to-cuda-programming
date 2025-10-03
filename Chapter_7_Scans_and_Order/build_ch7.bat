
@echo off
setlocal enableextensions enabledelayedexpansion

REM =====================================================
REM  CUDA Chapter 7 Build Script
REM  This script compiles and links all CUDA demo and verification
REM  programs for Chapter 7: Scans and Order.
REM
REM  Files built:
REM    - demo_scan_apps.cu         (main demo)
REM    - demo_scan_verify_apps.cu  (with --verify, for §7.6)
REM    - scan_global_stitch.cu     (scan kernel)
REM    - verify_scan.cu            (extra verification)
REM
REM  Usage:
REM    build_ch7.bat [SM]
REM      [SM] = optional CUDA compute capability (e.g. 86 for sm_86)
REM    build_ch7.bat clean
REM
REM  Output:
REM    Binaries in .\bin\
REM =====================================================


REM Check for clean command
if /i "%~1"=="clean" goto :clean


REM Optional: set CUDA architecture (e.g. 86 for sm_86)
set "SM=%~1"
set "NVCC=nvcc"
set "CUDACFLAGS=-O2 -std=c++14 -lineinfo"
set "GENCODE="


REM If SM is set, add -gencode for that architecture
if not "%SM%"=="" (
  set "GENCODE=-gencode arch=compute_%SM%,code=sm_%SM%"
)


REM Check for nvcc in PATH
where nvcc >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Open a Developer Command Prompt with CUDA or add nvcc to PATH.
  exit /b 1
)


REM Create build and bin folders if needed
if not exist build mkdir build
if not exist bin   mkdir bin


REM Step 1: Compile scan_global_stitch.cu
echo [1/6] Compiling scan_global_stitch.cu ...
%NVCC% %CUDACFLAGS% %GENCODE% -c scan_global_stitch.cu -o build\scan_global_stitch.obj
if errorlevel 1 goto :err


REM Step 2: Compile verify_scan.cu
echo [2/6] Compiling verify_scan.cu ...
%NVCC% %CUDACFLAGS% %GENCODE% -c verify_scan.cu -o build\verify_scan.obj
if errorlevel 1 goto :err


REM Step 3: Compile demo_scan_apps.cu
echo [3/6] Compiling demo_scan_apps.cu ...
%NVCC% %CUDACFLAGS% %GENCODE% -c demo_scan_apps.cu -o build\demo_scan_apps.obj
if errorlevel 1 goto :err


REM Step 4: Compile demo_scan_verify_apps.cu with HAVE_VERIFY
echo [4/6] Compiling demo_scan_verify_apps.cu (HAVE_VERIFY) ...
%NVCC% %CUDACFLAGS% %GENCODE% -DHAVE_VERIFY -c demo_scan_verify_apps.cu -o build\demo_scan_verify_apps.obj
if errorlevel 1 goto :err


REM Step 5: Link demo.exe
echo [5/6] Linking bin\demo.exe ...
%NVCC% %CUDACFLAGS% %GENCODE% build\demo_scan_apps.obj build\scan_global_stitch.obj -o bin\demo.exe
if errorlevel 1 goto :err


REM Step 6: Link demo_verify.exe
echo [6/6] Linking bin\demo_verify.exe ...
%NVCC% %CUDACFLAGS% %GENCODE% build\demo_scan_verify_apps.obj build\scan_global_stitch.obj build\verify_scan.obj -o bin\demo_verify.exe
if errorlevel 1 goto :err


REM Success message and usage
echo.
echo [OK] Build complete. Binaries are in .\bin
echo     Run: bin\demo.exe --mode compact --N 1000 --BLOCK 256
echo     Or : bin\demo_verify.exe --verify --mode hist --N 5000 --BINS 64
exit /b 0


:clean
REM Clean build artifacts
echo Cleaning build artifacts...
if exist build rmdir /s /q build
if exist bin   rmdir /s /q bin
echo Clean complete.
exit /b 0


:err
REM Error handler
echo.
echo [FAIL] Build step failed. See errors above.
exit /b 1
