@echo off
REM ============================================================
REM build_ch2.bat — Build & run Chapter 2 CUDA examples (Windows)
REM
REM Programs:
REM   - vector_scale_10x_plus_5.cu    -> bin\vec_scale.exe
REM   - too_many_threads_demo.cu      -> bin\demo.exe
REM   - compare_cpu_gpu_scale.cu      -> bin\compare.exe
REM   - run_grid_block_experiments.cu -> bin\grid_expts.exe
REM
REM Usage (from the folder containing the .cu files):
REM   build_ch2.bat                 (build all)
REM   build_ch2.bat clean           (remove bin)
REM   build_ch2.bat vec             (build vec_scale only)
REM   build_ch2.bat demo            (build demo only)
REM   build_ch2.bat compare         (build compare only)
REM   build_ch2.bat grid            (build grid_expts only)
REM   build_ch2.bat run-vec         (run vec_scale)
REM   build_ch2.bat run-demo        (run demo)
REM   build_ch2.bat run-compare     (run compare)
REM   build_ch2.bat run-grid        (run grid_expts)
REM   build_ch2.bat run-grid 37 "32,64,128"    (pass args)
REM
REM Customize:
REM   Set SM (e.g., 70, 75, 86, 89, 90) and NVCC path below if needed.
REM ============================================================

setlocal ENABLEDELAYEDEXPANSION

REM ---- Config ----
if "%SM%"=="" set SM=86
if "%NVCC%"=="" set NVCC=nvcc

set ARCH=-arch=sm_%SM%
set NVCCFLAGS=-O2 -std=c++17 %ARCH% -Xcompiler "/W3"

set BIN=bin
set SRC_VEC=vector_scale_10x_plus_5.cu
set SRC_DEMO=too_many_threads_demo.cu
set SRC_COMP=compare_cpu_gpu_scale.cu
set SRC_GRID=run_grid_block_experiments.cu

set EXE_VEC=%BIN%\vec_scale.exe
set EXE_DEMO=%BIN%\demo.exe
set EXE_COMP=%BIN%\compare.exe
set EXE_GRID=%BIN%\grid_expts.exe

REM ---- Ensure nvcc is available ----
where %NVCC% >nul 2>nul
if errorlevel 1 (
  echo [ERROR] nvcc not found. Add it to PATH or set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin\nvcc.exe"
  exit /b 1
)

REM ---- Ensure bin dir exists ----
if not exist "%BIN%" mkdir "%BIN%"

REM ---- Helper: compile one .cu -> .exe ----
REM call :build_one source.cu out.exe
:build_one
  if not exist "%~1" (
    echo [ERROR] Source not found: %~1
    exit /b 1
  )
  echo [BUILD] %~1 -> %~2
  "%NVCC%" %NVCCFLAGS% "%~1" -o "%~2"
  if errorlevel 1 (
    echo [ERROR] Build failed: %~1
    exit /b 1
  )
  exit /b 0

REM ---- Commands ----
if "%~1"=="" goto :build_all

if /I "%~1"=="clean" (
  echo [CLEAN] Removing %BIN%
  rmdir /S /Q "%BIN%" 2>nul
  goto :eof
)

if /I "%~1"=="vec" (
  call :build_one "%SRC_VEC" "%EXE_VEC"
  goto :eof
)

if /I "%~1"=="demo" (
  call :build_one "%SRC_DEMO" "%EXE_DEMO"
  goto :eof
)

if /I "%~1"=="compare" (
  call :build_one "%SRC_COMP" "%EXE_COMP"
  goto :eof
)

if /I "%~1"=="grid" (
  call :build_one "%SRC_GRID" "%EXE_GRID"
  goto :eof
)

if /I "%~1"=="run-vec" (
  if not exist "%EXE_VEC%" call :build_one "%SRC_VEC" "%EXE_VEC%"
  echo [RUN] %EXE_VEC%
  "%EXE_VEC%"
  goto :eof
)

if /I "%~1"=="run-demo" (
  if not exist "%EXE_DEMO%" call :build_one "%SRC_DEMO" "%EXE_DEMO%"
  echo [RUN] %EXE_DEMO%
  "%EXE_DEMO%"
  goto :eof
)

if /I "%~1"=="run-compare" (
  if not exist "%EXE_COMP%" call :build_one "%SRC_COMP" "%EXE_COMP%"
  echo [RUN] %EXE_COMP% %~2 %~3
  "%EXE_COMP%" %~2 %~3
  goto :eof
)

if /I "%~1"=="run-grid" (
  if not exist "%EXE_GRID%" call :build_one "%SRC_GRID" "%EXE_GRID%"
  REM Forward up to two args to grid_expts: N and "TPB_CSV[,blocks]"
  echo [RUN] %EXE_GRID% %~2 %~3
  "%EXE_GRID%" %~2 %~3
  goto :eof
)

echo [INFO] Unknown command: %~1
echo Usage:
echo   build_ch2.bat ^| clean ^| vec ^| demo ^| compare ^| grid ^| run-vec ^| run-demo ^| run-compare ^| run-grid
goto :eof

REM ---- Default: build all ----
:build_all
call :build_one "%SRC_VEC"  "%EXE_VEC%"
call :build_one "%SRC_DEMO" "%EXE_DEMO%"
call :build_one "%SRC_COMP" "%EXE_COMP%"
call :build_one "%SRC_GRID" "%EXE_GRID%"
echo [OK] All built in %BIN%
echo.
echo Tips:
echo   set SM=90 ^&^& build_ch2.bat       (build for sm_90)
echo   build_ch2.bat run-compare          (run compare with defaults)
echo   build_ch2.bat run-grid 37 "32,64,128"
