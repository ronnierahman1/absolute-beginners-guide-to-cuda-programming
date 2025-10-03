@echo off
setlocal enabledelayedexpansion

REM ============================================
REM Build script for Chapter 5 programs (Windows)
REM - bin\invert_pgm.exe     (CUDA)
REM - bin\gradient_pgm.exe   (C++)
REM Requirements:
REM   - NVIDIA CUDA Toolkit (nvcc in PATH)
REM   - A C++ compiler (cl from VS Dev Prompt OR g++/clang++)
REM ============================================

set SRCDIR=%~dp0
set BINDIR=%SRCDIR%bin
set CUDA_SRC=%SRCDIR%invert_pgm.cu
set CPP_SRC=%SRCDIR%gradient_pgm.cpp

set NVCC_FLAGS=-O2 -std=c++17
set CPP_FLAGS=-O2 -std=c++17

if not exist "%BINDIR%" mkdir "%BINDIR%"

echo.
echo === Checking tools ===
where nvcc >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Install CUDA Toolkit or open "x64 Native Tools Command Prompt for VS" after setting CUDA.
  exit /b 1
) else (
  for /f "delims=" %%i in ('nvcc --version ^| findstr /i "release"') do echo [nvcc] %%i
)

REM Prefer cl if available; else try g++; else try clang++
set CXX=
where cl >NUL 2>&1 && set CXX=cl
if "%CXX%"=="" ( where g++ >NUL 2>&1 && set CXX=g++ )
if "%CXX%"=="" ( where clang++ >NUL 2>&1 && set CXX=clang++ )

if "%CXX%"=="" (
  echo [ERROR] No C++ compiler found (cl, g++, or clang++). Install one or use a Dev Prompt.
  exit /b 1
) else (
  echo [CXX] Using: %CXX%
)

echo.
echo === Building invert_pgm (CUDA) ===
nvcc %NVCC_FLAGS% "%CUDA_SRC%" -o "%BINDIR%\invert_pgm.exe"
if errorlevel 1 (
  echo [ERROR] Failed to build invert_pgm.exe
  exit /b 1
)

echo.
echo === Building gradient_pgm (C++) ===
if /i "%CXX%"=="cl" (
  REM cl uses different flags/output handling
  pushd "%BINDIR%"
  "%CXX%" /nologo /O2 /std:c++17 "%CPP_SRC%" /Fe:gradient_pgm.exe
  set ERR=%ERRORLEVEL%
  popd
  if not "%ERR%"=="0" (
    echo [ERROR] Failed to build gradient_pgm.exe with cl
    exit /b 1
  )
) else (
  "%CXX%" %CPP_FLAGS% "%CPP_SRC%" -o "%BINDIR%\gradient_pgm.exe"
  if errorlevel 1 (
    echo [ERROR] Failed to build gradient_pgm.exe
    exit /b 1
  )
)

echo.
echo === Build Succeeded ===
echo   %BINDIR%\invert_pgm.exe
echo   %BINDIR%\gradient_pgm.exe
echo.
echo Usage:
echo   invert_pgm.exe input.pgm output_negative.pgm [--verify]
echo   gradient_pgm.exe WIDTH HEIGHT h^|v out.pgm
echo.

endlocal
