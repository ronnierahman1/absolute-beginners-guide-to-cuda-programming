@echo off
REM ============================================================
REM build_ch9.bat — Chapter 9 (CUDA + CSV utils) for Windows
REM
REM Usage examples:
REM   build_ch9                 -> build all binaries into .\bin
REM   build_ch9 chapter9        -> build main all-in-one program
REM   build_ch9 test            -> build SMA test harness
REM   build_ch9 utils           -> build CSV utilities
REM   build_ch9 run N=1048576 W=5 EDGE=0.0 ALPHA=0.2 THRESH=64
REM   build_ch9 run_csv CSV=data.csv COL=1 OP=sma WIN=5 EDGE=0.0 OUT=out.csv THRESH=64
REM   build_ch9 clean           -> remove .\bin
REM
REM Requirements:
REM   - NVIDIA CUDA Toolkit (nvcc in PATH)
REM   - MSVC (cl in PATH) or g++ (MinGW) for the tiny C++ utils
REM ============================================================

setlocal enableextensions enabledelayedexpansion

REM ---- Defaults (override via env or KEY=VALUE pairs with `build run ...`) ----
if not defined SM set SM=75
if not defined OPT set OPT=/O2
if not defined CXXSTD set CXXSTD=/std:c++17

set NVCC=nvcc
set CPPCL=cl
set GPP=g++

set GEN=-gencode arch=compute_%SM%,code=sm_%SM%
set NVCCFLAGS=-Xcompiler "/W3" %GEN% -std=c++17

set BIN=bin
if not exist "%BIN%" mkdir "%BIN%"

REM ---- Parse first arg as target ----
set TARGET=%1%
if "%TARGET%"=="" set TARGET=all

REM ---- Helper: parse KEY=VALUE from the rest of args into env vars ----
:parse_kv
shift
if "%~1"=="" goto after_kv
for /f "tokens=1,2 delims==" %%A in ("%~1") do (
  set "%%~A=%%~B"
)
goto parse_kv
:after_kv

REM ---- Detect tools ----
where %NVCC% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Install CUDA Toolkit or add nvcc to PATH.
  exit /b 1
)

where %CPPCL% >nul 2>&1
if errorlevel 1 (
  REM cl not found; try g++
  where %GPP% >nul 2>&1
  if errorlevel 1 (
    echo [WARN] Neither 'cl' nor 'g++' found. CSV utilities will be skipped.
    set HAVE_CPP=0
  ) else (
    set HAVE_CPP=2
  )
) else (
  set HAVE_CPP=1
)

REM ---- Files ----
set CU_MAIN=chapter9_timeseries.cu
set CU_BASIC=moving_average.cu
set CU_AUTO=moving_average_auto.cu
set CU_TEST=moving_average_test.cu

set CPP_HEAD=csv_head.cpp
set CPP_PLOT=csv_to_plot.cpp

set BIN_MAIN=%BIN%\chapter9_timeseries.exe
set BIN_BASIC=%BIN%\moving_average.exe
set BIN_AUTO=%BIN%\moving_average_auto.exe
set BIN_TEST=%BIN%\moving_average_test.exe
set BIN_HEAD=%BIN%\csv_head.exe
set BIN_PLOT=%BIN%\csv_to_plot.exe

goto %TARGET% 2>nul || goto help

:all
call :build_main || exit /b 1
call :build_basic || exit /b 1
call :build_auto  || exit /b 1
call :build_test  || exit /b 1
call :build_utils
echo.
echo [OK] All targets built in %BIN%
exit /b 0

:chapter9
call :build_main
exit /b %errorlevel%

:test
call :build_test
exit /b %errorlevel%

:utils
call :build_utils
exit /b %errorlevel%

:clean
if exist "%BIN%" rd /s /q "%BIN%"
echo [OK] Cleaned %BIN%
exit /b 0

:run
if not exist "%BIN_MAIN%" call :build_main || exit /b 1
if "%N%"=="" set N=1048576
if "%W%"=="" set W=5
if "%EDGE%"=="" set EDGE=0.0
if "%ALPHA%"=="" set ALPHA=0.2
if "%THRESH%"=="" set THRESH=64
echo Running: %BIN_MAIN% %N% %W% %EDGE% %ALPHA% %THRESH%
"%BIN_MAIN%" %N% %W% %EDGE% %ALPHA% %THRESH%
exit /b %errorlevel%

:run_csv
if not exist "%BIN_MAIN%" call :build_main || exit /b 1
if "%CSV%"=="" (
  echo Usage: build run_csv CSV=path\in.csv OUT=path\out.csv [COL=1] [OP=sma^|ema] [WIN=5] [EDGE=0.0] [ALPHA=0.2] [THRESH=64]
  exit /b 2
)
if "%OUT%"=="" (
  echo Usage: build run_csv CSV=path\in.csv OUT=path\out.csv ...
  exit /b 2
)
if "%COL%"=="" set COL=1
if "%OP%"=="" set OP=sma
if "%WIN%"=="" set WIN=5
if "%EDGE%"=="" set EDGE=0.0
if "%ALPHA%"=="" set ALPHA=0.2
if "%THRESH%"=="" set THRESH=64

if /i "%OP%"=="ema" (
  echo Running EMA CSV pipeline...
  "%BIN_MAIN%" --csv "%CSV%" --col %COL% --op ema --alpha %ALPHA% --out "%OUT%"
) else (
  echo Running SMA CSV pipeline...
  "%BIN_MAIN%" --csv "%CSV%" --col %COL% --op sma --win %WIN% --edge %EDGE% --thresh %THRESH% --out "%OUT%"
)
exit /b %errorlevel%

REM ============================================================
REM Build functions
REM ============================================================
:build_main
if not exist "%BIN%" mkdir "%BIN%"
if not exist "%CU_MAIN%" (
  echo [ERROR] Missing %CU_MAIN%
  exit /b 1
)
echo [NVCC] %CU_MAIN%  ->  %BIN_MAIN%
%NVCC% %NVCCFLAGS% -o "%BIN_MAIN%" "%CU_MAIN%"
exit /b %errorlevel%

:build_basic
if not exist "%CU_BASIC%" goto :eof
echo [NVCC] %CU_BASIC% -> %BIN_BASIC%
%NVCC% %NVCCFLAGS% -o "%BIN_BASIC%" "%CU_BASIC%"
exit /b %errorlevel%

:build_auto
if not exist "%CU_AUTO%" goto :eof
echo [NVCC] %CU_AUTO% -> %BIN_AUTO%
%NVCC% %NVCCFLAGS% -o "%BIN_AUTO%" "%CU_AUTO%"
exit /b %errorlevel%

:build_test
if not exist "%CU_TEST%" goto :eof
echo [NVCC] %CU_TEST% -> %BIN_TEST%
%NVCC% %NVCCFLAGS% -o "%BIN_TEST%" "%CU_TEST%"
exit /b %errorlevel%

:build_utils
if "%HAVE_CPP%"=="0" (
  echo [WARN] Skipping CSV utilities (no C++ compiler found).
  exit /b 0
)
if "%HAVE_CPP%"=="1" (
  if exist "%CPP_HEAD%" (
    echo [CL ] %CPP_HEAD% -> %BIN_HEAD%
    "%CPPCL%" /nologo %OPT% %CXXSTD% "%CPP_HEAD%" /Fe"%BIN_HEAD%"
  )
  if exist "%CPP_PLOT%" (
    echo [CL ] %CPP_PLOT% -> %BIN_PLOT%
    "%CPPCL%" /nologo %OPT% %CXXSTD% "%CPP_PLOT%" /Fe"%BIN_PLOT%"
  )
  exit /b 0
)
if "%HAVE_CPP%"=="2" (
  if exist "%CPP_HEAD%" (
    echo [G++] %CPP_HEAD% -> %BIN_HEAD%
    "%GPP%" -O2 -std=c++17 "%CPP_HEAD%" -o "%BIN_HEAD%"
  )
  if exist "%CPP_PLOT%" (
    echo [G++] %CPP_PLOT% -> %BIN_PLOT%
    "%GPP%" -O2 -std=c++17 "%CPP_PLOT%" -o "%BIN_PLOT%"
  )
  exit /b 0
)
exit /b 0

:help
echo.
echo build_ch9.bat targets:
echo   all          - build all CUDA programs and CSV utilities
echo   chapter9     - build main all-in-one program
echo   test         - build SMA test harness
echo   utils        - build CSV utilities
echo   run          - run main program (use N, W, EDGE, ALPHA, THRESH)
echo   run_csv      - run CSV pipeline (CSV, OUT, COL, OP, WIN, EDGE, ALPHA, THRESH)
echo   clean        - delete .\bin
echo.
echo Examples:
echo   build_ch9
echo   build_ch9 chapter9
echo   build_ch9 run N=1048576 W=9 EDGE=nan ALPHA=0.2 THRESH=64
echo   build_ch9 run_csv CSV=data\prices.csv OUT=out_sma.csv OP=sma WIN=9 EDGE=0.0
echo   build_ch9 run_csv CSV=data\sensor.csv OUT=out_ema.csv OP=ema ALPHA=0.15
exit /b 0
