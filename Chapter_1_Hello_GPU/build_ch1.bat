@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem Absolute Beginners Guide to CUDA Programming in C/C++
rem Chapter 1 — Build/Run Helper (Windows)
rem ----------------------------------------------------------------------------
rem Usage:
rem   build_ch1.bat                 -> build all programs
rem   build_ch1.bat run             -> build and run all programs
rem   build_ch1.bat clean           -> remove executables
rem   build_ch1.bat help            -> show this help
rem
rem   build_ch1.bat hello           -> build just hello
rem   build_ch1.bat run-hello       -> build & run just hello
rem   (same for: hello_grid, hello_independent, cpu_reference_check)
rem
rem Requirements:
rem   - NVIDIA CUDA Toolkit installed (so "nvcc" is in PATH)
rem   - Run from a "x64 Native Tools" or Developer Command Prompt (recommended)
rem ============================================================================

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

rem ---- Check for nvcc --------------------------------------------------------
where nvcc >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] "nvcc" not found in PATH.
  echo         Please install the CUDA Toolkit or add its "bin" folder to PATH.
  echo         Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
  echo.
  goto :end_fail
)

rem ---- Settings --------------------------------------------------------------
set NVCC=nvcc
set CFLAGS=-O2

rem ---- Targets ---------------------------------------------------------------
set TARGETS=hello hello_grid hello_independent cpu_reference_check

rem ---- Entry point -----------------------------------------------------------
if "%~1"==""         goto :build_all
if /I "%~1"=="build" goto :build_all
if /I "%~1"=="run"   goto :run_all
if /I "%~1"=="clean" goto :clean
if /I "%~1"=="help"  goto :help

rem Per-program shortcuts:
if /I "%~1"=="hello"                 call :build_one hello & goto :end
if /I "%~1"=="hello_grid"            call :build_one hello_grid & goto :end
if /I "%~1"=="hello_independent"     call :build_one hello_independent & goto :end
if /I "%~1"=="cpu_reference_check"   call :build_one cpu_reference_check & goto :end

if /I "%~1"=="run-hello"               call :build_one hello               && call :run_one hello               & goto :end
if /I "%~1"=="run-hello_grid"          call :build_one hello_grid          && call :run_one hello_grid          & goto :end
if /I "%~1"=="run-hello_independent"   call :build_one hello_independent   && call :run_one hello_independent   & goto :end
if /I "%~1"=="run-cpu_reference_check" call :build_one cpu_reference_check && call :run_one cpu_reference_check & goto :end

echo.
echo [ERROR] Unknown command "%~1"
echo        Try: build_ch1.bat help
echo.
goto :end_fail

rem ----------------------------------------------------------------------------
:help
echo.
echo Absolute Beginners Guide to CUDA Programming in C/C++ — Chapter 1
echo.
echo Commands:
echo   build_ch1.bat               ^> build all programs
echo   build_ch1.bat run           ^> build and run all programs
echo   build_ch1.bat clean         ^> remove executables
echo   build_ch1.bat help          ^> show this help
echo.
echo   build_ch1.bat hello                 ^> build just "hello"
echo   build_ch1.bat run-hello             ^> build and run "hello"
echo   (also: hello_grid, hello_independent, cpu_reference_check)
echo.
goto :end

rem ----------------------------------------------------------------------------
:build_all
echo.
echo === Building all Chapter 1 programs ===
for %%T in (%TARGETS%) do (
  call :build_one %%T || goto :end_fail
)
echo Done.
goto :end

rem ----------------------------------------------------------------------------
:run_all
echo.
echo === Building all Chapter 1 programs (then running) ===
for %%T in (%TARGETS%) do (
  call :build_one %%T || goto :end_fail
)
echo.
for %%T in (%TARGETS%) do (
  call :run_one %%T || goto :end_fail
)
echo.
echo All programs ran successfully.
goto :end

rem ----------------------------------------------------------------------------
:build_one
rem %1 = target name without extension (expects a matching .cu)
set NAME=%~1
set SRC=%NAME%.cu
set EXE=%NAME%.exe

if not exist "%SRC%" (
  echo [WARN] Source not found: %SRC%  (skipping)
  exit /b 0
)

echo.
echo [BUILD] %SRC%  ->  %EXE%
"%NVCC%" %CFLAGS% "%SRC%" -o "%EXE%"
if errorlevel 1 (
  echo [FAIL] Build failed for %SRC%
  exit /b 1
)
echo [OK] Built %EXE%
exit /b 0

rem ----------------------------------------------------------------------------
:run_one
rem %1 = target name (expects .exe present)
set NAME=%~1
set EXE=%NAME%.exe

if not exist "%EXE%" (
  echo [WARN] Executable not found: %EXE%  (building now)
  call :build_one %NAME% || exit /b 1
)

echo.
echo ---- Running: %EXE% ----
"%EXE%"
if errorlevel 1 (
  echo [FAIL] Program returned a non-zero exit code.
  exit /b 1
)
exit /b 0

rem ----------------------------------------------------------------------------
:clean
echo.
set ERR=0
for %%T in (%TARGETS%) do (
  if exist "%%T.exe" (
    del /q "%%T.exe" || set ERR=1
  )
)
if %ERR%==0 (echo Cleaned.) else (echo Some files could not be removed.)
goto :end

rem ----------------------------------------------------------------------------
:end
popd
exit /b 0

:end_fail
popd
exit /b 1
