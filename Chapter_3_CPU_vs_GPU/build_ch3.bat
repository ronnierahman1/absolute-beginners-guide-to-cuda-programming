@echo off
setlocal ENABLEDELAYEDEXPANSION
setlocal ENABLEEXTENSIONS

rem ============================================================
rem  CUDA Chapter 3 Build Script (Windows .bat)
rem
rem  Usage:
rem    build_ch3.bat                 (build all, release, sm_70)
rem    build_ch3.bat MODE=debug      (debug flags -G -O0 -lineinfo)
rem    build_ch3.bat ARCH=sm_86      (set GPU arch)
rem    build_ch3.bat vector_add_verification   (build one target)
rem    build_ch3.bat clean           (remove bin\)
rem    build_ch3.bat help            (show help)
rem
rem  Notes:
rem    - Requires NVCC in PATH (CUDA Toolkit).
rem    - Outputs .exe files into bin\
rem ============================================================

rem -------- Default variables (can be overridden via args) ----
set "ARCH=sm_70"
set "MODE=release"
set "BIN_DIR=bin"
set "NVCC=nvcc"
set "STDFLAG=-std=c++14"

rem -------- Targets (filenames must match .cu names) ----------
set TARGETS=device_alloc_free memcpy_roundtrip vector_add_minimal vector_add_workflow vector_add_with_checks vector_add_verification

rem -------- Parse key=value args and single-target arg --------
set "SINGLE_TARGET="

for %%A in (%*) do (
  echo %%A | findstr /b /c:"ARCH=" >nul  && (for /f "tokens=2 delims==" %%X in ("%%A") do set "ARCH=%%~X" & goto :continue_arch) &:continue_arch
  echo %%A | findstr /b /c:"MODE=" >nul  && (for /f "tokens=2 delims==" %%X in ("%%A") do set "MODE=%%~X" & goto :continue_mode) &:continue_mode
)

rem If the last non key=value token is present, treat as SINGLE_TARGET / verb
for %%A in (%*) do (
  echo %%A | findstr /r /c:"^[A-Za-z0-9_][A-Za-z0-9_]*$" >nul
  if not errorlevel 1 (
    echo %%A | findstr /b /c:"ARCH=" >nul || echo %%A | findstr /b /c:"MODE=" >nul || set "SINGLE_TARGET=%%A"
  )
)

rem -------- Help / Clean handling -----------------------------
if /I "%SINGLE_TARGET%"=="help" (
  call :help
  exit /b 0
)
if /I "%SINGLE_TARGET%"=="clean" (
  call :clean
  exit /b 0
)

rem -------- Check for nvcc ------------------------------------
where %NVCC% >nul 2>nul
if errorlevel 1 (
  echo [ERROR] nvcc not found in PATH. Install CUDA Toolkit or open a "x64 Native Tools Command Prompt" with CUDA in PATH.
  echo        Example CUDA bin path: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
  exit /b 1
)

rem -------- Flags by MODE -------------------------------------
if /I "%MODE%"=="debug" (
  set "NVCCFLAGS=%STDFLAG% -G -O0 -lineinfo"
) else (
  set "NVCCFLAGS=%STDFLAG% -O2 -lineinfo"
)

rem -------- Create bin\ if needed ------------------------------
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

echo.
echo === CUDA Chapter 3 Build ===
echo ARCH   : %ARCH%
echo MODE   : %MODE%
echo NVCC   : %NVCC%
echo OUTDIR : %BIN_DIR%
echo.

rem -------- Build single target or all ------------------------
if defined SINGLE_TARGET (
  call :build_one "%SINGLE_TARGET%"
  exit /b %ERRORLEVEL%
) else (
  for %%T in (%TARGETS%) do (
    call :build_one "%%T"
    if errorlevel 1 exit /b 1
  )
)

echo.
echo Build complete (%MODE%, %ARCH%). EXEs are in "%BIN_DIR%".
exit /b 0

::--------------------------------------------------------------
:: build_one <targetname>
:: Compiles .\targetname.cu -> .\bin\targetname.exe
::--------------------------------------------------------------
:build_one
setlocal
set "TNAME=%~1"
if "%TNAME%"=="" (
  echo [ERROR] No target specified to build_one.
  exit /b 1
)

set "SRC=%TNAME%.cu"
set "OUT=%BIN_DIR%\%TNAME%.exe"

if not exist "%SRC%" (
  echo [WARN] Skipping "%TNAME%" (source "%SRC%" not found).
  endlocal & exit /b 0
)

echo.
echo   NVCC  %SRC%  ->  %OUT%
%NVCC% %NVCCFLAGS% -arch=%ARCH% "%SRC%" -o "%OUT%"
if errorlevel 1 (
  echo [ERROR] Build failed for %TNAME%.
  endlocal & exit /b 1
)

echo   OK    %TNAME%
endlocal & exit /b 0

::--------------------------------------------------------------
:: clean
::--------------------------------------------------------------
:clean
if exist "%BIN_DIR%" (
  echo Deleting "%BIN_DIR%"\ ...
  rmdir /s /q "%BIN_DIR%"
) else (
  echo Nothing to clean.
)
exit /b 0

::--------------------------------------------------------------
:: help
::--------------------------------------------------------------
:help
echo.
echo CUDA Chapter 3 Windows Build Script
echo -----------------------------------
echo Usage:
echo   build_ch3.bat [ARCH=sm_86] [MODE=debug^|release] [target^|all^|clean^|help]
echo.
echo Examples:
echo   build_ch3.bat
echo   build_ch3.bat MODE=debug
echo   build_ch3.bat ARCH=sm_86
echo   build_ch3.bat vector_add_verification
echo   build_ch3.bat clean
echo.
echo Targets:
echo   device_alloc_free
echo   memcpy_roundtrip
echo   vector_add_minimal
echo   vector_add_workflow
echo   vector_add_with_checks
echo   vector_add_verification
echo.
echo Notes:
echo   - Ensure nvcc is in PATH (e.g., "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin").
echo   - Output binaries are written to .\bin\
echo.
exit /b 0
