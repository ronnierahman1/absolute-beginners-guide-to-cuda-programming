@echo off
REM ch10_build — Build & demo Chapter 10 Mandelbrot programs on Windows
REM Usage:
REM   ch10_build build
REM   ch10_build run-samples
REM   ch10_build convert
REM   ch10_build clean
REM   ch10_build all   (build + run-samples + convert)

setlocal

REM --- Config ---------------------------------------------------------------
set NVCC=nvcc
set PYTHON=python
set BIN=bin
set OUT=out

set SRC_GRAY=mandelbrot.cu
set SRC_COLOR=mandelbrot_color.cu
set SRC_FINAL=mandelbrot_final.cu
set SRC_PACK=mandelbrot_final_palette_pack.cu

set EXE_GRAY=%BIN%\mandelbrot.exe
set EXE_COLOR=%BIN%\mandelbrot_color.exe
set EXE_FINAL=%BIN%\mandelbrot_final.exe
set EXE_PACK=%BIN%\mandelbrot_final_palette_pack.exe

set PGM_OUT=%OUT%\mandelbrot.pgm
set PPM_OUT=%OUT%\mandelbrot.ppm
set PNG_GRAY=%OUT%\mandelbrot.png
set PNG_COLOR=%OUT%\mandelbrot_color.png

REM --- Parse command --------------------------------------------------------
if "%~1"=="" goto :help
if /I "%~1"=="build"       goto :build
if /I "%~1"=="run-samples" goto :runSamples
if /I "%~1"=="convert"     goto :convert
if /I "%~1"=="clean"       goto :clean
if /I "%~1"=="all"         goto :all

:help
echo.
echo Usage:
echo   %~nx0 build        ^(compile all CUDA programs^)
echo   %~nx0 run-samples  ^(render sample PGM/PPM outputs into .\out^)
echo   %~nx0 convert      ^(convert .pgm/.ppm to .png using pgm2png.py / ppm2png.py^)
echo   %~nx0 clean        ^(remove .\bin and .\out^)
echo   %~nx0 all          ^(build + run-samples + convert^)
echo.
exit /b 0

REM --- Helpers --------------------------------------------------------------
:needTools
where %NVCC% >nul 2>nul || (echo [ERROR] nvcc not found on PATH. Install CUDA Toolkit or open an NVIDIA CUDA shell.& exit /b 1)
mkdir "%BIN%" 2>nul
mkdir "%OUT%" 2>nul
exit /b 0

:build
call :needTools || exit /b 1
echo [build] %SRC_GRAY%  -> %EXE_GRAY%
%NVCC% "%SRC_GRAY%" -o "%EXE_GRAY%" || (echo [ERROR] build failed.& exit /b 1)

echo [build] %SRC_COLOR% -> %EXE_COLOR%
%NVCC% "%SRC_COLOR%" -o "%EXE_COLOR%" || (echo [ERROR] build failed.& exit /b 1)

echo [build] %SRC_FINAL% -> %EXE_FINAL%
%NVCC% "%SRC_FINAL%" -o "%EXE_FINAL%" || (echo [ERROR] build failed.& exit /b 1)

echo [build] %SRC_PACK%  -> %EXE_PACK%
%NVCC% "%SRC_PACK%" -o "%EXE_PACK%" || (echo [ERROR] build failed.& exit /b 1)

echo [build] done.
exit /b 0

:runSamples
call :build || exit /b 1
echo [render] grayscale -> %PGM_OUT%
"%EXE_GRAY%" --width 1024 --height 768 --max-iter 200 --out "%PGM_OUT%"

echo [render] color (HSV) -> %PPM_OUT%
"%EXE_COLOR%" --width 1024 --height 768 --max-iter 500 --out "%PPM_OUT%"

echo [render] palette-pack (fire) -> %OUT%\mandelbrot_fire.ppm
"%EXE_PACK%" --width 1920 --height 1080 --max-iter 800 --color --smooth --binary ^
  --palette fire --out "%OUT%\mandelbrot_fire.ppm"

echo [render] done. Outputs in "%OUT%".
exit /b 0

:convert
REM Convert only if the helper scripts are present
if exist pgm2png.py (
  echo [convert] PGM -> PNG
  %PYTHON% pgm2png.py "%PGM_OUT%" "%PNG_GRAY%"
) else (
  echo [convert] Skipping pgm2png.py (not found)
)

if exist ppm2png.py (
  echo [convert] PPM -> PNG
  %PYTHON% ppm2png.py "%PPM_OUT%" "%PNG_COLOR%"
) else (
  echo [convert] Skipping ppm2png.py (not found)
)

echo [convert] done.
exit /b 0

:clean
echo [clean] removing "%BIN%" and "%OUT%"
rmdir /s /q "%BIN%" 2>nul
rmdir /s /q "%OUT%" 2>nul
echo [clean] done.
exit /b 0

:all
call :runSamples || exit /b 1
call :convert
exit /b 0
