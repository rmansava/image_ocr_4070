@echo off
setlocal

echo ==========================================================
echo  Image OCR Pipeline (vLLM batched engine)
echo  Qwen3-VL-8B @ 1280px  ^|  rep_penalty=1.15
echo  Progress saved per batch - Ctrl+C to stop, re-run to resume
echo ==========================================================
echo.

echo Step 1: Starting vLLM server in WSL...
echo (A WSL window will open. Keep it running while processing.)
echo.
start "vLLM Server" wsl -e bash -c "bash /mnt/c/Users/rmans/image_ocr_4070/start_vllm_server.sh"

echo Step 2: Waiting for vLLM server to be ready (~10 min for model load)...
echo.
cd /d "%~dp0"
python wait_for_vllm.py
if %errorlevel% neq 0 (
    echo ERROR: Could not connect to vLLM server. Check the WSL window for errors.
    pause
    exit /b 1
)

echo.
set /p "SCAN_DIR=Directory to scan (e.g. T:\archiverelated\board games): "

if "%SCAN_DIR%"=="" (
    echo No directory entered. Exiting.
    pause
    exit /b 1
)

if not exist "%SCAN_DIR%" (
    echo Directory not found: %SCAN_DIR%
    pause
    exit /b 1
)

echo.
echo Scanning: %SCAN_DIR%
echo.

python -m image_ocr "%SCAN_DIR%" --engine vllm
if %errorlevel% equ 0 (
    echo.
    echo Pipeline finished successfully.
) else (
    echo.
    echo Pipeline exited with error. Re-run this script to resume.
)

:end
pause
