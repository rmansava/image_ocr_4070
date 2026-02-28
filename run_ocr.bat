@echo off
REM ==========================================================================
REM  Comic OCR Pipeline - processes all images under T:\archiverelated\comics
REM  Writes .txt files next to each image with extracted text
REM  Skips images already processed - safe to re-run at any time
REM  Auto-restarts on failure (waits 30s between retries)
REM ==========================================================================

REM --- Ollama performance settings (Vulkan + parallel requests) ---
set OLLAMA_NUM_PARALLEL=4
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KEEP_ALIVE=24h
set OLLAMA_VULKAN=1
set OLLAMA_LLM_LIBRARY=vulkan
set HIP_VISIBLE_DEVICES=-1

REM --- Kill existing Ollama so it restarts with our settings ---
taskkill /IM ollama.exe /F >NUL 2>&1
timeout /t 3 /nobreak >NUL

echo Starting Ollama...
start "" "C:\Users\rick\AppData\Local\Programs\Ollama\ollama.exe" serve
echo Waiting 10 seconds for Ollama to start...
timeout /t 10 /nobreak >NUL

REM --- Run the OCR pipeline in a restart loop ---
echo.
echo Starting OCR on T:\archiverelated\comics
echo Workers: 4 (Vulkan)
echo Max image dimension: 1024px
echo Staging: copies batches of 500 to C: SSD before processing
echo Output: .txt files next to each image on NAS
echo Auto-restart: enabled (30s delay on failure)
echo.
echo Press Ctrl+C to stop. Progress is saved - you can re-run to resume.
echo.

:loop
python -m image_ocr "T:\archiverelated\comics" --workers 4 --max-dim 1024 --stage-dir "C:\Users\rick\image_ocr\staging" --batch-size 500
if %errorlevel% equ 0 goto done

echo.
echo [%date% %time%] Pipeline exited with error code %errorlevel%. Restarting in 30 seconds...
echo.

REM Make sure Ollama is still running, restart if not
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >NUL
if %errorlevel% neq 0 (
    echo Ollama is not running. Restarting Ollama...
    start "" "C:\Users\rick\AppData\Local\Programs\Ollama\ollama.exe" serve
    echo Waiting 10 seconds for Ollama to start...
    timeout /t 10 /nobreak >NUL
) else (
    timeout /t 30 /nobreak >NUL
)

goto loop

:done
echo.
echo OCR pipeline finished successfully.
pause
