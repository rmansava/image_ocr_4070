@echo off
setlocal

echo ==========================================================
echo  Image OCR Pipeline
echo  Scans images recursively, writes .txt to T:\archive
echo  Progress saved per batch - Ctrl+C to stop, re-run to resume
echo ==========================================================
echo.

set /p "SCAN_DIR=Directory to scan (e.g. T:\archiverelated\albums): "

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
echo Scanning: %SCAN_DIR% (all subfolders)
echo.

:loop
python -m image_ocr "%SCAN_DIR%"
if %errorlevel% equ 0 goto done

echo.
echo [%date% %time%] Pipeline exited with error code %errorlevel%. Restarting in 30 seconds...
echo.
timeout /t 30 /nobreak >NUL
goto loop

:done
echo.
echo Pipeline finished successfully.
pause
