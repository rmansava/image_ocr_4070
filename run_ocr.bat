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

python -m image_ocr "%SCAN_DIR%" --quantize 4bit
if %errorlevel% equ 0 goto done

echo.
echo [%date% %time%] Pipeline exited with error code %errorlevel%.
echo Check ocr_scan.log for details.
echo Re-run this script to resume.
goto end

:done
echo.
echo Pipeline finished successfully.

:end
pause
