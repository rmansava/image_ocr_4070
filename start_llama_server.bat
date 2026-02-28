@echo off
echo Starting llama-server with Vulkan backend...
"C:\Users\rick\image_ocr\llama-server\llama-server.exe" -m "C:\Users\rick\.ollama\models\blobs\sha256-65493e1f85b9ea4ba3ed793515fde13cbdbea7d74ad2c662b566b146eab0081e" -ngl 999 --port 8080 -c 4096 --parallel 4
echo.
echo Exit code: %ERRORLEVEL%
pause
