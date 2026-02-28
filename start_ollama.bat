@echo off
REM Force Ollama to use Vulkan backend (not ROCm) for proper unified memory detection
REM Vulkan sees 62GB on Strix Halo vs ROCm's incorrect 1GB
set OLLAMA_NUM_PARALLEL=4
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KEEP_ALIVE=24h
set OLLAMA_VULKAN=1
set OLLAMA_LLM_LIBRARY=vulkan
REM Disable ROCm/HIP so Ollama doesn't prefer it over Vulkan
set HIP_VISIBLE_DEVICES=-1
echo Starting Ollama with Vulkan backend (OLLAMA_LLM_LIBRARY=vulkan)
start "" "C:\Users\rick\AppData\Local\Programs\Ollama\ollama.exe" serve
