Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Reload user env vars into current session
$env:OLLAMA_NUM_PARALLEL = [System.Environment]::GetEnvironmentVariable('OLLAMA_NUM_PARALLEL', 'User')
$env:OLLAMA_FLASH_ATTENTION = [System.Environment]::GetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', 'User')
$env:OLLAMA_VULKAN = [System.Environment]::GetEnvironmentVariable('OLLAMA_VULKAN', 'User')

Write-Host "Starting Ollama with:"
Write-Host "  OLLAMA_NUM_PARALLEL=$env:OLLAMA_NUM_PARALLEL"
Write-Host "  OLLAMA_FLASH_ATTENTION=$env:OLLAMA_FLASH_ATTENTION"
Write-Host "  OLLAMA_VULKAN=$env:OLLAMA_VULKAN"

Start-Process "C:\Users\rick\AppData\Local\Programs\Ollama\ollama.exe" -ArgumentList "serve"
Start-Sleep -Seconds 8

$ps = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ps) {
    Write-Host "Ollama running (PID: $($ps.Id))"
} else {
    Write-Host "ERROR: Ollama failed to start"
}
