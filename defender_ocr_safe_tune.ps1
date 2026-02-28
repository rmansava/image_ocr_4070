param(
    [ValidateSet('Apply','Restore','Status')]
    [string]$Mode = 'Apply',

    [ValidateRange(5,100)]
    [int]$ScanCpuLoad = 20,

    [string]$BackupPath = "$PSScriptRoot\defender_ocr_tune_backup.json"
)

$ErrorActionPreference = 'Stop'

function Test-IsAdmin {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-TuneSnapshot {
    $pref = Get-MpPreference
    [ordered]@{
        Timestamp = (Get-Date).ToString('o')
        ScanAvgCPULoadFactor = [int]$pref.ScanAvgCPULoadFactor
        EnableLowCpuPriority = [bool]$pref.EnableLowCpuPriority
        ThrottleForScheduledScanOnly = [bool]$pref.ThrottleForScheduledScanOnly
        ScanOnlyIfIdleEnabled = [bool]$pref.ScanOnlyIfIdleEnabled
        DisableRealtimeMonitoring = [bool]$pref.DisableRealtimeMonitoring
    }
}

function Show-Status {
    $status = Get-TuneSnapshot
    $status.GetEnumerator() | ForEach-Object {
        "{0}={1}" -f $_.Key, $_.Value
    }
}

if (-not (Test-IsAdmin)) {
    throw 'Run this script in an elevated PowerShell session (Run as Administrator).'
}

switch ($Mode) {
    'Status' {
        Show-Status
        return
    }

    'Apply' {
        $before = Get-TuneSnapshot
        $before | ConvertTo-Json -Depth 4 | Set-Content -Path $BackupPath -Encoding UTF8

        Set-MpPreference -ScanAvgCPULoadFactor ([byte]$ScanCpuLoad)
        Set-MpPreference -EnableLowCpuPriority $true
        Set-MpPreference -ThrottleForScheduledScanOnly $true
        Set-MpPreference -ScanOnlyIfIdleEnabled $true

        "BackupWritten=$BackupPath"
        "Applied=true"
        Show-Status
        return
    }

    'Restore' {
        if (-not (Test-Path -Path $BackupPath)) {
            throw "Backup file not found: $BackupPath"
        }

        $backup = Get-Content -Path $BackupPath -Raw | ConvertFrom-Json

        Set-MpPreference -ScanAvgCPULoadFactor ([byte]$backup.ScanAvgCPULoadFactor)
        Set-MpPreference -EnableLowCpuPriority ([bool]$backup.EnableLowCpuPriority)
        Set-MpPreference -ThrottleForScheduledScanOnly ([bool]$backup.ThrottleForScheduledScanOnly)
        Set-MpPreference -ScanOnlyIfIdleEnabled ([bool]$backup.ScanOnlyIfIdleEnabled)

        "RestoredFrom=$BackupPath"
        Show-Status
        return
    }
}
