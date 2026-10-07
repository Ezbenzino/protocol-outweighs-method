# Standalone plain_unet launcher - runs python as independent process
# Output redirected to files, not dependent on PowerShell session
$ErrorActionPreference = "Continue"
$projectDir = $PSScriptRoot
$python = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$stdoutLog = "$projectDir\outputs\runs\plain_unet_stdout.txt"
$stderrLog = "$projectDir\outputs\runs\plain_unet_stderr.txt"

Set-Location $projectDir
$env:PYTHONUNBUFFERED = "1"

Write-Host "Starting plain_unet as independent process..."
Write-Host "Stdout: $stdoutLog"
Write-Host "Stderr: $stderrLog"

$proc = Start-Process -FilePath $python `
    -ArgumentList "scripts/train.py","--config","configs/plain_unet.yaml","--base","configs/default.yaml","--name","plain_unet" `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -NoNewWindow -PassThru

Write-Host "Process ID: $($proc.Id)"
Write-Host "Use Get-Process -Id $($proc.Id) to check status"
Write-Host "Check train.log for progress"
