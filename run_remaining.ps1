# Remaining experiments after plain_unet completion
# 1. weight_brbc01_lov01 (rerun - was interrupted)
# 2. unet3d baseline
$env:PYTHON = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot

$startTime = Get-Date
Write-Host "=== Remaining experiments started at $startTime ===" -ForegroundColor Yellow

# --- weight_brbc01_lov01 (rerun) ---
Write-Host "`n[1/2] Starting weight_brbc01_lov01 (rerun)..." -ForegroundColor Cyan
& $env:PYTHON scripts/train.py --config configs/ablation/weight_brbc01_lov01.yaml --base configs/default.yaml --name weight_brbc01_lov01
Write-Host "weight_brbc01_lov01 exited with code $LASTEXITCODE" -ForegroundColor Green

# --- unet3d baseline ---
Write-Host "`n[2/2] Starting unet3d baseline..." -ForegroundColor Cyan
& $env:PYTHON scripts/train_3d.py --config configs/unet3d.yaml --name unet3d
Write-Host "unet3d exited with code $LASTEXITCODE" -ForegroundColor Green

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n=== All remaining experiments complete in $($duration.ToString('hh\:mm\:ss')) ===" -ForegroundColor Yellow
