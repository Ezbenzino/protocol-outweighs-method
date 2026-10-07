# Batch training v2: rerun failed experiments (C drive now has 42GB free)
# Runs: union_fold3, union_fold4, 4 weight sweeps, plain_unet, unet3d
$env:PYTHON = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$env:PROJECT_DIR = $PSScriptRoot
Set-Location $env:PROJECT_DIR

function Run-Training {
    param([string]$Name, [string]$Config, [string]$Base = "configs/default.yaml", [string]$SplitTag = "")
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STARTING: $Name" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    $args = @("scripts/train.py", "--config", $Config, "--base", $Base, "--name", $Name)
    if ($SplitTag) { $args += @("--split-tag", $SplitTag) }
    & $env:PYTHON @args
    Write-Host "COMPLETED: $Name (exit code: $LASTEXITCODE)" -ForegroundColor Green
}

function Run-Training3D {
    param([string]$Name, [string]$Config)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STARTING 3D: $Name" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    & $env:PYTHON scripts/train_3d.py --config $Config --name $Name
    Write-Host "COMPLETED: $Name (exit code: $LASTEXITCODE)" -ForegroundColor Green
}

$startTime = Get-Date
Write-Host "Batch experiments v2 started at $startTime" -ForegroundColor Yellow

# --- Remaining union folds ---
Run-Training -Name "union_fold3" -Config "configs/ablation/union_target.yaml" -SplitTag "fold3"
Run-Training -Name "union_fold4" -Config "configs/ablation/union_target.yaml" -SplitTag "fold4"

# --- Weight sweep (4 configs) ---
Run-Training -Name "weight_brbc01_lov01" -Config "configs/ablation/weight_brbc01_lov01.yaml"
Run-Training -Name "weight_brbc01_lov05" -Config "configs/ablation/weight_brbc01_lov05.yaml"
Run-Training -Name "weight_brbc10_lov01" -Config "configs/ablation/weight_brbc10_lov01.yaml"
Run-Training -Name "weight_brbc10_lov05" -Config "configs/ablation/weight_brbc10_lov05.yaml"

# --- Plain U-Net baseline ---
Run-Training -Name "plain_unet" -Config "configs/plain_unet.yaml"

# --- 3D U-Net baseline ---
Run-Training3D -Name "unet3d" -Config "configs/unet3d.yaml"

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "ALL BATCH EXPERIMENTS V2 COMPLETE" -ForegroundColor Yellow
Write-Host "Total duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
