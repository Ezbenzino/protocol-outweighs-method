# Batch training script: runs all remaining experiments sequentially
# Usage: .\run_batch_experiments.ps1
# Estimated total time: 5-8 hours

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
Write-Host "Batch experiments started at $startTime" -ForegroundColor Yellow

# --- Phase 2a: Union target 5-fold CV (folds 1-4, fold0 already running) ---
Run-Training -Name "union_fold1" -Config "configs/ablation/union_target.yaml" -SplitTag "fold1"
Run-Training -Name "union_fold2" -Config "configs/ablation/union_target.yaml" -SplitTag "fold2"
Run-Training -Name "union_fold3" -Config "configs/ablation/union_target.yaml" -SplitTag "fold3"
Run-Training -Name "union_fold4" -Config "configs/ablation/union_target.yaml" -SplitTag "fold4"

# --- Phase 2b: Auxiliary loss weight sweep (4 corner configs) ---
Run-Training -Name "weight_brbc01_lov01" -Config "configs/ablation/weight_brbc01_lov01.yaml"
Run-Training -Name "weight_brbc01_lov05" -Config "configs/ablation/weight_brbc01_lov05.yaml"
Run-Training -Name "weight_brbc10_lov01" -Config "configs/ablation/weight_brbc10_lov01.yaml"
Run-Training -Name "weight_brbc10_lov05" -Config "configs/ablation/weight_brbc10_lov05.yaml"

# --- Phase 2c: Standard plain U-Net baseline ---
Run-Training -Name "plain_unet" -Config "configs/plain_unet.yaml"

# --- Phase 3: 3D U-Net baseline ---
Run-Training3D -Name "unet3d" -Config "configs/unet3d.yaml"

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "ALL BATCH EXPERIMENTS COMPLETE" -ForegroundColor Yellow
Write-Host "Total duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
