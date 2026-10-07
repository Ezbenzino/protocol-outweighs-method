# nnU-Net 2D training pipeline for LIDC-IDRI
# Requires: data converted by scripts/convert_to_nnunet.py
# Usage: .\run_nnunet.ps1

$env:PYTHON = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$env:NNUNET_SCRIPTS = (if ($env:SEG_PYTHON) { Join-Path (Split-Path $env:SEG_PYTHON) 'Scripts' } else { 'Scripts' })
$env:PROJECT_DIR = $PSScriptRoot

# nnU-Net environment variables
$env:nnUNet_raw = "$env:PROJECT_DIR\data\nnunet"
$env:nnUNet_preprocessed = "$env:PROJECT_DIR\data\nnunet_preprocessed"
$env:nnUNet_results = "$env:PROJECT_DIR\outputs\nnunet_results"

New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null

Set-Location $env:PROJECT_DIR

Write-Host "=== Step 1: Plan and preprocess ===" -ForegroundColor Cyan
& "$env:NNUNET_SCRIPTS\nnUNetv2_plan_and_preprocess.exe" -d 101 --verify_dataset_integrity
Write-Host "Plan and preprocess exit code: $LASTEXITCODE"

Write-Host "`n=== Step 2: Train 2D U-Net (5-fold CV) ===" -ForegroundColor Cyan
for ($fold = 0; $fold -lt 5; $fold++) {
    Write-Host "`n--- Training fold $fold ---" -ForegroundColor Yellow
    & "$env:NNUNET_SCRIPTS\nnUNetv2_train.exe" 101 2d $fold
    Write-Host "Fold $fold exit code: $LASTEXITCODE"
}

Write-Host "`n=== Step 3: Find best configuration ===" -ForegroundColor Cyan
& "$env:NNUNET_SCRIPTS\nnUNetv2_find_best_configuration.exe" 101 -c 2d

Write-Host "`n=== nnU-Net pipeline complete ===" -ForegroundColor Green
