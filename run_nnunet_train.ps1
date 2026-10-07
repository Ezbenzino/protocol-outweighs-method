# Dataset root holding the raw LIDC-IDRI / nnU-Net trees; override with $env:SEG_DATA_ROOT.
$DataRoot = if ($env:SEG_DATA_ROOT) { $env:SEG_DATA_ROOT } else { Join-Path $HOME 'datasets' }
# nnU-Net 2D training for Dataset101_LIDC, fold 0
$env:nnUNet_raw = "$DataRoot\nnunet"
$env:nnUNet_preprocessed = "$DataRoot\nnunet_preprocessed"
$env:nnUNet_results = "$DataRoot\nnunet_results"
$env:PYTHONUNBUFFERED = "1"
# Try to limit workers via env vars (nnU-Net v2 may use these)
$env:nnUNet_n_proc_DA = "2"
$env:nnUNet_num_workers = "2"
$env:OMP_NUM_THREADS = "2"

$nnUNetTrain = 'nnUNetv2_train'

Write-Host "=== nnU-Net 2D Training ===" -ForegroundColor Cyan
Write-Host "Dataset: 101 (LIDC_IDRI)"
Write-Host "Config: 2d"
Write-Host "Fold: 0"
Write-Host "Start time: $(Get-Date)"
Write-Host ""

& $nnUNetTrain 101 2d 0

Write-Host ""
Write-Host "Training exited with code: $LASTEXITCODE"
Write-Host "End time: $(Get-Date)"
