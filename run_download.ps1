# Dataset root holding the raw LIDC-IDRI / nnU-Net trees; override with $env:SEG_DATA_ROOT.
$DataRoot = if ($env:SEG_DATA_ROOT) { $env:SEG_DATA_ROOT } else { Join-Path $HOME 'datasets' }
param([int]$Workers = 32)
# Download LIDC-IDRI (CT images + original XML annotations).
# Resumable: re-run the same command to skip already-downloaded files.
# Data is written to $DataRoot\LIDC-IDRI and mapped to data\raw\LIDC-IDRI via a junction.
$ErrorActionPreference = "Stop"

$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root = $PSScriptRoot
& $Py (Join-Path $Root "preprocess\download_lidc.py") --workers $Workers
exit $LASTEXITCODE
