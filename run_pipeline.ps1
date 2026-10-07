# Full pipeline: download -> preprocess -> train (resumable).
# Safe to re-run: download skips existing files, preprocess skips existing npz.
$ErrorActionPreference = "Stop"

$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root   = $PSScriptRoot
$Raw    = Join-Path $Root "data\raw\LIDC-IDRI"
$Proc   = Join-Path $Root "data\processed"
$Splits = Join-Path $Root "data\splits"
$Config = Join-Path $Root "configs\default.yaml"

Write-Host "=== [1/3] Downloading LIDC-IDRI (resumable) ===" -ForegroundColor Cyan
& $Py (Join-Path $Root "preprocess\download_lidc.py") --workers 32
if ($LASTEXITCODE -ne 0) { Write-Host "download failed, exit $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "=== [2/3] Preprocessing (parse -> npz -> split) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path (Join-Path $Proc "npz") | Out-Null
New-Item -ItemType Directory -Force -Path $Splits                 | Out-Null

& $Py (Join-Path $Root "preprocess\parse_lidc_xml.py") --raw_dir $Raw --out (Join-Path $Proc "annotations.json")
if ($LASTEXITCODE -ne 0) { Write-Host "parse failed"; exit $LASTEXITCODE }

& $Py (Join-Path $Root "preprocess\process_lidc.py") --raw-dir $Raw --annotations (Join-Path $Proc "annotations.json") --out-dir (Join-Path $Proc "npz") --workers 4
if ($LASTEXITCODE -ne 0) { Write-Host "process failed"; exit $LASTEXITCODE }

& $Py (Join-Path $Root "preprocess\make_splits.py") --npz_dir (Join-Path $Proc "npz") --out_dir $Splits --stratify
if ($LASTEXITCODE -ne 0) { Write-Host "split failed"; exit $LASTEXITCODE }

Write-Host "=== [3/3] Training ===" -ForegroundColor Cyan
& $Py (Join-Path $Root "scripts\train.py") --config $Config
exit $LASTEXITCODE
