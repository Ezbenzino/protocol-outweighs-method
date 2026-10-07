# One-click data preparation: parse XML -> process to npz -> split
$ErrorActionPreference = "Stop"

$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root = $PSScriptRoot
$Raw  = Join-Path $Root "data\raw\LIDC-IDRI"
$Proc = Join-Path $Root "data\processed"
$Splits = Join-Path $Root "data\splits"

if (-not (Test-Path $Raw)) {
    Write-Host "[error] Raw LIDC-IDRI data not found at:" -ForegroundColor Red
    Write-Host "        $Raw" -ForegroundColor Red
    Write-Host "        Run .\run_download.ps1 first (or put the case folders there)."
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $Proc "npz") | Out-Null
New-Item -ItemType Directory -Force -Path $Splits                 | Out-Null

Write-Host "[1/3] Parsing LIDC XML annotations..." -ForegroundColor Cyan
& $Py (Join-Path $Root "preprocess\parse_lidc_xml.py") --raw_dir $Raw --out (Join-Path $Proc "annotations.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] Windowing slices + consensus -> npz ..." -ForegroundColor Cyan
& $Py (Join-Path $Root "preprocess\process_lidc.py") --raw-dir $Raw --annotations (Join-Path $Proc "annotations.json") --out-dir (Join-Path $Proc "npz")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/3] Splitting cases into train/val/test..." -ForegroundColor Cyan
& $Py (Join-Path $Root "preprocess\make_splits.py") --npz_dir (Join-Path $Proc "npz") --out_dir $Splits --stratify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Data preparation done." -ForegroundColor Green
