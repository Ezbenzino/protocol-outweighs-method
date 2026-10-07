# ---------------------------------------------------------------------------
# Second fix from the 2026-10-02 audit: re-run the window-offset ladder.
#
# Problem 1: diag_offset_allarms.json / diag_offset.json were produced on
#   2026-08-21 with thresholds from an early calibration (the old
#   outputs\analysis\symmetric_analysis.json, now overwritten), e.g. arm B used
#   0.50, arm C 0.50, arm D 0.25, while the calibrated thresholds reported in
#   the paper (Table 3, outputs\analysis_full) are 0.55, 0.65 and 0.50.
# Problem 2: the plain U-Net in the ten-arm ladder is archPU_fold0 (lr = 1e-4,
#   not the selected configuration); it must be archPU1e3_fold0.
#
# This script backs up the two JSON files and re-runs both ladders with the
# thresholds of outputs\analysis_full\symmetric_analysis.json.
# Expected time on the RTX 5060 laptop: about 5-10 minutes (21 model runs,
# 76 test-set nodules x 8 offsets x 6 directions each).
# Run from the repository root: powershell -ExecutionPolicy Bypass -File .\run_fix_offset_20261002.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
Set-Location $PSScriptRoot
$thr = "outputs\analysis_full\symmetric_analysis.json"

$bk = "outputs\_backup_offset_20261002"
if (-not (Test-Path $bk)) {
  New-Item -ItemType Directory $bk | Out-Null
  Copy-Item "outputs\analysis_scope\diag_offset_allarms.json" $bk
  Copy-Item "outputs\analysis_scope\diag_offset.json" $bk
  Write-Host "[backup] -> $bk"
} else { Write-Host "[backup] $bk already exists, not overwritten" }

Write-Host ""
Write-Host "==== [1/2] Ten arms + translation arm, fold-0 models  $(Get-Date -Format HH:mm:ss) ===="
$runs = "tgtA_fold0,tgtB_fold0,tgtC_fold0,tgtD_fold0,tgtE_fold0,tgtF_fold0,tgtBshift_fold0,archR18_fold0,archCNX_fold0,archPVT_fold0,archPU1e3_fold0"
& $py scripts\diag_offset.py --runs $runs --thr-from $thr --out "outputs\analysis_scope\diag_offset_allarms.json"
if ($LASTEXITCODE -ne 0) { throw "diag_offset (all arms) failed" }

Write-Host ""
Write-Host "==== [2/2] Arm B and translation arm, five fold models  $(Get-Date -Format HH:mm:ss) ===="
$runs5 = "tgtB_fold0,tgtB_fold1,tgtB_fold2,tgtB_fold3,tgtB_fold4,tgtBshift_fold0,tgtBshift_fold1,tgtBshift_fold2,tgtBshift_fold3,tgtBshift_fold4"
& $py scripts\diag_offset.py --runs $runs5 --thr-from $thr --out "outputs\analysis_scope\diag_offset.json"
if ($LASTEXITCODE -ne 0) { throw "diag_offset (five folds) failed" }

Write-Host ""
Write-Host "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). The figures and the manuscript are updated from these two files."
