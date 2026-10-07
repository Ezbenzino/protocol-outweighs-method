# ---------------------------------------------------------------------------
# Fix found in the 2026-10-02 audit (test-set plain U-Net arm -> archPU1e3).
#
# Problem: on the held-out test set, the "Plain U-Net" arm is archPU_fold0-4,
# i.e. the lr = 1e-4 runs of 2026-08-20. The paper (and the cross-validation
# analysis) use the selected lr = 1e-3 runs, archPU1e3_fold0-4, which were
# trained on 2026-08-30, after the test set had been evaluated (2026-08-21).
# Because archPU is absent from the validation threshold table, the test
# analysis also fell back to choosing that arm's thresholds on the test set.
#
# This script: backs up outputs\analysis_test, evaluates archPU1e3 on the
# test split, removes the stale archPU rows, re-runs the test-set analysis
# with thresholds imported from the validation folds, and re-runs every
# downstream step that uses the test-set architecture range.
#
# Expected time on the RTX 5060 laptop: about 15-30 minutes in total
# (GPU inference is a few minutes; the rest is CPU work on a ~600 MB CSV).
# Run from the repository root:  .\run_fix_testset_plainunet_20261002.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
Set-Location $PSScriptRoot

function Step($n, $msg) { Write-Host ""; Write-Host "==== [$n/8] $msg  $(Get-Date -Format HH:mm:ss) ====" }
function Run($argsList) {
  & $py @argsList
  if ($LASTEXITCODE -ne 0) { throw "FAILED: python $($argsList -join ' ')" }
}

$bk = "outputs\_backup_analysis_test_20261002"
Step 1 "Back up outputs\analysis_test to $bk"
if (-not (Test-Path $bk)) {
  New-Item -ItemType Directory $bk | Out-Null
  Copy-Item "outputs\analysis_test\*" $bk
} else { Write-Host "backup already exists, not overwritten" }

Step 2 "Evaluate archPU1e3 (selected lr = 1e-3) on the held-out test set"
Run @("scripts\eval_symmetric.py", "--config", "configs\arch\plain_unet_lr1e3.yaml", "--base", "configs\default.yaml",
      "--folds", "0,1,2,3,4", "--prefixes", "archPU1e3", "--split", "test", "--out-dir", "outputs\analysis_test")

Step 3 "Remove the stale archPU (lr = 1e-4) rows"
Run @("scripts\drop_runs_from_csv.py", "outputs\analysis_test\per_sample_test.csv", "archPU_fold")
Run @("scripts\drop_runs_from_csv.py", "outputs\analysis_test\per_sample_free_test.csv", "archPU_fold")

Step 4 "Test-set analysis, thresholds imported from the validation folds"
Run @("scripts\analyze_symmetric.py", "--out-dir", "outputs\analysis_test",
      "--thr-from", "outputs\analysis_full\symmetric_analysis.json")

Step 5 "Effect-size uncertainty (bootstrap)"
Run @("scripts\effect_uncertainty.py")

Step 6 "Case / lesion unit sensitivity"
Run @("scripts\case_lesion_sensitivity.py")

Step 7 "Independent rebuild check (the two architecture checks are EXPECTED to differ from 1.87 / 12.0)"
& $py scripts\cross_validate_rebuild.py

Step 8 "Redraw Fig. 3 and refresh paper_numbers"
Run @("scripts\make_fig3_ranked.py")
Run @("scripts\build_paper_numbers.py")

Write-Host ""
Write-Host "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Record the whole window output (or at least steps 4-8)"
Write-Host "so the manuscript numbers (Table 8 plain U-Net row, architecture range, ratios, Fig. 3) can be updated as v22."
