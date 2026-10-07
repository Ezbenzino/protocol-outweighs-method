# Batch experiment runner (for the SCI experiment matrix).
# Stages:
#   main      - main model (ResNet-34 + full loss)
#   ablation  - baseline / +SITL / +CSL / +BRBC  (vs main)
#   baseline  - U-Net style baseline (ResNet-18 + plain Dice/BCE)
#   cv        - 5-fold CV of the main model (report mean+/-std in the paper)
#   all       - main + ablation + baseline + cv
#
# Every run writes to outputs\runs\<name>\ (independent checkpoints/logs).
# A failed run does NOT stop the rest; failures are listed at the end.
# Usage:
#   .\run_experiments.ps1 -Stage all
#   .\run_experiments.ps1 -Stage cv -Folds 5
param(
    [ValidateSet('all', 'main', 'ablation', 'baseline', 'cv')]
    [string]$Stage = 'all',
    [int]$Folds = 5
)

$ErrorActionPreference = "Continue"
$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root   = $PSScriptRoot
$Splits = Join-Path $Root "data\splits"
$Runs   = Join-Path $Root "outputs\runs"
$script:Failed = @()

function Run-Train([string]$Name, [string]$Config, [string]$Base, [string]$Tag) {
    $runDir = Join-Path $Runs $Name
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $logFile = Join-Path $runDir "console.log"
    $a = @('scripts\train.py', '--config', $Config)
    if ($Base) { $a += '--base'; $a += $Base }
    $a += '--name'; $a += $Name
    if ($Tag)  { $a += '--split-tag'; $a += $Tag }
    Write-Host ("==> [{0}] config={1} base={2} tag={3}" -f $Name, $Config, $Base, $Tag) -ForegroundColor Cyan
    & $Py @a 2>&1 | Tee-Object -FilePath $logFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("!! [{0}] FAILED" -f $Name) -ForegroundColor Red
        $script:Failed += $Name
    } else {
        Write-Host ("OK  [{0}]" -f $Name) -ForegroundColor Green
    }
}

if ($Stage -in @('all', 'main')) {
    Run-Train 'main' 'configs\default.yaml' '' ''
}

if ($Stage -in @('all', 'ablation')) {
    Run-Train 'abl_baseline' 'configs\ablation\baseline.yaml' 'configs\default.yaml' ''
    Run-Train 'abl_sitl'     'configs\ablation\sitl.yaml'     'configs\default.yaml' ''
    Run-Train 'abl_csl'      'configs\ablation\csl.yaml'      'configs\default.yaml' ''
    Run-Train 'abl_brbc'     'configs\ablation\brbc.yaml'     'configs\default.yaml' ''
    # optional: +Lovasz on top of full is already 'main'; lovasz.yaml is the single-component probe
    Run-Train 'abl_lovasz'   'configs\ablation\lovasz.yaml'   'configs\default.yaml' ''
    # full stack but classic union-mask targets (vs main's consensus targets)
    Run-Train 'abl_uniontgt' 'configs\ablation\union_target.yaml' 'configs\default.yaml' ''
}

if ($Stage -in @('all', 'baseline')) {
    Run-Train 'base_unet' 'configs\baseline_unet.yaml' 'configs\default.yaml' ''
}

if ($Stage -in @('all', 'cv')) {
    Write-Host "generating $Folds-fold splits (fixed 20% test hold-out) ..." -ForegroundColor Cyan
    & $Py (Join-Path $Root 'preprocess\make_splits.py') --npz_dir (Join-Path $Root 'data\processed\npz') --out_dir $Splits --n_folds $Folds --stratify
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! fold split generation failed" -ForegroundColor Red
        $script:Failed += 'cv:make_splits'
    } else {
        for ($k = 0; $k -lt $Folds; $k++) {
            Run-Train ("main_fold{0}" -f $k) 'configs\default.yaml' '' ("fold{0}" -f $k)
        }
    }
}

Write-Host ""
if ($script:Failed.Count -eq 0) {
    Write-Host "ALL EXPERIMENTS DONE (no failures)" -ForegroundColor Green
} else {
    Write-Host ("FINISHED WITH FAILURES: {0}" -f ($script:Failed -join ', ')) -ForegroundColor Red
}
