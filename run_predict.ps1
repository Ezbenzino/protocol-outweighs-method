param(
    [string]$Config = "configs\default.yaml",
    [string]$Ckpt = "outputs\checkpoints\best.pth",
    [int]$Num = 4
)
# Inference / visualization entry: .\run_predict.ps1 [-Num 4]
$ErrorActionPreference = "Stop"

$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root = $PSScriptRoot
$cfg  = Join-Path $Root $Config
$ckpt = Join-Path $Root $Ckpt

& $Py (Join-Path $Root "scripts\predict.py") --config $cfg --ckpt $ckpt --num $Num
exit $LASTEXITCODE
