param(
    [string]$Config = "configs\default.yaml",
    [string]$Ckpt = "outputs\checkpoints\best.pth",
    [string]$Split = "test"
)
# Evaluation entry: .\run_evaluate.ps1 [-Ckpt outputs\checkpoints\best.pth -Split test]
$ErrorActionPreference = "Stop"

$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root = $PSScriptRoot
$cfg   = Join-Path $Root $Config
$ckpt  = Join-Path $Root $Ckpt

& $Py (Join-Path $Root "scripts\evaluate.py") --config $cfg --ckpt $ckpt --split $Split
exit $LASTEXITCODE
