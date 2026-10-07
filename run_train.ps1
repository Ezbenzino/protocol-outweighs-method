param(
    [string]$Config = "configs\default.yaml",
    [string]$Base = "",
    [string]$Name = "",
    [string]$SplitTag = ""
)
# Training entry:
#   .\run_train.ps1 [-Config configs\ablation\baseline.yaml -Base configs\default.yaml]
#   .\run_train.ps1 -Name main_fold0 -SplitTag fold0
$ErrorActionPreference = "Stop"

$Py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
if (-not (Test-Path $Py)) { $Py = "python" }

$Root = $PSScriptRoot
$cfg  = Join-Path $Root $Config

Write-Host "Using python: $Py"

$a = @((Join-Path $Root "scripts\train.py"), '--config', $cfg)
if ($Base)     { $a += '--base';      $a += (Join-Path $Root $Base) }
if ($Name)     { $a += '--name';      $a += $Name }
if ($SplitTag) { $a += '--split-tag'; $a += $SplitTag }

& $Py @a
exit $LASTEXITCODE
