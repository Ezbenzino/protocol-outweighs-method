# P1(b) 合并臂评测 + 分析：训练完成后执行
# 产出：tgtBshift_bg 并入 analysis_full → symmetric_analysis.json → §4.13 数据
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$base = "configs/default.yaml"
$out = "outputs\analysis_full"

Write-Output "=== 合并臂评测 (fold0/fold1) $(Get-Date -Format HH:mm:ss) ==="
& $py scripts\eval_symmetric.py --config configs\aug\B_majority_shift_bg.yaml --base $base --folds 0,1 --prefixes tgtBshift_bg --out-dir $out --n-raters 4 2>&1 | Select-Object -Last 2
Write-Output "=== 评测结束 exit=$LASTEXITCODE ==="

Write-Output "=== 重新 analyze $(Get-Date -Format HH:mm:ss) ==="
& $py scripts\analyze_symmetric.py --out-dir $out 2>&1 | Select-Object -Last 3
Write-Output "=== analyze 结束 exit=$LASTEXITCODE ==="

Write-Output "=== 完成 $(Get-Date -Format HH:mm:ss) ==="
