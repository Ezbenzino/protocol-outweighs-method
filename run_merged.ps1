# P1(b) 合并臂训练：translation augmentation + background sampling
# 回答 §4.13 报告：两种补救叠加后，win128→win512 视野效应是否只剩协议项。
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$runs = @(
  @{ n = "tgtBshift_bg_fold0"; fold = "fold0" },
  @{ n = "tgtBshift_bg_fold1"; fold = "fold1" }
)
foreach ($r in $runs) {
  $n = $r.n; $fold = $r.fold
  Write-Output "=== 启动 $n (fold=$fold) $(Get-Date -Format HH:mm:ss) ==="
  & $py scripts\train.py --config configs\aug\B_majority_shift_bg.yaml --base configs\default.yaml --name $n --split-tag $fold --seed 1 2>&1 | Tee-Object -FilePath "outputs\logs\$n.log"
  Write-Output "=== $n 结束 $(Get-Date -Format HH:mm:ss) exit=$LASTEXITCODE ==="
}
Write-Output "=== 合并臂训练全部完成 $(Get-Date -Format HH:mm:ss) ==="
