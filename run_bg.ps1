# P1-3 去混淆臂重启：bg_maj_fold0 + bg_maj_fold1（使用加速后的背景 bank 采样）
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$runs = @(
  @{ n = "bg_maj_fold0"; fold = "fold0" },
  @{ n = "bg_maj_fold1"; fold = "fold1" }
)
foreach ($r in $runs) {
  $n = $r.n; $fold = $r.fold
  Write-Output "=== 启动 $n (fold=$fold) $(Get-Date -Format HH:mm:ss) ==="
  & $py scripts\train.py --config configs\targets\B_maj_bg.yaml --base configs\default.yaml --name $n --split-tag $fold --seed 1 2>&1 | Tee-Object -FilePath "outputs\logs\$n.log"
  Write-Output "=== $n 结束 $(Get-Date -Format HH:mm:ss) exit=$LASTEXITCODE ==="
}
Write-Output "=== 全部完成 $(Get-Date -Format HH:mm:ss) ==="
