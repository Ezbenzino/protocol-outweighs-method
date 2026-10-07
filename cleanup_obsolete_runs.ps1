# cleanup_obsolete_runs.ps1
# 清理选题重构后作废的实验 checkpoint, 保留全部训练日志.
#
# 默认是预演模式, 不会删任何东西:
#   .\cleanup_obsolete_runs.ps1              只列出将要删除的文件和总大小
#   .\cleanup_obsolete_runs.ps1 -Execute     确认无误后真正删除
#
# 为什么只删 checkpoint 不删日志:
#   每个 .pth 约 109 MB, 是磁盘占用的全部来源; train.log 只有几十 KB,
#   却是"辅助损失消融"这一阴性结果的原始证据, 将来审稿人质询时要拿得出来.

param([switch]$Execute)

$Root = Join-Path $PSScriptRoot "outputs\runs"

# 作废的实验: 六项损失消融, 权重搜索, 中间版本, 口径不一致的基线.
$Obsolete = @(
    "abl_baseline", "abl_brbc", "abl_csl", "abl_lovasz", "abl_sitl", "abl_uniontgt",
    "weight_brbc01_lov01", "weight_brbc01_lov05", "weight_brbc10_lov01", "weight_brbc10_lov05",
    "main", "main_v1_uniontgt", "main_v2_consensus_partial", "main_v3_uniondef_data", "base_unet",
    "union_fold0", "union_fold1", "union_fold2", "union_fold3", "union_fold4",
    "nnunet_2d_fold0", "unet3d"
)

# 暂时保留: 六损失版主模型, 等新主实验跑完再决定.
$KeepForNow = @("main_fold0","main_fold1","main_fold2","main_fold3","main_fold4","plain_unet")

Write-Host ""
if ($Execute) {
    Write-Host "执行模式: 将真正删除文件" -ForegroundColor Red
} else {
    Write-Host "预演模式: 不会删除任何文件, 加 -Execute 才真正删除" -ForegroundColor Yellow
}
Write-Host ""

$total = 0
$count = 0
foreach ($name in $Obsolete) {
    $ckptDir = Join-Path $Root ("{0}\checkpoints" -f $name)
    if (-not (Test-Path $ckptDir)) {
        Write-Host ("  不存在: {0}" -f $name) -ForegroundColor DarkGray
        continue
    }
    $files = @(Get-ChildItem $ckptDir -Filter *.pth -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { continue }

    $size = ($files | Measure-Object -Property Length -Sum).Sum
    $total = $total + $size
    $count = $count + $files.Count
    Write-Host ("  {0,-28} {1,2} 个文件 {2,8:N1} MB" -f $name, $files.Count, ($size / 1MB))
    if ($Execute) { $files | Remove-Item -Force }
}

Write-Host ""
Write-Host ("合计: {0} 个 checkpoint, {1:N2} GB" -f $count, ($total / 1GB)) -ForegroundColor Cyan
Write-Host ""
Write-Host "保留, 日志全部保留, checkpoint 也暂留:" -ForegroundColor Green
foreach ($k in $KeepForNow) { Write-Host ("  {0}" -f $k) -ForegroundColor Green }
Write-Host ""
if (-not $Execute) {
    Write-Host "确认无误后运行: .\cleanup_obsolete_runs.ps1 -Execute" -ForegroundColor Yellow
} else {
    Write-Host "已删除. train.log 与 tensorboard 事件文件均未受影响. " -ForegroundColor Cyan
}
Write-Host ""

