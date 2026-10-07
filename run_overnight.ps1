# 通宵批处理：把今晚所有剩余的 GPU 任务串起来，可断点续跑。
#
# 设计要点（都是这个项目踩过坑之后的结果）：
#   1. 每一步先检查产物是否已存在，存在就跳过 -> 中断后重跑不会白干
#   2. 单步失败【不中止整条链】，记下来继续跑下一步 -> 凌晨三点挂一步不会毁掉整晚
#   3. 全程写日志到 outputs/overnight_<时间戳>.log
#   4. 纯 ASCII 标点、CRLF、UTF-8 BOM -> PowerShell 5.1 不会把中文字节切错
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1

Set-Location $PSScriptRoot
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "outputs\overnight_$stamp.log"
$failed = @()
$skipped = @()
$done = @()

function Step($name, $probe, $cmd) {
    if ($probe -and (Test-Path $probe)) {
        Write-Host "[跳过] $name  (已存在 $probe)"
        $script:skipped += $name
        return
    }
    Write-Host ""
    Write-Host "=== [$(Get-Date -Format 'HH:mm:ss')] $name ==="
    $t0 = Get-Date
    & $py @cmd 2>&1 | Tee-Object -FilePath $log -Append
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[失败] $name  退出码 $LASTEXITCODE  用时 $mins 分钟"
        $script:failed += $name
    } else {
        Write-Host "[完成] $name  用时 $mins 分钟"
        $script:done += $name
    }
}

Write-Host "通宵批处理开始 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "日志: $log"

# ---- 1. SVLS 五折训练 (约 100 分钟) ----
foreach ($k in 0..4) {
    Step "训练 tgtE_fold$k (SVLS)" "outputs\runs\tgtE_fold$k\checkpoints\best.pth" `
        @("scripts\train.py", "--config", "configs\targets\E_svls.yaml",
          "--base", "configs\default.yaml", "--name", "tgtE_fold$k", "--split-tag", "fold$k")
}

# ---- 2. SoftSeg 五折训练 (约 100 分钟) ----
# 前置：verify_softseg.py 必须先通过。这里再自动跑一次，不过就跳过整组。
Write-Host ""
Write-Host "=== SoftSeg 上线自检 ==="
& $py scripts\verify_softseg.py 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "[砍掉] SoftSeg 自检未通过 - 跳过它的全部训练与评测"
    $failed += "SoftSeg 自检"
} else {
    foreach ($k in 0..4) {
        Step "训练 tgtF_fold$k (SoftSeg)" "outputs\runs\tgtF_fold$k\checkpoints\best.pth" `
            @("scripts\train.py", "--config", "configs\targets\F_softseg.yaml",
              "--base", "configs\default.yaml", "--name", "tgtF_fold$k", "--split-tag", "fold$k")
    }
}

# ---- 3. 验证集评测：新臂 (约 15 分钟) ----
$newArms = @()
if (Test-Path "outputs\runs\tgtE_fold4\checkpoints\best.pth") { $newArms += "tgtE" }
if (Test-Path "outputs\runs\tgtF_fold4\checkpoints\best.pth") { $newArms += "tgtF" }
if ($newArms.Count -gt 0) {
    Step "验证集评测 $($newArms -join ',')" "" `
        @("scripts\eval_symmetric.py", "--config", "configs\default.yaml",
          "--folds", "0,1,2,3,4", "--prefixes", ($newArms -join ","))
    Step "重算阈值校准" "" @("scripts\analyze_symmetric.py")
}

# ---- 4. 测试集补评：archCNX, archPVT + 新臂 (约 25 分钟) ----
$testArms = @("archCNX", "archPVT") + $newArms
Step "测试集评测 $($testArms -join ',')" "" `
    @("scripts\eval_symmetric.py", "--config", "configs\default.yaml",
      "--folds", "0,1,2,3,4", "--prefixes", ($testArms -join ","),
      "--split", "test", "--out-dir", "outputs\analysis_test")
Step "测试集分析" "" `
    @("scripts\analyze_symmetric.py", "--out-dir", "outputs\analysis_test",
      "--thr-from", "outputs\analysis\symmetric_analysis.json")

# ---- 5. 位置先验的跨臂验证 (约 15 分钟) ----
# 目前结论只在 tgtB 上测过。跑遍所有臂的 fold0，回答"这是不是某个架构特有的毛病"。
$offRuns = @("tgtA_fold0", "tgtB_fold0", "tgtC_fold0", "tgtD_fold0",
             "tgtBshift_fold0", "archR18_fold0", "archPU_fold0",
             "archCNX_fold0", "archPVT_fold0")
foreach ($a in $newArms) { $offRuns += "${a}_fold0" }
Step "位置先验跨臂验证" "outputs\analysis_scope\diag_offset_allarms.json" `
    @("scripts\diag_offset.py", "--runs", ($offRuns -join ","),
      "--out", "outputs\analysis_scope\diag_offset_allarms.json")

# ---- 6. 不占 GPU 的小任务 ----
Step "SVLS 核宽标定" "outputs\analysis\svls_calibration.json" @("scripts\calibrate_svls.py")
Step "架构参数量表" "" @("scripts\check_arch.py", "--batch", "8")

# ---- 汇总 ----
Write-Host ""
Write-Host "==================================================================="
Write-Host "全部结束 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "完成 $($done.Count) 步  跳过 $($skipped.Count) 步  失败 $($failed.Count) 步"
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "失败的步骤 - 把这几行贴给我:"
    $failed | ForEach-Object { Write-Host "  - $_" }
}
Write-Host "日志: $log"
Write-Host "==================================================================="
