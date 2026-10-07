# 可断点续跑的 QUBIQ 训练 launcher —— 覆盖尚未训练的剩余任务。
#
# 用途：把 QUBIQ 外部验证从当前 3 个点（brain-growth / kidney / prostate-task1）
#       扩到 9 个任务点。跳过已完成 run（outputs/runs/<name>/checkpoints/best.pth 存在即跳过），
#       可反复运行以断点续跑。
#
# 用法（仓库根目录，PowerShell）：
#     powershell -ExecutionPolicy Bypass -File scripts\run_qubiq_campaign.ps1
#     # 或只跑某组任务：powershell ... -Tasks brain_tumor_task01,pancreas
#
# 训练完成后评估：
#     powershell -ExecutionPolicy Bypass -File scripts\eval_qubiq_all.ps1
param(
    [string]$Tasks = ""
)
$ErrorActionPreference = "Stop"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$base = "configs\default.yaml"
Set-Location (Split-Path $PSScriptRoot -Parent)

# 任务 -> (config 前缀, 三臂名)
$tasks = @{
    "brain_tumor_task01"   = "brain_tumor_task01"
    "brain_tumor_task02"   = "brain_tumor_task02"
    "brain_tumor_task03"   = "brain_tumor_task03"
    "prostate_task02"      = "prostate_task02"
    "pancreas"             = "pancreas"
    "pancreatic_lesion"    = "pancreatic_lesion"
}
$arms = @("A", "B", "C")
$folds = 0..4

$selected = if ($Tasks -ne "") { $Tasks.Split(",") | ForEach-Object { $_.Trim() } } else { $tasks.Keys }
$total = 0; $done = 0; $toRun = 0

foreach ($t in $selected) {
    if (-not $tasks.ContainsKey($t)) { Write-Host "[跳过] 未知任务 $t"; continue }
    foreach ($a in $arms) {
        foreach ($f in $folds) {
            $total++
            $name = "qubiq_${t}_${a}_fold$f"
            $marker = "outputs\runs\$name\checkpoints\best.pth"
            if (Test-Path $marker) { $done++; continue }
            $toRun++
            Write-Host ("[训练] {0}  (已训练 {1}/{2})" -f $name, $done, $total)
            & $py scripts\train.py --config "configs\qubiq\${t}_${a}.yaml" --base $base `
                --name $name --split-tag "fold$f" 2>&1 | Select-Object -Last 3
            if ($LASTEXITCODE -ne 0) { Write-Warning "run $name 失败，继续下一个（可重跑续跑）" }
        }
    }
}
Write-Host ""
Write-Host ("QUBIQ 训练完成：共 {0} 个 run，跳过已完成 {1}，本次训练 {2}" -f $total, $done, $toRun)
