# QUBIQ 管线幂等推进脚本（供定时任务每 20 分钟触发）
#
# 状态机（每轮调用只前进一步，重复调用安全）：
#   STAGE 0 训练完成检查：
#      - 有 train.py 进程在跑 -> 报告进度，退出
#      - 无 train.py 进程 且 135 个 run 全部有 best.pth -> 标记训练完成，进入 STAGE 1
#      - 无进程但 run 数不足 -> 报告"训练中断/待续"，退出（不评估不完整数据）
#   STAGE 1 评估（detached 后台进程，跨定时任务存活）：
#      - 未启动 -> Start-Process 启动 scripts\eval_qubiq_all.ps1（自动跳过已评估任务）
#      - 已启动 -> 等 eval_symmetric 进程结束 且 9 个任务都有 per_sample_val_fold4.csv -> 标记评估完成
#   STAGE 2 汇总：
#      - 跑 analyze_qubiq_all.py（-> qubiq_all_effects.json）+ qubiq_trend_analysis.py（-> qubiq_trend.json）
#      - 写 ALL_DONE 标记
#
# 状态文件：outputs/logs/qubiq_pipeline_state.json
# 输出约定：stdout 最后一行输出 [TRAIN_DONE]/[EVAL_DONE]/[ALL_DONE]/[进行中...]，供定时任务解析。
$ErrorActionPreference = "Continue"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
Set-Location (Split-Path $PSScriptRoot -Parent)
$stateFile = "outputs\logs\qubiq_pipeline_state.json"

$tasks = @(
    @{ label = "qubiq_brain";              n = 7 },
    @{ label = "qubiq_kidney";             n = 3 },
    @{ label = "qubiq_brain_tumor_task01"; n = 3 },
    @{ label = "qubiq_brain_tumor_task02"; n = 3 },
    @{ label = "qubiq_brain_tumor_task03"; n = 3 },
    @{ label = "qubiq_prostate_task01";    n = 6 },
    @{ label = "qubiq_prostate_task02";    n = 6 },
    @{ label = "qubiq_pancreas";           n = 2 },
    @{ label = "qubiq_pancreatic_lesion";  n = 2 }
)
$arms = @("A", "B", "C"); $folds = 0..4
$TOTAL = 135

# ---------- 读取/初始化状态 ----------
$state = @{}
if (Test-Path $stateFile) { $state = Get-Content $stateFile -Raw | ConvertFrom-Json }
foreach ($k in @("training_done", "eval_launched", "eval_done", "analyze_done")) {
    if ($null -eq $state.$k) { $state.$k = $false }
}
function Save-State { $state | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 $stateFile }

# ---------- STAGE 0：训练完成检查 ----------
if (-not $state.training_done) {
    $trainProc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'scripts[\\/]train\.py' }
    $best = 0
    foreach ($t in $tasks) { foreach ($a in $arms) { foreach ($f in $folds) {
        $armName = $t.label.Substring(6)   # 去掉 qubiq_ 前缀（6 字符）
        if (Test-Path "outputs\runs\qubiq_${armName}_${a}_fold$f\checkpoints\best.pth") { $best++ }
    } } }
    if ($trainProc) {
        Write-Output "[进行中] 训练中：$best/$TOTAL run 已有 checkpoint；正在训练：$($trainProc[0].CommandLine)"
        exit 0
    }
    if ($best -ge $TOTAL) {
        $state.training_done = $true; $state.training_done_ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Save-State
        Write-Output "[TRAIN_DONE] 训练完成（$best/$TOTAL run）"
    } else {
        Write-Output "[待续] 无训练进程但仅 $best/$TOTAL run，训练可能中断或尚未全部完成——本次不推进"
        exit 0
    }
}

# ---------- STAGE 1：评估 ----------
if (-not $state.eval_done) {
    if (-not $state.eval_launched) {
        $log = "outputs\logs\qubiq_eval_$(Get-Date -Format yyyyMMdd_HHmmss).log"
        $p = Start-Process powershell -WindowStyle Hidden -PassThru `
            -ArgumentList '-ExecutionPolicy','Bypass','-File',(Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts\eval_qubiq_all.ps1') `
            -RedirectStandardOutput $log -RedirectStandardError ($log + ".err")
        $state.eval_launched = $true; $state.eval_launch_ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        $state.eval_pid = $p.Id
        Save-State
        Write-Output "[EVAL_LAUNCHED] 评估已在后台启动（PID $($p.Id)，日志 $log）"
        exit 0
    }
    # 检查评估是否结束：无 eval_symmetric 进程 且 9 任务都有 fold4 per_sample
    $evalProc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'eval_symmetric\.py' }
    $missing = @()
    foreach ($t in $tasks) {
        if (-not (Test-Path "outputs\analysis\$($t.label)\per_sample_val_fold4.csv")) { $missing += $t.label }
    }
    if (-not $evalProc -and $missing.Count -eq 0) {
        $state.eval_done = $true; $state.eval_done_ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Save-State
        Write-Output "[EVAL_DONE] 评估完成（9 任务 per_sample 齐全）"
    } else {
        Write-Output "[进行中] 评估中：未完成 $($missing.Count)/9 任务；eval 进程 $([bool]$evalProc)"
        exit 0
    }
}

# ---------- STAGE 2：汇总 ----------
if (-not $state.analyze_done) {
    $o1 = & $py scripts\analyze_qubiq_all.py 2>&1 | Select-Object -Last 3
    $o2 = & $py scripts\qubiq_trend_analysis.py 2>&1 | Select-Object -Last 3
    $state.analyze_done = $true; $state.analyze_done_ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Save-State
    Write-Output "[ALL_DONE] QUBIQ 全流程完成（9 点 qubiq_all_effects.json + qubiq_trend.json 已更新）"
    $o1; $o2
    exit 0
}

Write-Output "[ALL_DONE] QUBIQ 全流程此前已完成"
