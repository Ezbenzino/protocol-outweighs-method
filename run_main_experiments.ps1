# run_main_experiments.ps1
# 重构后主实验的批处理入口. 可断点续跑: 已存在 best.pth 的实验会自动跳过,
# 所以中途 Ctrl+C 或关机后直接重跑本脚本即可接着做.
#
# 用法:
#   .\run_main_experiments.ps1 -Stage list      只列清单和预计耗时, 不训练
#   .\run_main_experiments.ps1 -Stage target    阶段1: 4 个训练靶 乘 5 折, 20 次, 约 12 小时
#   .\run_main_experiments.ps1 -Stage arch      阶段2: 2 个额外架构 乘 5 折, 10 次, 约 6 小时
#   .\run_main_experiments.ps1 -Stage lr        阶段0: plain U-Net 学习率预筛, 2 次, 约 1.5 小时
#   .\run_main_experiments.ps1 -Stage seed      阶段3: 种子重复, 4 次, 约 2.5 小时
#   .\run_main_experiments.ps1 -Stage all       全部
#
# 建议顺序: list, target 今晚, arch 明晚, lr 加 seed 第三晚.
# 先跑 target 是因为它决定论文主表能不能成立, 其余都是补强.

param(
    [ValidateSet("list","lr","target","arch","seed","all")]
    [string]$Stage = "list"
)

$ErrorActionPreference = "Continue"
$Py   = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$Root = $PSScriptRoot
$Base = "configs\default.yaml"
Set-Location $Root

# 每个实验一条记录: name 决定输出目录 outputs\runs\<name>\,
# config 会与 default.yaml 深合并, 只覆盖它显式写出的字段.
function New-Job($name, $config, $fold, $seed) {
    return New-Object PSObject -Property @{
        Name = $name; Config = $config; Fold = $fold; Seed = $seed
    }
}

function Get-Jobs([string]$stage) {
    $jobs = New-Object System.Collections.ArrayList

    if ($stage -eq "lr" -or $stage -eq "all") {
        # 阶段0: plain U-Net 没有预训练编码器, 学习率不能沿用 ResNet 臂的设置.
        # 在 fold0 上把 1e-4 与 1e-3 各跑一次, 取优者用于阶段2 的全部 5 折.
        # 这是按架构单独选学习率, 比强行统一更公平, 论文里要写明这一点.
        [void]$jobs.Add((New-Job "lrscan_pu_1e4" "configs\arch\plain_unet.yaml"      0 $null))
        [void]$jobs.Add((New-Job "lrscan_pu_1e3" "configs\arch\plain_unet_lr1e3.yaml" 0 $null))
    }

    if ($stage -eq "target" -or $stage -eq "all") {
        # 阶段1 主表: 4 个训练靶乘 5 折. 架构固定 ResNet-34, 损失固定 Dice 加 BCE.
        # tgtB 同时充当阶段2 的 ResNet-34 架构臂, 不重复跑.
        # 先跑 tgtB 和 tgtC, 因为它们直接回答"软概率监督到底有没有独立贡献".
        $order   = @("tgtB","tgtC","tgtA","tgtD")
        $cfgMap  = @{}
        $cfgMap["tgtA"] = "configs\targets\A_union.yaml"
        $cfgMap["tgtB"] = "configs\targets\B_majority.yaml"
        $cfgMap["tgtC"] = "configs\targets\C_maj_soft.yaml"
        $cfgMap["tgtD"] = "configs\targets\D_soft.yaml"
        foreach ($t in $order) {
            foreach ($f in 0..4) {
                [void]$jobs.Add((New-Job ("{0}_fold{1}" -f $t, $f) $cfgMap[$t] $f $null))
            }
        }
    }

    if ($stage -eq "arch" -or $stage -eq "all") {
        # 阶段2 架构对照: 训练靶固定为 B, 只变 backbone.
        # ResNet-34 臂复用 tgtB_fold*, 此处只跑另外两个.
        foreach ($f in 0..4) {
            [void]$jobs.Add((New-Job ("archR18_fold{0}" -f $f) "configs\arch\resnet18.yaml"   $f $null))
            [void]$jobs.Add((New-Job ("archPU_fold{0}"  -f $f) "configs\arch\plain_unet.yaml" $f $null))
        }
    }

    if ($stage -eq "seed" -or $stage -eq "all") {
        # 阶段3: 单次训练的随机波动有多大.
        # 5 折结果里 fold2 的组间差值是其余四折的三倍, 怀疑那次 union 训练没训好.
        # 同一配置跑多个种子, 才能判断折间差异里有多少其实是训练随机性.
        foreach ($s in @(1,2)) {
            [void]$jobs.Add((New-Job ("seed_tgtA_s{0}" -f $s) "configs\targets\A_union.yaml"    0 $s))
            [void]$jobs.Add((New-Job ("seed_tgtB_s{0}" -f $s) "configs\targets\B_majority.yaml" 0 $s))
        }
    }
    return $jobs
}

$jobs = Get-Jobs $Stage
$todo = New-Object System.Collections.ArrayList
$doneCount = 0
foreach ($j in $jobs) {
    $ck = Join-Path $Root ("outputs\runs\{0}\checkpoints\best.pth" -f $j.Name)
    if (Test-Path $ck) { $doneCount = $doneCount + 1 } else { [void]$todo.Add($j) }
}

Write-Host ""
Write-Host ("阶段 {0}: 共 {1} 个实验, 已完成 {2}, 待跑 {3}" -f $Stage, $jobs.Count, $doneCount, $todo.Count) -ForegroundColor Cyan
Write-Host ("预计耗时约 {0:N1} 小时, 按每次 35 分钟估. 实测 main_fold0=43min, union_fold0=26min" -f ($todo.Count * 35 / 60)) -ForegroundColor Yellow
Write-Host ""

if ($Stage -eq "list") {
    foreach ($j in $todo) {
        $s = ""
        if ($j.Seed -ne $null) { $s = " --seed " + $j.Seed }
        Write-Host ("  {0,-18} {1,-38} fold{2}{3}" -f $j.Name, $j.Config, $j.Fold, $s)
    }
    Write-Host ""
    Write-Host "确认无误后用 -Stage target 开始跑主表. " -ForegroundColor Yellow
    exit 0
}

$t0 = Get-Date
$i  = 0
foreach ($j in $todo) {
    $i = $i + 1
    Write-Host ""
    Write-Host ("[{0}] ({1}/{2}) {3}  config={4}  fold{5}" -f (Get-Date -Format "HH:mm:ss"), $i, $todo.Count, $j.Name, $j.Config, $j.Fold) -ForegroundColor Green

    $jt0 = Get-Date
    if ($j.Seed -ne $null) {
        & $Py "scripts\train.py" --config $j.Config --base $Base --name $j.Name --split-tag ("fold" + $j.Fold) --seed $j.Seed
    } else {
        & $Py "scripts\train.py" --config $j.Config --base $Base --name $j.Name --split-tag ("fold" + $j.Fold)
    }
    $code = $LASTEXITCODE
    $mins = ((Get-Date) - $jt0).TotalMinutes

    if ($code -ne 0) {
        Write-Host ("  失败: {0} 退出码 {1}, 耗时 {2:N1} 分钟, 继续下一个" -f $j.Name, $code, $mins) -ForegroundColor Red
    } else {
        Write-Host ("  完成: {0} 耗时 {1:N1} 分钟" -f $j.Name, $mins) -ForegroundColor DarkGreen
    }
}

Write-Host ""
Write-Host ("阶段 {0} 结束, 总耗时 {1:N1} 小时" -f $Stage, ((Get-Date) - $t0).TotalHours) -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host ("  {0} scripts\eval_symmetric.py --config {1} --folds 0,1,2,3,4 --prefixes tgtA,tgtB,tgtC,tgtD" -f $Py, $Base)
Write-Host ("  {0} scripts\analyze_symmetric.py" -f $Py)
Write-Host ""

