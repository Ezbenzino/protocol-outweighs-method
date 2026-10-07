# QUBIQ 全任务评估 launcher —— 训练完成后运行。
# 对每个任务用 eval_symmetric.py 评估三臂并写 per_sample CSV，然后跑
# analyze_qubiq_all.py 汇总为 outputs/analysis/qubiq_all_effects.json。
#
# 用法（仓库根目录，PowerShell）：
#     powershell -ExecutionPolicy Bypass -File scripts\eval_qubiq_all.ps1
$ErrorActionPreference = "Continue"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
Set-Location (Split-Path $PSScriptRoot -Parent)

# 任务 -> (config 前缀, n_raters)
$tasks = @(
    @{ cfg = "brain";            n = 7;  label = "qubiq_brain" },
    @{ cfg = "kidney";           n = 3;  label = "qubiq_kidney" },
    @{ cfg = "brain_tumor_task01_A"; n = 3; label = "qubiq_brain_tumor_task01" },
    @{ cfg = "brain_tumor_task02_A"; n = 3; label = "qubiq_brain_tumor_task02" },
    @{ cfg = "brain_tumor_task03_A"; n = 3; label = "qubiq_brain_tumor_task03" },
    @{ cfg = "prostate_task01_A"; n = 6;  label = "qubiq_prostate_task01" },
    @{ cfg = "prostate_task02_A"; n = 6;  label = "qubiq_prostate_task02" },
    @{ cfg = "pancreas_A";        n = 2;  label = "qubiq_pancreas" },
    @{ cfg = "pancreatic_lesion_A"; n = 2; label = "qubiq_pancreatic_lesion" }
)

foreach ($t in $tasks) {
    $label = $t.label
    $arm = $label.Substring(6)   # 去掉 qubiq_ 前缀（6 字符）-> 数据集名
    $prefixes = "qubiq_${arm}_A,qubiq_${arm}_B,qubiq_${arm}_C"
    # 幂等：该任务 5 折 per_sample 已存在则跳过（避免重复评估已完成的 brain/kidney/prostate-task1）
    $doneMarker = "outputs\analysis\$label\per_sample_val_fold4.csv"
    if (Test-Path $doneMarker) {
        Write-Host "[跳过] $label （已评估，$doneMarker 存在）"
        continue
    }
    Write-Host "[评估] $label  (n_raters=$($t.n))"
    & $py scripts\eval_symmetric.py --config "configs\qubiq\$($t.cfg).yaml" --base configs\default.yaml `
        --folds 0,1,2,3,4 --prefixes $prefixes --n-raters $t.n `
        --out-dir "outputs\analysis\$label" 2>&1 | Select-Object -Last 1
}
Write-Host ""
Write-Host "[汇总] analyze_qubiq_all.py"
& $py scripts\analyze_qubiq_all.py 2>&1
