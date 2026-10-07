# QUBIQ 外部数据集验证：全量训练批处理（30 个模型：2 数据集 x 3 训练靶 x 5 折）
# 串行运行，每模型输出追加到 outputs/logs/qubiq_batch.log
Set-Location $PSScriptRoot
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$log = "outputs\logs\qubiq_batch.log"

foreach ($ds in @('kidney', 'brain')) {
    foreach ($arm in @('A', 'B', 'C')) {
        foreach ($k in 0..4) {
            $name = "qubiq_${ds}_${arm}_fold${k}"
            $cfg = "configs\qubiq\${ds}_${arm}.yaml"
            Write-Output ("===== [{0}] 开始 {1} =====" -f (Get-Date -Format 'HH:mm:ss'), $name) | Tee-Object -FilePath $log -Append
            & $py scripts\train.py --config $cfg --base configs\default.yaml --name $name --split-tag "fold${k}" 2>&1 | Tee-Object -FilePath $log -Append
            $best = (Select-String -Path "outputs\runs\$name\logs\train.log" -Pattern 'saved best' | Select-Object -Last 1).Line
            Write-Output ("  [done] {0}  ->  {1}" -f $name, ($best -replace '.*saved best','best')) | Tee-Object -FilePath $log -Append
        }
    }
}
Write-Output ("===== 全部完成 {0} =====" -f (Get-Date -Format 'HH:mm:ss')) | Tee-Object -FilePath $log -Append
