# 排队脚本：等 QUBIQ 外部验证（30 模型）全部结束后，自动接跑 plain U-Net @ 1e-3 的 LIDC 5 折。
# 用法：后台运行。GPU 串行，不并发抢资源。
Set-Location $PSScriptRoot
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$log = "outputs\logs\plain_batch.log"

Write-Output ("===== [{0}] 排队脚本启动：等待 QUBIQ 训练结束 =====" -f (Get-Date -Format 'HH:mm:ss')) | Tee-Object -FilePath $log -Append
do {
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'train\.py' }
    if ($running) {
        Start-Sleep -Seconds 60
    }
} while ($running)

Write-Output ("===== [{0}] QUBIQ 全部结束，开始 plain U-Net @ 1e-3 5 折 =====" -f (Get-Date -Format 'HH:mm:ss')) | Tee-Object -FilePath $log -Append
foreach ($k in 0..4) {
    $name = "archPU1e3_fold${k}"
    Write-Output ("  [{0}] 开始 {1}" -f (Get-Date -Format 'HH:mm:ss'), $name) | Tee-Object -FilePath $log -Append
    & $py scripts\train.py --config configs\arch\plain_unet_lr1e3.yaml --base configs\default.yaml --name $name --split-tag "fold${k}" 2>&1 | Tee-Object -FilePath $log -Append
    $best = (Select-String -Path "outputs\runs\$name\logs\train.log" -Pattern 'saved best' | Select-Object -Last 1).Line
    Write-Output ("  [done] {0} -> {1}" -f $name, ($best -replace '.*saved best','')) | Tee-Object -FilePath $log -Append
}
Write-Output ("===== [{0}] 全部训练完成 =====" -f (Get-Date -Format 'HH:mm:ss')) | Tee-Object -FilePath $log -Append
