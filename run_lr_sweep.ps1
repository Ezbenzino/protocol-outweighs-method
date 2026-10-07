# P1-4: plain U-Net 学习率调优（1e-4 / 3e-4 / 1e-3 × fold0，取优者后续跑 5 折）
# 依据 补强清单 P1-4：plain U-Net 曾用单一 lr=1e-4（来自早期配置），需 mini-sweep。
$ErrorActionPreference = "Continue"
$Py   = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$Root = $PSScriptRoot
$Base = "configs\default.yaml"
Set-Location $Root

$jobs = @(
    @{ Name="lrscan_pu_1e4"; Cfg="configs\arch\plain_unet.yaml" },
    @{ Name="lrscan_pu_3e4"; Cfg="configs\arch\plain_unet_lr3e4.yaml" },
    @{ Name="lrscan_pu_1e3"; Cfg="configs\arch\plain_unet_lr1e3.yaml" }
)

foreach ($j in $jobs) {
    $ck = Join-Path $Root ("outputs\runs\{0}\checkpoints\best.pth" -f $j.Name)
    if (Test-Path $ck) { Write-Host "跳过（已完成）: $($j.Name)"; continue }
    Write-Host ("[{0}] 开始 {1}  config={2}" -f (Get-Date -Format "HH:mm:ss"), $j.Name, $j.Cfg)
    $t0 = Get-Date
    & $Py "scripts\train.py" --config $j.Cfg --base $Base --name $j.Name --split-tag fold0
    $code = $LASTEXITCODE
    Write-Host ("  {0} 退出码={1} 耗时 {2:N1} 分钟" -f $j.Name, $code, ((Get-Date)-$t0).TotalMinutes)
}
Write-Host "P1-4 LR sweep 全部结束。"
