# P1-2 种子扩展 + P1-3 去混淆臂，串行后台跑
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
Set-Location $PSScriptRoot
$base = "configs\default.yaml"
$logdir = "outputs\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

function Run-Train($cfg, $name, $fold, $seed) {
    $tag = "fold$fold"
    Write-Output "=== START $name (fold$fold seed$seed) $(Get-Date -Format 'HH:mm') ==="
    & $py scripts\train.py --config $cfg --base $base --name $name --split-tag $tag --seed $seed 2>&1 | Tee-Object -FilePath "$logdir\$name.log"
    Write-Output "=== DONE $name exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm') ==="
}

# ---- P1-2 种子扩展（补 fold0_s3 + fold1 s1-3）----
Run-Train "configs\targets\A_union.yaml"   "seed_tgtA_s3"         0 3
Run-Train "configs\targets\A_union.yaml"   "seed_tgtA_fold1_s1"   1 1
Run-Train "configs\targets\A_union.yaml"   "seed_tgtA_fold1_s2"   1 2
Run-Train "configs\targets\A_union.yaml"   "seed_tgtA_fold1_s3"   1 3
Run-Train "configs\targets\B_majority.yaml" "seed_tgtB_s3"         0 3
Run-Train "configs\targets\B_majority.yaml" "seed_tgtB_fold1_s1"   1 1
Run-Train "configs\targets\B_majority.yaml" "seed_tgtB_fold1_s2"   1 2
Run-Train "configs\targets\B_majority.yaml" "seed_tgtB_fold1_s3"   1 3

# ---- P1-3 去混淆臂（背景采样 bg=0.5，训练靶 B）----
Run-Train "configs\targets\B_maj_bg.yaml"  "bg_maj_fold0"         0 1
Run-Train "configs\targets\B_maj_bg.yaml"  "bg_maj_fold1"         1 1

Write-Output "ALL P1-2/P1-3 TRAINING DONE $(Get-Date -Format 'HH:mm')"
