# P0 恢复：重跑全部主臂 + seed + bg 到 analysis_full/（口径统一）
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$base = "configs/default.yaml"
$out = "outputs\analysis_full"

function Run-Folds($cfg, $prefix, $tag) {
  Write-Output "=== $tag ($prefix) $(Get-Date -Format HH:mm:ss) ==="
  & $py scripts\eval_symmetric.py --config $cfg --base $base --folds 0,1,2,3,4 --prefixes $prefix --out-dir $out --n-raters 4 2>&1 | Select-Object -Last 2
  Write-Output "=== $tag 结束 exit=$LASTEXITCODE ==="
}
function Run-Runs($cfg, $runs, $split, $tag) {
  Write-Output "=== $tag ($runs / $split) $(Get-Date -Format HH:mm:ss) ==="
  & $py scripts\eval_symmetric.py --config $cfg --base $base --runs $runs --split $split --out-dir $out --n-raters 4 2>&1 | Select-Object -Last 2
  Write-Output "=== $tag 结束 exit=$LASTEXITCODE ==="
}

# 主臂（监督 6 + 架构 4 + 增强 1）
Run-Folds "configs\targets\B_majority.yaml" "tgtB" "tgtB"
Run-Folds "configs\targets\C_maj_soft.yaml" "tgtC" "tgtC"
Run-Folds "configs\targets\D_soft.yaml" "tgtD" "tgtD"
Run-Folds "configs\targets\E_svls.yaml" "tgtE" "tgtE"
Run-Folds "configs\targets\F_softseg.yaml" "tgtF" "tgtF"
Run-Folds "configs\arch\resnet18.yaml" "archR18" "archR18"
Run-Folds "configs\arch\plain_unet_lr1e3.yaml" "archPU1e3" "archPU1e3"
Run-Folds "configs\arch\convnext_tiny.yaml" "archCNX" "archCNX"
Run-Folds "configs\arch\pvt_v2_b1.yaml" "archPVT" "archPVT"
Run-Folds "configs\aug\B_majority_shift.yaml" "tgtBshift" "tgtBshift"

# seed 臂（fold0 s1/s2/s3 + fold1 s1/s2/s3）
Run-Runs "configs\targets\A_union.yaml" "seed_tgtA_s1,seed_tgtA_s2,seed_tgtA_s3" "val_fold0" "seedA_f0"
Run-Runs "configs\targets\A_union.yaml" "seed_tgtA_fold1_s1,seed_tgtA_fold1_s2,seed_tgtA_fold1_s3" "val_fold1" "seedA_f1"
Run-Runs "configs\targets\B_majority.yaml" "seed_tgtB_s1,seed_tgtB_s2,seed_tgtB_s3" "val_fold0" "seedB_f0"
Run-Runs "configs\targets\B_majority.yaml" "seed_tgtB_fold1_s1,seed_tgtB_fold1_s2,seed_tgtB_fold1_s3" "val_fold1" "seedB_f1"

# bg 臂
Run-Runs "configs\targets\B_maj_bg.yaml" "bg_maj_fold0" "val_fold0" "bg_f0"
Run-Runs "configs\targets\B_maj_bg.yaml" "bg_maj_fold1" "val_fold1" "bg_f1"

Write-Output "=== 全部重跑完成 $(Get-Date -Format HH:mm:ss) ==="
