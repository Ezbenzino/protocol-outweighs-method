# 对称评测：8 个 seed runs + 2 个 bg runs（center-crop 协议，多 V>=1..N 参考）
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$base = "configs/default.yaml"

function Run-Eval($cfg, $runs, $split, $tag) {
  Write-Output "=== $tag : $runs / $split $(Get-Date -Format HH:mm:ss) ==="
  & $py scripts\eval_symmetric.py --config $cfg --base $base --runs $runs --split $split --n-raters 4 2>&1 | Tee-Object -FilePath "outputs\logs\eval_$tag.log"
  Write-Output "=== $tag 结束 exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss) ==="
}

Run-Eval "configs\targets\A_union.yaml" "seed_tgtA_s3" "val_fold0" "A_fold0"
Run-Eval "configs\targets\A_union.yaml" "seed_tgtA_fold1_s1,seed_tgtA_fold1_s2,seed_tgtA_fold1_s3" "val_fold1" "A_fold1"
Run-Eval "configs\targets\B_majority.yaml" "seed_tgtB_s3" "val_fold0" "B_fold0"
Run-Eval "configs\targets\B_majority.yaml" "seed_tgtB_fold1_s1,seed_tgtB_fold1_s2,seed_tgtB_fold1_s3" "val_fold1" "B_fold1"
Run-Eval "configs\targets\B_maj_bg.yaml" "bg_maj_fold0" "val_fold0" "bg_fold0"
Run-Eval "configs\targets\B_maj_bg.yaml" "bg_maj_fold1" "val_fold1" "bg_fold1"

Write-Output "=== 全部评测完成 $(Get-Date -Format HH:mm:ss) ==="
