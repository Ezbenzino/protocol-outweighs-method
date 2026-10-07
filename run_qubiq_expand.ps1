# QUBIQ 外部验证扩展：把 §4.14 的一致性-协议关系从 3 个点扩到 10 个点
# 在 Windows PowerShell 里跑（需要 skin_seg 环境的 nibabel 与 GPU）
#
# 背景：QUBIQ 2021 训练集共 6 个子数据集 / 9 个任务，论文只用了 2 个。
#       其余 7 个任务的原始数据已在 data/raw/qubiq/extracted/ 下，无需再下载。
#
# 一致性指数（x 轴）已由 scripts/qubiq_agreement_index.py 算出，不需要 GPU：
#   brain-tumor/task02 0.130 < LIDC 0.333 < brain-growth 0.501
#   < pancreatic-lesion 0.556 < brain-tumor/task03 0.660 < prostate/task02 0.707
#   < pancreas 0.724 < brain-tumor/task01 0.782 < prostate/task01 0.787 < kidney 0.862
# 本脚本产生的是 y 轴（协议效应 / 监督效应比值），需要训练。
#
# ⚠️ pancreas 与 pancreatic-lesion 只有 2 位标注者，评测靶阶梯只有 G1/G2 两级，
#    比值可比性弱于其它任务。建议在正文图里标出，或只放补充材料。

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py  = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$raw = 'data\raw\qubiq\extracted\training_data_v3_QC'
$log = 'outputs\logs\qubiq_expand.log'
New-Item -ItemType Directory -Force -Path outputs\logs | Out-Null

# short, 病例父目录, 任务, 标注者数, 归一化
$TASKS = @(
  @{s='btum1';      d="$raw\brain-tumor\Training";  t='task01'; n=3; norm='minmax'},
  @{s='btum2';      d="$raw\brain-tumor\Training";  t='task02'; n=3; norm='minmax'},
  @{s='btum3';      d="$raw\brain-tumor\Training";  t='task03'; n=3; norm='minmax'},
  @{s='panc';       d="$raw\pancreas";              t='task01'; n=2; norm='hu'},
  @{s='panclesion'; d="$raw\pancreatic-lesion";     t='task01'; n=2; norm='hu'},
  @{s='pros1';      d="$raw\prostate\Training";     t='task01'; n=6; norm='minmax'},
  @{s='pros2';      d="$raw\prostate\Training";     t='task02'; n=6; norm='minmax'}
)

# ---------- 1. 预处理 + 划分（CPU，几分钟）----------
foreach ($T in $TASKS) {
  Write-Output ("=== 预处理 {0} ({1}, {2} raters) ===" -f $T.s, $T.t, $T.n) | Tee-Object $log -Append
  & $py preprocess\build_qubiq.py --raw-dir $T.d --out-dir ("data\qubiq\{0}\npz" -f $T.s) `
        --task $T.t --n-raters $T.n --normalize $T.norm 2>&1 | Tee-Object $log -Append
  & $py preprocess\make_splits.py --npz_dir ("data\qubiq\{0}\npz" -f $T.s) `
        --out_dir ("data\qubiq\{0}\splits" -f $T.s) --n_folds 5 2>&1 | Tee-Object $log -Append
}

# ---------- 2. 训练：7 任务 x 3 臂 x 5 折 = 105 个模型 ----------
# 单个模型在这些小数据集上很快；先跑一个计时再决定要不要过夜。
foreach ($T in $TASKS) {
  foreach ($arm in @('A','B','C')) {
    foreach ($k in 0..4) {
      $name = "qubiq_{0}_{1}_fold{2}" -f $T.s, $arm, $k
      $cfg  = "configs\qubiq\{0}_{1}.yaml" -f $T.s, $arm
      Write-Output ("===== [{0}] {1} =====" -f (Get-Date -Format 'HH:mm:ss'), $name) | Tee-Object $log -Append
      & $py scripts\train.py --config $cfg --base configs\default.yaml --name $name --split-tag ("fold{0}" -f $k) 2>&1 |
        Tee-Object $log -Append
    }
  }
}

# ---------- 3. 评测 ----------
foreach ($T in $TASKS) {
  $pref = ("qubiq_{0}_A,qubiq_{0}_B,qubiq_{0}_C" -f $T.s)
  & $py scripts\eval_qubiq.py --prefixes $pref --n-raters $T.n `
        --out-dir ("outputs\analysis\qubiq_{0}" -f $T.s) --folds 0,1,2,3,4 2>&1 | Tee-Object $log -Append
}

Write-Output '=== 完成。接下来需要把新任务接进 make_table_generalisation.py 的 KID/BRA 结构里 ==='
Write-Output '   （那一步我来做，把 outputs\analysis\qubiq_* 的目录名发我即可）'
