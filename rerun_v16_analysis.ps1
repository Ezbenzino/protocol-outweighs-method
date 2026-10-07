# v16 分析链复核 —— 在 skin_seg 环境重跑一遍，确认与我在 Linux 侧算出的 JSON 完全一致
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
& $py scripts\compute_disagreement.py
& $py scripts\analyze_symmetric.py --out-dir outputs\analysis_full
& $py scripts\analyze_symmetric.py --out-dir outputs\analysis_test --thr-from outputs\analysis_full\symmetric_analysis.json
& $py scripts\save_noise_floor.py
& $py scripts\make_table_generalisation.py
& $py scripts\build_paper_numbers.py
Write-Output ''
Write-Output '若上述输出与 2026-09-04 的分析表格一致，则 scipy 替身无影响。'
