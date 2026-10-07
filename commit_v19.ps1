$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Test-Path .git\index.lock) { Remove-Item .git\index.lock -Force }
Get-ChildItem .git\objects -Recurse -Filter 'tmp_obj_*' | Remove-Item -Force -ErrorAction SilentlyContinue
git rm --cached _tmp_pipeline_t3.ps1 _tmp_qubiq_status.html _tmp_train_extra.ps1 _tmp_train_fixed.ps1 2>$null
git add -A
git commit -F docs\_commit_v19_message.txt
git log --oneline | Select-Object -First 3
git status --porcelain | Measure-Object | Select-Object -ExpandProperty Count
