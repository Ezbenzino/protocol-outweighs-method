# v16 提交脚本 —— 在 Windows PowerShell 里跑（我这边删不掉 .git\index.lock，只能你本机提交）
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Test-Path .git\index.lock) { Remove-Item .git\index.lock -Force }
Get-ChildItem .git\objects -Recurse -Filter 'tmp_obj_*' | Remove-Item -Force -ErrorAction SilentlyContinue
git add -A
git commit -F docs\_commit_v17_message.txt
git log --oneline | Select-Object -First 3
