# 渲染验证：用 Word COM 把论文 docx 另存为 PDF（pymupdf 扫残留旧数字用）
# 用法: powershell scripts\_render_pdf.ps1 [-Docx <路径>] [-Pdf <路径>]
param(
    [string]$Docx = (Join-Path $PSScriptRoot '..\outputs\paper\论文_协议效应_v13_CIBM.docx'),
    [string]$Pdf  = (Join-Path $PSScriptRoot '..\outputs\paper\论文_协议效应_v13_CIBM_preview.pdf')
)
$ErrorActionPreference = 'Stop'
Get-Process WINWORD -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open((Resolve-Path $Docx).Path)
$doc.SaveAs([ref]$Pdf, [ref]17)
$doc.Close($false)
$word.Quit()
Write-Output 'rendered ok'
