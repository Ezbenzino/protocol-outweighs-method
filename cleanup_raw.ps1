# Dataset root holding the raw LIDC-IDRI / nnU-Net trees; override with $env:SEG_DATA_ROOT.
$DataRoot = if ($env:SEG_DATA_ROOT) { $env:SEG_DATA_ROOT } else { Join-Path $HOME 'datasets' }
# Wait for preprocessing to finish, then delete the raw DICOM data (frees ~120 GB).
# The training pipeline only reads npz files, so deleting DICOMs is safe.
# XML annotations (a few MB) are kept in case you need to rebuild labels.
$ErrorActionPreference = "Continue"

$Root    = $PSScriptRoot
$NpzDir  = "$DataRoot\LIDC-IDRI-processed\npz"
$Splits  = Join-Path $Root "data\splits\train.csv"
$Raw     = "$DataRoot\LIDC-IDRI"
param([int]$Target = 1010)

Write-Host "cleanup: waiting for preprocessing (target $Target npz + splits) ..."

for ($i = 0; $i -lt 600; $i++) {
    $n = 0
    try { $n = (Get-ChildItem $NpzDir -Filter *.npz -ErrorAction SilentlyContinue).Count } catch {}
    $done = (Test-Path $Splits)
    if ($n -ge $Target -and $done) { break }
    Start-Sleep -Seconds 60
}

$n = 0
try { $n = (Get-ChildItem $NpzDir -Filter *.npz -ErrorAction SilentlyContinue).Count } catch {}
if ($n -lt $Target -or -not (Test-Path $Splits)) {
    Write-Host "cleanup: timeout waiting (npz=$n, splits=$done), skipping deletion" -ForegroundColor Yellow
    exit 2
}

Write-Host "preprocessing done (npz=$n). deleting raw DICOMs ..."
$before = 0
try { $before = ((Get-ChildItem $Raw -Recurse -File -Filter *.dcm -ErrorAction SilentlyContinue) | Measure-Object Length -Sum).Sum / 1GB } catch {}

Get-ChildItem $Raw -Recurse -File -Filter *.dcm -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# remove now-empty series subfolders
Get-ChildItem $Raw -Recurse -Directory -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending |
    ForEach-Object {
        if (-not (Get-ChildItem $_.FullName -Force -ErrorAction SilentlyContinue)) {
            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
        }
    }

$after = 0
try { $after = ((Get-ChildItem $Raw -Recurse -File -ErrorAction SilentlyContinue) | Measure-Object Length -Sum).Sum / 1GB } catch {}
$free  = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='D:'").FreeSpace / 1GB

Write-Host ("cleanup done: freed ~{0:N1} GB (dcm {1:N1}GB -> remaining {2:N2} GB xml/zip). D: free now {3:N1} GB" -f ($before - $after), $before, $after, $free) -ForegroundColor Green
