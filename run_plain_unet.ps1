$env:PYTHON = (if ($env:SEG_PYTHON) { $env:SEG_PYTHON } else { 'python' })
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot
Write-Host "Starting plain_unet..."
& $env:PYTHON scripts/train.py --config configs/plain_unet.yaml --base configs/default.yaml --name plain_unet
Write-Host "plain_unet exited with code $LASTEXITCODE"
