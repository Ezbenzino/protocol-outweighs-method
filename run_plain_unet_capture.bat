@echo off
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
echo Starting plain_unet with error capture at %date% %time% > outputs\runs\plain_unet_crash.log
python scripts/train.py --config configs/plain_unet.yaml --base configs/default.yaml --name plain_unet 2>> outputs\runs\plain_unet_crash.log
echo Exit code: %ERRORLEVEL% at %date% %time% >> outputs\runs\plain_unet_crash.log
