"""3D U-Net smoke test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.dataset3d import LungNoduleDataset3D
from src.models.unet3d import UNet3D
from src.utils.config import load_config
from src.losses.dice import soft_dice_loss, bce_with_logits

cfg = load_config('configs/unet3d.yaml')
print('Config loaded OK')

ds = LungNoduleDataset3D(cfg.data.npz_dir, 'data/splits/train_smoke.csv', cfg, phase='train')
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Image shape: {sample["image"].shape}')
print(f'Majority mask shape: {sample["majority_mask"].shape}')
print(f'Consensus shape: {sample["consensus"].shape}')
print(f'Diameter: {sample["diameter_mm"].item():.1f}mm')

device = torch.device('cuda')
model = UNet3D(in_channels=1, num_classes=1, base_channels=32).to(device)
x = sample['image'].unsqueeze(0).to(device)
print(f'Input shape: {x.shape}')
region, boundary = model(x)
print(f'Output shape: {region.shape}, boundary={boundary}')

prob = torch.sigmoid(region)
maj = sample['majority_mask'].unsqueeze(0).to(device)
con = sample['consensus'].unsqueeze(0).to(device)
loss_dice = soft_dice_loss(prob, maj)
loss_bce = bce_with_logits(region, con)
print(f'Loss dice: {loss_dice.item():.4f}, bce: {loss_bce.item():.4f}')
loss = loss_dice + loss_bce
loss.backward()
print('Backward pass OK')

# Test DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))
print(f'Batch image shape: {batch["image"].shape}')
out, _ = model(batch['image'].to(device))
print(f'Batch output shape: {out.shape}')
print('ALL 3D SMOKE TESTS PASSED')
