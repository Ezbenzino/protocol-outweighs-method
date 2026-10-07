"""
Real-time converter: read nnU-Net training_log.txt and write TensorBoard events.
Runs in background, updates every 30 seconds.
"""
import time
import os
import re
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
# nnU-Net log directory (override via env var NNUNET_LOG_DIR; default assumes local datasets mirror)
nnunet_log_dir = Path(os.environ.get(
    'NNUNET_LOG_DIR',
    str(ROOT / 'data' / 'nnunet_results' / 'Dataset101_LIDC' / 'nnUNetTrainer__nnUNetPlans__2d' / 'fold_0')
))
# TensorBoard output directory (shared with our other runs)
tb_dir = ROOT / 'outputs' / 'runs' / 'nnunet_2d_fold0' / 'logs'
tb_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=str(tb_dir))
print(f"TensorBoard writer started: {tb_dir}")

last_epoch = -1
log_file = None

while True:
    try:
        # Find the latest training log
        log_files = sorted(nnunet_log_dir.glob("training_log_*.txt"))
        if not log_files:
            time.sleep(30)
            continue
        
        current_log = log_files[-1]
        
        # Read new lines
        with open(current_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse epoch data
        current_epoch = None
        train_loss = None
        val_loss = None
        pseudo_dice = None
        epoch_time = None
        lr = None
        
        for line in lines:
            # Epoch marker
            m = re.search(r'Epoch (\d+)', line)
            if m:
                # If we have data for previous epoch, write it
                if current_epoch is not None and train_loss is not None and current_epoch > last_epoch:
                    writer.add_scalar('Loss/train', train_loss, current_epoch)
                    if val_loss is not None:
                        writer.add_scalar('Loss/val', val_loss, current_epoch)
                    if pseudo_dice is not None:
                        writer.add_scalar('Dice/pseudo', pseudo_dice, current_epoch)
                    if epoch_time is not None:
                        writer.add_scalar('Time/epoch', epoch_time, current_epoch)
                    if lr is not None:
                        writer.add_scalar('LR', lr, current_epoch)
                    last_epoch = current_epoch
                    print(f"  Wrote epoch {current_epoch}: train_loss={train_loss}, val_loss={val_loss}, dice={pseudo_dice}")
                
                current_epoch = int(m.group(1))
                train_loss = None
                val_loss = None
                pseudo_dice = None
                epoch_time = None
            
            # Learning rate
            m = re.search(r'Current learning rate: ([\d.eE+-]+)', line)
            if m:
                lr = float(m.group(1))
            
            # Train loss
            m = re.search(r'train_loss ([-\d.eE+-]+)', line)
            if m:
                train_loss = float(m.group(1))
            
            # Val loss
            m = re.search(r'val_loss ([-\d.eE+-]+)', line)
            if m:
                val_loss = float(m.group(1))
            
            # Pseudo dice
            m = re.search(r'Pseudo dice \[np\.float32\(([-\d.eE+-]+)\)\]', line)
            if m:
                pseudo_dice = float(m.group(1))
            
            # Epoch time
            m = re.search(r'Epoch time: ([\d.]+) s', line)
            if m:
                epoch_time = float(m.group(1))
        
        # Write last epoch if not yet written
        if current_epoch is not None and train_loss is not None and current_epoch > last_epoch:
            writer.add_scalar('Loss/train', train_loss, current_epoch)
            if val_loss is not None:
                writer.add_scalar('Loss/val', val_loss, current_epoch)
            if pseudo_dice is not None:
                writer.add_scalar('Dice/pseudo', pseudo_dice, current_epoch)
            if epoch_time is not None:
                writer.add_scalar('Time/epoch', epoch_time, current_epoch)
            if lr is not None:
                writer.add_scalar('LR', lr, current_epoch)
            last_epoch = current_epoch
            print(f"  Wrote epoch {current_epoch}: train_loss={train_loss}, val_loss={val_loss}, dice={pseudo_dice}")
        
        writer.flush()
        
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(30)
