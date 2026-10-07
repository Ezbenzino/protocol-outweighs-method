"""Extract all experiment metrics from training logs."""
import re, json, os

def parse_log(logpath):
    epochs, train_loss, val_dice, val_maj, val_iou, val_hd95 = [], [], [], [], [], []
    val_micro, val_small, val_medium = [], [], []
    if not os.path.exists(logpath):
        return None
    for line in open(logpath, encoding='utf-8'):
        m = re.search(r'Epoch (\d+)/\d+ \| train_loss=([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
        m = re.search(r'val \| dice=([\d.]+), dice_majority=([\d.]+), iou=([\d.]+), hd95=([\d.]+), assd=[\d.]+, dice_micro=([\d.nan]+), dice_small=([\d.nan]+), dice_medium=([\d.nan]+)', line)
        if m and len(val_maj) < len(epochs):
            val_dice.append(float(m.group(1)))
            val_maj.append(float(m.group(2)))
            val_iou.append(float(m.group(3)))
            val_hd95.append(float(m.group(4)))
            val_micro.append(float(m.group(5)) if m.group(5) != 'nan' else None)
            val_small.append(float(m.group(6)) if m.group(6) != 'nan' else None)
            val_medium.append(float(m.group(7)) if m.group(7) != 'nan' else None)
    return {'epochs': epochs, 'train_loss': train_loss, 'val_dice': val_dice,
            'val_maj': val_maj, 'val_iou': val_iou, 'val_hd95': val_hd95,
            'val_micro': val_micro, 'val_small': val_small, 'val_medium': val_medium}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base = os.path.join(ROOT, 'outputs', 'runs')
experiments = ['main', 'abl_baseline', 'abl_sitl', 'abl_csl', 'abl_brbc',
               'abl_lovasz', 'abl_uniontgt', 'base_unet',
               'main_fold0', 'main_fold1', 'main_fold2', 'main_fold3', 'main_fold4']

results = {}
for exp in experiments:
    log = os.path.join(base, exp, 'logs', 'train.log')
    data = parse_log(log)
    if data and data['val_maj']:
        best_idx = max(range(len(data['val_maj'])), key=lambda i: data['val_maj'][i])
        results[exp] = {
            'best_epoch': data['epochs'][best_idx],
            'best_dice_maj': data['val_maj'][best_idx],
            'best_dice': data['val_dice'][best_idx],
            'best_iou': data['val_iou'][best_idx],
            'best_hd95': data['val_hd95'][best_idx],
            'best_micro': data['val_micro'][best_idx],
            'best_small': data['val_small'][best_idx],
            'best_medium': data['val_medium'][best_idx],
            'num_epochs': len(data['epochs']),
            'full_epochs': data['epochs'],
            'full_train_loss': data['train_loss'],
            'full_val_maj': data['val_maj'],
        }
        print(f'{exp:20s}: best_epoch={data["epochs"][best_idx]:3d}  dice_maj={data["val_maj"][best_idx]:.4f}  micro={data["val_micro"][best_idx]}')
    else:
        print(f'{exp:20s}: NO DATA')

# Also extract full per-epoch data for main model (for training curves)
main_data = parse_log(os.path.join(base, 'main', 'logs', 'train.log'))
results['main_full'] = main_data

with open(os.path.join(ROOT, 'outputs', 'paper', 'extracted_metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved extracted_metrics.json')
