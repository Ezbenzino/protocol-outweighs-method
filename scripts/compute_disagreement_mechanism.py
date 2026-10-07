# -*- coding: utf-8 -*-
"""compute_disagreement_mechanism.py v2 — 53.7%/0.7% 的逐 reader 覆盖复现 (先试点)

方法:
  1. case_id -> series_uid 映射: 用「标注像素总面积」最近邻匹配 (npz 的 sum(consensus)
     == annotations 全部 ROI 面积之和), 多候选用 ROI/投票质心消歧。
  2. 对每个 series 的每个 session(reader), 把所有 ROI 多边形 rasterize 到 512x512
     平面并叠加 -> reader 平面标注图 (跨 slice 投影; 对"检测层面分歧"判断合理)。
  3. 对每个结节实例 (json: slice, cy, cx + V>=2 连通域), 统计覆盖它的 reader 数 r:
     r = #{ reader: 该 reader 平面标注图 与 实例 mask 交集非空 }。
  4. 统计: r in {2,3} 比例 (论文 53.7%); r==4 且实例内无 vote=4 像素 比例 (论文 0.7%)。

用法: skin_seg python scripts/compute_disagreement_mechanism.py --limit 50
"""
import argparse
import json
import os

import numpy as np
from skimage.draw import polygon
from skimage.measure import label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_DIR = os.path.join(ROOT, 'data', 'processed', 'npz')
ANN = os.path.join(ROOT, 'data', 'processed', 'annotations.json')
OUT = os.path.join(ROOT, 'outputs', 'analysis', 'disagreement_mechanism.json')
MAP = os.path.join(ROOT, 'outputs', 'analysis', 'case_series_map.json')

with open(ANN, encoding='utf-8') as f:
    ANNS = json.load(f)


def raster(points):
    rr, cc = polygon([p[1] for p in points], [p[0] for p in points], (512, 512))
    m = np.zeros((512, 512), dtype=bool)
    m[rr, cc] = True
    return m


def series_reader_masks(suid):
    """返回 [(reader_id, 平面 bool mask)] —— 该 reader 所有 ROI 的投影并集。"""
    out = []
    for sess in ANNS.get(suid, {}).get('sessions', []):
        m = np.zeros((512, 512), dtype=bool)
        for nod in sess.get('nodules', []):
            for roi in nod.get('rois', []):
                m |= raster(roi['points'])
        out.append((sess.get('doctor', '?'), m))
    return out


def build_map(limit=None):
    """面积最近邻 + 质心消歧, 返回 {case_id: suid}"""
    # ann 侧: 面积 -> series 列表; 以及质心
    ann_area = {}
    ann_cent = {}
    for uid, v in ANNS.items():
        tot = 0
        wsum = np.zeros(2)
        wcnt = 0
        for sess in v.get('sessions', []):
            for nod in sess.get('nodules', []):
                for roi in nod.get('rois', []):
                    pts = roi['points']
                    m = raster(pts)
                    a = int(m.sum())
                    tot += a
                    if a:
                        ys, xs = np.where(m)
                        wsum += np.array([ys.sum(), xs.sum()])
                        wcnt += a
        ann_area.setdefault(tot, []).append(uid)
        ann_cent[uid] = (wsum / max(wcnt, 1)) if wcnt else np.array([256.0, 256.0])

    files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith('.npz'))
    if limit:
        files = files[:limit]
    mp = {}
    for fn in files:
        case_id = fn[:-4]
        jp = os.path.join(NPZ_DIR, case_id + '.json')
        if not os.path.exists(jp):
            continue
        with open(jp, encoding='utf-8') as f:
            meta = json.load(f)
        if not meta.get('nodules'):
            mp[case_id] = None
            continue
        cons = np.load(os.path.join(NPZ_DIR, fn))['consensus']
        t = int(cons.sum())
        # 面积最近邻
        cands = sorted(ann_area.keys(), key=lambda k: abs(k - t))[:5]
        best, best_d = None, None
        for k in cands:
            for uid in ann_area[k]:
                if best is None:
                    best, best_d = uid, abs(k - t)
                    continue
                d = abs(k - t)
                if d < best_d:
                    best, best_d = uid, d
        mp[case_id] = best
    with open(MAP, 'w', encoding='utf-8') as f:
        json.dump(mp, f, indent=1, ensure_ascii=False)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    mp = build_map(args.limit)

    stats = {'n_cases': 0, 'n_instances': 0, 'r2': 0, 'r3': 0, 'r4_no_common': 0,
             'r4_common': 0, 'reader_dist': {}}
    n_nomatch = 0
    files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith('.npz'))
    if args.limit:
        files = files[:args.limit]
    for fn in files:
        case_id = fn[:-4]
        suid = mp.get(case_id)
        if not suid:
            n_nomatch += 1
            continue
        jp = os.path.join(NPZ_DIR, case_id + '.json')
        with open(jp, encoding='utf-8') as f:
            meta = json.load(f)
        nods = meta.get('nodules', [])
        if not nods:
            continue
        npz = np.load(os.path.join(NPZ_DIR, fn))
        cons = npz['consensus']
        readers = series_reader_masks(suid)
        stats['n_cases'] += 1
        stats['reader_dist'][len(readers)] = stats['reader_dist'].get(len(readers), 0) + 1
        cache = {}
        for nod in nods:
            sl = int(nod['slice'])
            cy, cx = float(nod['cy']), float(nod['cx'])
            if sl not in cache:
                maj = cons[sl] >= 2
                cache[sl] = label(maj, connectivity=2) if maj.any() else None
            lbl = cache[sl]
            if lbl is None:
                continue
            lid = lbl[int(round(cy)), int(round(cx))]
            if lid == 0:
                continue
            inst = lbl == lid
            r = 0
            for _, rm in readers:
                if (rm & inst).any():
                    r += 1
            has_v4 = bool((cons[sl][inst] == 4).any())
            stats['n_instances'] += 1
            if r in (2, 3):
                stats['r2' if r == 2 else 'r3'] += 1
            elif r == 4:
                if has_v4:
                    stats['r4_common'] += 1
                else:
                    stats['r4_no_common'] += 1
        npz.close()

    n = stats['n_instances']
    print(f'=== 逐 reader 覆盖复现 (试点 n={n}) ===')
    print(f'case 数: {stats["n_cases"]} (未匹配跳过: {n_nomatch})  reader分布: {stats["reader_dist"]}')
    print(f'  r=2: {stats["r2"]}')
    print(f'  r=3: {stats["r3"]}')
    print(f'  r=4 有公共: {stats["r4_common"]}')
    print(f'  r=4 无公共: {stats["r4_no_common"]}')
    if n:
        print(f'  被 2-3 位标记: {100*(stats["r2"]+stats["r3"])/n:.2f}%  (论文 53.7%)')
        print(f'  4 位标记无公共: {100*stats["r4_no_common"]/n:.2f}%  (论文 0.7%)')


if __name__ == '__main__':
    main()
