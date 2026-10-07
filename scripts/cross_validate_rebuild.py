# -*- coding: utf-8 -*-
"""cross_validate_rebuild.py — v20 独立复核：论文头条数字 vs 从权威 JSON 独立重算。

与旧版（v16 防回归）的区别：
  - 旧版 check 4/5/6 是 paper==paper 自比较，且口径停留在 8 臂/8.03/3.24x；
  - 本版所有 rebuilt 从 table1 / table4 / scope JSON 独立计算，paper 值为 v20 正文权威值；
  - 覆盖 CV 与 held-out test 两套口径；
  - 覆盖 supervision range (A-E @V4)、architecture range (5 models @V2)、
    evaluation-target effect (fixed 0.5, V1-V4)、threshold effect (tgtA @V4)、
    以及头条比值 7.1x / 18.7x（v22）。
  - 2026-10-03：视野分子改为从 scope JSON 的 table4_paired 独立读取（旧版的 runs 键并不存在，
    实际一直落到写死的 22.51 兜底值，等于拿论文数字自比）；新增视野两行与十臂位移中位数的复核。

用法：python scripts/cross_validate_rebuild.py
退出码：全部通过 0，任一失败 1。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CV_SRC = os.path.join(ROOT, 'outputs', 'analysis_full', 'symmetric_analysis.json')
TEST_SRC = os.path.join(ROOT, 'outputs', 'analysis_test', 'symmetric_analysis_applied.json')
SCOPE_SRC = os.path.join(ROOT, 'outputs', 'analysis_scope', 'scope_analysis_merged2.json')
OUT = os.path.join(ROOT, 'outputs', 'analysis_full', 'cross_validation_report.json')

SUP = ['tgtA', 'tgtB', 'tgtC', 'tgtD', 'tgtE']
# 2026-10-02：CV 与 test 都用选定学习率的 plain U-Net（archPU1e3）；v22 正文已按新结果更新（1.19 / 18.7x）。
ARCH_CV = ['tgtB', 'archR18', 'archCNX', 'archPVT', 'archPU1e3']
ARCH_TEST = ['tgtB', 'archR18', 'archCNX', 'archPVT', 'archPU1e3']


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def range_pts(t1, arms, target, key='dice_opt'):
    """从 table1 独立计算指定 arms 在指定 consensus target 下的 dice 极差（points）。"""
    vals = [t1[a][str(target)][key] for a in arms if a in t1 and str(target) in t1[a]]
    if len(vals) < 2:
        return None
    return 100 * (max(vals) - min(vals))


def eval_target_fixed05(t1, t4, arm='tgtA'):
    """评测靶效应 fixed 0.5：V>=1 到 V>=4 的 Dice 落差。优先 table4，否则从 table1 算。"""
    if arm in t4 and 'fixed05' in t4[arm]:
        return t4[arm]['fixed05'][2] * 100
    return 100 * (t1[arm]['1']['dice_05'] - t1[arm]['4']['dice_05'])


def threshold_effect(t1, arm='tgtA', target=4):
    """阈值效应：arm @ target 下 dice_opt - dice_05。"""
    return 100 * (t1[arm][str(target)]['dice_opt'] - t1[arm][str(target)]['dice_05'])


def add(checks, name, rebuilt, paper, tol=0.05, note=''):
    ok = (rebuilt is not None) and (paper is not None) and (abs(rebuilt - paper) < tol)
    checks.append({
        'check': name,
        'rebuilt': round(rebuilt, 3) if rebuilt is not None else None,
        'paper': round(paper, 3) if paper is not None else None,
        'tol': tol,
        'ok': ok,
        'note': note,
    })
    return ok


def main():
    cv = load(CV_SRC)
    test = load(TEST_SRC)
    scope = load(SCOPE_SRC)

    cv_t1, cv_t4 = cv['table1'], cv['table4']
    test_t1, test_t4 = test['table1'], test.get('table4', {})

    # 视野效应 —— 头条比值的分子。直接取 scope JSON 的逐病例配对表（table4_paired，
    # V>=2，win128 -> win512 段），与 effect_uncertainty.py 的计算相互独立；不设兜底值。
    def fov(arm):
        r = [x for x in scope['table4_paired']
             if x['arm'] == arm and x['level'] == 2 and x['segment'] == '纯视野效应']
        assert len(r) == 1, arm
        return abs(r[0]['diff'])
    fov_conservative = fov('tgtBshift_bg')
    fov_position_invariant = fov('tgtBshift')
    # 十臂位移中位数：直接从 diag_offset_allarms.json 的 fold0 曲线重算
    off = load(os.path.join(ROOT, 'outputs', 'analysis_scope', 'diag_offset_allarms.json'))['runs']
    drops = sorted(100 * (r['mean_by_offset']['0'] - r['mean_by_offset']['8'])
                   for k, r in off.items() if 'shift' not in k)
    assert len(drops) == 10, len(drops)
    offset_median = (drops[4] + drops[5]) / 2

    checks = []

    # ---- CV 口径 ----
    add(checks, 'CV: supervision range A-E @V4 calibrated',
        range_pts(cv_t1, SUP, 4), 2.65,
        note='five controlled supervision targets, max-min dice_opt at V>=4')
    add(checks, 'CV: architecture range 5 models @V2 calibrated',
        range_pts(cv_t1, ARCH_CV, 2), 0.96,
        note='tgtB/R18/CNX/PVT/PU1e3, max-min dice_opt at V>=2')
    add(checks, 'CV: evaluation-target effect fixed0.5 V1-V4 (arm A)',
        eval_target_fixed05(cv_t1, cv_t4, 'tgtA'), 25.58,
        note='table4 fixed05 gap, V>=1 minus V>=4')
    add(checks, 'CV: threshold effect tgtA @V4 (0.5 vs optimal)',
        threshold_effect(cv_t1, 'tgtA', 4), 9.42,
        note='dice_opt minus dice_05 at V>=4')

    # ---- TEST 口径 ----
    add(checks, 'TEST: supervision range A-E @V4 calibrated',
        range_pts(test_t1, SUP, 4), 3.12,
        note='held-out test set, five supervision targets')
    add(checks, 'TEST: architecture range 5 models @V2 calibrated',
        range_pts(test_t1, ARCH_TEST, 2), 1.19,
        note='held-out test set, tgtB/R18/CNX/PVT/PU1e3')
    add(checks, 'TEST: evaluation-target effect fixed0.5 V1-V4 (arm A)',
        eval_target_fixed05(test_t1, test_t4, 'tgtA'), 28.58,
        note='held-out test set')
    add(checks, 'TEST: threshold effect tgtA @V4 (0.5 vs optimal)',
        threshold_effect(test_t1, 'tgtA', 4), 9.99,
        note='held-out test set; Fig.3 uses this value (test-set figure)')

    # ---- 视野与位移（test 口径，各臂验证集校准阈值）----
    add(checks, 'TEST: FOV win128->win512, both confounds removed (merged arm)',
        fov_conservative, 22.29, note='scope table4_paired, tgtBshift_bg @V>=2, thr 0.65')
    add(checks, 'TEST: FOV win128->win512, translation-augmented arm',
        fov_position_invariant, 43.54, note='scope table4_paired, tgtBshift @V>=2, thr 0.55')
    add(checks, 'TEST: offset d0->d8, median of ten arms (fold0)',
        offset_median, 26.40, note='diag_offset_allarms.json, each arm at its calibrated threshold')

    # ---- 头条比值（test 口径，分子=视野保守 22.29）----
    test_sup = range_pts(test_t1, SUP, 4)
    test_arch = range_pts(test_t1, ARCH_TEST, 2)
    ratio_sup = fov_conservative / test_sup if test_sup else None
    ratio_arch = fov_conservative / test_arch if test_arch else None
    add(checks, 'HEADLINE: FOV / supervision-target ratio (test)',
        ratio_sup, 7.1, tol=0.06,
        note=f'= {fov_conservative:.2f} / {test_sup:.2f}; paper reports 7.1x')
    add(checks, 'HEADLINE: FOV / architecture ratio (test)',
        ratio_arch, 18.7, tol=0.06,
        note=f'= {fov_conservative:.2f} / {test_arch:.2f}; paper reports 18.7x')

    # ---- 输出 ----
    n_ok = sum(1 for c in checks if c['ok'])
    n_total = len(checks)
    report = {
        'date': '2026-10-03',
        'version': 'v22 case-level, independent rebuild',
        'sources': {
            'cv': 'outputs/analysis_full/symmetric_analysis.json',
            'test': 'outputs/analysis_test/symmetric_analysis_applied.json',
            'scope': 'outputs/analysis_scope/scope_analysis_merged2.json',
        },
        'method': 'all rebuilt values computed independently from table1/table4/scope; paper values are v22 manuscript numbers',
        'passed': n_ok,
        'total': n_total,
        'checks': checks,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'独立复核 {n_ok}/{n_total} passed')
    for c in checks:
        mark = 'OK ' if c['ok'] else 'FAIL'
        print(f"  [{mark}] {c['check']}: rebuilt={c['rebuilt']} vs paper={c['paper']}"
              + (f"  ({c['note']})" if c['note'] else ''))
    if n_ok < n_total:
        print('!! 存在不一致，先排查再发布。')
        sys.exit(1)
    print(f'报告已写入: {OUT}')


if __name__ == '__main__':
    main()
