"""Split cases into train/val/test by case id (no slice leakage).

Modes:
  1. Single split (default): train/val/test at 70/15/15 (stratified by nodule presence).
  2. K-fold CV + fixed test: --n_folds N. A fixed 20% test hold-out is split off
     first (stratified), the remaining 80% is used for N-fold CV, producing
     train_fold{K}.csv / val_fold{K}.csv plus test.csv.
     Use --fold K to generate a single fold (default -1 = generate all folds).
"""
import argparse
import glob
import json
import os
import random


def _has_nodule(npz_dir, case):
    with open(os.path.join(npz_dir, f"{case}.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return 1 if meta.get("nodules") else 0


def main(npz_dir, out_dir, train=0.7, val=0.15, seed=42, stratify=False,
         n_folds=0, fold=-1, test_frac=0.2):
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(npz_dir, "*.npz"))
    )
    random.seed(seed)

    if n_folds > 1:
        from sklearn.model_selection import StratifiedKFold, train_test_split
        has_nod = [_has_nodule(npz_dir, c) for c in cases]
        cv_pool, test = train_test_split(
            cases, test_size=test_frac, random_state=seed, stratify=has_nod)
        with open(os.path.join(out_dir, "test.csv"), "w", encoding="utf-8") as f:
            f.write("case_id\n" + "\n".join(sorted(test)) + "\n")
        pool_idx = {c: i for i, c in enumerate(cv_pool)}
        pool_has = [has_nod[cases.index(c)] for c in cv_pool]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (tr_idx, va_idx) in enumerate(skf.split(cv_pool, pool_has)):
            if fold != -1 and k != fold:
                continue
            tr = [cv_pool[i] for i in tr_idx]
            va = [cv_pool[i] for i in va_idx]
            with open(os.path.join(out_dir, f"train_fold{k}.csv"), "w", encoding="utf-8") as f:
                f.write("case_id\n" + "\n".join(sorted(tr)) + "\n")
            with open(os.path.join(out_dir, f"val_fold{k}.csv"), "w", encoding="utf-8") as f:
                f.write("case_id\n" + "\n".join(sorted(va)) + "\n")
            print(f"fold {k}: train={len(tr)} val={len(va)} (test={len(test)} hold-out)")
        return

    if stratify:
        from sklearn.model_selection import train_test_split
        has_nod = [_has_nodule(npz_dir, c) for c in cases]
        rest, test = train_test_split(
            cases, test_size=1 - train - val, random_state=seed, stratify=has_nod)
        rest_has = [has_nod[cases.index(c)] for c in rest]
        tr, va = train_test_split(
            rest, test_size=val / (train + val), random_state=seed, stratify=rest_has)
        splits = {"train": tr, "val": va, "test": test}
    else:
        random.shuffle(cases)
        n = len(cases)
        nt = int(n * train)
        nv = int(n * val)
        splits = {
            "train": cases[:nt],
            "val": cases[nt:nt + nv],
            "test": cases[nt + nv:],
        }

    for name, ids in splits.items():
        with open(os.path.join(out_dir, f"{name}.csv"), "w", encoding="utf-8") as f:
            f.write("case_id\n" + "\n".join(ids) + "\n")
    print(f"split {len(cases)} cases: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--n_folds", type=int, default=0, help="k-fold CV (over 80% of cases)")
    ap.add_argument("--fold", type=int, default=-1, help="-1 = all folds")
    args = ap.parse_args()
    main(args.npz_dir, args.out_dir, args.train, args.val, args.seed,
         args.stratify, args.n_folds, args.fold)
