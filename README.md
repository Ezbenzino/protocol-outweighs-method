# Lung Nodule Segmentation — Protocol Effects on LIDC-IDRI

**Protocol outweighs method: patch sampling, evaluation scope, consensus level and threshold dominate architecture and supervision-target choice in lung nodule segmentation on LIDC-IDRI.**

This repository contains the full pipeline for the paper: training, evaluation, analysis, figures, and the manuscript. Every number reported in the paper is reproducible from the released per-case results CSVs and the analysis scripts. Manuscript version **v22** (2026-10-03), formatted for submission to *Computerized Medical Imaging and Graphics* (CMIG). The earlier target, *Computers in Biology and Medicine*, was removed from Web of Science (SCIE) on 2025-11-17. All point estimates use the **case** as the unit of analysis.

---

## 1. Repository layout

```
configs/     training/evaluation YAML configs
data/        processed data (LIDC npz patches, splits, QUBIQ) — NOT distributed, see §2
outputs/
  analysis/            QUBIQ 8-point analysis + generalisation + disagreement index
                       (per_sample_*.csv here is a DAMAGED partial copy from 2026-08-31 — use analysis_full/)
  analysis_full/       main 5-fold analysis + per-case CSVs + noise floor + Holm table
  analysis_scope/      field-of-view ladder + position-prior offsets
  analysis_test/       held-out test set (158 cases) — primary inference
  figures/             paper figures (fig1–fig9; PDF+PNG)
  paper/               manuscript revision/verification scripts + metric traceability
                       (the manuscript docx itself is not distributed with this repository)
  release/             aggregated release package + `manifest.json` (per-file SHA-256)
scripts/     every stage, numbered by pipeline order (see §3)
src/         package (data / engine / losses / models / utils)
tests/       unit tests
```

This repository distributes **code, configs and aggregated results only**. `data/`, `docs/`, `external_repos/`, `refs/` and the manuscript binaries (`.docx` / `.pptx` / `.pdf`) are excluded by `.gitignore`; see §2 for how to obtain the data and §3 for the full reproduction path.

## 2. Data

- **LIDC-IDRI**: public (The Cancer Imaging Archive). Parsing yields 1,018 CT series; after the nodule definition 1,010 enter the study: 808 cases in 5-fold cross-validation (`data/splits/`, 634 of them contain a qualifying nodule) plus a 202-case held-out test set (158 contain a qualifying nodule), nodule-level patches from `data/processed/patches/*.npz`. Preprocessing: `preprocess/` (see §3, step 1–2).
- **QUBIQ 2021 (7 multi-rater tasks)**: brain-growth (7 raters), kidney (3 raters), brain-tumour task 1–3 (3 raters each), prostate task 1–2 (6 raters each), downloaded from grand-challenge.org under the challenge terms (see manuscript §3.8/§5.5). Data lives in `outputs/analysis/qubiq_*`.

Per-case results are released as CSV (`outputs/analysis_full/per_sample_val_fold0-4.csv` for cross-validation, `outputs/analysis_test/per_sample_test.csv` for the held-out test set; columns: `run, fold, case_id, threshold, dice_v1..v4, gt_area_v1..v4, area_pred`). **All paper confidence intervals can be recomputed from these CSVs**; `scripts/noise_recompute.py` and `scripts/stat_significance.py` show how.

## 3. End-to-end reproduction path (from raw LIDC to every paper table)

> Commands assume the repo root and the conda env from `environment.yml`. GPU: one CUDA GPU (reproduced on RTX 5060 Laptop, 8 GB).

| Step | Script | Produces |
|---|---|---|
| 1. Build LIDC patches | `preprocess/` + `scripts/compress_npz.py` | `data/processed/patches/*.npz` |
| 2. 5-fold splits | `preprocess/make_splits.py` (or `data/splits/*.csv`) | fold 0–4 |
| 3. Train all arms (5-fold) | `scripts/train.py --config configs/default.yaml` (per arm) | `outputs/runs/<arm>_fold*` |
| 4. Evaluate (threshold sweep) | `scripts/eval_symmetric.py` | `outputs/analysis_full/per_sample_val_fold*.csv` |
| 5. Cross-validation matrix / effect sizes | `scripts/analyze_symmetric.py` → `symmetric_analysis.json` | Table 3/5/6/8 numbers (25.58, 9.42, 2.65, 0.96, ...) |
| 6. Held-out test evaluation | `scripts/eval_symmetric.py --split test` → `analysis_test/` | **primary inference, five-fold ensemble** (55 runs = 11 configs × 5 folds; 28.58 / 3.12 / 1.19; ratios 7.1×/18.7×) |
| 7. Noise floor (3 seeds × 2 folds) | `scripts/noise_recompute.py` + `save_noise_floor.py` | `noise_floor.json` (2σ = 1.19) |
| 8. Multiple-comparison control | `scripts/rebuild_holm_table.py` (Holm over the 20 contrasts of §4.3; Table 4 shows 16) | `analysis_full/holm_table3.json` (11/20 survive; rebuilt at case level 2026-10-02) |
| 9. Field-of-view ladder + merged arm | `scripts/eval_scope.py` + `analyze_scope.py` | `analysis_scope/scope_analysis_merged2.json` (43.54 / 22.29; each arm at its case-level validation-calibrated threshold, re-scored 2026-10-02) |
| 10. Position prior (offset ladder) | `scripts/diag_offset.py` | `analysis_scope/diag_offset_allarms.json` (median 26.40 @ 8 px); paired statistics in `offset_paired_stats.json` (`scripts/offset_paired_stats.py`) |
| 11. QUBIQ training & evaluation (7 tasks, 18 A/B/C configs) | `scripts/gen_qubiq_configs.py` + `run_qubiq_expand.ps1` + `eval_qubiq.py` | `outputs/analysis/qubiq_*/` per-case CSVs |
| 12. QUBIQ effects + generalisation | `scripts/make_table_generalisation.py` + `build_paper_numbers.py` | `generalisation_table.json` (8 points), `qubiq_all_effects.json` |
| 13. Trend + disagreement index | `scripts/qubiq_agreement_index.py` + `compute_disagreement.py` | `qubiq_trend.json` (ρ = −0.786, p = 0.048), `qubiq_agreement_index.json`, `disagreement_index.json` (8 points) |
| 14. Figures | `scripts/make_figure_source_data.py` (curves for Figs 2 and 6) + `scripts/make_figures_v22.py` (all nine figures; the older per-figure scripts are superseded) | `outputs/figures/v22/Fig1–9` (png + pdf) + `figure_values_v22.json` |
| 14b. Figure input JSON | `scripts/build_paper_numbers.py` | `outputs/analysis/paper_numbers_v11.json` |
| 15. Manuscript | `outputs/paper/论文_协议效应_v22_CMIG.docx` (authoritative) + `论文_协议效应_v22_补充材料.docx`, generated from v21 by `scripts/revise_paper_v22.py`, checked by `scripts/verify_paper_v22.py` | |

**Number traceability**: `scripts/revise_paper_v22.py` reads every number it writes into the manuscript from the analysis JSONs and asserts it against the text; `outputs/figures/v22/figure_values_v22.json` records every number drawn on a figure; `outputs/analysis_full/cross_validation_report.json` independently rebuilds the headline values (case-level, 13/13 pass). If a number disagrees, treat the docx (via the source file) as authoritative and investigate before changing anything.

## 4. Verification of headline results

Primary inference is the held-out test set (thresholds imported from the validation folds, not re-selected); cross-validation is reported as the development analysis.

| Claim | Test (primary) | 5-fold CV | Source |
|---|---|---|---|
| Evaluation-target effect (fixed 0.5, G₁→G₄) | **28.58 pts** | 25.58 pts | `analysis_test/`, `symmetric_analysis.json` table1[tgtA] |
| Supervision-target range (5 arms) | **3.12 pts** | 2.65 pts | table1[tgtA-E][4].dice_opt |
| Architecture range (5 models) | **1.19 pts** | 0.96 pts | table1[archR18..tgtB][2].dice_opt |
| Protocol ≫ learning design ratio | **7.1× / 18.7×** | — | scope merged arm ÷ supervision / architecture |
| Threshold effect (arm A, G₄) | **9.99 pts** | 9.42 pts | table1[tgtA][4] dice_opt vs dice_05 |
| Noise floor 2σ | — | 1.19 pts (CI [0.97, 1.59]) | `noise_floor.json` |
| Field-of-view cost | **43.54 (tgtBshift) / 22.29 (tgtBshift_bg), V≥2** | — | `analysis_scope/scope_analysis_merged2.json` |
| Position-prior cost @ 8 px offset | **median 26.40 (10 arms, fold-0 models)** | — | `analysis_scope/diag_offset_allarms.json` |
| Disagreement index (8 points) | — | LIDC 0.333; QUBIQ 0.175–0.960 | `disagreement_index.json` |
| Protocol effect vs agreement (8 points) | — | **ρ = −0.86, exact perm. p = 0.011** (7 QUBIQ: ρ = −0.79, p = 0.048) | `qubiq_trend.json` + LIDC reference (`generalisation_table.json`) |

## 5. Code and data release

- Per-case CSVs are ~1.1 GB in total (the held-out test set alone is 3.74 M rows). They are the strongest reproducibility asset, so they are published in full on Zenodo rather than downsampled.
- **Archiving.** `.zenodo.json` carries the record metadata; enabling the GitHub–Zenodo integration archives each release automatically and mints the DOI. To build the upload payload by hand, run `python scripts/build_zenodo_release.py`, which stages `dist/zenodo/` with the aggregated release package, the per-case CSVs, `CHECKSUMS.sha256`, `zenodo-metadata.json` and a deposit README. Every packed file is verified against `outputs/release/manifest.json` first, and the script aborts on any size or SHA-256 mismatch.
- This repository is public. The Zenodo DOI is assigned at submission time (author step) and will be added here once available.
- `scripts/release_prep.py` was run before publication to strip absolute paths and check for leaked local information. The scripts here resolve the interpreter from `$env:SEG_PYTHON` (falling back to `python` on `PATH`) and the dataset root from `$env:SEG_DATA_ROOT`, so no machine-specific paths are baked in. Rerun the checker after any change that adds paths.
- Aggregated analysis JSONs + paper figures are staged in `outputs/release/` (see `outputs/release/README.md` for the v22 key numbers and QUBIQ 8-point results, and `outputs/release/manifest.json` for per-file SHA-256 sums and the `source_commit` these results were generated from).

## 6. License / terms

Code: MIT — see [LICENSE](LICENSE). Data: per LIDC-IDRI TCIA terms and QUBIQ grand-challenge terms (see manuscript Data Availability). Manuscript text: CC BY 4.0.

---

*Internally, the manuscript docx is the single source of truth: edit the docx (or the revision scripts that regenerate it), not any markdown mirror.*
