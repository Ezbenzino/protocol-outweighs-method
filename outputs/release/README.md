# 逐病例结果数据包（released with this work）— v22

> **2026-10-03 v22**：配套论文 `outputs/paper/论文_协议效应_v22_CMIG.docx`。本次数据层改动三项，均为口径统一，
> 结论方向不变：① 测试集 plain U-Net 改为选定学习率的 `archPU1e3`（架构极差 1.87 → 1.19）；
> ② 8 px 位移与视野阶梯改用病例级验证集校准阈值（位移中位数 26.36 → 26.40；视野 22.51 → 22.29、43.84 → 43.54）；
> ③ 9 张插图全部由 `scripts/make_figures_v22.py` 从本包 JSON 重画，`figures/` 已整体替换（旧图移至
> `outputs/_backup_release_figures_20261003/`），并附 `figures/source_data/` 与 `figure_values_v22.json`。
> 新增文件：`scope_analysis_merged2.json`、`diag_offset_allarms.json`、`diag_offset.json`、`offset_paired_stats.json`、
> `effect_uncertainty.json`、`symmetric_analysis_test_applied.json`（保留测试集，阈值从验证集导入）。


> 重建日期：2026-09-05（release 链同步到 v20，QUBIQ 8 点扩展 + 冻结后审计修复已收口）
> 配套论文：《Protocol outweighs method...》**v20（病例级口径）**，`outputs/paper/论文_协议效应_v20_CIBM.docx`
> 配套 commit：见 `manifest.json`。
> v20 相对 v19 的变更：Fig.3 换病例级测试集版（此前是废弃实例级数字）、Fig.9 p 值统一为 8 点 0.011、
> Table 9 补标题、§4.14 读者数修正、删 7 个孤儿 media 部件（docx 4.39→2.44 MB）。本包 figures/ 已同步病例级最终版。

> **2026-10-02 补记**：论文升为 v21（改投 CMIG，所有实验数字未变）；`holm_table3.json` 按病例级重建——此前的文件生成于 2026-09-02（实例级口径），20 个对比中 12 个的差值/CI/p 与论文 Table 4 不一致。旧文件已移至 `outputs/_backup_holm_stale_20261002/`，manifest 已重算。

## 本包内容（v20 同步后）

| 路径 | 内容 | 口径 / 备注 |
|---|---|---|
| `symmetric_analysis.json` | 主分析全量：table1/3/4、效应分解、配对检验 | **病例级**，论文 Table 3/5/6/8 数字的唯一来源（21 臂） |
| `noise_floor.json` | 噪声底 σ/2σ/分靶/CI | 2σ = 1.19 [0.97, 1.59]（fixed-0.5） |
| `holm_table3.json` | Table 4（原 Table 3）二十个预设对比的 Holm 校正 | 11/20 校正后仍显著 |
| `cross_validation_report.json` | 独立复核（v22：视野分子改为从 scope JSON 读取，不再用写死值） | 13/13 通过 |
| `symmetric_analysis_test_applied.json` | 保留测试集全量（阈值从验证集导入） | 论文 §4.9、Table 8、Fig. 3 的数据源 |
| `scope_analysis_merged2.json` | 视野阶梯（§4.11/4.13、Fig. 8） | 各臂病例级验证集校准阈值（2026-10-02 重算） |
| `diag_offset_allarms.json` / `diag_offset.json` / `offset_paired_stats.json` | 位置先验（§4.10、Fig. 7） | 十臂 fold0 + tgtB/tgtBshift 五折；配对统计由 `scripts/offset_paired_stats.py` 生成 |
| `effect_uncertainty.json` | 效应量 bootstrap CI 与比值（Fig. 3 误差线、§4.5） | 病例级，5000 次 |
| `dataset_stats.json` | 划分与结节统计 | cv_test_overlap=0、fold_val_duplicates=0 |
| `generalisation_table.json` | QUBIQ 外验（Table 9 数据源，LIDC 参考点） | 8 点（LIDC + 7 QUBIQ 任务） |
| `disagreement_index.json` | 一致性指数（8 点） | LIDC 0.333 / brain-tumor t2 0.175 / brain-growth 0.489 / prostate t2 0.766 / brain-tumor t3 0.636 / prostate t1 0.960 / brain-tumor t1 0.777 / kidney 0.868 |
| `qubiq_all_effects.json` | 7 个 QUBIQ 任务的协议/监督效应（fixed0.5 + opt_thr） | 每任务含 protocol / supervision / ratio |
| `qubiq_trend.json` | 7 QUBIQ 点趋势：协议效应 vs 一致性 | **ρ = −0.786，精确置换 p = 0.048（5040 置换）**；比值趋势 ρ = −0.071（p = 0.906，不显著） |
| `qubiq_agreement_index.json` | QUBIQ 一致性指数（3D 体积口径） | ⚠️ 与论文 2D 切片口径不同（prostate t1 2D 0.96 vs 3D 0.79，论文 §3.8 已注明） |
| `svls_calibration.json` / `training_curves.json` | SVLS 核校准 / 训练曲线 | 未变 |
| `figures/` | 论文插图 Fig1–Fig9（png 600 dpi + 矢量 pdf）+ `figure_values_v22.json` + `source_data/` | 2026-10-03 由 `scripts/make_figures_v22.py` 整体重画 |
| `manifest.json` | 文件清单 + SHA-256 + 关键数字快照 | 发布核验用 |

## 关键口径（务必按此解读，与 v22 论文一致）

- **分析单元 = 病例级**：病例内多个结节实例先对 Dice 求平均，再跨病例检验（§3.9）。
  此前 v14/v15 的实例级数字已全部废弃。
- **头条数字**（held-out 测试集，**五折模型集成**，55 runs = 11 配置 × 5 折）：
  评测靶效应 25.58（CV）/ 28.58（test）、阈值效应 9.42、监督靶极差 2.65/3.12、
  架构极差 0.96/1.19、噪声底 1.19。
- **比值**：评测协议超出学习设计 7.1×（对监督）/ 18.7×（对架构），分子用保守口径 22.29；124 例公共子集联合 bootstrap 为 7.0× [5.8, 8.7] / 15.6× [9.3, 30.9]。
- **QUBIQ 8 点外验（Table 9，opt_thr）**：协议效应随标注一致性衰减——
  含 LIDC 共 8 点 ρ = −0.86（精确置换 p = 0.011）；仅 7 个 QUBIQ 任务 ρ = −0.79（p = 0.048）。
  最低一致性任务（brain-tumor t2，0.175）协议效应最大（30.35 点），最高一致性任务（prostate t1，0.960）最小（2.40 点）。
  **比值指标不可单用**：高一致性端分母趋零使比值失真（prostate t1 的 4.58× 是假象），比值趋势不显著。
- **视野效应**：43.54 点（平移增强模型）/ 22.29 点（训练侧混杂清除后）；8 px 位移十臂中位数 26.40 点（范围 7.1–35.8）；d = 40 五折配对差 +60.79 [+55.47, +65.80]。

## 复现命令（Windows / conda `skin_seg`，见 README 顶层）

```powershell
# 主分析全链（逐条跑，输出与 analysis_full/ 应逐位一致）
python scripts\compute_disagreement.py
python scripts\analyze_symmetric.py --splits-csv ...   # CV 表
python scripts\analyze_symmetric.py --thr-from ...     # 测试集
python scripts\save_noise_floor.py
python scripts\make_table_generalisation.py
python scripts\build_paper_numbers.py
python scripts\analyze_scope.py --scope outputs\analysis_scope\scope_test_merged.csv --out outputs\analysis_scope\scope_analysis_merged2.json
python scripts\offset_paired_stats.py                  # §4.10 配对统计
python scripts\effect_uncertainty.py                   # Fig. 3 误差线与比值
python scripts\cross_validate_rebuild.py               # 独立复核，13/13
python scripts\make_figure_source_data.py              # Fig. 2/6 曲线源数据
python scripts\make_figures_v22.py                     # 9 张论文插图

# QUBIQ 8 点外验
python scripts\gen_qubiq_configs.py                    # 18 个 A/B/C 配置
python scripts\qubiq_agreement_index.py                # 一致性指数（3D 口径）
```

## 发布策略（待作者决策，勿直接发布本目录）

- **逐病例完整 CSV**：LIDC（~600 MB，`outputs/analysis_full/per_sample_val_fold*.csv`）与
  QUBIQ（`outputs/analysis/qubiq_*/per_sample_*.csv`）建议随 Zenodo 归档，不放入代码仓库。本包仅含聚合 JSON 与论文插图。
- **不包含**：模型权重（`outputs/runs/`）、原始影像（`data/raw/`）、QUBIQ 原始数据
  （来源 grand-challenge.org QUBIQ 2021，arXiv:2405.18435，使用条款见 §5.5 引用）。
- **新增 5 个 QUBIQ 任务的 per-case CSV 未在本包导出**（brain-tumor t1–3、prostate t1–2）；
  如需完整逐病例发布，从 `outputs/analysis/qubiq_brain_tumor_task*/`、`qubiq_prostate_task*/`
  按病例级重新导出（聚合结果已含于 `qubiq_all_effects.json` / `qubiq_trend.json`）。
