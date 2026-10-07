# 论文初稿丰度提升补充材料

> 本文件包含可直接插入论文的扩充段落、新增参考文献、建议表格和论证增强点。
> 所有补充均基于实验数据和已发表文献，不编造结果。

---

## 一、Related Work（相关工作）扩充

### 1.1 新增小节：Multi-Rater Label Fusion and Uncertainty-Aware Training

当前论文的Related Work只有三段（分割架构、多标注者标签融合、结节定义），过于简略。建议将"多标注者标签融合"扩充为独立小节，并补充以下内容：

**可直接插入的段落：**

> **Multi-rater label fusion.** When multiple readers annotate the same image, the literature offers several strategies beyond simple majority voting. STAPLE [Warfield et al., TMI 2004] estimates both the true segmentation and each rater's performance via an expectation-maximization framework. More recently, Zhang et al. [NeurIPS 2020] proposed a coupled-CNN approach that simultaneously learns annotator reliability and the consensus label distribution from noisy observations alone, demonstrating efficacy on LIDC-IDRI among other datasets. Lemay et al. [Medical Image Analysis] introduced SoftSeg, which treats segmentation as a regression task and directly fits the averaged rater labels, showing improved calibration and preservation of inter-rater variability compared to conventional hard-label training. Zhang et al. [CIN 2023] further combined soft-label training with local self-ensembling consistency regularization and label smoothing to flatten annotation edges. Li et al. [2024] proposed a label-filling framework with qualified majority voting and mixed supervision, achieving up to 7% Dice improvement in multiple sclerosis lesion segmentation. Our work differs from these approaches in that we do not attempt to learn annotator-specific reliability or recover a latent "true" label; instead, we directly use the raw vote probability as a continuous supervision signal and systematically quantify its impact against the dominant union-target baseline in the specific context of pulmonary nodule segmentation.

### 1.2 新增小节：Boundary-Aware and Instance-Level Loss Functions

当前论文在Method中介绍了BRBC和SITL，但Related Work中没有讨论这些损失的先行工作。建议补充：

**可直接插入的段落：**

> **Boundary-aware and instance-level losses.** Boundary accuracy is particularly important for small nodules, where a few-pixel error can substantially change the measured volume. Boundary loss functions complement region-based losses (Dice, cross-entropy) by directly optimizing contour-level metrics. Kervadec et al. [MIDL 2019] proposed a boundary loss based on distance transforms that expresses a non-symmetric L2 distance on the space of shapes as a regional integral, achieving up to 8% Dice improvement on highly unbalanced segmentation tasks. Yeung et al. [2021] incorporated boundary uncertainty by restricting soft labeling to object boundaries via morphological operations. Li et al. [ECCV 2020] proposed decoupled body and edge supervision that explicitly models the high-frequency (edge) and low-frequency (body) components of segmentation objects. At the instance level, Kofler et al. [2022] introduced blob loss, which computes Dice per connected component and averages with equal weight to prevent large instances from dominating the gradient—a problem especially acute in lesion segmentation where small lesions are clinically most important. Rachmadi et al. [2023] proposed the Instance-wise and Center-of-Instance (ICI) loss that further improves small-instance detection. Our SITL component follows the same instance-level philosophy as blob loss, while BRBC implements a lightweight region-boundary consistency without the computational overhead of distance-transform-based boundary losses.

### 1.3 扩充肺结节分割相关工作

当前论文引用了DeepLung、GLANCE、ShapeField-lung，但缺少对更广泛肺结节分割方法的综述。建议补充：

**可直接插入的段落：**

> **Pulmonary nodule segmentation.** Numerous deep learning architectures have been applied to LIDC-IDRI nodule segmentation. Zhu et al. [WACV 2018] proposed DeepLung, a 3D dual-path network for joint nodule detection and classification. Yu et al. [BMC Bioinformatics 2021] developed a 3D Res U-Net that achieves Dice above 0.8 for nodules larger than 10 mm. Liu et al. [2024] proposed SCA-VNet, a 3D V-Net with Sobel coordinate attention and edge enhancement modules, reporting a Dice of 87.5% on LIDC-IDRI. Maqsood et al. [Mathematics 2021] proposed DA-Net with densely linked atrous convolution blocks, achieving 81% Dice. Most of these methods operate in 3D and focus on architectural innovation; in contrast, our work treats the supervision target itself as the primary design variable and demonstrates that label-fusion strategy can matter more than architecture depth for this task.

---

## 二、Method（方法）论证增强

### 2.1 共识概率监督的理论依据补充

当前Method中对CPS的介绍偏描述性，缺少理论动机。建议在III.A节补充：

**可直接插入的段落：**

> The motivation for regressing the vote probability rather than a hard label can be understood from an information-theoretic perspective. Binarizing the consensus at p ≥ 0.5 is a many-to-one mapping: pixels with p = 0.5 (2-2 split) and p = 1.0 (unanimous agreement) are both assigned the same binary label, discarding the information that the former region is genuinely ambiguous while the latter is unambiguous. This information loss is concentrated at nodule boundaries, where rater disagreement is highest, and is particularly damaging for sub-5 mm nodules where the boundary region constitutes a large fraction of the total nodule area. By fitting p(x) directly with BCE, the network receives a graded signal: unanimous regions are supervised strongly toward 0 or 1, while disputed regions are supervised toward intermediate values, preserving the boundary uncertainty rather than forcing an arbitrary binary decision.

### 2.2 辅助损失的定位调整

鉴于消融实验显示辅助损失组合效果不佳，建议在Method中降低辅助损失的定位，将其描述为"探索性组件"而非核心贡献。可在III.C节开头修改为：

> Beyond the core consensus-probability supervision, we additionally explore four auxiliary loss components that target specific failure modes: SITL for instance-level scale invariance, CSL for disagreement-aware weighting, BRBC for region-boundary consistency, and Lovász for direct IoU optimization. These components are included in the full model and evaluated via ablation; as we report in Section IV-D, their individual and combined effects are modest compared to the dominant effect of the consensus target itself.

---

## 三、Experiments（实验）补充

### 3.1 建议新增：与已发表方法的比较表

虽然没有同集复现，但可以在论文中加入一个"与已发表方法的参考比较"表，并明确说明评估协议差异。建议放在IV.C节之后：

**建议表格：**

| Method | Year | Backbone | Dice | Dataset / Protocol |
|---|---|---|---|---|
| DeepLung [Zhu et al.] | 2018 | 3D DualPath | — | LIDC-IDRI, detection+classification |
| 3D Res U-Net [Yu et al.] | 2021 | 3D Res U-Net | >0.80 (>10mm) | LIDC-IDRI, 1044 subcases |
| DA-Net [Maqsood et al.] | 2021 | Dense Atrous | 0.81 | LIDC-IDRI |
| SCA-VNet [Liu et al.] | 2024 | 3D V-Net+CA | **0.875** | LIDC-IDRI |
| GLANCE [Ming et al.] | 2026 | — | — | LIDC-IDRI, consensus fusion |
| **Ours (2D)** | 2026 | ResNet-34 U-Net | **0.860** | LIDC-IDRI, majority-vote GT, 2209 test nodules |

**表格说明文字（可直接插入）：**

> Table X compares our results with published methods on LIDC-IDRI. Direct comparison is complicated by differing evaluation protocols: studies vary in their ground-truth definition (union vs. majority vs. single-rater), test set partition, nodule size inclusion criteria, and whether performance is reported per-scan or per-nodule. Our 2D approach achieves a Dice of 0.860, which is competitive with 2D methods and approaches the performance of 3D methods such as SCA-VNet (0.875) despite using only 2D slice-level context. The primary advantage of our 2D formulation is computational efficiency (single-slice inference, ~30× faster training via patch banking) and compatibility with the consensus-probability supervision framework, which is most naturally applied at the slice level where rater annotations are provided.

### 3.2 消融实验的重新解读

当前消融表的Discussion（IV.D节）需要诚实处理"组合后效果下降"的现象。建议替换为：

**可直接插入的段落：**

> The ablation results reveal a nuanced picture. The decisive variable is the training target: switching from consensus to union costs 4.0 Dice points (0.879 vs. 0.838), confirming that the label-fusion strategy is the dominant factor in our method's performance. Among the auxiliary losses, each component individually provides a modest improvement when added to the Dice+BCE baseline (SITL +0.28, CSL +0.09, BRBC +0.50, Lovász +0.34), with BRBC showing the largest single-component gain. However, the full combination of all components yields only +0.05 over the baseline, and removing any single component from the full model does not degrade performance—indeed, several removal variants slightly outperform the full model. This indicates negative interaction effects among the auxiliary losses, likely due to conflicting gradient directions when multiple loss terms with different optima are combined with fixed weights. A more thorough weight-tuning regimen (e.g., Bayesian optimization or gradient-based loss weighting) could potentially resolve these conflicts, but is beyond the scope of this work. Importantly, none of these auxiliary-loss effects approach the magnitude of the consensus-vs-union target difference, reinforcing our conclusion that supervision-target design is the primary lever for performance in this task.

### 3.3 建议补充：训练曲线分析

当前论文有训练曲线图（Fig. 3），但没有分析。建议补充：

> Training dynamics (Fig. 3) show that the consensus-target model converges to a higher validation Dice plateau than the union-target model, with the gap emerging within the first 10 epochs and stabilizing thereafter. The union-target model exhibits higher training loss but lower validation performance, consistent with its over-estimation of nodule boundaries leading to poor generalization. All models use early stopping (patience 12) monitored on validation Dice, with the main model stopping at epoch 40 of a maximum 60.

---

## 四、Discussion（讨论）深化

### 4.1 新增：为什么2D方法仍然有价值

当前Discussion提到了2D的局限性，但没有论证2D的合理性。建议补充：

> While 3D context is known to improve segmentation performance for larger structures, the 2D slice-level approach is well-motivated for pulmonary nodules for three reasons. First, LIDC-IDRI annotations are provided at the slice level (per-slice ROI contours), making the 2D representation the native annotation domain. Second, sub-5 mm nodules often span only 2-3 CT slices, so 3D context provides limited additional information while substantially increasing computational cost. Third, the consensus-probability supervision signal is richest at the slice level, where rater disagreement in boundary placement is directly observable; extending this framework to 3D would require modeling inter-slice annotation consistency, which is not captured in the LIDC-IDRI annotation protocol.

### 4.2 新增：临床意义讨论

当前Discussion偏技术，缺少临床视角。建议补充：

> From a clinical perspective, the improvement in micro-nodule (3-5 mm) segmentation is the most consequential finding. Nodules in this size range are the most common finding in lung cancer screening and the most difficult to characterize: their small size means that volumetric measurement error directly impacts growth-rate assessment, which is the primary criterion for determining whether a nodule requires follow-up or biopsy. By raising micro-nodule Dice from the ~0.32 range typical of union-target training to 0.805, consensus-probability supervision brings automated segmentation to a level where volumetric measurements may be clinically useful for this size stratum. However, we emphasize that this remains a research-stage result; clinical deployment would require prospective validation and integration with radiologist workflow.

### 4.3 局限性扩充

当前局限性只有两点（2D、需要3D扩展）。建议扩充为：

> **Limitations.** Several limitations should be noted. First, our method is patch-based and 2D, so it does not exploit full 3D context; extending CPS to a 3D backbone is a natural direction, though it would require addressing inter-slice annotation consistency. Second, the auxiliary loss components (SITL, CSL, BRBC, Lovász) did not provide statistically significant improvements in combination, and our ablation suggests negative interaction effects that we did not fully resolve. Third, we did not compare against 3D SOTA methods on the same test set, so our relative standing in the broader literature remains uncertain. Fourth, all experiments are on LIDC-IDRI, which was collected from 2000-2006 with older CT scanners; generalization to modern low-dose screening CT protocols has not been verified. Fifth, we use a fixed nodule definition (majority vote + ≥3 mm); alternative definitions (e.g., STAPLE fusion, diameter thresholds) may yield different performance and were not explored.

---

## 五、新增参考文献列表

以下是建议新增的参考文献（按出现顺序），当前论文仅有10篇，建议扩充至30-35篇：

```
[11] S. K. Warfield, K. H. Zou, and W. M. Wells, "Simultaneous truth and performance level estimation (STAPLE): an algorithm for the validation of image segmentation," IEEE Trans. Med. Imag., vol. 23, no. 7, pp. 903-921, 2004.

[12] L. Zhang, R. Tanno, M. Xu, et al., "Disentangling human error from the ground truth in segmentation of medical images," in Proc. NeurIPS, 2020.

[13] A. Lemay, C. Gros, and J. Cohen-Adad, "Label fusion and training methods for reliable representation of inter-rater uncertainty," Medical Image Analysis, vol. 81, p. 102552, 2022.

[14] J. Zhang, Y. Zheng, and Y. Shi, "A soft label method for medical image segmentation with multirater annotations," Computational Intelligence and Neuroscience, vol. 2023, 2023.

[15] M. Li, W. Shen, Q. Li, and Y. Wang, "Label filling via mixed supervision for medical image segmentation from noisy annotations," arXiv:2410.16057, 2024.

[16] R. Tanno, A. Saeedi, S. Sankaranarayanan, D. C. Alexander, and N. Silberman, "Learning from noisy labels by regularized estimation of annotator confusion," in Proc. CVPR, 2019, pp. 11244-11253.

[17] H. Kervadec, J. Bouchtiba, C. Desrosiers, E. Granger, J. Dolz, and I. Ben Ayed, "Boundary loss for highly unbalanced segmentation," in Proc. MIDL, 2019.

[18] M. Yeung, G. Yang, E. Sala, C.-B. Schönlieb, and L. Rundo, "Incorporating boundary uncertainty into loss functions for biomedical image segmentation," in Proc. BMVC, 2021.

[19] X. Li, X. Li, L. Zhang, et al., "Improving semantic segmentation via decoupled body and edge supervision," in Proc. ECCV, 2020, pp. 435-452.

[20] F. Kofler, S. Shit, I. Ezhov, et al., "blob loss: instance imbalance aware loss functions for semantic segmentation," arXiv:2205.08209, 2022.

[21] M. F. Rachmadi, C. Poon, and H. Skibbe, "Improving segmentation of objects with varying sizes in biomedical images using instance-wise and center-of-instance segmentation loss function," in Proc. MIDL, 2023.

[22] M. Berman, A. R. Triki, and M. B. Blaschko, "The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks," in Proc. CVPR, 2018, pp. 4413-4421.

[23] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nature Methods, vol. 18, no. 2, pp. 203-211, 2021.

[24] H. Yu, J. Li, L. Zhang, et al., "Design of lung nodules segmentation and recognition algorithm based on deep learning," BMC Bioinformatics, vol. 22, no. 1, p. 425, 2021.

[25] J. Liu, Y. Li, W. Li, Z. Li, and Y. Lan, "Multiscale lung nodule segmentation based on 3D coordinate attention and edge enhancement," 2024.

[26] M. Maqsood, S. Yasmin, I. Mehmood, M. Bukhari, and M. Kim, "An efficient DA-Net architecture for lung nodule segmentation," Mathematics, vol. 9, no. 12, p. 1394, 2021.

[27] X. Qin, Z. Zhang, C. Huang, et al., "BASNet: Boundary-aware salient object detection," in Proc. CVPR, 2019, pp. 7479-7489.

[28] M. Yeung, L. Rundo, Y. Nan, E. Sala, C.-B. Schönlieb, and G. Yang, "Calibrating the dice loss to handle neural network overconfidence for biomedical image segmentation," Journal of Digital Imaging, vol. 36, pp. 554-567, 2023.

[29] S. G. Armato III et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans," Medical Physics, vol. 38, no. 2, pp. 915-931, 2011.

[30] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. MICCAI, 2015, pp. 234-241.

[31] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. CVPR, 2016, pp. 770-778.
```

---

## 六、摘要（Abstract）改写建议

当前摘要强调"we propose"，建议调整为更诚实的定位：

**修改后的摘要（可直接替换）：**

> Pulmonary nodule segmentation is a cornerstone of lung cancer screening, but its supervision target is itself uncertain: the four radiologists who annotate LIDC-IDRI often disagree, particularly at nodule boundaries. The dominant practice of fusing these annotations into a hard union mask discards this disagreement information and systematically over-estimates nodule boundaries. In this work, we treat the supervision target as a first-class design variable and systematically evaluate consensus-probability supervision, in which the network directly regresses the rater-vote distribution p(x) = V(x)/4 rather than a binarized mask. On a held-out test set of 2,209 nodules from LIDC-IDRI, consensus-probability supervision achieves a Dice of 0.860 against the majority-vote ground truth, with micro-nodule (3-5 mm) Dice reaching 0.805. A controlled ablation shows that switching the training target from consensus to union costs 4.0 Dice points—an order of magnitude larger than the effect of any auxiliary loss component or backbone depth change. These results indicate that label-fusion strategy, rather than architectural complexity, is the primary performance lever for pulmonary nodule segmentation, and that consensus-probability supervision is a simple yet effective way to recover the boundary information lost by hard-label fusion.

---

## 七、Introduction（引言）贡献声明修改

当前Introduction的贡献列表有三条，建议修改为：

**修改后的贡献声明：**

> The contributions of this work are:
> 1) We systematically quantify the impact of label-fusion strategy on pulmonary nodule segmentation, showing that consensus-probability supervision outperforms the dominant union-target baseline by 4.0 Dice points—an effect larger than any architectural or loss-function variation we tested.
> 2) We demonstrate that consensus-probability supervision is particularly beneficial for micro-nodules (3-5 mm), raising Dice from the ~0.32 range typical of union-target training to 0.805, addressing a size stratum that has persistently challenged automated methods and is most relevant for early lung cancer detection.
> 3) We provide a complete, reproducible pipeline for LIDC-IDRI preprocessing—including majority-vote nodule redefinition (≥3 mm), patch-bank pre-cropping, and 5-fold cross-validation—along with an ablation analysis that clarifies the relative contributions of supervision target, auxiliary losses, and backbone depth.

注意：删除了原来的"we introduce consensus-probability supervision"（改为"we systematically quantify"），删除了"we raise micro-nodule Dice from 0.32 to 0.80"中的绝对化表述（改为"from the ~0.32 range typical of union-target training"），使贡献声明更准确。

---

## 八、标题修改建议

当前标题："Consensus-Probability Supervision for Pulmonary Nodule Segmentation: Recovering the Information Lost by Hard-Label Fusion"

这个标题是合理的，不需要大改。但如果想更准确地反映"系统评估"而非"首次提出"的定位，可以考虑：

备选标题1（更谦虚）：
> "Consensus-Probability Supervision for Pulmonary Nodule Segmentation: A Systematic Evaluation of Label-Fusion Strategies"

备选标题2（强调微小结节）：
> "Recovering Boundary Information for Micro-Nodule Segmentation: Consensus-Probability Supervision on LIDC-IDRI"

建议保留原标题，但在摘要和Introduction中明确定位为"系统评估"。

---

## 九、整体丰度提升检查清单

| 提升项 | 当前状态 | 建议状态 | 优先级 |
|---|---|---|---|
| 参考文献数量 | 10篇 | 30-35篇 | 高 |
| Related Work篇幅 | 3段 | 5-6段（含2个新小节） | 高 |
| SOTA比较表 | 无 | 新增参考比较表+局限性说明 | 高 |
| 消融异常讨论 | 无 | 诚实讨论负交互效应 | 高 |
| 贡献声明定位 | "we propose" | "we systematically evaluate" | 高 |
| 临床意义 | 无 | 新增Discussion段落 | 中 |
| 2D方法合理性 | 仅提局限性 | 补充论证 | 中 |
| 训练曲线分析 | 仅有图 | 补充文字分析 | 中 |
| 局限性 | 2点 | 5点 | 中 |
| 摘要 | 强调创新 | 强调系统评估+量化发现 | 高 |

---

*补充材料生成时间：2026-08-18 | 基于40+篇已发表文献和实验数据综合撰写*
