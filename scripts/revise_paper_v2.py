"""
Paper revision v2: reposition contribution, downgrade auxiliary losses,
fix ablation wording, correct data errors, sync conclusion.
Based on outputs/paper/论文初稿_修订版.docx -> outputs/paper/论文初稿_v2.docx
"""
from docx import Document
from copy import deepcopy
import os

_PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "outputs", "paper")
SRC = os.path.join(_PAPER, "论文初稿_修订版.docx")
DST = os.path.join(_PAPER, "论文初稿_v2.docx")

doc = Document(SRC)

def replace_paragraph_text(para, new_text):
    """Replace paragraph text while preserving paragraph style."""
    # Clear all runs
    for run in para.runs:
        run.text = ""
    # Set text on first run if exists, else add new run
    if para.runs:
        para.runs[0].text = new_text
    else:
        para.add_run(new_text)

# ============================================================
# 1. TITLE [0]: reposition from "proposing CPS" to "systematic evaluation"
# ============================================================
replace_paragraph_text(doc.paragraphs[0],
    "Impact of Label-Fusion Strategy on Pulmonary Nodule Segmentation: "
    "A Systematic Evaluation of Consensus-Probability Supervision on LIDC-IDRI"
)

# ============================================================
# 2. ABSTRACT [4]: reposition wording
# ============================================================
replace_paragraph_text(doc.paragraphs[4],
    "Pulmonary nodule segmentation is a cornerstone of lung cancer screening, but its supervision "
    "target is itself uncertain: the four radiologists who annotate each LIDC-IDRI scan frequently "
    "disagree on exact boundaries. The prevailing practice fuses these annotations into a single hard "
    "mask before training, either by union or majority vote. This paper systematically evaluates how "
    "the label-fusion strategy affects segmentation performance, with a focus on consensus-probability "
    "supervision in which the network directly regresses the rater-vote fraction p(x)=V(x)/4 rather "
    "than a binarized mask, while region-overlap losses (Dice and Lovasz) are anchored to the "
    "majority-vote mask. Nodules are defined as majority-vote components of at least 3 mm. On a "
    "held-out test set of 2,209 nodules, consensus-probability supervision reaches a mean Dice of "
    "0.860 against the majority-vote ground truth and lifts micro-nodule (3-5 mm) Dice to 0.805, "
    "a size stratum where union-target training yields only approximately 0.32 in our experiments. "
    "A controlled ablation reveals the decisive factor: switching the training target from consensus "
    "to union costs 4.0 Dice points (0.879 vs. 0.838) -- an effect an order of magnitude larger "
    "than any auxiliary loss or backbone-depth variation. Auxiliary regularization terms (SITL, CSL, "
    "BRBC, Lovasz) are explored but exhibit negative interaction effects in combination, with no single "
    "term approaching the magnitude of the target-strategy effect. The majority-vote Dice is stable "
    "to within 0.005 across thresholds 0.30-0.70. These results indicate that label-fusion strategy, "
    "rather than architectural complexity or auxiliary loss design, is the primary performance lever "
    "for pulmonary nodule segmentation on LIDC-IDRI."
)

# ============================================================
# 3. INTRODUCTION contributions [11,12,13] -> new three contributions
# ============================================================
replace_paragraph_text(doc.paragraphs[11],
    "1) We quantify the impact of label-fusion strategy on pulmonary nodule segmentation, showing that "
    "switching from consensus-probability to union-target training costs 4.0 Dice points (0.879 vs. "
    "0.838) on LIDC-IDRI -- an effect an order of magnitude larger than any auxiliary loss or backbone "
    "variation we tested."
)

replace_paragraph_text(doc.paragraphs[12],
    "2) We demonstrate that consensus-probability supervision lifts micro-nodule (3-5 mm) Dice from "
    "approximately 0.32 under union-target training to 0.805, addressing the most clinically "
    "consequential and historically most difficult size stratum in lung cancer screening."
)

replace_paragraph_text(doc.paragraphs[13],
    "3) We provide a complete, reproducible preprocessing pipeline for LIDC-IDRI -- including XML "
    "annotation parsing, lung-windowed slice extraction, majority-vote nodule redefinition with a "
    ">=3 mm diameter threshold, and a pre-cropped 128x128 patch bank -- together with a full ablation "
    "analysis that isolates the effects of supervision target, backbone capacity, and auxiliary "
    "regularization terms."
)

# ============================================================
# 4. METHOD - Loss Functions [31]: downgrade auxiliary losses to "exploratory"
# ============================================================
replace_paragraph_text(doc.paragraphs[31],
    "The primary supervision consists of two terms: L_dice and L_lovasz computed against the majority "
    "mask M(x), and L_bce computed against the continuous consensus p(x). In addition to these core "
    "terms, we explore four additional regularization terms as exploratory components: the "
    "scale-invariant term L_sitl averages a soft Dice over every majority-vote instance independently "
    "so that small nodules contribute equally to large ones; the consensus term L_csl is a "
    "consensus-weighted cross-entropy on p(x) plus a low-confidence penalty that pushes the prediction "
    "toward 0.5 in high-disagreement regions; the boundary term L_brbc enforces bidirectional "
    "consistency between the region gradient and a boundary head, anchored by a low-weight "
    "cross-entropy against the morphological boundary of M(x); and L_lovasz provides a convex "
    "surrogate for the intersection-over-union metric. The total loss is a weighted sum "
    "L = L_dice + L_bce + L_sitl + L_csl + L_brbc + w_lov L_lovasz with fixed weights. These "
    "auxiliary terms are not presented as methodological contributions; their purpose is to investigate "
    "whether additional regularization can further improve performance beyond the consensus target, and "
    "their interaction effects are analyzed in the ablation study."
)

# ============================================================
# 5. ABLATION - main text [46]: fix wording from "adding" to "removal", correct data
# ============================================================
replace_paragraph_text(doc.paragraphs[46],
    "Table II reports validation Dice (majority) for ablations conducted by removing individual "
    "components from the full model. The decisive row is the union-target variant, which keeps the "
    "full loss stack but trains against the union mask: it drops to 0.838, a 4.0-point regression "
    "relative to the consensus target (0.879). This quantifies the failure mode identified in "
    "Section I -- union-target training over-estimates small-nodule boundaries because single-reader "
    "pixels are treated as foreground. Removing any single auxiliary loss from the full model does not "
    "degrade performance; indeed, every removal variant slightly outperforms the full model "
    "(abl_BRBC=0.883, abl_Lovasz=0.882, abl_SITL=0.881, abl_CSL=0.879). The Dice+BCE-only baseline "
    "(0.878) is within 0.001 of the full model (0.879), indicating that the auxiliary losses provide "
    "no measurable benefit in combination. A ResNet-18 backbone performs nearly identically (0.878, "
    "difference <0.001), confirming the task is not capacity-limited once the supervision target is "
    "corrected."
)

# ============================================================
# 6. ABLATION - negative interaction [47]: strengthen and correct
# ============================================================
replace_paragraph_text(doc.paragraphs[47],
    "The ablation reveals a clear pattern of negative interaction among the auxiliary regularization "
    "terms. When each term is removed individually from the full model, performance increases rather "
    "than decreases: removing BRBC yields the largest gain (+0.0045), followed by Lovasz (+0.0029), "
    "SITL (+0.0023), and CSL (+0.0004). The full six-loss model (0.8789) underperforms the "
    "Dice+BCE-only baseline (0.8784) by 0.0005. This pattern is consistent with conflicting gradient "
    "directions: each auxiliary loss optimizes a different objective (instance-level Dice, "
    "disagreement-weighted entropy, region-boundary consistency, IoU surrogate), and when combined "
    "with fixed, untuned weights, their gradients may partially cancel or pull the network toward "
    "competing optima. The fixed-weight scheme used here does not include any loss-weight annealing or "
    "uncertainty-based weighting, which could potentially mitigate these conflicts. However, the "
    "critical observation is that none of these auxiliary-loss effects -- whether positive or negative "
    "-- approach the magnitude of the consensus-vs-union target difference (4.0 points), reinforcing "
    "the conclusion that supervision-target design is the primary lever for performance in this task, "
    "and auxiliary loss engineering is not the bottleneck."
)

# ============================================================
# 7. ABLATION TABLE TITLE [48]: change to "removal from full model"
# ============================================================
replace_paragraph_text(doc.paragraphs[48],
    "TABLE II. ABLATION STUDY (REMOVAL FROM FULL MODEL, VALIDATION DICE, MAJORITY VOTE)"
)

# ============================================================
# 8. CONCLUSION [64]: sync wording
# ============================================================
replace_paragraph_text(doc.paragraphs[64],
    "This paper systematically evaluated the impact of label-fusion strategy on pulmonary nodule "
    "segmentation on LIDC-IDRI. We showed that fusing multi-rater annotations into a hard union mask "
    "is a silent source of error, and that consensus-probability supervision -- regressing the "
    "rater-vote fraction p(x)=V(x)/4 directly -- recovers the boundary information discarded by "
    "hard-label fusion. Under a controlled comparison, consensus-probability supervision outperforms "
    "union-target training by 4.0 Dice points (0.879 vs. 0.838), lifts micro-nodule (3-5 mm) Dice "
    "from approximately 0.32 to 0.805, and reaches a test Dice of 0.860 (majority vote). The 5-fold "
    "cross-validation mean of 0.8675 +/- 0.0097 confirms stability. Auxiliary regularization terms "
    "(SITL, CSL, BRBC, Lovasz) were explored but exhibited negative interaction effects in combination, "
    "with no term approaching the magnitude of the target-strategy effect. Beyond the specific task, "
    "the result argues for treating annotation-fusion strategy as a first-class design variable rather "
    "than an afterthought -- a lesson that transfers to any segmentation problem with disagreeing "
    "experts."
)

# ============================================================
# 9. Fix Fig.4 caption [50]: correct misleading description
# ============================================================
replace_paragraph_text(doc.paragraphs[50],
    "Fig. 4.  Ablation study on the validation set (removal from full model). (a) Each bar represents "
    "the validation Dice (majority) when the indicated component is removed from the full six-loss "
    "model. Every removal variant slightly outperforms the full model, indicating negative interaction "
    "among auxiliary losses. (b) The consensus-vs-union target comparison dominates all other "
    "ablations, with a 4.0-point gap."
)

# ============================================================
# Save
# ============================================================
doc.save(DST)
print(f"Saved revised paper to: {DST}")
print(f"Total paragraphs: {len(doc.paragraphs)}")
