"""PyTorch Dataset for LIDC-IDRI lung nodule segmentation (2D slice level).

Each case is stored as:
  {case_id}.npz  -> slices (uint8 lung-windowed CT) + consensus (uint8 vote 0..4)
  {case_id}.json -> nodule metadata [{slice, cy, cx, diameter_mm}]

union mask / disagreement / instance ids / boundary are derived on the fly.
"""
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import utils as data_utils
from src.data.transforms import build_transforms


class _CaseCache(OrderedDict):
    """Small LRU cache for memory-mapped npz files."""

    def __init__(self, max_cases=2):
        super().__init__()
        self.max_cases = max_cases

    def get(self, case_id, data_dir):
        if case_id in self:
            self.move_to_end(case_id)
            return self[case_id]
        npz = np.load(os.path.join(data_dir, f"{case_id}.npz"), mmap_mode="r")
        self[case_id] = npz
        if len(self) > self.max_cases:
            self.popitem(last=False)
        return npz


class LungNoduleDataset(Dataset):
    def __init__(self, data_dir, split_csv, cfg, phase="train"):
        self.data_dir = data_dir
        self.phase = phase
        self.cfg = cfg
        self.patch_size = cfg.data.patch_size
        self.in_channels = cfg.model.in_channels
        self.transforms = build_transforms(cfg, phase)
        self.case_ids = self._load_cases(split_csv)
        self.samples = self._build_index()
        self.cache = _CaseCache(max_cases=2)
        # fast path: pre-cropped 128px nodule bank (build with preprocess/build_patch_bank.py)
        self.patch_dir = getattr(cfg.data, "patch_dir", None)
        self.bank_mode = bool(self.patch_dir) and os.path.isdir(self.patch_dir)
        # 随机平移增强：仅训练阶段生效，且只在“库比 patch_size 大”时才有余量。
        # 现有全部实验 shift_px=0（default.yaml 不设该键），行为与改动前逐位相同。
        self.shift_px = int(getattr(cfg.data, "random_shift_px", 0) or 0)
        # 平移时要判断窗口是否超出真实切片，需要切片尺寸。Bank 模式下不读 npz，
        # 所以在这里用 mmap 只读一次文件头拿到形状（不解压像素），代价可忽略。
        self.slice_hw = (512, 512)
        if self.shift_px > 0 and self.samples:
            try:
                z = np.load(os.path.join(self.data_dir, f"{self.samples[0]['case_id']}.npz"),
                            mmap_mode="r")
                self.slice_hw = tuple(int(v) for v in z["slices"].shape[1:3])
            except Exception:
                pass
        self.bank_cache = OrderedDict()  # case_id -> (images, votes) small LRU
        # P1-3 去混淆臂：训练时以 background_ratio 概率采样“无结节切片”作为负样本。
        # 默认 0（现有全部实验行为不变）；>0 时仅训练阶段生效。
        self.bg_ratio = float(getattr(cfg.data, "background_ratio", 0) or 0)
        # 标注者数量（LIDC=4；QUBIQ 肾脏=3、脑生长=7…）。默认 4，向后兼容。
        self.n_raters = int(getattr(cfg.data, "n_raters", 4) or 4)
        self.majority_thr = int((self.n_raters + 1) // 2)  # ceil(N/2)，4->2, 3->2, 7->4
        # P1-3 加速：预提取无结节背景 patch 到内存，避免训练中反复读 npz（曾导致 40x 慢）。
        self.bg_patches = []
        if self.phase == "train" and self.bg_ratio > 0:
            self._preload_background()

    def _load_cases(self, split_csv):
        ids = []
        with open(split_csv, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and line != "case_id":
                    ids.append(line)
        return ids

    def _build_index(self):
        samples = []
        for case_id in self.case_ids:
            meta_path = os.path.join(self.data_dir, f"{case_id}.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for j, nod in enumerate(meta.get("nodules", [])):
                samples.append({
                    "case_id": case_id,
                    "nodule_idx": j,
                    "slice": int(nod["slice"]),
                    "cy": float(nod["cy"]),
                    "cx": float(nod["cx"]),
                    "diameter_mm": float(nod["diameter_mm"]),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_bank_patch(self, s):
        case_id = s["case_id"]
        entry = self.bank_cache.get(case_id)
        if entry is None:
            npz = np.load(os.path.join(self.patch_dir, f"{case_id}.npz"))
            entry = (npz["images"], npz["votes"])
            self.bank_cache[case_id] = entry
            if len(self.bank_cache) > 16:
                self.bank_cache.popitem(last=False)
        else:
            self.bank_cache.move_to_end(case_id)
        return entry[0][s["nodule_idx"]], entry[1][s["nodule_idx"]]

    def _shift(self, bank_size, cy, cx):
        """训练阶段的随机平移偏移；验证/测试恒为 0。

        偏移范围是两个约束的交集：

        1) 库内余量 m = (bank_size - patch_size) // 2。
           超出它，裁剪窗就伸到 bank 数组之外。
        2) 窗口仍完整落在【真实切片】内。
           建库时对靠近切片边缘的结节做过零填充，若平移把窗口推向那一侧，
           样本里会出现一条硬零边。零边是训练时才有的伪影，模型可能去学它，
           对照实验就不干净了。全数据集有 15.95% 的样本在 +-32 平移下会碰到这种情况。

        若结节太靠边导致交集为空（全数据集 4.06% 的样本，它们在 d=0 时就已经在
        补零了，现有全部实验同样如此），退回只用约束 1，不比现状更差。

        注意：这里用 np.random，与 transforms.py 的既有增强同源。DataLoader 未设
        worker_init_fn，多 worker 下 numpy 种子会重复，这是本代码库已有的行为，
        对所有实验一视同仁；此处刻意不修，以免平移臂与基线臂在第二个地方产生差异。
        """
        if self.phase != "train" or self.shift_px <= 0:
            return 0, 0
        m = max(0, (bank_size - self.patch_size) // 2)
        sh = min(self.shift_px, m)
        if sh <= 0:
            return 0, 0
        h = self.patch_size // 2
        out = []
        for c, size in ((cy, self.slice_hw[0]), (cx, self.slice_hw[1])):
            c = int(round(c))
            lo, hi = max(-sh, h - c), min(sh, size - h - c)   # 窗口不超出真实切片
            if lo > hi:                                       # 交集为空 -> 退回库内余量
                lo, hi = -sh, sh
            out.append(int(np.random.randint(lo, hi + 1)))
        return out[0], out[1]

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.bank_mode:
            image, vote = self._load_bank_patch(s)
            image = image.astype(np.float32) / 255.0   # uint8 -> [0,1]
            vote = vote.astype(np.float32)             # 0..N
            if image.shape[0] != self.patch_size:
                # center-crop bank patch -> config patch size; identical to
                # cropping the original slice at (cy, cx) with this size
                c = image.shape[0] // 2
                dy, dx = self._shift(image.shape[0], s["cy"], s["cx"])
                image = data_utils.roi_center_crop(image, (c + dy, c + dx), self.patch_size)
                vote = data_utils.roi_center_crop(vote, (c + dy, c + dx), self.patch_size)
            elif self.shift_px > 0 and self.phase == "train":
                raise ValueError(
                    f"data.random_shift_px={self.shift_px} 需要一个比 patch_size 更大的"
                    f"patch 库才能平移取样（当前库 {image.shape[0]}px = patch_size）。"
                    f"用 preprocess/build_patch_bank.py --bank_size {self.patch_size + 64} "
                    f"重建，并把 data.patch_dir 指向它。"
                    f"若在现有库内平移就必须补零，会引入训练时才有的零边伪影，"
                    f"模型可能去学边界线索，对照实验就不干净了。")
        else:
            npz = self.cache.get(s["case_id"], self.data_dir)
            sl = s["slice"]

            image = npz["slices"][sl].astype(np.float32) / 255.0   # uint8 -> [0,1]
            vote = npz["consensus"][sl].astype(np.float32)         # 0..N

            center = (s["cy"], s["cx"])
            image = data_utils.roi_center_crop(image, center, self.patch_size)
            vote = data_utils.roi_center_crop(vote, center, self.patch_size)

        consensus = vote / self.n_raters                                 # soft label [0,1]
        disagreement = self.n_raters * consensus * (1.0 - consensus)     # uncertainty [0,1]
        union = (vote >= 1).astype(np.float32)
        majority_mask = (vote >= self.majority_thr).astype(np.float32) # >= ceil(N/2) raters
        instance_ids = data_utils.label_instances(union, connectivity=2)
        boundary = data_utils.boundary_from_mask(union, width=1).astype(np.float32)
        maj_instance_ids = data_utils.label_instances(majority_mask, connectivity=2)
        maj_boundary = data_utils.boundary_from_mask(majority_mask, width=1).astype(np.float32)

        sample = {
            "image": image,
            "union_mask": union,
            "majority_mask": majority_mask,
            "consensus": consensus,
            "disagreement": disagreement,
            "instance_ids": instance_ids,
            "boundary": boundary,
            "maj_instance_ids": maj_instance_ids,
            "maj_boundary": maj_boundary,
            "diameter_mm": s["diameter_mm"],
            "case_id": s["case_id"],
        }

        # P1-3 去混淆臂：以 background_ratio 概率返回无结节背景样本（目标全零）。
        if self.phase == "train" and self.bg_ratio > 0 and np.random.rand() < self.bg_ratio:
            bg = self._sample_background()
            if bg is not None:
                image, vote, case_id = bg
                return self._finalize(self._sample_from_vote(image, vote, case_id, 0.0))
            # 采样失败（很少见）则回落到正常结节样本

        return self._finalize(sample)

    def _sample_from_vote(self, image, vote, case_id, diameter_mm):
        """由 (image, vote) 派生完整 sample dict（普通样本与背景样本共用）。"""
        consensus = vote / self.n_raters                                 # soft label [0,1]
        disagreement = self.n_raters * consensus * (1.0 - consensus)     # uncertainty [0,1]
        union = (vote >= 1).astype(np.float32)
        majority_mask = (vote >= self.majority_thr).astype(np.float32) # >= ceil(N/2) raters
        instance_ids = data_utils.label_instances(union, connectivity=2)
        boundary = data_utils.boundary_from_mask(union, width=1).astype(np.float32)
        maj_instance_ids = data_utils.label_instances(majority_mask, connectivity=2)
        maj_boundary = data_utils.boundary_from_mask(majority_mask, width=1).astype(np.float32)
        return {
            "image": image,
            "union_mask": union,
            "majority_mask": majority_mask,
            "consensus": consensus,
            "disagreement": disagreement,
            "instance_ids": instance_ids,
            "boundary": boundary,
            "maj_instance_ids": maj_instance_ids,
            "maj_boundary": maj_boundary,
            "diameter_mm": diameter_mm,
            "case_id": case_id,
        }

    def _finalize(self, sample):
        """增强 + 通道扩展 + 张量化（normal 与 background 共用收尾）。"""
        if self.transforms is not None:
            sample = self.transforms(sample)

        image = sample["image"][np.newaxis, ...]  # (1, H, W)
        if self.in_channels == 3:
            image = np.repeat(image, 3, axis=0)   # (3, H, W)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "union_mask": torch.from_numpy(np.ascontiguousarray(sample["union_mask"][np.newaxis, ...])),
            "majority_mask": torch.from_numpy(np.ascontiguousarray(sample["majority_mask"][np.newaxis, ...])),
            "consensus": torch.from_numpy(np.ascontiguousarray(sample["consensus"][np.newaxis, ...])),
            "disagreement": torch.from_numpy(np.ascontiguousarray(sample["disagreement"][np.newaxis, ...])),
            "boundary": torch.from_numpy(np.ascontiguousarray(sample["boundary"][np.newaxis, ...])),
            "instance_ids": torch.from_numpy(np.ascontiguousarray(sample["instance_ids"])),
            "maj_boundary": torch.from_numpy(np.ascontiguousarray(sample["maj_boundary"][np.newaxis, ...])),
            "maj_instance_ids": torch.from_numpy(np.ascontiguousarray(sample["maj_instance_ids"])),
            "diameter_mm": torch.tensor(sample["diameter_mm"], dtype=torch.float32),
            "case_id": sample["case_id"],
        }

    def _preload_background(self, n_target=2000):
        """预提取无结节背景 patch 到内存，训练中直接取用（避免每次读压缩 npz 的 40x 慢）。

        优先加载预生成的 bank 文件 data/processed/bg_bank.npz（快，一次性扫描在
        preprocess/build_bg_bank.py）；文件不存在时回落在本进程内扫描。
        """
        ph = self.patch_size
        # 优先按尺寸找 bank（bg_bank128.npz / bg_bank192.npz），fallback 到旧名 bg_bank.npz
        bank_path = os.path.join(self.data_dir, "..", f"bg_bank{ph}.npz")
        bank_path = os.path.normpath(bank_path)
        if not os.path.exists(bank_path):
            bank_path = os.path.join(self.data_dir, "..", "bg_bank.npz")
            bank_path = os.path.normpath(bank_path)
        if os.path.exists(bank_path):
            try:
                bank = np.load(bank_path)
                images = bank["images"]  # (N,H,W) uint8
                votes = bank["votes"]
                hw = images.shape[1]
                if hw == ph and len(images) > 0:
                    for i in range(len(images)):
                        self.bg_patches.append(
                            (images[i].astype(np.float32) / 255.0,
                             votes[i].astype(np.float32), "bg_bank"))
                    print(f"[dataset] 从 bg_bank.npz 加载背景 patch: {len(self.bg_patches)} 个", flush=True)
                    return
            except Exception as e:
                print(f"[dataset] bg_bank.npz 加载失败({e})，回落扫描", flush=True)
        per_case_cap = 4
        for case_id in self.case_ids:
            if len(self.bg_patches) >= n_target:
                break
            npz_path = os.path.join(self.data_dir, f"{case_id}.npz")
            if not os.path.exists(npz_path):
                continue
            npz = self.cache.get(case_id, self.data_dir)
            try:
                n_slices = int(npz["slices"].shape[0])
                h, w = int(npz["slices"].shape[1]), int(npz["slices"].shape[2])
            except Exception:
                continue
            if h < ph or w < ph:
                continue
            added = 0
            for sl in range(n_slices):
                vote_slice = np.asarray(npz["consensus"][sl])
                if vote_slice.max() > 0:
                    continue
                cy = np.random.randint(ph // 2, h - ph // 2)
                cx = np.random.randint(ph // 2, w - ph // 2)
                img = npz["slices"][sl].astype(np.float32) / 255.0
                img = data_utils.roi_center_crop(img, (cy, cx), ph)
                vote = data_utils.roi_center_crop(vote_slice, (cy, cx), ph).astype(np.float32)
                self.bg_patches.append((img, vote, case_id))
                added += 1
                if added >= per_case_cap or len(self.bg_patches) >= n_target:
                    break
        print(f"[dataset] 预提取背景 patch: {len(self.bg_patches)} 个（train, bg_ratio={self.bg_ratio}）", flush=True)

    def _sample_background(self):
        """采样一个无结节切片上的随机 128x128 裁剪，返回 (image, vote, case_id)；失败返回 None。

        优先从内存预提取池取用（快）；池为空时回落逐次读 npz。
        """
        if self.bg_patches:
            return self.bg_patches[np.random.randint(len(self.bg_patches))]
        if not self.case_ids:
            return None
        ph = self.patch_size
        for _ in range(10):
            case_id = self.case_ids[np.random.randint(len(self.case_ids))]
            npz_path = os.path.join(self.data_dir, f"{case_id}.npz")
            if not os.path.exists(npz_path):
                continue  # 个别划分文件含异常行/缺 npz 时跳过
            npz = self.cache.get(case_id, self.data_dir)
            n_slices = int(npz["slices"].shape[0])
            if n_slices <= 0:
                continue
            sl = np.random.randint(n_slices)
            vote_slice = np.asarray(npz["consensus"][sl])
            if vote_slice.max() > 0:
                continue
            h, w = vote_slice.shape
            if h < ph or w < ph:
                continue
            cy = np.random.randint(ph // 2, h - ph // 2)
            cx = np.random.randint(ph // 2, w - ph // 2)
            image = npz["slices"][sl].astype(np.float32) / 255.0
            image = data_utils.roi_center_crop(image, (cy, cx), ph)
            vote = data_utils.roi_center_crop(vote_slice, (cy, cx), ph).astype(np.float32)
            return image, vote, case_id
        return None
