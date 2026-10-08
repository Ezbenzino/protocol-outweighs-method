# -*- coding: utf-8 -*-
"""build_zenodo_release.py —— 构建 Zenodo 上传件。

用法：
    python scripts/build_zenodo_release.py [--out dist/zenodo] [--skip-csv]

产出（默认 dist/zenodo/）：
    protocol-outweighs-method-v22-release-package.zip    聚合结果包（outputs/release 全部文件）
    protocol-outweighs-method-v22-lidc-per-case-csv.zip  LIDC 逐病例 CSV（5 折验证集 + 保留测试集）
    CHECKSUMS.sha256                                     两个 zip 及所有原始 CSV 的 SHA-256
    zenodo-metadata.json                                 可直接提交给 Zenodo API 的记录元数据
    DEPOSIT_README.md                                    数据包说明与校验方法

打包前用 outputs/release/manifest.json 逐文件校验 size 与 sha256，任何不一致立即中止，
因此上传件与 manifest 记录的 source_commit 始终对得上。zip 内时间戳固定，产物可复现。
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'outputs' / 'release'
PREFIX = 'protocol-outweighs-method-v22'
ZIP_STAMP = (2026, 10, 3, 0, 0, 0)          # 固定时间戳 -> 可复现的 zip
BUF = 1 << 20

# 逐病例 CSV：目录 -> 文件名
CSV_GROUPS = [
    ('analysis_full', ['per_sample_val_fold%d.csv' % i for i in range(5)]
                    + ['per_sample_free_val_fold%d.csv' % i for i in range(5)]),
    ('analysis_test', ['per_sample_test.csv', 'per_sample_free_test.csv']),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(BUF), b''):
            h.update(chunk)
    return h.hexdigest()


def add_to_zip(zf, src, arcname):
    """按固定时间戳写入，保证同一输入产生逐位相同的 zip。"""
    zi = zipfile.ZipInfo(arcname, date_time=ZIP_STAMP)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    with open(src, 'rb') as f, zf.open(zi, 'w', force_zip64=True) as dst:
        shutil.copyfileobj(f, dst, BUF)


def verify_manifest():
    """逐文件校验 outputs/release 与 manifest.json 是否一致。"""
    manifest = json.loads((RELEASE / 'manifest.json').read_text(encoding='utf-8'))
    bad = []
    for entry in manifest['files']:
        p = RELEASE / entry['path']
        if not p.exists():
            bad.append((entry['path'], 'MISSING'))
        elif p.stat().st_size != entry['size']:
            bad.append((entry['path'], 'size %d != %d' % (p.stat().st_size, entry['size'])))
        elif sha256(p) != entry['sha256']:
            bad.append((entry['path'], 'sha256 mismatch'))
    return manifest, bad


def fmt_mb(n):
    return '%.1f MB' % (n / (1 << 20))


def main():
    ap = argparse.ArgumentParser(description='构建 Zenodo 上传件')
    ap.add_argument('--out', default=str(ROOT / 'dist' / 'zenodo'), help='输出目录')
    ap.add_argument('--skip-csv', action='store_true', help='只打聚合结果包，跳过约 1.1 GB 的逐病例 CSV')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checksums = []          # (sha256, 展示路径)

    # ---- 1. 校验 manifest -------------------------------------------------
    print('[1/4] 校验 outputs/release 与 manifest.json ...')
    manifest, bad = verify_manifest()
    if bad:
        for path, why in bad:
            print('      MISMATCH  %-58s %s' % (path, why))
        sys.exit('\nmanifest 校验失败。先运行： python scripts/rebuild_release_manifest.py')
    print('      %d 个文件全部匹配（source_commit=%s）' % (len(manifest['files']), manifest['source_commit']))

    # ---- 2. 聚合结果包 ----------------------------------------------------
    rel_zip = out / (PREFIX + '-release-package.zip')
    print('[2/4] 打包聚合结果包 -> %s' % rel_zip.name)
    entries = sorted(manifest['files'], key=lambda e: e['path'])
    total = 0
    with zipfile.ZipFile(rel_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for entry in entries:
            src = RELEASE / entry['path']
            add_to_zip(zf, src, PREFIX + '/' + entry['path'])
            total += src.stat().st_size
        add_to_zip(zf, RELEASE / 'manifest.json', PREFIX + '/manifest.json')
    checksums.append((sha256(rel_zip), rel_zip.name))
    print('      %d 个文件 + manifest.json，原始 %s -> zip %s'
          % (len(entries), fmt_mb(total), fmt_mb(rel_zip.stat().st_size)))

    # ---- 3. 逐病例 CSV ----------------------------------------------------
    csv_zip = out / (PREFIX + '-lidc-per-case-csv.zip')
    csv_rows = []
    if args.skip_csv:
        print('[3/4] 跳过逐病例 CSV（--skip-csv）')
    else:
        print('[3/4] 打包 LIDC 逐病例 CSV -> %s' % csv_zip.name)
        total = 0
        with zipfile.ZipFile(csv_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for sub, names in CSV_GROUPS:
                for name in names:
                    src = ROOT / 'outputs' / sub / name
                    if not src.exists():
                        sys.exit('缺少逐病例 CSV：%s' % src)
                    add_to_zip(zf, src, '%s-per-case-csv/%s/%s' % (PREFIX, sub, name))
                    digest = sha256(src)
                    checksums.append((digest, 'outputs/%s/%s' % (sub, name)))
                    csv_rows.append('| %s/%s | %s | %s... |'
                                    % (sub, name, fmt_mb(src.stat().st_size), digest[:16]))
                    total += src.stat().st_size
                    print('      + %-34s %s' % (name, fmt_mb(src.stat().st_size)))
        checksums.append((sha256(csv_zip), csv_zip.name))
        print('      原始 %s -> zip %s' % (fmt_mb(total), fmt_mb(csv_zip.stat().st_size)))

    # ---- 4. 校验和 / 元数据 / 说明 ---------------------------------------
    print('[4/4] 写出 CHECKSUMS.sha256 / zenodo-metadata.json / DEPOSIT_README.md')
    (out / 'CHECKSUMS.sha256').write_text(
        ''.join('%s  %s\n' % (d, n) for d, n in checksums), encoding='utf-8')

    meta = {
        'metadata': {
            'title': ('Protocol outweighs method: patch sampling, evaluation scope, consensus level and threshold '
                      'dominate architecture and supervision-target choice in lung nodule segmentation on LIDC-IDRI'),
            'upload_type': 'dataset',
            'description': (
                '<p>Per-case results and aggregated analysis for a measurement study of <strong>protocol effects</strong> '
                'in lung nodule segmentation on LIDC-IDRI.</p>'
                '<p>The evaluation-target effect is 28.58 Dice points on the held-out test set, against a supervision-target '
                'range of 3.12, an architecture range of 1.19 and a two-sigma noise floor of 1.19: protocol choices outweigh '
                'learning-design choices by 7.1x and 18.7x. The effect generalises to the seven multi-rater QUBIQ 2021 tasks '
                'and decays with annotation agreement (eight points, Spearman rho = -0.86, exact permutation p = 0.011).</p>'
                '<p><strong>protocol-outweighs-method-v22-release-package.zip</strong> holds the aggregated analysis JSONs, '
                'the nine paper figures with their source data, and the QUBIQ per-case outputs, together with '
                '<code>manifest.json</code> (per-file size and SHA-256, plus the <code>source_commit</code> the results were '
                'generated from). <strong>protocol-outweighs-method-v22-lidc-per-case-csv.zip</strong> holds the LIDC-IDRI '
                'per-case CSVs (five validation folds plus the 202-case held-out test set) from which every confidence '
                'interval in the paper can be recomputed.</p>'
                '<p><strong>Attribution.</strong> LIDC-IDRI is distributed by The Cancer Imaging Archive under '
                'CC BY 3.0, and users must abide by the TCIA Data Usage Policy. Data citation: Armato III, S. G., '
                'McLennan, G., Bidaut, L., et al. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. '
                'https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX. The authors acknowledge the National Cancer Institute '
                'and the Foundation for the National Institutes of Health, and their critical role in the creation of '
                'the free publicly available LIDC/IDRI Database used in this study.</p>'
                '<p>QUBIQ 2021 is hosted on grand-challenge.org, whose Terms of Service state that Radboudumc claims '
                'no rights to Results generated by use of the platform. Please cite: Li, H. B., Navarro, F., Ezhov, I., '
                'et al. (2024). QUBIQ: Uncertainty Quantification for Biomedical Image Segmentation Challenge. '
                'arXiv:2405.18435.</p>'
                '<p><strong>Licensing.</strong> These derived per-case results are distributed under open access '
                '(recorded on Zenodo as other-open) with no restrictions beyond the upstream dataset terms: the '
                'LIDC-IDRI licence (CC BY 3.0, attribution as above, no ShareAlike condition) and the '
                'grand-challenge Terms of Service for QUBIQ 2021. The analysis code is MIT licensed at '
                'https://github.com/Ezbenzino/protocol-outweighs-method.</p>'
            ),
            'creators': [{
                'name': 'Li, Yize',
                'orcid': '0009-0002-9940-5317',
                'affiliation': 'Hangzhou Medical College',
            }],
            'license': 'other-open',
            'access_right': 'open',
            'version': 'v22',
            'publication_date': '2026-10-03',
            'language': 'eng',
            'keywords': ['lung nodule segmentation', 'LIDC-IDRI', 'QUBIQ', 'protocol effects',
                         'benchmark methodology', 'annotation variability', 'reproducibility'],
            'related_identifiers': [
                {'identifier': 'https://github.com/Ezbenzino/protocol-outweighs-method',
                 'relation': 'is supplemented by', 'scheme': 'url'},
                {'identifier': '10.7937/K9/TCIA.2015.LO9QL9SX',
                 'relation': 'is derived from', 'scheme': 'doi'},
                {'identifier': 'https://arxiv.org/abs/2405.18435',
                 'relation': 'references', 'scheme': 'url'},
            ],
            'notes': 'Source commit: %s. LIDC-IDRI (CC BY 3.0) and QUBIQ 2021 remain under their original '
                     'terms; raw images of either collection are not redistributed here.' % manifest['source_commit'],
        }
    }
    (out / 'zenodo-metadata.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    rows = '\n'.join(csv_rows) if csv_rows else '| (skipped with --skip-csv) | | |'
    readme = '''# Zenodo deposit — %(prefix)s

Per-case results and aggregated analysis for:

> Protocol outweighs method: patch sampling, evaluation scope, consensus level and threshold
> dominate architecture and supervision-target choice in lung nodule segmentation on LIDC-IDRI.

Generated by scripts/build_zenodo_release.py from source commit %(commit)s.
Every packed file was verified against outputs/release/manifest.json (size + SHA-256) before packing.

## Files

| File | Contents |
|---|---|
| %(relzip)s | Aggregated analysis JSONs, nine paper figures (PNG + PDF) with source data, QUBIQ per-case outputs, manifest.json |
| %(csvzip)s | LIDC-IDRI per-case results CSVs (5 validation folds + held-out test set) |
| CHECKSUMS.sha256 | SHA-256 of both archives and of every raw per-case CSV |
| zenodo-metadata.json | Record metadata for the Zenodo API |

## Per-case CSVs

| File | Size | SHA-256 (prefix) |
|---|---|---|
%(rows)s

- per_sample_*_val_fold*.csv and per_sample_test.csv — threshold sweep. Columns:
  run, fold, case_id, diameter_mm, threshold, dice_v1..v4, gt_area_v1..v4, area_pred.
  dice_v1..v4 are the Dice values against the four nested consensus references V>=1 (union) through V>=4 (unanimous).
- per_sample_free_*.csv — threshold-free probabilistic metrics (soft_dice, brier, prob_mass).
- The unit of analysis is the case: nodule instances are averaged within a case before any test.

## Verify

    sha256sum -c CHECKSUMS.sha256

## Not included

Raw LIDC-IDRI DICOM (The Cancer Imaging Archive) and QUBIQ 2021 images (grand-challenge.org);
both remain under their original terms. See the repository README for how to obtain them and for
the end-to-end reproduction path.

## Attribution

LIDC-IDRI is distributed by The Cancer Imaging Archive under CC BY 3.0, and users must abide by the
TCIA Data Usage Policy.

- Data citation: Armato III, S. G., McLennan, G., Bidaut, L., et al. (2015). Data From LIDC-IDRI
  [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
- Acknowledgement: the authors acknowledge the National Cancer Institute and the Foundation for the
  National Institutes of Health, and their critical role in the creation of the free publicly
  available LIDC/IDRI Database used in this study.

QUBIQ 2021 is hosted on grand-challenge.org. Its Terms of Service state that Radboudumc claims no
rights to Results generated by use of the platform. Please cite:

- Li, H. B., Navarro, F., Ezhov, I., et al. (2024). QUBIQ: Uncertainty Quantification for Biomedical
  Image Segmentation Challenge. arXiv:2405.18435

## License

These derived per-case results are distributed under open access (recorded on Zenodo as other-open)
with no restrictions beyond the upstream dataset terms: the LIDC-IDRI licence (CC BY 3.0, attribution
as above, no ShareAlike condition) and the grand-challenge Terms of Service for QUBIQ 2021. The
analysis code is MIT licensed; the manuscript text is CC BY 4.0.
''' % {
        'prefix': PREFIX,
        'commit': manifest['source_commit'],
        'relzip': rel_zip.name,
        'csvzip': csv_zip.name if not args.skip_csv else '(skipped)',
        'rows': rows,
    }
    (out / 'DEPOSIT_README.md').write_text(readme, encoding='utf-8')

    print('\n完成，输出目录：%s' % out)
    for p in sorted(out.iterdir()):
        print('  %-58s %s' % (p.name, fmt_mb(p.stat().st_size)))


if __name__ == '__main__':
    main()
