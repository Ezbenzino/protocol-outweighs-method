# -*- coding: utf-8 -*-
"""upload_to_zenodo.py —— 把 dist/zenodo/ 上传为 Zenodo 记录。

令牌读取顺序：--token > 环境变量 ZENODO_TOKEN > ~/.zenodo_token > ~/.config/zenodo/token。

默认只创建**草稿**：上传全部文件并写入元数据，然后停下来等你在网页上确认。
发布（publish）会立刻签发永久 DOI，因此必须显式加 --publish 才会执行。

用法：
    python scripts/upload_to_zenodo.py                 # 上传为草稿（推荐）
    python scripts/upload_to_zenodo.py --sandbox       # 先到 sandbox.zenodo.org 试跑
    python scripts/upload_to_zenodo.py --dry-run       # 只校验本地文件与元数据
    python scripts/upload_to_zenodo.py --publish       # 上传后直接发布（不可撤回 DOI）
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist' / 'zenodo'
PROD = 'https://zenodo.org/api'
SAND = 'https://sandbox.zenodo.org/api'


def find_token(explicit):
    if explicit:
        return explicit.strip()
    env = os.environ.get('ZENODO_TOKEN')
    if env:
        return env.strip()
    for p in (Path.home() / '.zenodo_token', Path.home() / '.config' / 'zenodo' / 'token'):
        if p.exists():
            return p.read_text(encoding='utf-8').strip()
    sys.exit('未找到 Zenodo 令牌。请设置环境变量 ZENODO_TOKEN，或把令牌写入 %s' % (Path.home() / '.zenodo_token'))


def call(url, token, method='GET', payload=None, expect_json=True):
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/json'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw.decode()) if raw and expect_json else raw)
    except urllib.error.HTTPError as exc:
        sys.exit('Zenodo API %s %s 失败 (%s):\n%s' % (method, url, exc.code, exc.read().decode()[:900]))


def upload_file(url, token, path):
    headers = {
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/octet-stream',
        'Content-Length': str(path.stat().st_size),
    }
    with open(path, 'rb') as fh:
        req = urllib.request.Request(url, data=fh, headers=headers, method='PUT')
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            sys.exit('上传 %s 失败 (%s):\n%s' % (path.name, exc.code, exc.read().decode()[:900]))


def main():
    ap = argparse.ArgumentParser(description='上传 dist/zenodo/ 到 Zenodo')
    ap.add_argument('--token', help='Zenodo 访问令牌（默认从环境变量或 ~/.zenodo_token 读取）')
    ap.add_argument('--sandbox', action='store_true', help='使用 sandbox.zenodo.org')
    ap.add_argument('--publish', action='store_true', help='上传后立即发布（签发永久 DOI）')
    ap.add_argument('--dry-run', action='store_true', help='只校验本地文件与元数据，不联网写入')
    args = ap.parse_args()

    if not DIST.is_dir():
        sys.exit('缺少 %s，先运行： python scripts/build_zenodo_release.py' % DIST)
    files = sorted(p for p in DIST.iterdir() if p.is_file())
    meta_path = DIST / 'zenodo-metadata.json'
    if not meta_path.exists():
        sys.exit('缺少 %s' % meta_path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    total = sum(p.stat().st_size for p in files)

    print('待上传 %d 个文件，共 %.1f MB' % (len(files), total / (1 << 20)))
    for p in files:
        print('   %-58s %8.2f MB' % (p.name, p.stat().st_size / (1 << 20)))
    print('元数据 upload_type=%s license=%s creators=%s'
          % (metadata['metadata']['upload_type'], metadata['metadata']['license'],
             ', '.join(c['name'] for c in metadata['metadata']['creators'])))
    if args.dry_run:
        print('\n[dry-run] 本地校验通过，未联网。')
        return

    token = find_token(args.token)
    base = SAND if args.sandbox else PROD
    print('\n目标：%s' % base)

    status, dep = call(base + '/deposit/depositions', token, 'POST', {})
    dep_id = dep['id']
    bucket = dep['links']['bucket']
    print('[1/4] 已创建草稿 id=%s' % dep_id)

    for p in files:
        code = upload_file(bucket + '/' + p.name, token, p)
        print('[2/4] 上传 %-58s HTTP %s' % (p.name, code))

    status, dep = call(base + '/deposit/depositions/%s' % dep_id, token, 'PUT', metadata)
    print('[3/4] 元数据已写入')

    status, dep = call(base + '/deposit/depositions/%s' % dep_id, token)
    uploaded = [f['filename'] for f in dep.get('files', [])]
    missing = sorted(p.name for p in files if p.name not in uploaded)
    print('[4/4] 服务端已接收 %d 个文件%s' % (len(uploaded), '' if not missing else '，缺少：%s' % missing))
    if missing:
        sys.exit('有文件未上传成功，草稿保留在 %s' % dep['links']['html'])

    if args.publish:
        status, dep = call(base + '/deposit/depositions/%s/actions/publish' % dep_id, token, 'POST')
        print('\n已发布。DOI: %s' % dep.get('doi'))
    else:
        print('\n草稿已就绪（未发布）：%s' % dep['links']['html'])
        print('请在网页上确认元数据后点击 Publish —— 发布即签发永久 DOI。')


if __name__ == '__main__':
    main()
