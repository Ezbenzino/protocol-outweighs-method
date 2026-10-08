# -*- coding: utf-8 -*-
"""upload_to_zenodo.py —— 把 dist/zenodo/ 上传为 Zenodo 记录。

令牌读取顺序：--token > 环境变量 ZENODO_TOKEN > ~/.zenodo_token > ~/.config/zenodo/token。

默认只创建**草稿**：上传全部文件并写入元数据，然后停下来等你在网页上确认。
发布（publish）会立刻签发永久 DOI，因此必须显式加 --publish 才会执行。

用法：
    python scripts/upload_to_zenodo.py                    # 新建草稿并上传
    python scripts/upload_to_zenodo.py --deposition-id N  # 续传进已有草稿（断点重试）
    python scripts/upload_to_zenodo.py --force            # 已存在的文件也重传
    python scripts/upload_to_zenodo.py --sandbox          # 用 sandbox.zenodo.org 试跑
    python scripts/upload_to_zenodo.py --dry-run          # 只校验本地文件与元数据
    python scripts/upload_to_zenodo.py --publish          # 上传后直接发布（不可撤回 DOI）

上传失败（连接被重置等）会自动重试；重跑时已存在且大小一致的文件默认跳过，
因此网络中断后直接重跑同一条命令即可续传。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist' / 'zenodo'
PROD = 'https://zenodo.org/api'
SAND = 'https://sandbox.zenodo.org/api'
ATTEMPTS = 4
BACKOFF = 10          # 秒，指数退避：10 / 20 / 40


class UploadFailed(Exception):
    """单个文件重试用尽后仍失败；不中断其余文件的上传。"""


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


# .zenodo.json 用的是 GitHub 集成 / 新版 Zenodo 的词表（"is derived from"），
# 而 deposit API 只认驼峰形式（"isDerivedFrom"）。这里统一转换，
# 让同一份元数据既能被 GitHub 集成读取，也能直接提交给 deposit API。
RELATION_MAP = {
    'is supplement to': 'isSupplementTo',
    'is supplemented by': 'isSupplementedBy',
    'is derived from': 'isDerivedFrom',
    'is source of': 'isSourceOf',
    'is part of': 'isPartOf',
    'is referenced by': 'isReferencedBy',
    'is documented by': 'isDocumentedBy',
    'is new version of': 'isNewVersionOf',
    'is previous version of': 'isPreviousVersionOf',
    'is cited by': 'isCitedBy',
    'is reviewed by': 'isReviewedBy',
    'is required by': 'isRequiredBy',
    'is variant form of': 'isVariantFormOf',
    'is original form of': 'isOriginalFormOf',
    'is compiled by': 'isCompiledBy',
    'is continued by': 'isContinuedBy',
    'is described by': 'isDescribedBy',
    'is identical to': 'isIdenticalTo',
    'has part': 'hasPart',
    'has metadata': 'hasMetadata',
    'is metadata for': 'isMetadataFor',
    'is obsoleted by': 'isObsoletedBy',
    'is alternate identifier': 'isAlternateIdentifier',
}


def normalise_metadata(metadata):
    """把 related_identifiers 的关系名转成 deposit API 要求的驼峰形式（原地修改并返回）。"""
    for rel in metadata.get('metadata', {}).get('related_identifiers') or []:
        name = rel.get('relation', '')
        if ' ' not in name:
            continue                       # cites / references / documents 等单词形式本来就合法
        if name in RELATION_MAP:
            rel['relation'] = RELATION_MAP[name]
        else:                              # 兜底：多词 -> 驼峰
            head, *tail = name.split()
            rel['relation'] = head + ''.join(w.capitalize() for w in tail)
    return metadata


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


def upload_file(url, token, path, log=print):
    """PUT 一个文件；连接被重置时指数退避重试。"""
    size = path.stat().st_size
    last = None
    for attempt in range(1, ATTEMPTS + 1):
        headers = {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(size),
        }
        started = time.time()
        try:
            with open(path, 'rb') as fh:
                req = urllib.request.Request(url, data=fh, headers=headers, method='PUT')
                with urllib.request.urlopen(req, timeout=7200) as resp:
                    return resp.status, time.time() - started
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            if exc.code < 500:
                sys.exit('上传 %s 失败 (%s):\n%s' % (path.name, exc.code, detail))
            last = 'HTTP %s %s' % (exc.code, detail)
        except Exception as exc:                      # URLError, ConnectionResetError, timeout...
            last = '%s: %s' % (type(exc).__name__, exc)
        if attempt < ATTEMPTS:
            wait = BACKOFF * (2 ** (attempt - 1))
            log('      第 %d/%d 次失败（%s），%d 秒后重试' % (attempt, ATTEMPTS, last, wait))
            time.sleep(wait)
    raise UploadFailed('连续 %d 次失败：%s' % (ATTEMPTS, last))


def main():
    ap = argparse.ArgumentParser(description='上传 dist/zenodo/ 到 Zenodo')
    ap.add_argument('--token', help='Zenodo 访问令牌（默认从环境变量或 ~/.zenodo_token 读取）')
    ap.add_argument('--sandbox', action='store_true', help='使用 sandbox.zenodo.org')
    ap.add_argument('--publish', action='store_true', help='上传后立即发布（签发永久 DOI）')
    ap.add_argument('--dry-run', action='store_true', help='只校验本地文件与元数据，不联网写入')
    ap.add_argument('--deposition-id', type=int, help='续传进已有草稿，而不是新建')
    ap.add_argument('--force', action='store_true', help='已存在且大小一致的文件也重新上传')
    ap.add_argument('--exclude', action='append', default=[],
                    help='文件名包含该子串则跳过（可重复），用于在慢速链路上先传其余文件')
    args = ap.parse_args()

    if not DIST.is_dir():
        sys.exit('缺少 %s，先运行： python scripts/build_zenodo_release.py' % DIST)
    # 小的先传：网络不稳时先拿下能拿下的，大文件失败也不拖累其余文件
    files = sorted((p for p in DIST.iterdir() if p.is_file()), key=lambda p: p.stat().st_size)
    if args.exclude:
        files = [p for p in files if not any(x in p.name for x in args.exclude)]
        print('已排除：%s' % ', '.join(args.exclude))
    meta_path = DIST / 'zenodo-metadata.json'
    if not meta_path.exists():
        sys.exit('缺少 %s' % meta_path)
    metadata = normalise_metadata(json.loads(meta_path.read_text(encoding='utf-8')))
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

    if args.deposition_id:
        status, dep = call('%s/deposit/depositions/%d' % (base, args.deposition_id), token)
        if dep.get('submitted'):
            sys.exit('草稿 %d 已发布，不能再改；请新建一个。' % args.deposition_id)
        dep_id = dep['id']
        bucket = dep['links']['bucket']
        print('[1/4] 续传已有草稿 id=%s' % dep_id)
    else:
        status, dep = call(base + '/deposit/depositions', token, 'POST', {})
        dep_id = dep['id']
        bucket = dep['links']['bucket']
        print('[1/4] 已创建草稿 id=%s' % dep_id)

    failed = []
    present = {f['filename']: f.get('filesize', f.get('size', 0)) for f in dep.get('files', [])}
    if present:
        print('      服务端已有 %d 个文件' % len(present))

    for i, p in enumerate(files, 1):
        have = present.get(p.name)
        if have == p.stat().st_size and not args.force:
            print('[2/4] (%d/%d) %-58s 已存在，跳过' % (i, len(files), p.name))
            continue
        print('[2/4] (%d/%d) %-58s 上传中 ...' % (i, len(files), p.name), flush=True)
        try:
            code, secs = upload_file(bucket + '/' + p.name, token, p)
        except UploadFailed as exc:
            failed.append(p.name)
            print('      放弃 %s —— %s' % (p.name, exc), flush=True)
            continue
        print('      HTTP %s  用时 %.1f 秒  (%.1f MB/s)'
              % (code, secs, p.stat().st_size / (1 << 20) / max(secs, 0.001)), flush=True)

    status, dep = call('%s/deposit/depositions/%s' % (base, dep_id), token, 'PUT', metadata)
    print('[3/4] 元数据已写入')

    status, dep = call('%s/deposit/depositions/%s' % (base, dep_id), token)
    uploaded = {f['filename']: f.get('filesize', f.get('size', 0)) for f in dep.get('files', [])}
    missing = [p.name for p in files if uploaded.get(p.name) != p.stat().st_size]
    print('[4/4] 服务端已接收 %d/%d 个文件%s'
          % (len(files) - len(missing), len(files), '' if not missing else '，仍缺：%s' % missing))
    if missing:
        sys.exit('有文件未上传成功%s。续传： python scripts/upload_to_zenodo.py --deposition-id %d'
                 % ('' if not failed else '（本轮放弃：%s）' % ', '.join(failed), dep_id))

    if args.publish:
        status, dep = call('%s/deposit/depositions/%s/actions/publish' % (base, dep_id), token, 'POST')
        print('\n已发布。DOI: %s' % dep.get('doi'))
    else:
        print('\n草稿已就绪（未发布）：%s' % dep['links']['html'])
        print('请在网页上确认元数据后点击 Publish —— 发布即签发永久 DOI。')


if __name__ == '__main__':
    main()
