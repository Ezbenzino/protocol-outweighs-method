"""Download LIDC-IDRI CT scans (from IDC GCS bucket) + original XML annotations.

Rationale:
  * TCIA's NBIA API host (services.cancerimagingarchive.net) is unreachable from
    some networks (read timeouts), but IDC's public GCS bucket and TCIA's wiki
    host are reachable and fast.
  * idc-index bundles s5cmd, which may be flagged by Windows Defender; this
    script downloads directly over HTTPS with urllib (no s5cmd, no auth).

Layout produced under --raw-dir:
  LIDC-IDRI-XXXX/<crdc_series_uuid>/*.dcm     (CT slices)
  LIDC-IDRI-XXXX/<...>.xml                    (4-reader annotation files)

Usage:
  python preprocess/download_lidc.py --raw-dir data/raw/LIDC-IDRI
  python preprocess/download_lidc.py --limit 3          # quick test
"""
import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.request
import zipfile

import pandas as pd

BUCKET = "idc-open-data"
GCS_LIST = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_GET = "https://storage.googleapis.com/{bucket}/{key}"
XML_URL = ("https://wiki.cancerimagingarchive.net/download/attachments/1966254/"
           "LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2")

_lock = threading.Lock()
_state = {"done": 0, "total": 0, "bytes": 0, "start": None}


def _http_json(url, timeout=60):
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def list_objects(prefix):
    out = []
    token = None
    while True:
        url = f"{GCS_LIST.format(bucket=BUCKET)}?prefix={prefix}&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        d = _http_json(url)
        for it in d.get("items", []):
            out.append((it["name"], int(it["size"])))
        token = d.get("nextPageToken")
        if not token:
            break
    return out


def build_file_list(rows, raw_dir, list_workers=8):
    """List all objects for all series -> [(url, dest, size), ...]."""
    jobs = []
    for row in rows:
        uuid = row["crdc_series_uuid"]
        patient = row["PatientID"]
        jobs.append((uuid, patient))

    files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=list_workers) as ex:
        futures = {ex.submit(list_objects, u + "/"): (u, p) for u, p in jobs}
        for fut in concurrent.futures.as_completed(futures):
            u, p = futures[fut]
            for name, size in fut.result():
                dest = os.path.join(raw_dir, p, u, os.path.basename(name))
                files.append((GCS_GET.format(bucket=BUCKET, key=name), dest, size))
    return files


def download_one(job):
    url, dest, size = job
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        with _lock:
            _state["done"] += 1
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            with _lock:
                _state["done"] += 1
                _state["bytes"] += size
            return size
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    # give up on this file (retried on next run) without crashing the pipeline
    with _lock:
        _state["done"] += 1
        _state["failed"] = _state.get("failed", 0) + 1
    return -1


def download_xml(raw_dir):
    zpath = os.path.join(raw_dir, "LIDC-XML-only.zip")
    if not os.path.exists(zpath):
        print("downloading XML annotations ...", flush=True)
        req = urllib.request.Request(XML_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(zpath, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    print("extracting XML ...", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw_dir)
    src_root = os.path.join(raw_dir, "tcia-lidc-xml")
    moved = 0
    if os.path.isdir(src_root):
        for name in os.listdir(src_root):
            sp = os.path.join(src_root, name)
            if not os.path.isdir(sp):
                continue
            m = re.search(r"LIDC-IDRI-\d{4}", name)
            case = m.group(0) if m else name
            dest = os.path.join(raw_dir, case)
            os.makedirs(dest, exist_ok=True)
            for fn in os.listdir(sp):
                if fn.lower().endswith(".xml"):
                    os.replace(os.path.join(sp, fn), os.path.join(dest, fn))
                    moved += 1
    print(f"moved {moved} xml files into case folders", flush=True)


def _log_periodically(total):
    while True:
        time.sleep(20)
        with _lock:
            done, bytes_, start = _state["done"], _state["bytes"], _state["start"]
        if done >= total:
            return
        rate = (bytes_ / 1e6) / max(1, time.time() - start)
        eta = (total - done) / max(1, done) * max(1, time.time() - start) if done else float("inf")
        print(f"[progress] {done}/{total} files  {bytes_/1e9:.2f} GB  "
              f"{rate:.1f} MB/s  eta~{eta/3600:.1f}h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "LIDC-IDRI"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N series")
    ap.add_argument("--xml-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)

    if args.xml_only:
        download_xml(args.raw_dir)
        print("DONE", flush=True)
        return

    import idc_index_data
    df = pd.read_parquet(idc_index_data.IDC_INDEX_PARQUET_FILEPATH)
    ct = df[(df["collection_id"] == "lidc_idri") & (df["Modality"] == "CT")].copy()
    rows = ct.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    print(f"listing objects for {len(rows)} CT series ...", flush=True)
    files = build_file_list(rows, args.raw_dir)
    _state["total"] = len(files)
    _state["start"] = time.time()
    print(f"total files: {len(files)} (~{sum(s for _, _, s in files)/1e9:.1f} GB)", flush=True)

    logger = threading.Thread(target=_log_periodically, args=(len(files),), daemon=True)
    logger.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(download_one, files))

    with _lock:
        print(f"finished {_state['done']}/{len(files)} files "
              f"{_state['bytes']/1e9:.2f} GB "
              f"failed={_state.get('failed', 0)}", flush=True)

    download_xml(args.raw_dir)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
