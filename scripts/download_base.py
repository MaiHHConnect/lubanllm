#!/usr/bin/env python3
"""Download official Qwen ModelScope files with pinned revisions and SHA-256.

No authentication or remote code execution. Interrupted files are resumable.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="/Volumes/移动数据库/lubanllm/models/Qwen3-1.7B")
parser.add_argument("--metadata-only", action="store_true")
args = parser.parse_args()
root = Path(args.root)
if str(root).startswith("/Volumes/"):
    volume = Path("/Volumes") / root.parts[2]
    if not volume.is_mount():
        raise SystemExit("Target volume is not mounted; refusing to create a substitute directory: " + str(volume))
root.mkdir(parents=True, exist_ok=True)
api = "https://modelscope.cn/api/v1/models/Qwen/Qwen3-1.7B"
with urlopen(api + "/repo/files?Revision=master&Recursive=true", timeout=30) as response:
    listing = json.load(response)
(root / "upstream-file-manifest.json").write_text(json.dumps(listing, indent=2) + "\n")
files = [f for f in listing["Data"]["Files"] if f["Type"] == "blob" and not f["Path"].startswith(".")]
if args.metadata_only:
    files = [f for f in files if not f["Path"].endswith(".safetensors")]
remaining = sum(max(0, f["Size"] - ((root / f["Path"]).stat().st_size
                  if (root / f["Path"]).exists() else 0)) for f in files)
if shutil.disk_usage(root).free < remaining + 6_000_000_000:
    raise SystemExit("Insufficient free disk space including 6 GB reserve")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(record):
    path = root / record["Path"]
    if path.exists() and path.stat().st_size == record["Size"] and sha256(path) == record["Sha256"]:
        print("verified " + path.name, flush=True)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    query = urlencode({"Revision": record["Revision"], "FilePath": record["Path"]})
    url = api + "/repo?" + query
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > record["Size"]:
                partial.unlink()
                offset = 0
            if offset < record["Size"]:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with urlopen(Request(url, headers=headers), timeout=45) as response:
                    resumed = offset and response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("incorrect HTTP resume range")
                    with partial.open("ab" if resumed else "wb") as handle:
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            handle.write(chunk)
            if partial.stat().st_size != record["Size"]:
                raise RuntimeError("download size mismatch")
            if sha256(partial) != record["Sha256"]:
                partial.unlink()
                raise RuntimeError("download SHA-256 mismatch")
            os.replace(partial, path)
            print("downloaded and verified " + path.name, flush=True)
            return
        except Exception as error:
            print(f"attempt {attempt+1}: {path.name}: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


with ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(download, files))
print("All selected files verified. " + str(root), flush=True)
