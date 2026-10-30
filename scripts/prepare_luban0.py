"""Fetch a pinned Apache-2.0 upstream into a new directory; verify LFS hashes."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx


def checksums(path):
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def verify(path, entry):
    sha256, blob = checksums(path)
    expected = entry.get("lfs", {}).get("sha256")
    size = entry.get("size", entry.get("lfs", {}).get("size"))
    if size is not None and path.stat().st_size != size:
        raise ValueError("upstream file size mismatch: " + entry["rfilename"])
    if expected:
        if sha256 != expected:
            raise ValueError("upstream LFS checksum mismatch: " + entry["rfilename"])
    elif not entry.get("blobId") or blob != entry["blobId"]:
        raise ValueError("upstream Git blob checksum mismatch: " + entry["rfilename"])
    return sha256


def download(client, urls, partial, entry):
    """Resume bounded streams; checksum prevents accepting a changed mirror revision."""
    for url in urls:
        for attempt in range(2):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with client.stream("GET", url, headers=headers) as stream:
                    if stream.status_code == 416:
                        return verify(partial, entry), url
                    stream.raise_for_status()
                    if stream.status_code == 206:
                        content_range = stream.headers.get("content-range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if not match or int(match[1]) != offset:
                            raise ValueError("invalid resume Content-Range")
                    else:
                        offset = 0  # A server that ignores Range must restart the partial file.
                    received, last_report, started = offset, time.monotonic(), time.monotonic()
                    with partial.open("ab" if offset else "wb") as output:
                        for block in stream.iter_bytes(4 * 1024 * 1024):
                            output.write(block)
                            received += len(block)
                            if time.monotonic() - last_report >= 10:
                                print(json.dumps({"file": entry["rfilename"], "bytes": received,
                                                  "MB_per_second": round((received - offset) / (time.monotonic() - started) / 1e6, 2)}), flush=True)
                                last_report = time.monotonic()
                return verify(partial, entry), url
            except ValueError as error:
                # A complete corrupt file cannot become valid by appending bytes.
                partial.unlink(missing_ok=True)
                print(f"rejected {entry['rfilename']} {type(error).__name__}: {error}", flush=True)
            except httpx.HTTPError as error:
                print(f"retry {entry['rfilename']} {type(error).__name__}", flush=True)
    raise RuntimeError("all official download sources failed: " + entry["rfilename"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--destination", default="/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    parser.add_argument("--metadata-file", help="Previously captured pinned Hugging Face API metadata")
    args = parser.parse_args()
    root = Path(args.destination)
    if str(root).startswith("/Volumes/"):
        assert (Path("/Volumes") / root.parts[2]).is_mount(), "storage not mounted"
    if (root / "upstream-file-manifest.json").exists():
        raise FileExistsError("Verified upstream already exists; refusing to overwrite")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=10)) as client:
        if args.metadata_file:
            metadata = json.loads(Path(args.metadata_file).read_text())
        else:
            response = client.get(f"{args.endpoint}/api/models/{args.model}/revision/{args.revision}?blobs=true")
            response.raise_for_status()
            metadata = response.json()
        revision = metadata["sha"]
        if revision != args.revision:
            raise ValueError("metadata does not match the requested pinned revision")
        files = []
        for entry in metadata["siblings"]:
            name = entry["rfilename"]
            if not name.endswith((".json", ".jinja", ".safetensors", ".txt")) and name not in {"README.md", "LICENSE"}:
                continue
            if "/" in name or name.startswith("."):
                continue
            target = root / name
            expected = entry.get("lfs", {}).get("sha256")
            partial = target.with_name(name + ".partial")
            if target.exists():
                digest, source = verify(target, entry), "previous pinned download; verified before reuse"
            else:
                urls = [f"https://modelscope.cn/models/{args.model}/resolve/master/{name}",
                        f"{args.endpoint}/{args.model}/resolve/{revision}/{name}",
                        f"https://huggingface.co/{args.model}/resolve/{revision}/{name}"]
                digest, source = download(client, urls, partial, entry)
                partial.replace(target)
            files.append({"Path": name, "Sha256": digest, "Size": target.stat().st_size,
                          "upstream_lfs_sha256": expected, "upstream_git_blob": entry.get("blobId"), "source": source})
            print(name, target.stat().st_size, flush=True)
        manifest = {"model": args.model, "revision": revision, "license": metadata.get("cardData", {}).get("license"),
                    "endpoint": args.endpoint, "created": time.time(), "elapsed_seconds": time.monotonic() - started,
                    "Data": {"Files": files}}
        (root / "upstream-file-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print("verified", revision, flush=True)


if __name__ == "__main__":
    main()
