#!/usr/bin/env python3
"""Pinned research downloads; bounded HTTPS ranges, one final integrity pass."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path('/Volumes/移动数据库/lubanllm/models/community-q4-study-20261017')
PLANS = [
    ('unsloth/Qwen3.5-2B-GGUF', 'f6d5376be1edb4d416d56da11e5397a961aca8ae',
     'unsloth-ud-q4-k-xl', 'Qwen3.5-2B-UD-Q4_K_XL.gguf'),
    ('mlx-community/Qwen3.5-2B-MLX-4bit', '93760be4f1f69842a46bc13dbdc0f19e291392a3',
     'mlx-community-4bit', 'model.safetensors'),
]
LOCAL = threading.local()


def chunk(url, fd, size, begin, end):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(3):
        try:
            with LOCAL.session.get(url, headers={'Range': f'bytes={begin}-{end}'},
                                   stream=True, timeout=(15, 30)) as response:
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{end}/{size}':
                    raise ValueError('Server did not return the exact requested range')
                at = begin
                for data in response.iter_content(1 << 20):
                    if at + len(data) > end + 1:
                        raise ValueError('Oversized range response')
                    view = memoryview(data)
                    while view:
                        written = os.pwrite(fd, view, at)
                        if written <= 0:
                            raise OSError('Short range write')
                        at += written
                        view = view[written:]
                if at != end + 1:
                    raise ValueError('Incomplete range response')
                return end - begin + 1
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise RuntimeError(f'Download range failed at {begin}; partial file not approved') from None


def main():
    assert Path('/Volumes/移动数据库').is_mount()
    api = HfApi()
    for repo, revision, directory, weight_name in PLANS:
        root = ROOT / directory
        root.mkdir(parents=True, exist_ok=True)
        info = api.model_info(repo, revision=revision, files_metadata=True)
        selected = [s for s in info.siblings if s.rfilename == weight_name or
                    (repo.startswith('unsloth/') and s.rfilename in ('README.md', 'LICENSE', 'LICENSE.txt')) or
                    (repo.startswith('mlx-community/') and (s.rfilename.endswith(('.json', '.jinja', '.txt', '.md')) or s.rfilename == 'LICENSE'))]
        receipt_path = root / 'download-receipt.json'
        receipt = {'repo': repo, 'revision': revision, 'started': time.time(),
                   'method': 'official HTTPS ranges, then one SHA256 against pinned HF LFS metadata', 'files': []}
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        for item in selected:
            path = root / item.rfilename
            if item.rfilename != weight_name:
                hf_hub_download(repo, item.rfilename, revision=revision, local_dir=root)
                assert path.stat().st_size == item.size
                receipt['files'].append({'name': item.rfilename, 'bytes': item.size, 'downloaded': True})
                continue
            if path.exists():
                approved = next((v for v in prior.get('files', []) if v['name'] == weight_name), {})
                assert prior.get('revision') == revision and approved.get('sha256') == item.lfs.sha256
                assert path.stat().st_size == item.size
            else:
                partial = path.with_suffix(path.suffix + '.https-partial')
                fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                try:
                    os.ftruncate(fd, item.size)
                    url = f'https://huggingface.co/{repo}/resolve/{revision}/{weight_name}'
                    completed = 0
                    with ThreadPoolExecutor(max_workers=16) as pool:
                        jobs = [pool.submit(chunk, url, fd, item.size, start, min(start + (8 << 20), item.size) - 1)
                                for start in range(0, item.size, 8 << 20)]
                        for job in as_completed(jobs):
                            completed += job.result()
                            print(json.dumps({'model': directory, 'completed': completed, 'total': item.size}), flush=True)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                digest = hashlib.sha256()
                with partial.open('rb') as stream:
                    for data in iter(lambda: stream.read(8 << 20), b''):
                        digest.update(data)
                assert digest.hexdigest() == item.lfs.sha256, 'Downloaded model integrity mismatch'
                assert not path.exists()
                partial.rename(path)
            receipt['files'].append({'name': item.rfilename, 'bytes': item.size,
                                     'sha256': item.lfs.sha256, 'downloaded': True})
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        receipt.update(complete=True, finished=time.time())
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print('COMPLETE ' + repo, flush=True)


if __name__ == '__main__':
    main()
