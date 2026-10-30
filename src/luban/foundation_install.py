"""Pinned first-install assets. No existing model, learned data or settings are overwritten."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import ssl
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile

from .settings import private_json, require_storage
from . import factory_expert

RELEASE_URL = "https://github.com/MaiHHConnect/lubanllm/releases/download/v0.3.217-beta.1/"
# These are release archive checksums, not a digital signature or model quality qualification.
ASSETS = ({'kind': 'mlx', 'name': 'Luban-MLX-Q4-model.zip', 'bytes': 1728259246, 'sha256': '2ff6cd2dfaa40b78275ba94aa55b94823114c3993451729a60d1b9d345e4588f', 'prefix': 'Luban-MLX-Q4', 'expanded_bytes': 1749093508, 'files': ['README.md', 'config.json', 'model.safetensors', 'model.safetensors.index.json', 'chat_template.jinja', 'preprocessor_config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json', 'LICENSE']}, {'kind': 'speech', 'name': 'Luban-Speech-model.zip', 'bytes': 1567082641, 'sha256': '553d5396f1ce6d21d7631934011b06865e5d115734e569acdfde9a6588ec647c', 'prefix': 'Luban-Speech', 'expanded_bytes': 1576391182, 'files': ['README.md', 'config.json', 'configuration.json', 'model.safetensors', 'chat_template.jinja', 'generation_config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'LICENSE-2.0.txt']}, {'kind': 'metadata', 'name': 'Luban-Model-Metadata.zip', 'bytes': 7041228, 'sha256': '2d27f94222429cc2e9281ece88974767811a76df5458b5ed00fcc2f436cf2ff1', 'prefix': 'Luban-Model-Metadata', 'expanded_bytes': 22928412, 'files': ['LICENSE', 'chat_template.jinja', 'config.json', 'merges.txt', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'upstream-file-manifest.json', 'video_preprocessor_config.json', 'vocab.json']})
ASSETS = (*ASSETS, factory_expert.MANIFEST["asset"])
CHUNK = 1024 * 1024
MLX_WEIGHT = {"name": "model.safetensors", "bytes": 1722271785,
              "sha256": "713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5", "downloaded": True}


def bundled_assets():
    """Only the four pinned public archives inside an installed application."""
    folder = Path(__file__).resolve().parents[3] / "foundation"
    if not folder.exists():
        return None
    if folder.is_symlink() or not folder.is_dir() or {p.name for p in folder.iterdir()} != {a["name"] for a in ASSETS}:
        raise ValueError("内置模型文件清单不一致，请重新下载完整安装包")
    for asset in ASSETS:
        path = folder / asset["name"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise ValueError("内置模型文件不完整，请重新下载完整安装包")
    return folder


def installation_status(settings):
    config = settings.snapshot()
    missing = not (Path(config["model_path"]) / "config.json").is_file()
    if config.get("native_inference_precision") == "mlx-q4":
        missing = missing or not (Path(config.get("native_mlx_model_path", "")) / "model.safetensors").is_file()
    return {"required": missing, "download_bytes": sum(row["bytes"] for row in ASSETS), "bundled": bundled_assets() is not None}


def _allowed_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
            or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("基础模型下载地址不在允许范围")
    if parsed.hostname == "github.com" and not url.startswith(RELEASE_URL):
        raise ValueError("基础模型下载地址不在允许范围")
    return url


class _ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _allowed_url(newurl))


def _download(asset, target, progress):
    folder = bundled_assets()
    if folder is not None:
        digest, done = hashlib.sha256(), 0
        with (folder / asset["name"]).open("rb") as source, target.open("xb") as output:
            os.chmod(target, 0o600)
            while block := source.read(CHUNK):
                done += len(block)
                if done > asset["bytes"]:
                    raise ValueError("内置模型文件尺寸不一致")
                digest.update(block)
                output.write(block)
                progress(done)
            output.flush()
            os.fsync(output.fileno())
        if done != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError("内置模型文件校验失败，请重新下载完整安装包")
        return
    import certifi
    opener = urllib.request.build_opener(_ReleaseRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())))
    request = urllib.request.Request(_allowed_url(RELEASE_URL + asset["name"]), headers={"User-Agent": "Luban-First-Install"})
    digest, done = hashlib.sha256(), 0
    with opener.open(request, timeout=60) as response, target.open("xb") as output:
        _allowed_url(response.geturl())
        length = response.headers.get("Content-Length")
        if response.status != 200 or length is not None and length != str(asset["bytes"]):
            raise ValueError("基础模型下载尺寸不一致")
        os.chmod(target, 0o600)
        while block := response.read(CHUNK):
            done += len(block)
            if done > asset["bytes"]:
                raise ValueError("基础模型下载超过允许尺寸")
            digest.update(block)
            output.write(block)
            progress(done)
        output.flush()
        os.fsync(output.fileno())
    if done != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
        raise ValueError("基础模型下载不完整或校验失败")


def _extract(archive, destination, asset):
    for name in asset["files"]:
        path = PurePosixPath(name)
        if not path.parts or path.is_absolute() or str(path) != name or ".." in path.parts or "\\" in name:
            raise ValueError("基础模型文件路径无效")
    expected = {asset["prefix"] + "/" + name for name in asset["files"]}
    with zipfile.ZipFile(archive) as package:
        entries = package.infolist()
        if len(entries) != len(expected) or {row.filename for row in entries} != expected:
            raise ValueError("基础模型文件清单不一致")
        if sum(row.file_size for row in entries) != asset["expanded_bytes"]:
            raise ValueError("基础模型解压尺寸不一致")
        for row in entries:
            mode = row.external_attr >> 16
            if (row.is_dir() or row.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or row.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or row.file_size < 0 or row.file_size > asset["expanded_bytes"]
                    or row.filename.endswith(".safetensors") and row.compress_type != zipfile.ZIP_STORED):
                raise ValueError("基础模型压缩格式无效")
        destination.mkdir(mode=0o700)
        for row in entries:
            # Nested expert paths must match the compiled whitelist exactly.
            target = destination / row.filename.split("/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with package.open(row) as source, target.open("xb") as output:
                os.chmod(target, 0o600)
                shutil.copyfileobj(source, output, CHUNK)
                output.flush()
                os.fsync(output.fileno())
            if target.stat().st_size != row.file_size:
                raise ValueError("基础模型文件解压不完整")


def install_models(settings, progress, *, download=_download):
    """Download into a fresh private directory, then commit paths; never start a model."""
    if not installation_status(settings)["required"]:
        raise ValueError("基础模型已安装")
    before = settings.snapshot()
    root = require_storage(before["storage_root"])
    store = root / "modules/luban0-v1"
    if any((store / name).exists() for name in ("active.json", "active-core.json", "active-router.json")):
        raise ValueError("已有学习模型，请恢复原模型目录或导入完整模型包")
    parent = root / "app/foundation"
    for directory in (root, root / "app", parent):
        if directory.is_symlink():
            raise ValueError("基础模型目录不能是符号链接")
    factory_expert.require_empty_storage(root)
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    required_space = sum(row["expanded_bytes"] for row in ASSETS) + max(row["bytes"] for row in ASSETS) + 256 * CHUNK
    if shutil.disk_usage(parent).free < required_space:
        raise ValueError("磁盘空间不足，基础模型安装需要至少6GB可用空间")
    installed = parent / ("release-" + uuid.uuid4().hex)
    total = sum(row["bytes"] for row in ASSETS)
    completed = 0
    with tempfile.TemporaryDirectory(prefix=".install-", dir=parent) as temporary:
        staged = Path(temporary) / "root"
        (staged / "models").mkdir(parents=True, mode=0o700)
        for asset in ASSETS:
            archive = Path(temporary) / asset["name"]
            phase = "copy" if bundled_assets() is not None else "download"
            progress({"phase": phase, "asset": asset["kind"], "bytes": completed, "total": total})
            download(asset, archive, lambda done: progress({"phase": phase, "asset": asset["kind"], "bytes": completed + done, "total": total}))
            progress({"phase": "extract", "asset": asset["kind"], "bytes": completed + asset["bytes"], "total": total})
            folder = {"mlx": "models/mlx", "speech": "models/asr", "metadata": "models/base", "expert": "seed"}[asset["kind"]]
            _extract(archive, staged / folder, asset)
            archive.unlink()
            completed += asset["bytes"]
        private_json(staged / "models/mlx/download-receipt.json", {
            "complete": True, "revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3", "finished": time.time(),
            "files": [MLX_WEIGHT], "installation": "pinned-release-archive", "archive_sha256": ASSETS[0]["sha256"]})
        # Rename only declared roots: macOS may move AppleDouble sidecars
        # together with their files on external disks.
        for name in sorted({PurePosixPath(name).parts[0] for name in factory_expert.MANIFEST["asset"]["files"]}):
            os.rename(staged / "seed" / name, staged / name)
        shutil.rmtree(staged / "seed")
        progress({"phase": "validate", "asset": "expert", "bytes": total, "total": total})
        payload = factory_expert.validate(staged)
        if settings.snapshot() != before:
            raise RuntimeError("安装期间设置发生变化，请重试")
        os.rename(staged, installed)
        try:
            factory_expert.install_journal(installed, payload)
            settings.update({"storage_root": str(installed),
                             "model_path": str(installed / "models/base"), "speech_model_path": str(installed / "models/asr"),
                             "native_mlx_model_path": str(installed / "models/mlx"), "native_inference_precision": "mlx-q4"})
        except BaseException:
            shutil.rmtree(installed)
            raise
    return {"installed": True, "model_started": False}
