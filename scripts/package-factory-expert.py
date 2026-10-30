#!/usr/bin/env python3
"""Package the already adopted core/expert as a new-install baseline, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luban import model_packages as packages

RUN_IDS = {"fdfa0ce5a75c", "bafb4050374f"}
PREFIX = "Luban-Factory-Expert"
NOTICE = """Luban Factory Expert / 鲁班零号出厂专家

此分包按产品所有者要求，复用已采纳的鲁班零号核心、专家和路由修改。
它以已授权的产品身份资料及项目自拟基础任务为起点；不含用户聊天、
API 凭据、教师原始响应、优化器或整份研究实验。
基础模型及其派生部分保留随包 Apache License 2.0；本说明不对应用源码、
独立品牌资料或其他未随包内容授予额外开源许可。
原质量/训练资格保留其实际测量及来源绑定；历史 local-trial 来源仍只是
本地试用范围，不被升级为通用能力、正式论文或第三方审查证明。
出厂学习历史仅作来源依据，不计作本设备用户的新学习或进化成果。

At the product owner's request, this package reuses the adopted Luban core,
expert and router modifications. Its materials consist of authorized product
identity facts and project-authored elementary tasks. It excludes user chats,
API credentials, raw teacher responses, optimizer state and full experiments.
The base model and its derivative portions retain the accompanying Apache
License 2.0. This notice grants no additional open-source license for the
application source, independent brand materials or other excluded content.
Original quality/training evidence and source bindings remain unchanged.
Historical local-trial evidence remains limited to local trials; it is not
general capability, publication or independent review certification.
Factory history is provenance, not new learning or evolution by this user.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "src/luban/factory-expert.json")
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    if args.output.exists():
        raise ValueError("目标分包已存在，拒绝覆盖")
    source_manifest = packages._read(root / "manifest.json")
    payload = packages._read(root / "accepted.json")
    payload["runs"] = [run for run in payload["runs"] if run["id"] in RUN_IDS]
    if {run["id"] for run in payload["runs"]} != RUN_IDS:
        raise ValueError("缺少已采纳的出厂核心或专家")
    payload["samples"] = []
    payload["material_archives"] = []
    payload["factory_runs"] = sorted(RUN_IDS)
    names = {"active-core.json", "active.json", "active-router.json"}
    for run in payload["runs"]:
        names.update(run["commit"]["artifacts"])
    source_names = {"modules/luban0-v1/" + name for name in names}
    source_names.update({"runtime/mlx/expert-proof.json", "runtime/mlx/training-proof.json"})
    if len(source_names) != 14:
        raise ValueError("出厂模型文件清单发生变化")
    files = {name: source_manifest["files"][name] for name in sorted(source_names)}
    packages._validate_journal(payload)
    packages._validate_modules(root, payload, files)
    generated = {"accepted.json": packages._encoded(payload), "NOTICE": NOTICE.encode("utf-8")}
    for name, content in generated.items():
        files[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    files["LICENSE"] = source_manifest["files"]["models/base/LICENSE"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".factory-expert-", dir=args.output.parent) as folder:
        target = Path(folder) / args.output.name
        with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
            for name in sorted(files):
                info = zipfile.ZipInfo(PREFIX + "/" + name, (2026, 10, 21, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                info.compress_type = zipfile.ZIP_STORED if name.endswith(".safetensors") else zipfile.ZIP_DEFLATED
                if name in generated:
                    archive.writestr(info, generated[name])
                    continue
                source = root / ("models/base/LICENSE" if name == "LICENSE" else name)
                packages._plain(source)
                if source.is_symlink() or not source.is_file():
                    raise ValueError("出厂模型来源不是普通文件")
                digest, size = hashlib.sha256(), 0
                with source.open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                    while block := reader.read(packages.CHUNK):
                        size += len(block)
                        digest.update(block)
                        writer.write(block)
                if {"size": size, "sha256": digest.hexdigest()} != files[name]:
                    raise ValueError("出厂模型来源已变化：" + name)
        digest = hashlib.sha256()
        with target.open("rb") as reader:
            while block := reader.read(packages.CHUNK):
                digest.update(block)
        result = {"schema": "luban-factory-expert-v1", "identity": packages.TRUSTED_IDENTITY,
                  "factory_runs": sorted(RUN_IDS), "files": files,
                  "asset": {"kind": "expert", "name": args.output.name, "prefix": PREFIX,
                            "bytes": target.stat().st_size, "sha256": digest.hexdigest(),
                            "expanded_bytes": sum(row["size"] for row in files.values()),
                            "files": sorted(files)}}
        with args.output.open("xb"):
            pass
        os.replace(target, args.output)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(args.output), "manifest": str(args.manifest), **result["asset"],
                      "validated_runs": sorted(RUN_IDS), "teacher_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
