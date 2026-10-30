#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the isolated Luban BF16 text prototype without running it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


UPSTREAM = "https://github.com/ashhart/TensorFold.git"
TAG = "v1.0.4"
COMMIT = "d4fc1964d0637c33e09ddac18aec936f962bb646"
ZIG_VERSION = "0.17.0"


def run(args, *, cwd=None, env=None, capture=False):
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def external_path(value):
    path = Path(value).expanduser().absolute()
    if path != path.resolve():
        raise ValueError(f"路径不能包含符号链接或 ..：{path}")
    parts = path.parts
    if len(parts) < 4 or parts[1] != "Volumes":
        raise ValueError(f"请指定 /Volumes 下的外置路径：{path}")
    volume = Path(*parts[:3])
    if not os.path.ismount(volume):
        raise ValueError(f"外置卷未挂载：{volume}")
    return path


def overlay_files(overlay):
    if overlay.is_symlink() or not overlay.is_dir():
        raise ValueError(f"缺少交付 overlay 目录：{overlay}")
    files = []
    for source in sorted(overlay.rglob("*")):
        relative = source.relative_to(overlay)
        # macOS creates these filesystem metadata files on external volumes.
        if any(part.startswith("._") for part in relative.parts):
            continue
        if source.is_symlink():
            raise ValueError(f"overlay 不允许符号链接：{relative}")
        if any(part in {".git", ".luban-zig-cache", ".luban-zig-global-cache", "zig-out"}
               for part in relative.parts):
            raise ValueError(f"overlay 包含受保护构建路径：{relative}")
        if source.is_file():
            files.append((source, relative))
        elif not source.is_dir():
            raise ValueError(f"overlay 不允许特殊文件：{relative}")
    required = {
        Path("luban-build.zig"),
        Path("zig/src/luban_native.zig"),
        Path("zig/src/luban_core.zig"),
        Path("zig/src/luban_kernel_check.zig"),
    }
    missing = required - {relative for _, relative in files}
    if missing:
        raise ValueError("overlay 不完整：" + ", ".join(str(path) for path in sorted(missing)))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, help="已挂载外置卷上的全新构建目录（必须不存在）")
    parser.add_argument("--zig", required=True, help="已安装 Zig 0.17.0 的外置可执行文件绝对路径")
    parser.add_argument("--source-cache", help="可选：只读复制已有外置 TensorFold Git 缓存，支持浅克隆")
    parser.add_argument("--install-product", action="store_true", help="将已构建worker和SHA回执复制到本机build/native，用于正式app打包")
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise ValueError("此研究原型仅支持 macOS Metal 构建")
    checkout = external_path(args.checkout)
    zig = external_path(args.zig)
    if checkout.exists():
        raise ValueError(f"拒绝覆盖已有目录；请使用全新 checkout：{checkout}")
    if not zig.is_file() or not os.access(zig, os.X_OK):
        raise ValueError(f"Zig 不是可执行文件：{zig}")
    if shutil.which("git") is None:
        raise ValueError("未找到 git；脚本不会自动安装依赖")
    source_cache = None
    if args.source_cache:
        source_cache = external_path(args.source_cache)
        git_dir = source_cache / ".git"
        if not git_dir.is_dir() or git_dir.is_symlink():
            raise ValueError(f"source-cache 必须是具有普通 .git 目录的已有仓库：{source_cache}")
        cache_origin = run(
            ["git", "remote", "get-url", "origin"], cwd=source_cache, capture=True,
        ).stdout.strip()
        if cache_origin not in {UPSTREAM, UPSTREAM.removesuffix(".git")}:
            raise ValueError(f"source-cache origin 不是固定 TensorFold 上游：{cache_origin!r}")
        cache_commit = run(
            ["git", "rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"],
            cwd=source_cache, capture=True,
        ).stdout.strip()
        if cache_commit != COMMIT:
            raise ValueError(f"source-cache 的 {TAG} 标签不是固定提交：{cache_commit!r}")
    version = run([zig, "version"], capture=True).stdout.strip()
    if version != ZIG_VERSION:
        raise ValueError(f"要求 Zig {ZIG_VERSION}，实际为 {version!r}")
    files = overlay_files(Path(__file__).resolve().parent / "overlay")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    checkout.mkdir()  # Exclusive creation also rejects a path created during validation.
    if source_cache is None:
        run(["git", "clone", "--depth", "1", "--branch", TAG, UPSTREAM, checkout])
    else:
        run(["git", "clone", "--no-hardlinks", "--branch", TAG, source_cache, checkout])
    # ExFAT AppleDouble files look like invalid pack indexes to Git. Remove
    # metadata only inside the new checkout created by this invocation.
    for metadata in (checkout / ".git").rglob("._*"):
        if metadata.is_file() and not metadata.is_symlink():
            metadata.unlink()
    if source_cache is not None:
        run(["git", "remote", "set-url", "origin", UPSTREAM], cwd=checkout)
    actual_commit = run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).stdout.strip()
    actual_origin = run(["git", "remote", "get-url", "origin"], cwd=checkout, capture=True).stdout.strip()
    if actual_commit != COMMIT or actual_origin != UPSTREAM:
        raise ValueError("上游提交或 origin 与固定版本不一致；未复制 overlay")
    if (checkout / ".git/objects/info/alternates").exists():
        raise ValueError("新 checkout 仍有外部 Git 对象依赖；未复制 overlay")
    run(["git", "diff", "--quiet"], cwd=checkout)
    run(["git", "diff", "--cached", "--quiet"], cwd=checkout)

    # Validate all destinations before copying any source into the new clone.
    for _, relative in files:
        target = checkout / relative
        if target.is_symlink() or target.resolve() != target:
            raise ValueError(f"上游目标包含符号链接：{relative}")
        if target.exists() and not target.is_file():
            raise ValueError(f"上游目标不是普通文件：{relative}")
    for source, relative in files:
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    env = os.environ.copy()
    env["ZIG_GLOBAL_CACHE_DIR"] = str(checkout / ".luban-zig-global-cache")
    run(
        [zig, "build", "--build-file", "luban-build.zig", "install", "kernel-check", "worker",
         "-j1", "--cache-dir", ".luban-zig-cache"],
        cwd=checkout,
        env=env,
    )
    worker = checkout / "zig-out/bin/luban-worker"
    receipt = {"protocol": "luban-worker-jsonl-v1", "upstream_commit": actual_commit,
               "zig_version": version, "binary_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
               "source_sha256": {str(relative): hashlib.sha256(source.read_bytes()).hexdigest() for source, relative in files}}
    worker.with_name(worker.name + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.install_product:
        product = Path(__file__).resolve().parents[2] / "build/native"
        product.mkdir(parents=True, exist_ok=True)
        shutil.copy2(worker, product / worker.name)
        shutil.copy2(worker.with_name(worker.name + ".json"), product / (worker.name + ".json"))
    print(f"worker_binary={worker}")
    print(f"upstream_commit={actual_commit}")
    print(f"zig_version={version}")
    print(f"native_binary={checkout / 'zig-out/bin/luban-native'}")
    print(f"kernel_check_binary={checkout / 'zig-out/bin/luban-kernel-check'}")
    print("仅完成构建；未运行 GPU 检查、模型推理或训练。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"构建失败：{error}", file=sys.stderr)
        sys.exit(1)
