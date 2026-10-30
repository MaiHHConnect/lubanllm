"""Bounded portable model packages; staging never activates or deletes live data."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import struct
import time
import unicodedata
import urllib.request
import uuid
import zipfile

from .settings import private_json, require_storage

RESET_PHRASE = "我同意重修"
FOUNDATION_URL = "https://ai.maihh.net/models/luban0/base.zip"
# No foundation package has been published or cryptographically pinned yet.
FOUNDATION_SHA256 = None
SCHEMA = "luban-model-package-v1"
MAX_FILES = 4096
MAX_BYTES = 16 * 1024**3
MAX_TEXT = 32 * 1024**2
CHUNK = 1024**2
HEX = re.compile(r"[a-f0-9]{64}")
TRUSTED = {
    'base': {
        'LICENSE': {'sha256': 'bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a', 'size': 11544},
        'README.md': {'sha256': 'c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275', 'size': 62814},
        'chat_template.jinja': {'sha256': '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80', 'size': 7755},
        'config.json': {'sha256': 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4', 'size': 2908},
        'merges.txt': {'sha256': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d', 'size': 3353259},
        'model.safetensors-00001-of-00001.safetensors': {'sha256': 'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1', 'size': 4548221488},
        'model.safetensors.index.json': {'sha256': 'aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9', 'size': 64460},
        'preprocessor_config.json': {'sha256': '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516', 'size': 390},
        'tokenizer.json': {'sha256': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42', 'size': 12807982},
        'tokenizer_config.json': {'sha256': '49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c', 'size': 16709},
        'upstream-file-manifest.json': {'sha256': 'd8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237', 'size': 4721},
        'video_preprocessor_config.json': {'sha256': '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13', 'size': 385},
        'vocab.json': {'sha256': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003', 'size': 6722759},
    },
    'asr': {
        '.gitattributes': {'sha256': '0aefab68978f25d82421f2039b23711121eb624b23fac3376f941aab386c05be', 'size': 2129},
        'LICENSE-2.0.txt': {'sha256': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30', 'size': 11358},
        'README.md': {'sha256': '742006d0f99ce6475b1c5e7b66e4a2f166cce836dd7ce3d7f20707252859195a', 'size': 16538},
        'chat_template.jinja': {'sha256': 'f50e6b694fbf4a683206e37869990d68333fe95d285730f084c838a34b0d98c2', 'size': 1434},
        'config.json': {'sha256': '9eecf6f1b383e343889c2e6010e632590fa57d4bc678e151c7d6a160a0dfb04a', 'size': 2398},
        'configuration.json': {'sha256': '0657e261c850f8d368b4da948a15279089f3484d40c152cc07fb4e9d1727ca3d', 'size': 63},
        'generation_config.json': {'sha256': '9939fc9388b79bd70757f938b87381e817173d6a6158f5af6506c0b73e775c3c', 'size': 165},
        'license-manifest.json': {'sha256': '163f4a137e5973856246c9be72ec593f259e80237ee3530b4b7d9bfc1fc1625f', 'size': 283},
        'model.safetensors': {'sha256': 'd3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35', 'size': 1564928088},
        'processor_config.json': {'sha256': 'bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a', 'size': 487},
        'tokenizer.json': {'sha256': 'fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05', 'size': 11429653},
        'tokenizer_config.json': {'sha256': '945e980986de2ca7768f3326bfdbb4fbea3406f972b8ae0be233089f2b253c11', 'size': 998},
        'upstream-file-manifest.json': {'sha256': '2488a3b5d7fe433d41c43771c98cf95bbfc4d548c4aa8a12990cffab597530c0', 'size': 4503},
    },
}
TRUSTED_IDENTITY = '673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df'

_MODULE = re.compile(r"(?:active(?:-core|-router)?\.json|router(?:-[a-f0-9]{12})?\.(?:json|safetensors)|core-[a-f0-9]{12}\.(?:json|safetensors|retention\.json)|layer-23/[a-zA-Z0-9_-]{1,80}\.(?:json|safetensors)|(?:learning|evolution|coverage-acceptance)-[a-f0-9]{12}\.json)")
_POINTERS = {"active.json", "active-core.json", "active-router.json"}
_PATH_KEYS = ("model_path", "speech_model_path", "storage_root")


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON 含重复字段")
        value[key] = item
    return value


def _json(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("非有限 JSON")))


def _read(path):
    if Path(path).stat().st_size > MAX_TEXT:
        raise ValueError("JSON 文件过大")
    return _json(Path(path).read_bytes())


def _plain(path, *, directory=False):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("模型包路径不能包含符号链接")
    if path.exists():
        mode = path.stat().st_mode
        if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
            raise ValueError("模型包路径必须是普通文件或目录")
    return path


def _name(name):
    if (not isinstance(name, str) or not name or len(name) > 240 or not name.isascii()
            or "\\" in name or ":" in name or name != unicodedata.normalize("NFC", name)
            or any(ord(char) < 32 for char in name)):
        raise ValueError("ZIP 文件路径无效")
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or "/".join(parts) != name or any(part in {".", ".."} or part.endswith((".", " ")) for part in parts):
        raise ValueError("ZIP 文件路径越界")
    if any(re.fullmatch(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part, re.I) for part in parts):
        raise ValueError("ZIP 含平台保留文件名")
    if name in {"manifest.json", "accepted.json"}:
        return name
    for key in ("base", "asr"):
        prefix = "models/" + key + "/"
        if name.startswith(prefix) and name[len(prefix):] in TRUSTED[key]:
            return name
    from .mlx_model_packages import allowed_name
    if allowed_name(name):
        return name
    prefix = "modules/luban0-v1/"
    if name.startswith(prefix) and _MODULE.fullmatch(name[len(prefix):]):
        return name
    raise ValueError("模型包包含非白名单文件")


def _record(path):
    path = _plain(path)
    return {"size": path.stat().st_size, "sha256": _sha(path)}


def _check_manifest(manifest):
    if (not isinstance(manifest, dict) or set(manifest) - {"engine"} != {"schema", "kind", "identity", "files", "package_id"}
            or manifest["schema"] != SCHEMA or manifest["identity"] != TRUSTED_IDENTITY
            or manifest["kind"] not in {"learned", "foundation"}
            or "engine" in manifest and manifest["engine"] != "mlx-q4"):
        raise ValueError("模型包版本或可信基座不兼容")
    files = manifest["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("模型包文件数量无效")
    total = 0
    for name, record in files.items():
        _name(name)
        if name == "manifest.json" or not isinstance(record, dict) or set(record) != {"size", "sha256"}:
            raise ValueError("模型包清单无效")
        if type(record["size"]) is not int or not 0 <= record["size"] <= MAX_BYTES or not HEX.fullmatch(record.get("sha256", "")):
            raise ValueError("模型包文件尺寸或摘要无效")
        if not name.endswith(".safetensors") and record["size"] > MAX_TEXT:
            raise ValueError("模型包文本过大")
        if name.startswith("modules/") and record["size"] > 128 * 1024**2:
            raise ValueError("已学模块文件尺寸超出当前 2B 图限制")
        total += record["size"]
    if total > MAX_BYTES or "accepted.json" not in files:
        raise ValueError("模型包大小超限或缺少采纳记录")
    from .mlx_model_packages import trusted_files
    trusted = (trusted_files() if manifest.get("engine") == "mlx-q4" else
               {"models/" + key + "/" + name: record for key, values in TRUSTED.items() for name, record in values.items()})
    for name, record in trusted.items():
            if files.get(name) != record:
                raise ValueError("基座/语音文件未匹配应用内固定的可信 SHA")
    if manifest.get("engine") != "mlx-q4" and any(name.startswith(("models/mlx/", "runtime/mlx/")) for name in files):
        raise ValueError("旧格式模型包不能包含MLX来源")
    content = {key: value for key, value in manifest.items() if key != "package_id"}
    if manifest["package_id"] != hashlib.sha256(_encoded(content)).hexdigest():
        raise ValueError("模型包身份校验失败")
    return total


def _zip_manifest(archive):
    infos = archive.infolist()
    if not 1 <= len(infos) <= MAX_FILES + 1:
        raise ValueError("ZIP 文件数量超限")
    seen, total = set(), 0
    for info in infos:
        _name(info.filename)
        key = unicodedata.normalize("NFKC", info.filename).casefold()
        mode = info.external_attr >> 16
        if (key in seen or info.is_dir() or info.flag_bits & 1
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
            raise ValueError("ZIP 包含重复、链接、特殊、加密或不支持的文件")
        if info.filename.endswith(".safetensors") and info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("权重必须原样存储，拒绝压缩权重包")
        if info.file_size > MAX_BYTES or (not info.filename.endswith(".safetensors") and info.file_size > MAX_TEXT):
            raise ValueError("ZIP 文件大小超限")
        if info.file_size > max(1, info.compress_size) * 1000:
            raise ValueError("ZIP 压缩比超限")
        seen.add(key)
        total += info.file_size
    if total > MAX_BYTES + MAX_TEXT:
        raise ValueError("ZIP 解压总量超限")
    manifest = _json(archive.read("manifest.json"))
    _check_manifest(manifest)
    if set(manifest["files"]) | {"manifest.json"} != {info.filename for info in infos}:
        raise ValueError("ZIP 内容与清单不一致")
    for info in infos:
        if info.filename != "manifest.json" and info.file_size != manifest["files"][info.filename]["size"]:
            raise ValueError("ZIP 文件尺寸与清单不一致")
    return manifest


@contextmanager
def _archive(path):
    # zipfile eagerly loads the central directory. Bound that allocation before
    # opening it, including ZIP64 offsets used by the multi-GB official weights.
    path = _plain(path)
    size = path.stat().st_size
    if not 22 <= size <= MAX_BYTES + MAX_TEXT:
        raise ValueError("模型包大小无效")
    with path.open("rb") as stream:
        stream.seek(max(0, size - 65557))
        tail = stream.read()
        offset = tail.rfind(b"PK\x05\x06")
        if offset < 0 or len(tail) - offset < 22:
            raise ValueError("ZIP 目录尾无效")
        end = struct.unpack("<4s4H2LH", tail[offset:offset + 22])
        if (end[1] or end[2] or end[3] != end[4] or end[4] > MAX_FILES + 1
                or end[5] > 4 * 1024**2 or offset + 22 + end[7] != len(tail)):
            raise ValueError("ZIP 目录超限或分卷不受支持")
        absolute = size - len(tail) + offset
        if absolute >= 20:
            stream.seek(absolute - 20)
            locator = stream.read(20)
            if locator[:4] == b"PK\x06\x07":
                _, disk, position, disks = struct.unpack("<4sLQL", locator)
                if disk or disks != 1 or position != absolute - 76:
                    raise ValueError("ZIP64 定位无效")
                stream.seek(position)
                data = stream.read(56)
                if len(data) != 56:
                    raise ValueError("ZIP64 目录截断")
                record = struct.unpack("<4sQ2H2L4Q", data)
                if (record[0] != b"PK\x06\x06" or record[1] != 44 or record[4] or record[5]
                        or record[6] != record[7] or record[7] > MAX_FILES + 1 or record[8] > 4 * 1024**2):
                    raise ValueError("ZIP64 目录超限")
    with zipfile.ZipFile(path) as archive:
        yield archive


def inspect_package(path):
    """Bounded structural inspection, explicitly not a content-verified import."""
    path = _plain(path)
    if path.stat().st_size > MAX_BYTES + MAX_TEXT:
        raise ValueError("模型包过大")
    with _archive(path) as archive:
        manifest = _zip_manifest(archive)
    return {"package_id": manifest["package_id"], "kind": manifest["kind"],
            "bytes": sum(item["size"] for item in manifest["files"].values()),
            "files": len(manifest["files"]), "content_verified": False}


def _no_private(value):
    if isinstance(value, dict):
        if any(str(key).lower() in {"api_key", "access_key", "ui_token", "credentials", "teacher_key", "optimizer", "raw_teacher", "raw_response"} for key in value):
            raise ValueError("采纳材料包含不允许导出的秘密或原始响应字段")
        for item in value.values():
            _no_private(item)
    elif isinstance(value, list):
        for item in value:
            _no_private(item)


def _sample(row):
    from .retention import provenance, normalized_context, verified_source
    if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
            or row.get("purpose") not in {"training", "validation"}
            or row.get("source") not in {"human", "teacher", "synthetic", "program"}
            or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()):
        raise ValueError("采纳样本格式无效")
    value = provenance(row)
    if (not row.get("approved") or not verified_source(row) or not isinstance(row.get("answer"), str)
            or not (row.get("training_allowed") or value.get("retention_generation_only") is True)):
        raise ValueError("仅允许导出已核验、授权的材料或已采纳生成保护")
    if value.get("base_identity", TRUSTED_IDENTITY) != TRUSTED_IDENTITY:
        raise ValueError("材料不属于可信基座")
    normalized_context(row)
    _no_private(row)
    _no_private(value)
    return copy.deepcopy(row)


def _record_rows(records):
    if isinstance(records, dict) and "id" in records and "answer" in records:
        yield _sample(records)
    elif isinstance(records, dict):
        if "directed_teaching" in records:
            from .evolution_progress import validate_product_records
            validate_product_records(records)
            values = (records[key] for key in ("groups", "protection", "retention_regressions"))
        else:
            values = records.values()
        for value in values:
            yield from _record_rows(value)
    elif isinstance(records, list):
        for value in records:
            yield from _record_rows(value)
    else:
        raise ValueError("采纳记录中的材料结构无效")


def _validate_product_partial(run, store=None):
    """A product partial receipt must also describe its actual saved candidate."""
    from .evolution_progress import PARTIAL_POLICY, validate_partial_progress, validate_product_records
    report, records, commit = run["commit"]["report"], run["records"], run["commit"]
    if not (run["kind"] == "evolution" and ("experience" in records or "directed_teaching" in records
            or report.get("steps") == 450 and report.get("partial_policy") == PARTIAL_POLICY)):
        return False
    if report.get("steps") != 450 or report.get("partial_policy") != PARTIAL_POLICY or run["scope"] != "product":
        raise ValueError("产品部分进化必须使用固定450政策")
    if run.get("arguments", {}).get("operation") == "validated-core-import":
        from .automatic import receipt_partial_policy
        if receipt_partial_policy(run) != PARTIAL_POLICY:
            raise ValueError("已验证核心导入政策不一致")
    elif run.get("arguments") != {"steps": 450, "learning_rate": 1e-5, "distillation_weight": 5,
            "preserve_replay_subspace": True, "partial_policy": PARTIAL_POLICY}:
        raise ValueError("产品部分进化冻结参数缺失或不一致")
    validate_product_records(records)
    version, metadata = report.get("version"), report.get("checkpoint", {})
    if (not isinstance(version, str) or not re.fullmatch(r"core-[a-f0-9]{12}", version)
            or not isinstance(metadata, dict) or metadata.get("version") != version
            or metadata.get("identity") != TRUSTED_IDENTITY
            or not isinstance(metadata.get("sha256"), str) or not HEX.fullmatch(metadata["sha256"])):
        raise ValueError("产品部分进化缺少核心文件绑定")
    validate_partial_progress(report, records, TRUSTED_IDENTITY, version, metadata["sha256"])
    names = (version + ".safetensors", version + ".json", version + ".retention.json", "evolution-" + run["id"] + ".json")
    if (not set(names) <= set(commit["artifacts"]) or metadata.get("report") != names[-1]
            or commit["artifacts"][names[0]] != metadata["sha256"]
            or commit["artifacts"][names[2]] != metadata.get("retention_sha256")
            or commit["pointers"][0] != {"name": "active-core.json", "before": commit["pointers"][0]["before"],
                "after": {"version": version, "identity": TRUSTED_IDENTITY, "sha256": metadata["sha256"]}}):
        raise ValueError("产品部分进化采纳文件或指针与报告不一致")
    if store is not None:
        from .retention import read_retention_file
        retention = read_retention_file(store / names[2], TRUSTED_IDENTITY, version)
        if (_read(store / names[1]) != metadata or _read(store / names[3]) != report
                or retention["groups"] != report["accepted_retention_groups"]
                or retention["regression_validation"] != report["accepted_retention_regressions"]):
            raise ValueError("产品部分进化实际报告或保护文件与采纳回执不一致")
    return True


def _validate_journal(payload):
    if not isinstance(payload, dict) or set(payload) - {"factory_runs"} != {"identity", "runs", "samples", "material_archives"} or payload["identity"] != TRUSTED_IDENTITY:
        raise ValueError("采纳账本格式无效")
    if not isinstance(payload["runs"], list) or not isinstance(payload["samples"], list) or len(_encoded(payload)) > MAX_TEXT:
        raise ValueError("采纳账本大小无效")
    seen, expected_archives = set(), {}
    for run in payload["runs"]:
        from .training_checkpoint import is_native_mlx_run
        if is_native_mlx_run(run):
            from .mlx_model_packages import validate_run
            validate_run(run, seen)
            seen.add(run["id"])
            continue
        if (not isinstance(run, dict) or set(run) - {"arguments"} != {"id", "kind", "state", "scope", "product_accepted", "binding", "records", "started", "commit"}
                or not re.fullmatch(r"[a-f0-9]{12}", run.get("id", "")) or run["id"] in seen
                or run["kind"] not in {"router", "learning", "evolution"} or run["state"] != "committed"
                or run["scope"] not in {"product", "research"}
                or run["scope"] == "research" and run["product_accepted"] is not True):
            raise ValueError("只允许已产品采纳的不可重复任务")
        seen.add(run["id"])
        if type(run["started"]) not in {int, float} or not 0 < run["started"] <= time.time() + 300:
            raise ValueError("采纳时间无效")
        expected_binding = {"identity": TRUSTED_IDENTITY, "records_sha256": hashlib.sha256(_encoded(run["records"])).hexdigest()}
        if run.get("arguments", {}).get("operation") == "validated-core-import":
            expected_binding["arguments"] = run["arguments"]
        if run["binding"] != expected_binding:
            raise ValueError("采纳材料 SHA 或基座绑定不符")
        commit = run["commit"]
        if not isinstance(commit, dict) or not isinstance(commit.get("report"), dict):
            raise ValueError("采纳回执格式无效")
        expected = {"router": {"splits"}, "learning": {"train", "valid", "protection"},
                    "evolution": {"groups", "protection", "retention_regressions"}}[run["kind"]]
        if run["kind"] == "evolution" and isinstance(run["records"], dict) and "experience" in run["records"]:
            expected = expected | {"experience"}
        if run["kind"] == "evolution" and isinstance(run["records"], dict) and "directed_teaching" in run["records"]:
            expected = expected | {"directed_teaching"}
        if not isinstance(run["records"], dict) or set(run["records"]) != expected:
            raise ValueError("采纳材料组不完整")
        list(_record_rows(run["records"]))
        commit = run["commit"]
        keys = set(commit) - {"material_publication"}
        if (keys != {"report", "pointers", "artifacts", "event_kind"}
                or commit["event_kind"] != run["kind"]
                or commit["report"].get("calibrated" if run["kind"] == "router" else "promoted") is not True
                or not isinstance(commit["pointers"], list) or len(commit["pointers"]) != 1
                or not isinstance(commit["artifacts"], dict) or not commit["artifacts"]):
            raise ValueError("采纳回执缺少成功证明或文件绑定")
        for pointer in commit["pointers"]:
            if set(pointer) != {"name", "before", "after"} or pointer["name"] not in _POINTERS:
                raise ValueError("采纳指针无效")
        for name, digest in commit["artifacts"].items():
            if not _MODULE.fullmatch(name) or name in _POINTERS or not isinstance(digest, str) or not HEX.fullmatch(digest):
                raise ValueError("采纳文件路径或 SHA 无效")
        _validate_product_partial(run)
        if "material_publication" in commit:
            from .training_checkpoint import _validate_publication
            _validate_publication(run)
            plan = commit["material_publication"]
            source = {row["id"]: _sample(row) for row in plan["source_rows"]}
            for identifier in plan["superseded_ids"]:
                expected_archives[(plan["version"], identifier)] = hashlib.sha256(_encoded(source[identifier])).hexdigest()
        _no_private(run)
    factory = payload.get("factory_runs", [])
    if (not isinstance(factory, list) or any(not isinstance(identifier, str) for identifier in factory)
            or len(factory) != len(set(factory)) or not set(factory) <= seen):
        raise ValueError("出厂基线必须引用不重复的已采纳任务")
    ids = set()
    for sample in payload["samples"]:
        _sample(sample)
        if sample["id"] in ids:
            raise ValueError("材料 ID 重复")
        ids.add(sample["id"])
    if not isinstance(payload["material_archives"], list):
        raise ValueError("材料归档摘要无效")
    actual_archives = {}
    for archive in payload["material_archives"]:
        if (not isinstance(archive, dict) or set(archive) != {"version", "sample_id", "original_sha256", "reason"}
                or not isinstance(archive["reason"], str) or not 1 <= len(archive["reason"]) <= 1000):
            raise ValueError("材料归档记录无效")
        key = (archive["version"], archive["sample_id"])
        if key in actual_archives:
            raise ValueError("材料归档重复")
        actual_archives[key] = archive["original_sha256"]
    if actual_archives != expected_archives:
        raise ValueError("材料归档与已采纳发布计划的来源 SHA 不符")


def _journal(runtime):
    from .training_checkpoint import is_product_run, is_native_mlx_run
    database = _plain(runtime.memory.path)
    payload = {"identity": TRUSTED_IDENTITY, "runs": [], "samples": [], "material_archives": []}
    if not database.exists():
        return payload
    with runtime.memory.lock:
        from .bun_bridge import readonly_connection
        db = readonly_connection(database)
        db.row_factory = sqlite3.Row
        try:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "training_runs" in tables:
                for row in db.execute("SELECT * FROM training_runs WHERE identity=? ORDER BY id", (TRUSTED_IDENTITY,)):
                    body = _json(row["body"])
                    if row["state"] in {"queued", "running", "paused", "committing"}:
                        raise ValueError("仍有未完成训练任务，暂不能导出模型")
                    if row["state"] != "committed" or not is_product_run(body):
                        continue
                    if is_native_mlx_run(body) and runtime.settings.snapshot().get("native_inference_precision") != "mlx-q4":
                        raise ValueError("MLX采纳任务必须随完整MLX模型包导出")
                    event = db.execute("SELECT kind,body FROM events WHERE id=?", ("training:" + row["id"],)).fetchone()
                    event_kind = body["commit"]["event_kind"]
                    factory = body.get("factory_baseline") is True
                    if ("factory_baseline" in body and not factory or not event
                            or event["kind"] != ("factory:" if factory else "") + event_kind
                            or _json(event["body"]) != body["commit"]["report"]):
                        raise ValueError("已采纳任务缺少匹配的持久事件")
                    if factory:
                        payload.setdefault("factory_runs", []).append(row["id"])
                    payload["runs"].append({"id": row["id"], "kind": row["kind"], "state": "committed",
                        "scope": body.get("scope", "product"), "product_accepted": body.get("product_accepted", False),
                        "binding": {key: body["binding"][key] for key in ("identity", "records_sha256")},
                        "records": body["records"], "started": body["started"], "commit": body["commit"]})
                    if "arguments" in body:
                        payload["runs"][-1]["arguments"] = body["arguments"]
                        if body["arguments"].get("operation") == "validated-core-import":
                            payload["runs"][-1]["binding"]["arguments"] = body["binding"].get("arguments")
                    if is_native_mlx_run(body):
                        from .mlx_model_packages import portable_run
                        payload["runs"][-1] = portable_run(payload["runs"][-1], body, runtime)
            if "samples" in tables:
                from .retention import verified_source, provenance
                # An external reference can be verified yet never adopted by
                # weights. Only matching accepted material travels with a model.
                accepted = {}
                fields = ("id", "prompt", "answer", "source", "purpose")
                for run in payload["runs"]:
                    for item in _record_rows(run["records"]):
                        accepted.setdefault(item["id"], []).append({**{key: item.get(key) for key in fields}, "provenance": provenance(item)})
                for row in db.execute("SELECT * FROM samples ORDER BY id"):
                    item = dict(row)
                    if (item["approved"] and item["training_allowed"] and verified_source(item)
                            and provenance(item).get("base_identity", TRUSTED_IDENTITY) == TRUSTED_IDENTITY
                            and {**{key: item.get(key) for key in fields}, "provenance": provenance(item)} in accepted.get(item["id"], [])):
                        payload["samples"].append(_sample(item))
            if "core_material_archives" in tables:
                versions = {run["commit"]["material_publication"]["version"] for run in payload["runs"] if "material_publication" in run["commit"]}
                payload["material_archives"] = [dict(row) for row in db.execute("SELECT version,sample_id,original_sha256,reason FROM core_material_archives ORDER BY version,sample_id") if row["version"] in versions]
        finally:
            db.close()
    _validate_journal(payload)
    return payload


def _active_names(store):
    from .backups import _file
    names = {name for name in _POINTERS if _file(store, name).exists()}
    for name in tuple(names):
        pointer = _read(_file(store, name))
        version = pointer.get("version") or pointer.get("id")
        if not isinstance(version, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", version):
            raise ValueError("活动版本名称无效")
        prefix = "layer-23/" if name == "active.json" else ""
        names.update({prefix + version + ".json", prefix + version + ".safetensors"})
        if name == "active-core.json" and _file(store, version + ".retention.json").exists():
            names.add(version + ".retention.json")
    if "active-router.json" not in names:
        names.update(name for name in ("router.json", "router.safetensors") if _file(store, name).exists())
    return names


def _validate_modules(root, payload, files):
    import tempfile
    from .backups import _validate, _tensor_shapes
    store = root / "modules/luban0-v1"
    # Header-only graph checks precede validators that load a replacement tensor.
    for full_name in files:
        if not full_name.startswith("modules/") or not full_name.endswith(".safetensors"):
            continue
        name = full_name.removeprefix("modules/luban0-v1/")
        shapes = _tensor_shapes(store / name)
        if name.startswith("core-") and shapes != {"model.language_model.layers.23.mlp.base.down_proj.weight": [2048, 6144]}:
            raise ValueError("核心不是当前 2B 图的末层权重")
        if name.startswith("layer-"):
            width = shapes.get("gate.weight", [0])[0]
            if (not 32 <= width <= 512 or shapes != {"gate.weight": [width, 2048], "up.weight": [width, 2048], "down.weight": [2048, width]}):
                raise ValueError("分支尺寸不属于 2B 基座")
        if name.startswith("router") and shapes != {"projection.weight": [3, 2048], "projection.bias": [3]}:
            raise ValueError("快脑尺寸不属于 2B 基座")
        import torch
        from safetensors.torch import load_file
        if any(not torch.isfinite(value).all() for value in load_file(str(store / name)).values()):
            raise ValueError("已学权重包含非有限值")
    active_names = _active_names(store)
    active_files = {name: files["modules/luban0-v1/" + name] for name in active_names}
    _validate(store, TRUSTED_IDENTITY, active_files)
    bound = set(active_names)
    for pointer in _POINTERS & active_names:
        value = _read(store / pointer)
        if not any(any(plan["name"] == pointer and plan["after"] == value for plan in run["commit"]["pointers"]) for run in payload["runs"]):
            raise ValueError("活动权重没有正式产品采纳回执，拒绝导出/导入研究候选")
    if {"router.json", "router.safetensors"} & active_names:
        raise ValueError("旧快脑没有不可变采纳回执，须先通过正式快脑验收")
    for run in payload["runs"]:
        commit = run["commit"]
        for name, digest in commit["artifacts"].items():
            entry = files.get("modules/luban0-v1/" + name)
            if not entry or entry["sha256"] != digest:
                raise ValueError("历史采纳文件缺失或绑定 SHA 不符")
            bound.add(name)
        _validate_product_partial(run, store)
        from .training_checkpoint import is_native_mlx_run
        if is_native_mlx_run(run):
            from .mlx_model_packages import validate_candidate
            validate_candidate(store, run, files)
            continue
        # Validate each historical candidate using its own pointer, not today's
        # selected pointer; no backbone or optimizer is loaded by this check.
        with tempfile.TemporaryDirectory(prefix=".verify-", dir=root) as folder:
            temporary = Path(folder)
            subset = {}
            for name in commit["artifacts"]:
                if name.startswith(("learning-", "evolution-", "coverage-acceptance-")):
                    continue
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(store / name, target)
                subset[name] = files["modules/luban0-v1/" + name]
            for pointer in commit["pointers"]:
                private_json(temporary / pointer["name"], pointer["after"])
                subset[pointer["name"]] = _record(temporary / pointer["name"])
            _validate(temporary, TRUSTED_IDENTITY, subset)
    if set(name.removeprefix("modules/luban0-v1/") for name in files if name.startswith("modules/")) != bound:
        raise ValueError("模型包包含未采纳、未绑定模块文件")


def _validate_tree(root, manifest):
    from .backups import _tensor_shapes
    _check_manifest(manifest)
    for name, record in manifest["files"].items():
        if _record(root / name) != record:
            raise ValueError("模型包文件内容与 SHA 不符")
        if name.endswith(".safetensors") and name != "models/mlx/model.safetensors":
            # The pinned MLX file also contains packed U32 tensors. Its exact
            # trusted content hash is checked above; learned tensors stay strict.
            _tensor_shapes(root / name)
    if _read(root / "models/base/config.json").get("model_type") != "qwen3_5":
        raise ValueError("仅支持官方 Qwen3.5 2B 基座")
    payload = _read(root / "accepted.json")
    _validate_journal(payload)
    _validate_modules(root, payload, manifest["files"])
    if manifest.get("engine") == "mlx-q4":
        from .mlx_model_packages import validate_tree
        validate_tree(root, manifest, payload)
    if manifest["kind"] == "foundation" and (payload["runs"] or payload["samples"] or any(name.startswith("modules/") for name in manifest["files"])):
        raise ValueError("基础重修包不能包含已学权重或材料")
    return payload


def _install_journal(root, payload):
    from .memory import Memory
    from .training_checkpoint import _schema
    memory = Memory(root)
    factory = set(payload.get("factory_runs", []))
    with memory.connect() as db:
        _schema(db)
        for sample in payload["samples"]:
            row = dict(sample)
            provenance = row.get("provenance", {})
            if not isinstance(provenance, str):
                provenance = _encoded(provenance).decode()
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", (row["id"], row["prompt"], row["answer"], row["source"], row["purpose"], row.get("created", 0), row["approved"], row["training_allowed"], provenance))
        for run in payload["runs"]:
            body = {key: value for key, value in run.items() if key != "state"}
            from .training_checkpoint import is_native_mlx_run
            if is_native_mlx_run(run):
                from .mlx_model_packages import installed_run
                body = installed_run(body, root)
            if run["id"] in factory:
                body["factory_baseline"] = True
            db.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?)", (run["id"], TRUSTED_IDENTITY, run["kind"], "committed", _encoded(body).decode(), run["started"]))
            event_kind = ("factory:" if run["id"] in factory else "") + run["commit"]["event_kind"]
            db.execute("INSERT INTO events VALUES (?,?,?,?)", ("training:" + run["id"], event_kind, _encoded(run["commit"]["report"]).decode(), run["started"]))
        db.execute("CREATE TABLE core_material_archives (version TEXT NOT NULL,sample_id TEXT NOT NULL,original_sha256 TEXT NOT NULL,reason TEXT NOT NULL,PRIMARY KEY(version,sample_id))")
        for row in payload["material_archives"]:
            db.execute("INSERT INTO core_material_archives VALUES (?,?,?,?)", tuple(row[key] for key in ("version", "sample_id", "original_sha256", "reason")))
    return {kind: sum(run["kind"] == kind and run["id"] not in factory for run in payload["runs"]) for kind in ("router", "learning", "evolution")}


def _new_root(parent):
    parent = _plain(parent, directory=True)
    require_storage(parent)
    root = parent / ("package-" + uuid.uuid4().hex)
    root.mkdir(mode=0o700)
    private_json(root / "package-owner.json", {"schema": SCHEMA, "id": root.name, "state": "staging"})
    return root


def export_package(runtime, target, *, progress=None):
    """Call after background work is idle; holds the shared runtime lock."""
    import tempfile
    target = _plain(target)
    if target.exists() or target.suffix.lower() != ".zip":
        raise ValueError("导出目标必须是尚不存在的 ZIP 文件")
    require_storage(target.parent)
    with runtime.lock:
        if (getattr(runtime, "training_session", None) is not None or runtime.state not in {"ready", "stopped"}
                or getattr(runtime, "identity", None) != TRUSTED_IDENTITY):
            raise ValueError("训练中或基座不兼容，暂不能导出")
        config = runtime.settings.snapshot()
        if (getattr(runtime, "native_inference_precision", None) == "mlx-q4"
                or config.get("native_inference_precision") == "mlx-q4"):
            from .mlx_model_packages import export_package as export_mlx
            return export_mlx(runtime, target, progress=progress)
        payload = _journal(runtime)
        sources = {}
        for key, field in (("base", "model_path"), ("asr", "speech_model_path")):
            # Only explicit local model settings may be historical symlink
            # aliases. Resolve their root once; contents still reject links.
            root = _plain(Path(config[field]).resolve(strict=True), directory=True)
            for name, record in TRUSTED[key].items():
                source = _plain(root / name)
                if _record(source) != record:
                    raise ValueError("当前基座或 ASR 文件与固定可信版本不符")
                sources["models/" + key + "/" + name] = source
        names = _active_names(Path(runtime.store))
        names.update(name for run in payload["runs"] for name in run["commit"]["artifacts"])
        for name in names:
            _name("modules/luban0-v1/" + name)
            sources["modules/luban0-v1/" + name] = _plain(Path(runtime.store) / name)
        with tempfile.TemporaryDirectory(prefix=".package-export-", dir=target.parent) as directory:
            scratch = Path(directory)
            private_json(scratch / "accepted.json", payload)
            sources["accepted.json"] = scratch / "accepted.json"
            files = {name: _record(path) for name, path in sources.items()}
            manifest = {"schema": SCHEMA, "kind": "learned", "identity": TRUSTED_IDENTITY, "files": files}
            manifest["package_id"] = hashlib.sha256(_encoded(manifest)).hexdigest()
            _check_manifest(manifest)
            # Only small learned artifacts are copied for existing validators.
            for name, path in sources.items():
                if name.startswith("modules/"):
                    destination = scratch / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            _validate_modules(scratch, payload, files)
            temporary = scratch / "result.zip"
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                archive.writestr("manifest.json", _encoded(manifest), compress_type=zipfile.ZIP_DEFLATED)
                done = 0
                for name, path in sources.items():
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    info.compress_type = zipfile.ZIP_STORED if name.endswith(".safetensors") else zipfile.ZIP_DEFLATED
                    digest, size = hashlib.sha256(), 0
                    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                        for block in iter(lambda: source.read(CHUNK), b""):
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                            done += len(block)
                            if progress:
                                progress({"phase": "export", "bytes": done, "total": sum(item["size"] for item in files.values())})
                    if {"size": size, "sha256": digest.hexdigest()} != files[name]:
                        raise ValueError("导出期间源文件发生变化")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            # Reserve the never-existing name, then atomically publish the
            # complete ZIP on the same volume. A crash can leave only an empty
            # reservation, which fails every import check.
            with target.open("xb"):
                pass
            os.replace(temporary, target)
    return {"path": str(target), "package_id": manifest["package_id"], "bytes": target.stat().st_size,
            "kind": "learned", "teacher_calls": 0}


def stage_import(path, managed_parent, *, progress=None, require_foundation=False):
    """Extract into a new owned directory; never change settings or old files."""
    path = _plain(path)
    inspect_package(path)
    if require_foundation and (FOUNDATION_SHA256 is None or _sha(path) != FOUNDATION_SHA256):
        raise ValueError("基础重修包尚未发布，或下载内容不符合固定可信摘要")
    root = _new_root(managed_parent)
    try:
        with _archive(path) as archive:
            manifest = _zip_manifest(archive)
            total = _check_manifest(manifest)
            if require_foundation and manifest["kind"] != "foundation":
                raise ValueError("重修必须使用官方基础包")
            if shutil.disk_usage(root).free < total + 256 * 1024**2:
                raise OSError("磁盘空间不足；当前模型保持不变")
            done = 0
            for name, record in manifest["files"].items():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with archive.open(name) as source, destination.open("xb") as output:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        size += len(block)
                        if size > record["size"]:
                            raise ValueError("解压内容超过声明尺寸")
                        output.write(block)
                        digest.update(block)
                        done += len(block)
                        if progress:
                            progress({"phase": "import", "bytes": done, "total": total})
                    output.flush()
                    os.fsync(output.fileno())
                if {"size": size, "sha256": digest.hexdigest()} != record:
                    raise ValueError("解压文件内容校验失败")
            private_json(root / "manifest.json", manifest)
        payload = _validate_tree(root, manifest)
        counts = _install_journal(root, payload)
        from .upgrades import read_upgrade_records
        read_upgrade_records(root)
        result = {"package_id": manifest["package_id"], "kind": manifest["kind"], "root": str(root),
                  "model_path": str(root / "models/base"), "speech_model_path": str(root / "models/asr"),
                  "storage_root": str(root), "counts": counts, "content_verified": True,
                  "journal_sha256": _sha(root / "app/memory.sqlite3"), "activation_intent": "import"}
        if manifest.get("engine") == "mlx-q4":
            from .mlx_model_packages import restore_weight_time
            restore_weight_time(root)
            result.update(native_inference_precision="mlx-q4", native_mlx_model_path=str(root / "models/mlx"))
        else:
            result.update(native_inference_precision="bf16", native_mlx_model_path="")
        private_json(root / "package-owner.json", {"schema": SCHEMA, "id": root.name, "state": "validated", "result": result})
        return result
    except BaseException as error:
        private_json(root / "package-owner.json", {"schema": SCHEMA, "id": root.name, "state": "failed", "error": str(error)})
        # Partial data stays in this uniquely owned directory, never in live paths.
        raise


def stage_reset(phrase, managed_parent, *, progress=None):
    if phrase != RESET_PHRASE:
        raise ValueError("请输入完整确认语：" + RESET_PHRASE)
    if FOUNDATION_SHA256 is None:
        raise ValueError("基础重修包尚未发布；当前模型和资料未作任何更改")
    import tempfile
    parent = _plain(managed_parent, directory=True)
    require_storage(parent)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            raise ValueError("基础模型下载不允许重定向")
    opener = urllib.request.build_opener(NoRedirect())
    with tempfile.TemporaryDirectory(prefix=".foundation-download-", dir=parent) as directory:
        path = Path(directory) / "base.zip"
        request = urllib.request.Request(FOUNDATION_URL, headers={"Accept-Encoding": "identity"})
        with opener.open(request, timeout=30) as response, path.open("xb") as output:
            if response.geturl() != FOUNDATION_URL or response.status != 200:
                raise ValueError("基础包下载地址或状态不符")
            declared = response.headers.get("Content-Length")
            if declared is not None and (not declared.isdecimal() or not 0 < int(declared) <= MAX_BYTES + MAX_TEXT):
                raise ValueError("基础包下载尺寸无效")
            if declared is not None and shutil.disk_usage(parent).free < int(declared) + 256 * 1024**2:
                raise OSError("基础包下载空间不足；当前模型保持不变")
            done = 0
            started = time.monotonic()
            while block := response.read(CHUNK):
                if time.monotonic() - started > 7200:
                    raise TimeoutError("基础包下载超过两小时上限")
                done += len(block)
                if done > MAX_BYTES + MAX_TEXT:
                    raise ValueError("基础包下载超限")
                output.write(block)
                if progress:
                    progress({"phase": "download", "bytes": done, "total": int(declared) if declared else None})
            if declared is not None and done != int(declared):
                raise ValueError("基础包下载不完整")
            output.flush()
            os.fsync(output.fileno())
        result = stage_import(path, parent, progress=progress, require_foundation=True)
        result["activation_intent"] = "reset"
        owner_path = Path(result["root"]) / "package-owner.json"
        owner = _read(owner_path)
        owner["result"] = result
        private_json(owner_path, owner)
        return result


def _activation_path(settings):
    # Settings.home is an explicit local directory, including macOS /var's
    # platform alias. Resolve that root only; a linked journal is still refused.
    root = _plain(Path(settings.home).resolve(strict=True), directory=True)
    return _plain(root / "package-activation.json")


def prepare_activation(settings, staged):
    """Caller holds runtime/task lock and has released the old model first."""
    journal = _activation_path(settings)
    if journal.exists() and _read(journal).get("state") not in {"committed", "rolled_back"}:
        raise ValueError("存在未恢复的模型包切换，请先恢复")
    root = _plain(staged["root"], directory=True)
    owner = _read(_plain(root / "package-owner.json"))
    if owner.get("state") != "validated" or owner.get("result") != staged or root.name != owner.get("id"):
        raise ValueError("暂存模型包未经完整验证")
    manifest = _read(root / "manifest.json")
    if (not re.fullmatch(r"package-[a-f0-9]{32}", root.name) or staged.get("package_id") != manifest.get("package_id")
            or staged.get("storage_root") != str(root) or staged.get("model_path") != str(root / "models/base")
            or staged.get("speech_model_path") != str(root / "models/asr")
            or staged.get("journal_sha256") != _sha(_plain(root / "app/memory.sqlite3"))):
        raise ValueError("暂存路径或重建账本已被修改")
    _validate_tree(root, manifest)
    keys = _PATH_KEYS + ("native_inference_precision", "native_mlx_model_path")
    if manifest.get("engine") == "mlx-q4":
        from .mlx_model_packages import check_activation
        check_activation(root, staged)
    elif staged.get("native_inference_precision") != "bf16" or staged.get("native_mlx_model_path") != "":
        raise ValueError("旧格式模型包的推理引擎设置已被修改")
    old = {key: settings.snapshot()[key] for key in keys}
    new = {key: staged[key] for key in keys}
    token = uuid.uuid4().hex
    _durable_json(journal, {"schema": SCHEMA, "token": token, "state": "prepared", "old": old, "new": new,
                           "package_id": staged["package_id"], "kind": staged["kind"], "intent": staged["activation_intent"]})
    try:
        settings.update(new)
    except BaseException:
        settings.update(old)
        raise
    return token


def finish_activation(settings, token):
    """Call only after loading/validating the new model; never before runtime.load."""
    path = _activation_path(settings)
    record = _read(path)
    if (record.get("token") != token or record.get("state") not in {"prepared", "committed"}
            or any(settings.snapshot()[key] != record["new"][key] for key in record["new"])):
        raise ValueError("模型包切换状态不一致")
    record["state"] = "committed"
    _durable_json(path, record)
    return {"committed": True, "package_id": record["package_id"], "old_paths_retained": record["old"]}


def recover_activation(settings):
    """Run before starting automatic work or loading a model. Crash => old paths."""
    path = _activation_path(settings)
    if not path.exists():
        return {"recovered": False}
    record = _read(path)
    if record.get("schema") != SCHEMA or record.get("state") not in {"prepared", "committed", "rolled_back"}:
        raise ValueError("模型包切换记录损坏；禁止自动加载")
    if record["state"] != "prepared":
        return {"recovered": False, "state": record["state"]}
    current = {key: settings.snapshot()[key] for key in record["new"]}
    if current not in (record["old"], record["new"]):
        raise ValueError("模型目录已被其他操作更改，禁止自动覆盖")
    settings.update(record["old"])
    record["state"] = "rolled_back"
    _durable_json(path, record)
    return {"recovered": True, "paths": record["old"]}


def _durable_json(path, value):
    private_json(path, value)
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def cleanup_retired_package(settings, token):
    """Optional after successful RESET activation; only an earlier owned package.

    Never rmtree a user-selected storage/model directory. Unknown legacy roots
    remain intact; their safe migration needs an explicit file inventory.
    """
    journal = _read(_activation_path(settings))
    if (journal.get("token") != token or journal.get("state") != "committed" or journal.get("kind") != "foundation"
            or journal.get("intent") != "reset"
            or any(settings.snapshot()[key] != journal["new"][key] for key in _PATH_KEYS)):
        raise ValueError("只有已成功切换的官方重修才允许清理旧包")
    root = _plain(journal["old"]["storage_root"], directory=True)
    owner_path = _plain(root / "package-owner.json")
    if not owner_path.exists():
        return {"deleted_files": 0, "retained": str(root), "reason": "旧目录不属于可验证的导入包，保留原文件"}
    owner = _read(owner_path)
    if (not re.fullmatch(r"package-[a-f0-9]{32}", root.name) or owner.get("schema") != SCHEMA
            or owner.get("id") != root.name or owner.get("state") != "validated"
            or journal["old"]["model_path"] != str(root / "models/base")
            or journal["old"]["speech_model_path"] != str(root / "models/asr")
            or Path(journal["new"]["storage_root"]).is_relative_to(root)):
        raise ValueError("旧包路径未满足安全清理边界")
    manifest = _read(_plain(root / "manifest.json"))
    _check_manifest(manifest)
    # Record-by-record unlink only: never traverse experiments, backups, sources,
    # arbitrary added files, or nested folders supplied by the user.
    names = list(manifest["files"]) + ["app/memory.sqlite3", "app/memory.sqlite3-wal", "app/memory.sqlite3-shm"]
    targets = [_plain(root / name) for name in names]
    for name, path in zip(names, targets):
        if name.startswith("models/") and path.exists() and _record(path) != manifest["files"][name]:
            raise ValueError("旧基座文件已经变化，停止清理")
    deleted = 0
    for path in targets:
        if path.exists():
            path.unlink()
            deleted += 1
    _durable_json(owner_path, {**owner, "state": "retired", "retired_by": journal["package_id"]})
    return {"deleted_files": deleted, "retained": str(root), "scope": "仅旧包清单内文件与应用账本；研究/备份/未知文件保留"}
