"""Research-only cached-tail Python/C++ comparison. Default CLI is stdlib-only.

No model loader, generation, teacher, publication or implicit resource consent.
CPU tiny export/run and raw tensor comparison need separate Root execution GO.
The producer remains Python full-Qwen; this is not full native-model training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ABI = "LUBAN_NATIVE_TRAINING_V1"
OUTPUT_ABI = "LUBAN_NATIVE_TRAINING_OUTPUT_V1"
PROTOCOL = "cached-tail-native-python-b1-numeric-v1"
ORIGINAL_IDENTITY = "673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df"
ORIGINAL_NATIVE_SHA = "1706e50833fffb25da45ad958af043a688e5ef495d45f33fa4b1ab623c696719"
SHA = re.compile(r"[0-9a-f]{64}")
DTYPE_BYTES = {"f32": 4, "bf16": 2, "i64": 8, "u8": 1}
INITIAL_STATE = ("candidate", "master", "exp_avg", "exp_avg_sq", "adam_step")
GLOBAL = set(INITIAL_STATE) | {"reference", "basis", "norm", "head"}
PROBES = ("ce_logits", "teacher_log_probs", "student_log_probs", "ce_grad", "kl_grad", "total_grad",
          "clipped_grad", "delta", "master", "candidate", "exp_avg", "exp_avg_sq", "adam_step")
FINAL = ("candidate", "master", "exp_avg", "exp_avg_sq", "adam_step", "reference", "basis")
TRACE = ("step", "ce_role", "ce_row", "replay_row", "ce", "kl", "preclip_norm", "loss", "wall_seconds")
# Preregistered engineering gates. Failures never adapt these constants.
TOLERANCES = {
    "cpu": {"native_bitwise": True, "candidate_agreement": 1.0,
            "loss_atol": 1e-6, "loss_rtol": 1e-6, "norm_atol": 1e-6, "norm_rtol": 1e-6,
            "state_atol": 1e-7, "state_rtol": 1e-5, "state_relative_l2": 1e-5,
            "delta_relative_l2": 1e-5},
    "mps": {"native_bitwise": False, "candidate_agreement": .9999,
            "logits_atol": 2**-10, "gradient_atol": 2**-16, "native_rtol": 2**-7,
            "native_relative_l2": 2**-7, "loss_atol": 1e-5, "loss_rtol": 1e-5,
            "norm_atol": 1e-5, "norm_rtol": 1e-4,
            "state_atol": 1e-6, "state_rtol": 1e-4, "state_relative_l2": 1e-5,
            "delta_relative_l2": 1e-3},
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def safe(path):
    path = Path(os.path.abspath(path))
    need(path.resolve() == path, "Symlink/noncanonical path rejected")
    return path


def fresh(path, forbidden=()):
    path = safe(path)
    need(path.parent.is_dir() and not path.exists(), "Fresh output with existing canonical parent required")
    for other in forbidden:
        other = safe(other)
        need(path != other and not path.is_relative_to(other) and not other.is_relative_to(path), "Input/output overlap")
    path.mkdir(mode=0o700)
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def integer(value):
    need(isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "Unsigned decimal integer required")
    return int(value)


def checkpoints(steps):
    need(type(steps) is int and steps in (2, 50, 450), "Only fixed2/50/450 steps supported")
    return {2: [1, 2], 50: [1, 2, 50], 450: [1, 2, 50, 450]}[steps]


def tensor_spec(parts):
    need(len(parts) == 7 and parts[0] == "tensor", "Malformed tensor descriptor")
    _, name, dtype, dimensions, file, size, hashed = parts
    need(re.fullmatch(r"[a-z][a-z0-9_]*", name) and re.fullmatch(r"[a-z][a-z0-9_]*\.raw", file), "Unsafe tensor name/path")
    need(dtype in DTYPE_BYTES and SHA.fullmatch(hashed), "Invalid dtype/SHA")
    shape = [] if dimensions == "-" else [integer(v) for v in dimensions.split(",")]
    need(len(shape) <= 3 and all(0 < v <= 2**31 for v in shape), "Invalid tensor shape")
    size = integer(size)
    need(math.prod(shape) * DTYPE_BYTES[dtype] == size and size <= 2**40, "Exact tensor bytes differ")
    return {"name": name, "dtype": dtype, "shape": shape, "file": file, "bytes": size, "sha256": hashed}


def descriptors(path, header, *, approved_sha=None):
    path = safe(path)
    if approved_sha is not None:
        need(SHA.fullmatch(approved_sha) and sha(path) == approved_sha, "Approved manifest SHA differs")
    lines = path.read_text(encoding="utf-8").splitlines()
    need(lines and lines[0] == header, "Wrong TSV ABI")
    tensors, files, rest = {}, set(), []
    for line in lines[1:]:
        parts = line.split("\t")
        if parts[0] == "tensor":
            item = tensor_spec(parts)
            need(item["name"] not in tensors and item["file"] not in files, "Duplicate tensor/file")
            tensors[item["name"]] = item
            files.add(item["file"])
        else:
            rest.append(parts)
    return tensors, rest


def fixture(path, approved_sha, *, verify_bytes=True):
    path = safe(path)
    tensors, rest = descriptors(path / "manifest.tsv", ABI, approved_sha=approved_sha)
    config, rows, steps = {}, {}, []
    for p in rest:
        if p[0] == "config":
            need(len(p) == 3 and p[1] not in config, "Duplicate/malformed config")
            config[p[1]] = p[2]
        elif p[0] == "row":
            need(len(p) == 6 and re.fullmatch(r"r[0-9]+", p[1]) and p[1] not in rows,
                 "Duplicate/malformed sample key")
            need(p[2] in ("new", "replay") and SHA.fullmatch(p[3]), "Invalid role/sample SHA")
            rows[p[1]] = {"role": p[2], "sample_sha256": p[3], "length": integer(p[4]), "gold": integer(p[5])}
            need(0 < rows[p[1]]["gold"] < rows[p[1]]["length"], "Invalid gold count")
        elif p[0] == "step":
            need(len(p) == 5, "Malformed schedule")
            steps.append({"ordinal": integer(p[1]), "role": p[2], "ce": p[3], "replay": p[4]})
        else:
            raise ValueError("Unknown manifest row")
    need(set(config) == {"identity", "device", "steps", "norm_eps", "train_eos", "checkpoint_steps"}, "Config contract differs")
    need(SHA.fullmatch(config["identity"]) and config["device"] in ("cpu", "mps") and config["norm_eps"] == "1e-6", "Invalid identity/device/norm")
    integer(config["train_eos"])
    count = integer(config["steps"])
    need(config["checkpoint_steps"] == ",".join(map(str, checkpoints(count))) and len(steps) == count, "Fixed schedule/checkpoints differ")
    allowed = GLOBAL | ({"down_bias"} if "down_bias" in tensors else set()) | ({"head_bias"} if "head_bias" in tensors else set())
    for key, row in rows.items():
        allowed |= {key + "_" + suffix for suffix in ("x", "residual", "ids", "labels")}
    need(set(tensors) == allowed, "Incomplete/extra tensor set")
    for i, step in enumerate(steps):
        role = "new" if i % 2 == 0 else "replay"
        need(step["ordinal"] == i + 1 and step["role"] == role and step["ce"] in rows and step["replay"] in rows,
             "Schedule ordinal/role/row differs")
        need(rows[step["ce"]]["role"] == role and rows[step["replay"]]["role"] == "replay", "CE/KL role mismatch")
    validate_shapes(tensors, rows)
    if verify_bytes:
        verify_files(path, tensors)
    return {"root": str(path), "config": config, "rows": rows, "steps": steps, "tensors": tensors, "manifest_sha256": approved_sha}


def validate_shapes(t, rows):
    shape = t["master"]["shape"]
    need(len(shape) == 2 and t["master"]["dtype"] == "f32", "FP32 matrix master required")
    hidden, intermediate = shape
    native = t["candidate"]["dtype"]
    need(native in ("f32", "bf16"), "Native dtype must FP32/BF16")
    for name in ("reference", "exp_avg", "exp_avg_sq"):
        need(t[name]["shape"] == shape and t[name]["dtype"] == "f32", "FP32 state shape/dtype differs")
    need(t["candidate"]["shape"] == shape and t["adam_step"]["shape"] == [] and t["adam_step"]["dtype"] == "f32", "Candidate/scalar Adam shape differs")
    basis = t["basis"]
    need(basis["dtype"] == "f32" and len(basis["shape"]) == 2 and basis["shape"][0] == intermediate
         and 0 < basis["shape"][1] < intermediate, "Basis needs nontrivial trainable complement")
    need(t["norm"]["shape"] == [hidden] and t["norm"]["dtype"] in ("f32", native), "Actual norm dtype/shape required")
    need(len(t["head"]["shape"]) == 2 and t["head"]["shape"][1] == hidden and t["head"]["dtype"] == native, "Native head shape/dtype differs")
    for name, n in (("down_bias", hidden), ("head_bias", t["head"]["shape"][0])):
        if name in t:
            need(t[name]["shape"] == [n] and t[name]["dtype"] == native, "Optional native bias differs")
    for key, row in rows.items():
        for suffix, size, dtype in (("x", [1, row["length"], intermediate], native),
                                    ("residual", [1, row["length"], hidden], native),
                                    ("ids", [1, row["length"]], "i64"), ("labels", [1, row["length"]], "i64")):
            need(t[key + "_" + suffix]["shape"] == size and t[key + "_" + suffix]["dtype"] == dtype, "Full sequence cache shape/dtype differs")


def verify_files(root, tensors):
    for value in tensors.values():
        path = safe(Path(root) / value["file"])
        need(path.is_file() and path.stat().st_size == value["bytes"] and sha(path) == value["sha256"], "Tensor file bytes/SHA differs")


def manifest_text(config, tensors, rows, steps):
    out = [ABI] + ["\t".join(("config", key, str(value))) for key, value in config.items()]
    for value in tensors.values():
        dimensions = ",".join(map(str, value["shape"])) or "-"
        out.append("\t".join(("tensor", value["name"], value["dtype"], dimensions, value["file"], str(value["bytes"]), value["sha256"])))
    out += ["\t".join(("row", key, row["role"], row["sample_sha256"], str(row["length"]), str(row["gold"]))) for key, row in rows.items()]
    out += ["\t".join(("step", str(s["ordinal"]), s["role"], s["ce"], s["replay"])) for s in steps]
    return "\n".join(out) + "\n"


def write_text(path, text):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def export_tensor(root, name, value):
    import torch
    need(sys.byteorder == "little", "LE ABI required")
    value = value.detach().cpu().contiguous()
    types = {torch.float32: "f32", torch.bfloat16: "bf16", torch.int64: "i64", torch.uint8: "u8"}
    need(value.dtype in types and torch.isfinite(value).all(), "Unsupported/nonfinite tensor")
    path = Path(root) / (name + ".raw")
    # numpy uint8 view writes storage directly; no full-head tobytes+clone copy.
    with path.open("xb") as stream:
        value.reshape(-1).view(torch.uint8).numpy().tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"name": name, "dtype": types[value.dtype], "shape": list(value.shape), "file": path.name,
            "bytes": value.numel() * value.element_size(), "sha256": sha(path)}


def export_runtime_fixture(runtime, new, replay, reference, basis, caches, destination, *, steps, metadata, schedule=None):
    """Caller owns model lock/qualification. No optimizer update or RNG consumption.

    Only an already frozen actual Python graph can provide its captured X/R.
    The independent actual producer binds official files/prod/qualification;
    this exporter does not load a model or substitute assertions for that proof.
    """
    import torch
    import verify_cached_training as cached
    from verify_cached_batch_training import _eos, _labels
    from luban.retention import provenance, eligible_learning_sample
    checkpoints(steps)
    need(new and replay and runtime.branch.expert is None, "Branch-OFF qualified TRAIN groups required")
    parameter = runtime.branch.base.down_proj.weight
    binding = cached._binding(runtime, parameter)
    need(float(runtime.decoder.norm.eps) == 1e-6, "Actual Qwen norm epsilon differs")
    need(metadata.get("scope") in ("cpu_tiny_fixture", "actual_python_frozen_prefix"), "Explicit producer scope required")
    actual = metadata["scope"] == "actual_python_frozen_prefix"
    if actual:
        need(runtime.identity == ORIGINAL_IDENTITY and getattr(runtime, "active_core", None) is None
             and cached.tensor_sha(parameter) == ORIGINAL_NATIVE_SHA and torch.equal(reference, parameter.detach().float()),
             "Actual producer requires exact originalNone1706 and original FP32 reference")
        need(metadata.get("identity") == runtime.identity and "core_version" in metadata and metadata["core_version"] is None
             and metadata.get("original_native_tensor_sha256") == ORIGINAL_NATIVE_SHA, "Actual fixture identity/core binding differs")
        official = metadata.get("official_files", {})
        need(isinstance(official, dict) and len(official) == 13 and all(isinstance(v, dict) and SHA.fullmatch(v.get("sha256", ""))
             and type(v.get("bytes")) is int and v["bytes"] > 0 for v in official.values()), "Actual13 official byte bindings required")
        need(isinstance(metadata.get("sources"), dict) and all(metadata["sources"].get(path) == hashed for path, hashed in binding["sources"].items()),
             "Actual frozen-prefix source bindings incomplete")
        need(schedule is not None and metadata.get("fixed_schedule_sha256") == digest(schedule), "Explicit actual schedule/SHA required")
    root = fresh(destination)
    original = parameter.detach().clone()
    config = {"identity": runtime.identity, "device": parameter.device.type, "steps": str(steps),
              "norm_eps": "1e-6", "train_eos": str(next(iter(_eos(runtime)))),
              "checkpoint_steps": ",".join(map(str, checkpoints(steps)))}
    need(actual or config["device"] == "cpu", "Tiny export must be CPU, never pretend actual MPS")
    values = {"candidate": parameter, "master": parameter.detach().float(), "reference": reference, "basis": basis,
              "exp_avg": torch.zeros_like(reference), "exp_avg_sq": torch.zeros_like(reference),
              "adam_step": torch.tensor(0., dtype=torch.float32), "norm": runtime.decoder.norm.weight,
              "head": runtime.model.lm_head.weight}
    for name, value in (("down_bias", runtime.branch.base.down_proj.bias), ("head_bias", runtime.model.lm_head.bias)):
        if value is not None:
            values[name] = value
    tensors = {name: export_tensor(root, name, value) for name, value in values.items()}
    rows, ids, samples = {}, {}, {}
    for i, (role, sample) in enumerate([(r, s) for r, group in (("new", new), ("replay", replay)) for s in group]):
        need(sample.get("purpose") == "training" and provenance(sample).get("evolution_role") == role,
             "Only bound role-specific TRAIN rows may be exported")
        need(not actual or eligible_learning_sample(sample), "Actual source is not independently verified/allowed TRAIN")
        key = "r" + str(i).zfill(3)
        need(sample["id"] not in ids, "Duplicate TRAIN identity")
        cache = caches[sample["id"]]
        need(cache["binding"] == binding and cache["sample_sha"] == hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest(), "Cache graph/sample binding differs")
        input_ids, labels = runtime._sample_tokens(sample)
        _labels(input_ids, labels, _eos(runtime))
        for suffix in ("x", "residual", "ids", "labels"):
            need(cached.tensor_sha(cache[suffix]) == cache["tensor_sha"][suffix], "Cache tensor changed")
            if suffix in ("ids", "labels"):
                need(torch.equal(cache[suffix], {"ids": input_ids, "labels": labels}[suffix].cpu()), "Cached input/gold labels differ")
            tensors[key + "_" + suffix] = export_tensor(root, key + "_" + suffix, cache[suffix])
        rows[key] = {"role": role, "sample_sha256": digest(sample), "length": input_ids.shape[1],
                     "gold": int((labels[:, 1:] != -100).sum())}
        ids[sample["id"]] = key
        samples[key] = sample
    if schedule is None:
        schedule = []
        for index in range(steps):
            pair = index // 2
            role = "new" if index % 2 == 0 else "replay"
            group = new if role == "new" else replay
            schedule.append({"ordinal": index + 1, "role": role, "ce": group[pair % len(group)]["id"],
                             "replay": replay[pair % len(replay)]["id"]})
    need(len(schedule) == steps, "Actual schedule count differs")
    schedule = [{**step, "ce": ids[step["ce"]], "replay": ids[step["replay"]]} for step in schedule]
    write_text(root / "manifest.tsv", manifest_text(config, tensors, rows, schedule))
    write_json(root / "fixture.json", {"protocol": PROTOCOL, "metadata": metadata, "samples": samples,
               "python_prefix_binding": binding, "manifest_sha256": sha(root / "manifest.tsv"), "tolerances": TOLERANCES,
               "producer_sources": binding["sources"], "shared_capture_timings": {key: caches[key]["timing"] for key in ids},
               "teacher_calls": 0, "generation_calls": 0, "publication": False})
    need(torch.equal(parameter, original) and cached._binding(runtime, parameter) == binding, "Producer graph/W changed")
    fixture(root, sha(root / "manifest.tsv"))
    return {"directory": str(root), "manifest_sha256": sha(root / "manifest.tsv"), "scope": metadata["scope"]}


def export_tiny(destination, dtype):
    """Actual offline tiny official Qwen CPU fixture, only after resource GO."""
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tests"))
    from test_cached_batch_training import fixture as tiny_graph
    import verify_cached_training as cached
    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as graph_store:
        torch.manual_seed(42)
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        runtime, new, replay, _ = tiny_graph(graph_store, torch.bfloat16 if dtype == "bf16" else torch.float32)
        # Existing helper's human-readable fixture identity is not the OS-lock
        # 64SHA ABI. Bind a synthetic identity before any cache is captured.
        runtime.identity = digest({"scope": "official_random_tiny_cpu", "seed": 42, "dtype": dtype,
                                   "config": runtime.model.config.to_json_string()})
        runtime.model.eval().requires_grad_(False)
        parameter = runtime.branch.base.down_proj.weight
        parameter.requires_grad_(True)
        reference = parameter.detach().float().clone()
        caches = {row["id"]: cached.capture_tail(runtime, row) for row in new + replay}
        basis, _ = cached._capture_replay_subspace(runtime, parameter, replay)
        return export_runtime_fixture(runtime, new, replay, reference, basis, caches, destination, steps=2,
            metadata={"scope": "cpu_tiny_fixture", "seed": 42, "actual_full_model": False})


def torch_load_raw(root, spec, device):
    import torch
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "i64": torch.int64, "u8": torch.uint8}[spec["dtype"]]
    return torch.from_file(str(Path(root) / spec["file"]), shared=False, size=math.prod(spec["shape"]), dtype=dtype).reshape(spec["shape"]).to(device)


def sync(device):
    if str(device) == "mps":
        import torch
        torch.mps.synchronize()


def memory():
    if sys.platform != "darwin":
        return {"scope": "Darwin physical-footprint comparison unavailable on this platform"}
    path = ROOT / "native/luban/footprint.py"
    spec = importlib.util.spec_from_file_location("native_training_footprint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {**module.footprint(os.getpid()), "api": "Darwin proc_pid_rusage RUSAGE_INFO_V4", "source_sha256": sha(path)}


def reference_run(source, output, approved_sha):
    """Minimal Python same-fixture engine: full-seq, two backwards, true Adam.

    Lock precedes raw tensor loading. No Runtime/full prefix remains resident.
    Source/runtime fixture creation cost is shared and excluded from this arm.
    """
    began = time.perf_counter()
    before_source = sha(__file__)
    m = fixture(source, approved_sha, verify_bytes=False)
    sys.path.insert(0, str(ROOT / "src"))
    from luban.process_lock import ModelProcessLock
    lock = ModelProcessLock(m["config"]["identity"], m["config"]["device"])
    root = None
    try:
        import torch
        from torch.nn import functional as F
        from luban.evolution import _answer_log_probs, _project_core_delta
        from verify_cached_training import _ce
        from verify_cached_batch_training import _labels
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device(m["config"]["device"])
        need(device.type != "mps" or torch.backends.mps.is_available(), "Actual MPS unavailable")
        root = fresh(output, (source,))
        loading = time.perf_counter()
        verify_files(source, m["tensors"])
        t = {key: torch_load_raw(source, value, device if key != "adam_step" else "cpu") for key, value in m["tensors"].items()}
        need(all(torch.isfinite(value).all() for value in t.values()), "Nonfinite fixture")
        need(torch.equal(t["master"].to(t["candidate"].dtype), t["candidate"]), "Candidate is not exact master cast")
        entry = float(t["adam_step"])
        need(0 <= entry <= 10000 and entry == int(entry) and (t["exp_avg_sq"] >= 0).all(), "Invalid Adam state")
        if entry == 0:
            need(torch.count_nonzero(t["exp_avg"]) == 0 and torch.count_nonzero(t["exp_avg_sq"]) == 0, "Fresh moments not zero")
        for key, row in m["rows"].items():
            _labels(t[key + "_ids"], t[key + "_labels"], {integer(m["config"]["train_eos"])})
            need(int((t[key + "_labels"][:, 1:] != -100).sum()) == row["gold"], "Actual gold count differs")
            need((t[key + "_ids"] >= 0).all() and (t[key + "_ids"] < t["head"].shape[0]).all(), "Token outside actual vocab")
        master = torch.nn.Parameter(t["master"])
        optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6, betas=(.9, .999), foreach=False, fused=False)
        if entry:
            optimizer.state[master] = {"step": t["adam_step"], "exp_avg": t["exp_avg"], "exp_avg_sq": t["exp_avg_sq"]}
        sync(device)
        load_seconds, load_memory = time.perf_counter() - loading, memory()
        tensors_out = {}

        def save(name, value):
            item = export_tensor(root, name, value)
            tensors_out[name] = item
            # Each durable partial output has an append-only tensor descriptor.
            listing.write("\t".join(("tensor", name, item["dtype"], ",".join(map(str, item["shape"])) or "-", item["file"], str(item["bytes"]), item["sha256"])) + "\n")
            listing.flush()

        def tail(key, weight):
            z = F.linear(t[key + "_x"], weight, t.get("down_bias")) + t[key + "_residual"]
            f = z.float()
            normalized = (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + t["norm"].float())).to(z.dtype)
            out = F.linear(normalized, t["head"], t.get("head_bias"))
            need(torch.isfinite(out).all(), "Full logits nonfinite")
            return out

        trace, phases, optimization_seconds = [], [], 0.
        with (root / "tensors.tsv").open("x") as listing, (root / "phases.jsonl").open("x") as phase_stream:
            listing.write(OUTPUT_ABI + "\n")
            for step in m["steps"]:
                sync(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                weight = master.detach().to(t["candidate"].dtype).clone().requires_grad_(True)
                incoming = []
                hook = weight.register_hook(lambda value: incoming.append(value.detach().clone()) or value)
                try:
                    start_ce = time.perf_counter()
                    logits = tail(step["ce"], weight)
                    labels = t[step["ce"] + "_labels"]
                    ce = _ce(logits, labels)
                    ce.backward()
                    sync(device)
                    ce_seconds = time.perf_counter() - start_ce
                    start_kl = time.perf_counter()
                    replay_labels = t[step["replay"] + "_labels"]
                    with torch.no_grad():
                        teacher = _answer_log_probs(tail(step["replay"], t["reference"].to(weight.dtype)), replay_labels).detach()
                    student = _answer_log_probs(tail(step["replay"], weight), replay_labels)
                    kl = F.kl_div(student, teacher, reduction="batchmean", log_target=True)
                    need(torch.isfinite(ce) and torch.isfinite(kl), "Nonfinite scalar objective")
                    (5 * kl).backward()
                    sync(device)
                    kl_seconds = time.perf_counter() - start_kl
                finally:
                    hook.remove()
                start_update = time.perf_counter()
                need(len(incoming) == 2 and torch.isfinite(weight.grad).all(), "Two finite native backwards required")
                total = weight.grad.detach().clone()
                master.grad = total.float().clone()
                norm = torch.nn.utils.clip_grad_norm_([master], 1, error_if_nonfinite=True)
                clipped = master.grad.detach().clone()
                optimizer.step()
                _project_core_delta(master, t["reference"], t["basis"])
                state = optimizer.state[master]
                need(int(float(state["step"])) == entry + step["ordinal"] and all(torch.isfinite(v).all() for v in (master, *state.values())), "Invalid cumulative Adam state")
                candidate = master.detach().to(weight.dtype).clone()
                sync(device)
                update_seconds = time.perf_counter() - start_update
                optimization_seconds += ce_seconds + kl_seconds + update_seconds
                saving = time.perf_counter()
                if step["ordinal"] in checkpoints(len(m["steps"])):
                    mask = labels[:, 1:] != -100
                    values = {"ce_logits": logits[:, :-1].float()[mask], "teacher_log_probs": teacher, "student_log_probs": student,
                              "ce_grad": incoming[0], "kl_grad": incoming[1], "total_grad": total, "clipped_grad": clipped,
                              "master": master, "candidate": candidate, "exp_avg": state["exp_avg"], "exp_avg_sq": state["exp_avg_sq"],
                              "adam_step": state["step"], "delta": master - t["reference"]}
                    for name, value in values.items():
                        save(f"checkpoint{step['ordinal']:06d}_" + name, value)
                whole = time.perf_counter() - start
                record = {**step, "adam_step": int(float(state["step"])), "ce_loss": float(ce.detach()), "kl": float(kl.detach()),
                          "preclip_norm": float(norm), "loss": float(ce.detach()) + 5 * float(kl.detach()), "wall_seconds": whole}
                trace.append(record)
                phase = {"step": step["ordinal"], "adam_step": record["adam_step"], "ce_role": step["role"], "ce_row": step["ce"], "kl_row": step["replay"],
                         "ce_seconds": ce_seconds, "kl_seconds": kl_seconds, "update_seconds": update_seconds, "raw_save_seconds": time.perf_counter() - saving,
                         "whole_step_seconds": whole}
                phases.append(phase)
                phase_stream.write(json.dumps(phase, allow_nan=False) + "\n")
                phase_stream.flush()
            for name, value in {"candidate": candidate, "master": master, "exp_avg": state["exp_avg"], "exp_avg_sq": state["exp_avg_sq"],
                                "adam_step": state["step"], "reference": t["reference"], "basis": t["basis"]}.items():
                save("final_" + name, value)
        with (root / "trace.tsv").open("x", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(TRACE)
            for value in trace:
                writer.writerow((value["ordinal"], value["role"], value["ce"], value["replay"], value["ce_loss"], value["kl"],
                                 value["preclip_norm"], value["loss"], value["wall_seconds"]))
        verify_files(source, m["tensors"])
        need(sha(Path(source) / "manifest.tsv") == approved_sha and sha(__file__) == before_source, "Input/source changed")
        report = {"abi": ABI, "worker": "Python-minimal-cached-tail-torch.optim.Adam", "adam_implementation": "installed single-tensor Adam foreach=False fused=False",
                  "passed_execution": True, "equivalence_tested": False, "manifest_sha256": approved_sha, "device": str(device), "steps": len(trace),
                  "entry_adam_step": int(entry), "final_adam_step": int(float(state["step"])), "new_ce_exposures": sum(v["role"] == "new" for v in trace),
                  "replay_ce_exposures": sum(v["role"] == "replay" for v in trace), "replay_kl_exposures": len(trace),
                  "load_hash_transfer_seconds": load_seconds, "pure_optimization_seconds": optimization_seconds,
                  "all_input_tensors_preserved": True, "input_tensor_count": len(m["tensors"]),
                  "whole_worker_seconds_before_result_write": time.perf_counter() - began, "load_memory": load_memory, "steady_memory": memory(),
                  "cpu_threads": 1, "cpu_interop_threads": 1, "teacher_calls": 0, "generation_calls": 0, "publication": False, "source_sha256": before_source}
        write_json(root / "result.json", report)
        print(json.dumps({"whole_worker_seconds_after_result_write": time.perf_counter() - began, "passed_execution": True}))
        return report
    except BaseException as error:
        if root is not None:
            write_json(root / "failure.json", {"passed_execution": False, "error": repr(error), "elapsed_seconds": time.perf_counter() - began})
        raise
    finally:
        lock.close()


def output_receipt(root, m):
    """Stdlib byte/schedule guard; does not claim tensor numeric equivalence."""
    root = safe(root)
    need(not (root / "failure.json").exists() and not (root / "failure.txt").exists(), "Failed/partial worker cannot mint successful receipt")
    tensors, rest = descriptors(root / "tensors.tsv", OUTPUT_ABI)
    need(not rest, "Unexpected output manifest row")
    required = {f"checkpoint{index:06d}_{key}" for index in checkpoints(len(m["steps"])) for key in PROBES}
    required |= {"final_" + name for name in FINAL}
    need(set(tensors) == required, "Incomplete/extra output state")
    for index in checkpoints(len(m["steps"])):
        step = m["steps"][index - 1]
        for key in PROBES:
            name = f"checkpoint{index:06d}_" + key
            actual = tensors[name]
            if key == "ce_logits":
                shape, dtype = [m["rows"][step["ce"]]["gold"], m["tensors"]["head"]["shape"][0]], "f32"
            elif key.endswith("log_probs"):
                shape, dtype = [m["rows"][step["replay"]]["gold"], m["tensors"]["head"]["shape"][0]], "f32"
            else:
                input_key = "candidate" if key in ("ce_grad", "kl_grad", "total_grad", "candidate") else key if key in GLOBAL else "master"
                shape, dtype = m["tensors"][input_key]["shape"], m["tensors"][input_key]["dtype"]
            need(actual["shape"] == shape and actual["dtype"] == dtype, "Output probe shape/dtype differs: " + name)
    for key in FINAL:
        need(tensors["final_" + key]["shape"] == m["tensors"][key]["shape"]
             and tensors["final_" + key]["dtype"] == m["tensors"][key]["dtype"], "Final state schema differs")
        if key in ("reference", "basis"):
            need(tensors["final_" + key]["sha256"] == m["tensors"][key]["sha256"], "Frozen reference/basis bytes changed")
    verify_files(root, tensors)
    report = json.loads((root / "result.json").read_text())
    need(report.get("abi") == ABI and report.get("passed_execution") is True
         and report.get("manifest_sha256") == m["manifest_sha256"] and report.get("device") == m["config"]["device"], "Worker input/execution binding differs")
    for key, expected in (("steps", len(m["steps"])), ("new_ce_exposures", len(m["steps"]) // 2),
                          ("replay_ce_exposures", len(m["steps"]) // 2), ("replay_kl_exposures", len(m["steps"])),
                          ("teacher_calls", 0), ("generation_calls", 0)):
        need(type(report.get(key)) is int and report[key] == expected, "Worker count differs: " + key)
    need(report.get("publication") is False and all(type(report.get(k)) is int and report[k] == 1 for k in ("cpu_threads", "cpu_interop_threads")),
         "No-publication/same-thread contract differs")
    need(report.get("all_input_tensors_preserved") is True and report.get("input_tensor_count") == len(m["tensors"]), "Input preservation receipt missing")
    entry = report.get("entry_adam_step")
    need(type(entry) is int and 0 <= entry <= 10000 and type(report.get("final_adam_step")) is int
         and report["final_adam_step"] == entry + len(m["steps"]), "Cumulative Adam step differs")
    with (root / "trace.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        need(tuple(reader.fieldnames or ()) == TRACE, "Trace header differs")
        trace = list(reader)
    need(len(trace) == len(m["steps"]), "Trace missing step")
    for value, step in zip(trace, m["steps"]):
        need(integer(value["step"]) == step["ordinal"] and value["ce_role"] == step["role"]
             and value["ce_row"] == step["ce"] and value["replay_row"] == step["replay"], "Actual schedule differs")
        for key in ("ce", "kl", "preclip_norm", "loss", "wall_seconds"):
            value[key] = float(value[key])
            need(math.isfinite(value[key]) and (key in ("kl", "loss") or value[key] >= 0), "Nonfinite/invalid metric")
        need(math.isclose(value["loss"], value["ce"] + 5 * value["kl"], abs_tol=1e-9, rel_tol=1e-9), "CE/KL total differs")
    return {"tensors": tensors, "report": report, "trace": trace,
            "receipt_sha256": digest({"manifest": sha(root / "tensors.tsv"), "report": sha(root / "result.json"), "trace": sha(root / "trace.tsv")})}


def tensor_comparison(expected, actual, *, atol, rtol, relative_l2, exact=False, agreement=None):
    import torch
    need(expected.shape == actual.shape and expected.dtype == actual.dtype and torch.isfinite(expected).all()
         and torch.isfinite(actual).all(), "Compared tensors shape/dtype/finite differ")
    a, b = expected.double(), actual.double()
    difference = (a - b).abs()
    norm = float(torch.linalg.vector_norm(a))
    error = float(torch.linalg.vector_norm(a - b))
    relative = error / norm if norm else (0. if error == 0 else None)
    # Exact uses raw bytes, including signed zeros; numeric tolerances are independent.
    raw_equal = torch.equal(expected.reshape(-1).view(torch.uint8), actual.reshape(-1).view(torch.uint8))
    matched = float((expected == actual).double().mean())
    allclose = bool((difference <= atol + rtol * a.abs()).all())
    norm_passed = error == 0 if norm == 0 else relative <= relative_l2
    passed = raw_equal if exact else (matched >= agreement if agreement is not None else allclose and norm_passed)
    return {"passed": passed, "raw_equal": raw_equal, "max_abs": float(difference.max()), "relative_l2": relative,
            "reference_l2": norm, "error_l2": error, "element_equal_fraction": matched,
            "allclose_passed": allclose, "relative_l2_passed": norm_passed, "atol": atol, "rtol": rtol,
            "relative_l2_limit": relative_l2, "agreement_minimum": agreement}


def compare_outputs(left_fixture, right_fixture, left, right, left_sha, right_sha, *, continuation=False):
    """Only after both engines exit; CPU tensor reads, no optimizer/generation."""
    lm, rm = fixture(left_fixture, left_sha), fixture(right_fixture, right_sha)
    need(lm["config"] == rm["config"] and lm["rows"] == rm["rows"] and lm["steps"] == rm["steps"], "Fixture schedule/config/materials differ")
    for key in lm["tensors"]:
        if not (continuation and key in INITIAL_STATE):
            need(lm["tensors"][key] == rm["tensors"][key], "Unapproved non-state fixture difference")
        else:
            need({k: v for k, v in lm["tensors"][key].items() if k != "sha256"}
                 == {k: v for k, v in rm["tensors"][key].items() if k != "sha256"}, "Continuation state schema differs")
    lr, rr = output_receipt(left, lm), output_receipt(right, rm)
    need(lr["report"]["worker"] == "Python-minimal-cached-tail-torch.optim.Adam"
         and rr["report"]["worker"] == "C++LibTorch-autograd-aligned-single-tensor-Adam", "Approved engines differ")
    need(lr["report"]["entry_adam_step"] == rr["report"]["entry_adam_step"], "Continuation entries differ")
    entry = lr["report"]["entry_adam_step"]
    import torch
    for fm in (lm, rm):
        actual_entry = float(torch_load_raw(fm["root"], fm["tensors"]["adam_step"], "cpu"))
        need(actual_entry == entry, "Reported entry is not actual input Adam state")
    settings = TOLERANCES[lm["config"]["device"]]
    metrics = {}
    for key in lr["tensors"]:
        a = torch_load_raw(left, lr["tensors"][key], "cpu")
        b = torch_load_raw(right, rr["tensors"][key], "cpu")
        kind = key.split("_", 1)[1]
        if kind == "adam_step":
            expected_step = entry + (len(lm["steps"]) if key.startswith("final_") else int(key[10:16]))
            need(float(a) == expected_step and float(b) == expected_step, "Actual output Adam state differs")
            gate = tensor_comparison(a, b, atol=0., rtol=0., relative_l2=0., exact=True)
        elif kind in ("reference", "basis"):
            need(lr["tensors"][key]["sha256"] == lm["tensors"][kind]["sha256"]
                 and rr["tensors"][key]["sha256"] == rm["tensors"][kind]["sha256"], "Frozen reference/basis changed")
            gate = tensor_comparison(a, b, atol=0., rtol=0., relative_l2=0., exact=True)
        elif kind == "candidate":
            gate = tensor_comparison(a, b, atol=0., rtol=0., relative_l2=0., exact=settings["native_bitwise"], agreement=settings["candidate_agreement"])
        elif kind in ("ce_logits", "teacher_log_probs", "student_log_probs", "ce_grad", "kl_grad", "total_grad"):
            is_logits = kind.endswith("log_probs") or kind == "ce_logits"
            gate = tensor_comparison(a, b, atol=settings.get("logits_atol" if is_logits else "gradient_atol", 0.),
                rtol=settings.get("native_rtol", 0.), relative_l2=settings.get("native_relative_l2", 0.), exact=settings["native_bitwise"])
        else:
            gate = tensor_comparison(a, b, atol=settings["state_atol"], rtol=settings["state_rtol"],
                relative_l2=settings["delta_relative_l2"] if kind == "delta" else settings["state_relative_l2"])
        metrics[key] = gate
    scalar_metrics = []
    for a, b in zip(lr["trace"], rr["trace"]):
        result = {"step": integer(a["step"])}
        for key in ("ce", "kl", "preclip_norm", "loss"):
            prefix = "norm" if key == "preclip_norm" else "loss"
            result[key] = {"python": a[key], "native": b[key], "abs_difference": abs(a[key] - b[key]),
                           "passed": abs(a[key] - b[key]) <= settings[prefix + "_atol"] + settings[prefix + "_rtol"] * abs(a[key])}
        scalar_metrics.append(result)
    return {"protocol": PROTOCOL, "passed_numeric": all(v["passed"] for v in metrics.values())
            and all(v["passed"] for row in scalar_metrics for v in row.values() if isinstance(v, dict)),
            "tolerances": settings, "tensor_metrics": metrics, "scalar_metrics": scalar_metrics,
            "python_receipt_sha256": lr["receipt_sha256"], "native_receipt_sha256": rr["receipt_sha256"],
            "exact_final_candidate_tensor_sha": lr["tensors"]["final_candidate"]["sha256"] == rr["tensors"]["final_candidate"]["sha256"],
            "python_final_candidate_sha256": lr["tensors"]["final_candidate"]["sha256"],
            "native_final_candidate_sha256": rr["tensors"]["final_candidate"]["sha256"],
            "actual_quality_proof": False, "quality_reuse_authorized": False, "continuation_own_states": continuation,
            "scope": "Numeric cached-tail executor comparison only; no169 closed-book generation or full-model native loader proof"}


def continue_fixture(source, state_output, destination, approved_sha):
    """Stdlib raw state import. Own-engine output, no PT/Adam/RNG restoration."""
    m = fixture(source, approved_sha)
    need(len(m["steps"]) == 2 and m["config"]["device"] == "cpu", "Only tiny2+2 continuation supported")
    receipt = output_receipt(state_output, m)
    need(receipt["report"]["entry_adam_step"] == 0 and receipt["report"]["final_adam_step"] == 2, "Second phase must start actualAdam2")
    root = fresh(destination, (source, state_output))
    tensors = {}
    for name, item in m["tensors"].items():
        state = receipt["tensors"]["final_" + name] if name in INITIAL_STATE else item
        file_root = Path(state_output) if name in INITIAL_STATE else Path(source)
        need(state["shape"] == item["shape"] and state["dtype"] == item["dtype"], "Imported state schema differs")
        dest = root / item["file"]
        shutil.copyfile(safe(file_root / state["file"]), dest)
        need(sha(dest) == state["sha256"], "Imported state bytes differ")
        tensors[name] = {**item, "sha256": state["sha256"]}
    write_text(root / "manifest.tsv", manifest_text(m["config"], tensors, m["rows"], m["steps"]))
    write_json(root / "continuation.json", {"protocol": PROTOCOL, "scope": "cpu_tiny_fixture",
               "parent_manifest_sha256": approved_sha, "own_engine_parent_receipt_sha256": receipt["receipt_sha256"],
               "own_engine": receipt["report"]["worker"], "entry_adam_step": 2, "no_optimizer_reset": True,
               "reference_basis_inputs_unchanged": True})
    return {"directory": str(root), "manifest_sha256": sha(root / "manifest.tsv")}


def run_tiny_pair(source, output, approved_sha, worker, worker_sha, *, seconds=900):
    """Root-only explicit execution: four serial children, no simultaneous GPU.

    No compiler/install. It requires an existing approved worker. Failed phase
    stops and retains immutable receipts; no retry or relaxed tolerance.
    """
    worker = safe(worker)
    need(SHA.fullmatch(worker_sha) and worker.is_file() and os.access(worker, os.X_OK) and sha(worker) == worker_sha, "Approved worker bytes differ")
    m = fixture(source, approved_sha)
    need(m["config"]["device"] == "cpu" and len(m["steps"]) == 2 and type(seconds) is int and 0 < seconds <= 900,
         "This first executor is CPU2+2 only with bounded900seconds")
    meta = json.loads((Path(source) / "fixture.json").read_text())
    need(meta.get("metadata", {}).get("scope") == "cpu_tiny_fixture" and meta.get("manifest_sha256") == approved_sha,
         "Actual tiny producer receipt required")
    root = fresh(output, (source, worker.parent))
    started = time.perf_counter()
    source_sha = sha(__file__)
    driver = ROOT / "scripts/native-training.ts"
    worker_source = ROOT / "native/Training.cpp"
    source_bindings = {str(path): sha(path) for path in (Path(__file__), driver, worker_source)}
    bun = shutil.which("bun")
    need(bun is not None, "Existing Bun native driver required; no installation")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")}

    def run(engine, inputs, target, hashed):
        need(sha(worker) == worker_sha and sha(__file__) == source_sha, "Executor source/binary changed")
        command = ([sys.executable, str(Path(__file__)), "--reference", str(inputs), "--output", str(target), "--approved-sha", hashed]
                   if engine == "python" else [bun, str(driver), "--run", str(inputs), "--output", str(target),
                                              "--binary", str(worker), "--binary-sha", worker_sha, "--manifest-sha", hashed])
        remaining = seconds - (time.perf_counter() - started)
        need(remaining > 0, "Pair deadline exceeded")
        with (root / (target.name + ".log")).open("xb") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(root / (target.name + ".invocation.json"), {"command": command, "pid": child.pid, "phase": target.name,
                       "manifest_sha256": hashed, "source_bindings": source_bindings, "binary_sha256": worker_sha})
            try:
                child.wait(timeout=remaining)
            except BaseException:
                # Bun's C++ child shares this group; killing only Bun could leave
                # a model process/OS lock live after the wrapper deadline.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                raise
        need(child.returncode == 0, "Engine failed; retained output/log; no retry")
        need(sha(worker) == worker_sha and all(sha(path) == expected for path, expected in source_bindings.items()), "Executor changed after child exit")
        if engine == "native":
            receipt = json.loads((root / (target.name + ".native-started.json.result.json")).read_text())
            need(receipt.get("fixture_binary_sources_preserved") is True and receipt.get("child_python_invocations") == 0
                 and receipt.get("actual_execution", {}).get("exit_code") == 0 and receipt.get("binary_sha256") == worker_sha
                 and receipt.get("fixture", {}).get("manifest_sha256") == hashed, "Bun native library/clean-environment receipt failed")

    try:
        run("python", source, root / "python1", approved_sha)
        run("native", source, root / "native1", approved_sha)
        first = compare_outputs(source, source, root / "python1", root / "native1", approved_sha, approved_sha)
        write_json(root / "phase1-comparison.json", first)
        need(first["passed_numeric"], "Phase1 numeric gate failed; no second phase")
        py2 = continue_fixture(source, root / "python1", root / "python-input2", approved_sha)
        cpp2 = continue_fixture(source, root / "native1", root / "native-input2", approved_sha)
        run("python", py2["directory"], root / "python2", py2["manifest_sha256"])
        run("native", cpp2["directory"], root / "native2", cpp2["manifest_sha256"])
        second = compare_outputs(py2["directory"], cpp2["directory"], root / "python2", root / "native2",
                                 py2["manifest_sha256"], cpp2["manifest_sha256"], continuation=True)
        write_json(root / "phase2-comparison.json", second)
        report = {"protocol": PROTOCOL, "passed_numeric": first["passed_numeric"] and second["passed_numeric"],
                  "actual_adam_steps": [0, 2, 4], "cpu_tiny_only": True, "full_model_quality_proof": False,
                  "manifest_sha256": approved_sha, "binary_sha256": worker_sha, "source_sha256": source_sha,
                  "source_bindings": source_bindings,
                  "whole_pair_seconds_before_result_write": time.perf_counter() - started, "teacher_calls": 0,
                  "generation_calls": 0, "publication": False, "execution_order": ["python1", "native1", "python2", "native2"],
                  "scope": "Each secondphase resumes its own moments/master and repeats the same2 CEevents; not4distinct uninterrupted events. Fixed order warm-disk caveat; no general speed claim."}
        need(report["whole_pair_seconds_before_result_write"] <= seconds, "Full pair deadline exceeded")
        write_json(root / "result.json", report)
        need(time.perf_counter() - started <= seconds, "Deadline exceeded after result save")
        return report
    except BaseException as error:
        write_json(root / "failure.json", {"passed_numeric": False, "error": repr(error), "elapsed_seconds": time.perf_counter() - started})
        raise


def plan():
    return {"protocol": PROTOCOL, "stage": "source_only", "actual_model_ready": False, "model_go": False,
            "abi": ABI, "cpu_tiny": {"phases": [2, 2], "entry_adam_steps": [0, 2], "final_adam_step": 4,
                                      "continuation": "Each engine resumes its own final native/master/moments; shared reference/basis/cache unchanged"},
            "mps_future": {"numeric_pilot_steps": 50, "full_steps": 450,
                           "actual_fixture_binding": "Separate actual producer owns official/prod/qualification/full-Qwen capture; no MPS GO in this source"},
            "tolerances": TOLERANCES, "B": 1, "G": 1, "lr": 1e-5, "eps": 1e-6, "betas": [.9, .999], "clip": 1., "kl_weight": 5.,
            "schedule": "1-based odd NEW CE/even REPLAY CE; each Adam step adds separate REPLAY KL backward",
            "native_optimizer": "aligned ATen lerp/addcmul/sqrt/addcdiv order, not default C++ Adam mul/add order",
            "quality": "Numeric agreement is not169 proof; reuse only exact candidateTensorSHA plus actual materials/method/order and closed quality evidence",
            "scope": "Cached Python-product prefix only; no native full-decoder loader/generation. Shared capture/SVD cost separately reported."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tiny-fixture")
    group.add_argument("--reference")
    group.add_argument("--tiny-pair")
    p.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    p.add_argument("--output")
    p.add_argument("--approved-sha")
    p.add_argument("--worker")
    p.add_argument("--worker-sha")
    p.add_argument("--seconds", type=int, default=900)
    args = p.parse_args()
    if args.tiny_fixture:
        print(json.dumps(export_tiny(args.tiny_fixture, args.dtype), indent=2))
    elif args.reference:
        need(args.output and args.approved_sha, "Explicit output and approved manifest SHA required")
        reference_run(args.reference, args.output, args.approved_sha)
    elif args.tiny_pair:
        need(args.output and args.approved_sha and args.worker and args.worker_sha, "Explicit fixture/source/binary approval required")
        print(json.dumps(run_tiny_pair(args.tiny_pair, args.output, args.approved_sha, args.worker, args.worker_sha, seconds=args.seconds), indent=2))
    else:
        print(json.dumps(plan(), indent=2))


if __name__ == "__main__":
    main()
