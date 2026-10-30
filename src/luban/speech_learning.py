"""Fixed-budget native ASR candidates. The server owns scheduling and adoption stays gated."""
from __future__ import annotations

from array import array
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import tempfile
import time
import unicodedata

import psutil

from .process_lock import ModelProcessLock
from .settings import private_json, require_storage
from .speech import decode_wav
from .training_checkpoint import TrainingPaused

ROUND_STEPS = 4
VOCAB = 151936
EOS = 151645
HOLDOUT_MAX_NEW_TOKENS = 512


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _cancelled(cancel):
    return bool(cancel and (cancel.is_set() if hasattr(cancel, "is_set") else cancel()))


def _check(cancel):
    if _cancelled(cancel):
        raise TrainingPaused("声音训练已暂停，保留最后完整候选")


def _binary(name):
    root = Path(__file__).resolve().parents[2]
    for path in (root / "native" / name, root / "build/native" / name):
        if path.is_file() and not path.is_symlink() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(f"原生声音训练组件未安装：{name}")


def capability(runtime):
    """Availability only: no compilation, weights, tokenizer parsing or hashes."""
    try:
        for name in ("luban-asr-training", "luban-asr-features", "luban-asr-tokenizer"):
            _binary(name)
        model = Path(runtime.settings.snapshot()["speech_model_path"])
        if not all((model / name).is_file() for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "upstream-file-manifest.json")):
            raise RuntimeError("请先安装原生声音模型与分词器")
        return {"available": True, "reason": "", "round_budget": ROUND_STEPS, "adoption_available": False}
    except (OSError, RuntimeError, KeyError) as error:
        return {"available": False, "reason": str(error), "round_budget": ROUND_STEPS, "adoption_available": False}


def _unchanged(fd, path, before):
    held, named = os.fstat(fd), os.lstat(path)
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
            or any(getattr(before, k) != getattr(held, k) or getattr(held, k) != getattr(named, k) for k in keys)):
        raise ValueError("声音输入文件在读取时发生变化")


def _read(path, limit):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= limit:
            raise ValueError("声音资料必须是有界普通文件")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) != before.st_size:
            raise ValueError("声音输入长度改变")
        _unchanged(fd, path, before)
        return raw
    finally:
        os.close(fd)


def validate_samples(samples):
    if not isinstance(samples, list) or not 2 <= len(samples) <= 64:
        raise ValueError("声音学习需要训练与独立检验录音，最多64条")
    families, ids, records = {}, set(), []
    for row in samples:
        if (not isinstance(row, dict) or row.get("kind") != "sound"
                or row.get("split") not in {"training", "validation"}
                or row.get("training_allowed") is not True or row.get("transcript_reviewed") is not True):
            raise ValueError("声音资料必须允许训练并已人工核对转写")
        key, family, answer = row.get("id"), row.get("family"), row.get("answer")
        if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", x) for x in (key, family)) or key in ids:
            raise ValueError("声音资料ID或来源分组无效")
        if (not isinstance(answer, str) or not answer.strip() or len(answer) > 4096
                or re.search(r"[\x00-\x1f\x7f]|<\|[^>]*\|>|<asr_text>", answer)):
            raise ValueError("请提供完整、无特殊控制标记的人工转写")
        if family in families and families[family] != row["split"]:
            raise ValueError("同一声音来源不能同时用于训练和独立检验")
        if not isinstance(row.get("asset_path"), str) or not Path(row["asset_path"]).is_absolute():
            raise ValueError("声音资料路径无效")
        families[family] = row["split"]
        ids.add(key)
        records.append({k: row[k] for k in ("id", "kind", "split", "family", "answer", "asset_path",
                                          "training_allowed", "transcript_reviewed")})
    if {r["split"] for r in records} != {"training", "validation"}:
        raise ValueError("声音训练与独立检验资料缺一不可")
    return records


def _run(argv, directory, cancel, fds=(), input_data=None, checkpoint=None):
    """Original subprocess cancellation/close boundary; no model or sample auditing loop."""
    stdout, stderr = directory / "stdout.txt", directory / "stderr.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
    started = time.monotonic()
    with stdout.open("w+b") as out, stderr.open("wb") as err:
        child = subprocess.Popen([str(v) for v in argv], stdin=subprocess.PIPE if input_data else subprocess.DEVNULL,
                                 stdout=out, stderr=err, pass_fds=tuple(fds), start_new_session=True, env=env)
        offset, terminating, rows = 0, None, []
        try:
            if input_data:
                child.stdin.write(input_data)
                child.stdin.close()
            while True:
                code = child.poll()
                with stdout.open("rb") as stream:
                    stream.seek(offset)
                    while line := stream.readline():
                        if not line.endswith(b"\n") and child.poll() is None:
                            break
                        offset += len(line)
                        value = json.loads(line)
                        if isinstance(value, dict) and value.get("event") == "checkpoint" and checkpoint:
                            checkpoint(value)
                        else:
                            rows.append(value)
                if code is not None:
                    break
                if _cancelled(cancel) or time.monotonic() - started > 2400:
                    if terminating is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        terminating = time.monotonic()
                    elif time.monotonic() - terminating > 20:
                        os.killpg(child.pid, signal.SIGKILL)
                time.sleep(.05)
            child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            receipt = {"argv": [str(v) for v in argv], "pid": child.pid, "exit_code": child.returncode,
                       "elapsed_seconds": time.monotonic() - started, "stdout": str(stdout), "stderr": str(stderr)}
            private_json(directory / "process.json", receipt)
    _check(cancel)
    if child.returncode or not rows:
        raise RuntimeError("原生声音训练失败：" + stderr.read_text(errors="replace")[-2048:])
    return rows[-1], receipt


def _filters():
    maximum = 15 + math.log(8) * 27 / math.log(6.4)
    step = maximum / 129
    centers = [1000 * math.exp(math.log(6.4) / 27 * ((maximum if i == 129 else step * i) - 15))
               if (maximum if i == 129 else step * i) >= 15 else 200 * (maximum if i == 129 else step * i) / 3 for i in range(130)]
    return array("f", (max(0, min((40 * f - centers[m]) / (centers[m + 1] - centers[m]),
                                 (centers[m + 2] - 40 * f) / (centers[m + 2] - centers[m + 1])))
                       * 2 / (centers[m + 2] - centers[m]) for f in range(201) for m in range(128))).tobytes()


def _prepare(record, wav, directory, filters_fd, model, binaries, cancel):
    _check(cancel)
    audio, duration = decode_wav(wav)
    audio_path = directory / "audio.raw"
    audio_path.write_bytes(audio.tobytes())
    with audio_path.open("rb") as source:
        features, _ = _run([binaries["features"], len(audio), 16000, len(audio), directory / "features",
                            source.fileno(), filters_fd], directory, cancel, (source.fileno(), filters_fd))
    active, padded = max(8000, len(audio)) // 160, math.ceil(max(8000, len(audio)) // 160 / 100) * 100
    if not features.get("passed") or features.get("active_frames") != active:
        raise ValueError("原生声音特征与原始音频长度不一致")
    audio_tokens = sum((min(100, active - at) + 7) // 8 for at in range(0, active, 100))
    target = "language Chinese<asr_text>" + record["answer"] + "<|im_end|>"
    request = [{"mode": "prompt", "language": "zh", "audio_tokens": audio_tokens}, {"text": target}]
    token_dir = directory / "tokenize"
    token_dir.mkdir(mode=0o700)
    tokens, _ = _run([binaries["tokenizer"], model], token_dir, cancel,
                     input_data=json.dumps(request, ensure_ascii=False).encode())
    # Tokenizer emits one JSON array rather than an ASR worker receipt.
    prefix, gold = tokens[0]["ids"], tokens[1]["ids"]
    if (not all(r.get("ok") for r in tokens) or tokens[1]["decode"] != target or gold[-1] != EOS
            or prefix.count(151676) != audio_tokens or len(prefix) + len(gold) > 512):
        raise ValueError("声音标签必须完整并包含EOS，不能截断")
    mel = _read(directory / "features/log.raw", 128 * padded * 4)
    mask = _read(directory / "features/mask.raw", padded * 4)
    if len(mel) != 128 * padded * 4 or len(mask) != padded * 4:
        raise ValueError("声音特征形状无效")
    if not all(math.isfinite(v) for v in struct.unpack("<" + "f" * (len(mel) // 4), mel)):
        raise ValueError("声音特征非有限")
    ids, labels = prefix + gold, [-100] * len(prefix) + gold
    chunks = [("input", "f32", [1, 128, padded], mel), ("mask", "i32", [1, padded], mask),
              ("ids", "i64", [1, len(ids)], struct.pack("<" + "q" * len(ids), *ids)),
              ("labels", "i64", [1, len(ids)], struct.pack("<" + "q" * len(labels), *labels))]
    prompt_chunks = chunks[:2] + [("ids", "i64", [1, len(prefix)], struct.pack("<" + "q" * len(prefix), *prefix))]
    return {"record": record, "chunks": chunks, "prompt_chunks": prompt_chunks, "prompt_tokens": len(prefix),
            "tokens": len(ids), "padded_frames": padded, "seconds": duration}


def _header(stream):
    before = os.fstat(stream.fileno())
    count = struct.unpack("<Q", stream.read(8))[0]
    if not 2 <= count <= 16 << 20:
        raise ValueError("声音模型头无效")
    header = json.loads(stream.read(count))
    rows, offset, parameters, largest = [], 8 + count, 0, 0
    for key, row in sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda item: item[1]["data_offsets"][0]):
        shape, ends = row["shape"], row["data_offsets"]
        if row["dtype"] != "BF16" or not shape or not all(type(d) is int and d > 0 for d in shape):
            raise ValueError("声音模型必须保留原BF16形状")
        size = math.prod(shape) * 2
        if ends != [offset - 8 - count, offset - 8 - count + size]:
            raise ValueError("声音模型完整范围不连续")
        rows.append((key, "bf16", shape, offset, size))
        offset += size
        parameters += size // 2
        largest = max(largest, size)
    if len(rows) != 611 or offset != before.st_size:
        raise ValueError("声音模型不是完整Qwen3-ASR-0.6B")
    if header["model.language_model.layers.27.mlp.down_proj.weight"]["shape"] != [1024, 3072]:
        raise ValueError("声音训练目标形状不匹配")
    return rows, parameters, largest, before


def _memory_required(parameters, largest, inputs):
    tokens, frames = max(i["tokens"] for i in inputs), max(i["padded_frames"] for i in inputs)
    target, weights = 1024 * 3072 * 4, parameters * 4
    cached_tokens = max(tokens, max((i.get("prompt_tokens", 0) + HOLDOUT_MAX_NEW_TOKENS
                                     for i in inputs if i.get("record", {}).get("split") == "validation"), default=0))
    training = weights + 11 * target + 2 * 28 * 8 * 128 * cached_tokens * 4 + 6 * tokens * VOCAB * 4
    training += (64 << 20) * (frames // 100) + sum(len(c[3]) for i in inputs for c in i["chunks"]) + 2 * 128 * frames * 4
    return max(weights + largest, training) + (512 << 20) + (128 << 20) + 1024 ** 3


def _descriptor(directory, model, model_rows, binding, selected, training_samples=None):
    raw, rows, offset = bytearray(), [], 0
    for index, item in enumerate(selected):
        for key, dtype, shape, value in item["chunks"]:
            rows.append("\t".join(map(str, ("tensor", f"sample{index}:{key}", dtype, ",".join(map(str, shape)), offset, len(value), 4))))
            raw.extend(value)
            offset += len(value)
    packed = directory / "round-input.raw"
    packed.write_bytes(raw)
    lines = ["LUBAN_ASR_TRAIN_OWNER_V2", "config\tmodel_path\t" + str(model),
             "config\tinput_path\t" + str(packed), "config\tbinding\t" + binding]
    if training_samples is not None:
        lines.append("config\ttraining_samples\t" + str(training_samples))
    lines += ["\t".join(map(str, ("tensor", k, dtype, ",".join(map(str, shape)), at, size, 3)))
              for k, dtype, shape, at, size in model_rows]
    path = directory / "owned.tsv"
    path.write_text("\n".join(lines + rows) + "\n")
    return path, packed



def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _recognition_quality(records, outputs):
    """Gold is used only here, after gold-free native greedy decoding. Never adopt."""
    if len(records) != len(outputs):
        raise ValueError("声音独立检验输出数量不一致")
    def words(text):
        text = unicodedata.normalize("NFKC", text).casefold()
        return "".join(c if unicodedata.category(c)[0] in "LN" or c == "'" else " " for c in text).split()
    rows = []
    for record, output in zip(records, outputs):
        if not output.get("ok") or not isinstance(output.get("decode"), str):
            raise ValueError("声音自由生成转写解码失败")
        raw, ids = output["decode"], output["ids"]
        ended = bool(ids and ids[-1] in {151643, EOS})
        text = raw.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
        # Preserve repetition/errors; the inference presentation repetition fix is not a quality metric.
        text = text.split("<asr_text>", 1)[-1].strip()
        reference, prediction = words(record["answer"]), words(text)
        ref_chars, pred_chars = list("".join(reference)), list("".join(prediction))
        rows.append({"id": record["id"], "reference": record["answer"], "raw_output": raw,
                     "transcription": text, "ended_eos": ended, "generated_tokens": len(ids),
                     "character_edits": _edit_distance(ref_chars, pred_chars), "reference_characters": len(ref_chars),
                     "word_edits": _edit_distance(reference, prediction), "reference_words": len(reference)})
    chars, char_edits = sum(r["reference_characters"] for r in rows), sum(r["character_edits"] for r in rows)
    words_count, word_edits = sum(r["reference_words"] for r in rows), sum(r["word_edits"] for r in rows)
    return {"method": "audio_prompt_only_greedy", "normalization": "NFKC_casefold_letters_numbers_apostrophe",
            "word_tokenization": "whitespace", "rows": rows, "character_edits": char_edits,
            "reference_characters": chars, "CER": char_edits / max(1, chars), "word_edits": word_edits,
            "reference_words": words_count, "WER": word_edits / max(1, words_count),
            "passed": bool(rows and chars and all(r["ended_eos"] and r["reference_characters"] for r in rows)
                           and char_edits == 0 and word_edits == 0)}


def _round_progress(candidate):
    if candidate is None:
        return 0, 0, ROUND_STEPS, 0
    total, completed = candidate.get("total_adam_step"), candidate.get("completed_steps")
    if (type(total) is not int or type(completed) is not int or not 0 <= completed <= ROUND_STEPS
            or not max(1, completed) <= total <= 100000 or candidate.get("round_budget") != ROUND_STEPS):
        raise ValueError("声音候选的固定轮次进度无效")
    done = completed if candidate.get("state") in {"paused", "candidate"} and completed < ROUND_STEPS else 0
    return total, done, ROUND_STEPS - done, total - done


def train_round(settings, samples, saved_candidate=None, cancel=None, progress=None):
    """Synchronous native4-update round; caller holds application runtime/job exclusion.

    settings is the existing Settings or its snapshot. Samples must be persisted,
    permission-checked raw WAV/transcript records. This function never adopts.
    """
    values = settings.snapshot() if hasattr(settings, "snapshot") else settings
    records = validate_samples(samples)
    binaries = {key: _binary(name) for key, name in {"training": "luban-asr-training", "features": "luban-asr-features",
                                                   "tokenizer": "luban-asr-tokenizer"}.items()}
    root, model = Path(values["storage_root"]), Path(values["speech_model_path"])
    for path in (root, model):
        if str(path).startswith("/Volumes/"):
            require_storage(Path("/Volumes") / path.parts[2])
    if model.resolve() != model or not model.is_dir():
        raise ValueError("声音模型目录不能包含符号链接")
    manifest = json.loads(_read(model / "upstream-file-manifest.json", 4 << 20))
    identity = next(r["Sha256"] for r in manifest["Data"]["Files"] if r["Path"] == "model.safetensors")
    if not re.fullmatch("[a-f0-9]{64}", identity):
        raise ValueError("声音模型来源身份无效")
    store = root / "modules/speech-learning"
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    if store.resolve() != store:
        raise ValueError("声音检查点目录不能包含符号链接")
    actual = Path(tempfile.mkdtemp(prefix="round-", dir=store))
    wavs = [_read(r["asset_path"], 8 << 20) for r in records]
    corpus = hashlib.sha256(_encoded(records).encode() + b"".join(wavs)).hexdigest()
    binding = _encoded({"schema": "luban-ASR-product-candidate-v1", "model_identity": identity,
                        "records": records, "original_wav_corpus": corpus, "round_budget": ROUND_STEPS,
                        "device": "cpu", "precision": "float32", "trainable": "last_language_down_projection"})
    candidate = None
    if saved_candidate:
        path = Path(saved_candidate)
        if path.resolve() != path or not path.is_relative_to(store):
            raise ValueError("声音候选检查点必须属于当前私有目录")
        candidate = json.loads(_read(path, 2 << 20))
        for key in ("checkpoint", "cpu_rng"):
            source = Path(candidate[key])
            if source.resolve() != source or not source.is_relative_to(store):
                raise ValueError("声音候选内部文件不属于当前私有目录")
        if candidate["binding"] != binding or candidate.get("adopted") is not False:
            raise ValueError("声音候选与当前模型或完整资料不一致")
    prior = candidate["checkpoint"] if candidate else None
    total, done, remaining, round_start = _round_progress(candidate)
    # A completed prior round is the next round's step 0 until a new save.
    published = {**candidate, "completed_steps": done, "state": "paused"} if candidate else None
    checkpoint_path = actual / "candidate.json"
    if progress:
        progress({"phase": "preparing", "step": 0, "steps": ROUND_STEPS})
    try:
        with ExitStack() as stack:
            cpu = ModelProcessLock(identity, "cpu")
            stack.callback(cpu.close)
            mps = ModelProcessLock(identity, "mps")
            stack.callback(mps.close)
            filters_path = actual / "filters.raw"
            filters_path.write_bytes(_filters())
            filters = stack.enter_context(filters_path.open("rb"))
            inputs = []
            for record, wav in zip(records, wavs):
                directory = actual / record["id"]
                directory.mkdir(mode=0o700)
                inputs.append(_prepare(record, wav, directory, filters.fileno(), model, binaries, cancel))
            held = stack.enter_context(os.fdopen(os.open(model / "model.safetensors", os.O_RDONLY | os.O_NOFOLLOW), "rb"))
            model_rows, parameters, largest, before = _header(held)
            required = _memory_required(parameters, largest, inputs)
            if psutil.virtual_memory().available < required:
                raise RuntimeError(f"声音训练内存不足，需要约{required / 1024 ** 3:.2f} GiB；未加载权重")
            train = [i for i in inputs if i["record"]["split"] == "training"]
            holdout = [i for i in inputs if i["record"]["split"] == "validation"]
            # Fixed four updates, never retry/extend based on a poor answer.
            schedule = [train[(total + i) % len(train)] for i in range(remaining)]
            train_dir = actual / "train"
            train_dir.mkdir(mode=0o700)
            selected = schedule + [{**item, "chunks": item["prompt_chunks"]} for item in holdout]
            descriptor, packed = _descriptor(train_dir, model / "model.safetensors", model_rows, binding, selected, remaining)
            def saved(event):
                nonlocal published
                step = event["step"]
                if not total < step <= total + remaining or event["sample"] != step - total - 1:
                    raise ValueError("原生声音实际更新进度不匹配")
                native = train_dir / "native" / ("segment-" + str(event["sample"]))
                published = {"state": "candidate", "checkpoint": str(native / "checkpoint.pt"), "cpu_rng": str(native / "cpu_rng.raw"),
                             "binding": binding, "total_adam_step": step, "completed_steps": step - round_start,
                             "round_budget": ROUND_STEPS, "quality_checked": False, "adopted": False,
                             "candidate_path": str(checkpoint_path), "records": records, "actual": str(actual)}
                private_json(checkpoint_path, published)
                if progress:
                    progress({"phase": "validating" if step - round_start == ROUND_STEPS else "training", "step": step - round_start, "steps": ROUND_STEPS,
                              "candidate": {**published, "state": "paused"}})
            with packed.open("rb") as packed_fd, ExitStack() as resume:
                rng = resume.enter_context(os.fdopen(os.open(candidate["cpu_rng"], os.O_RDONLY | os.O_NOFOLLOW), "rb")) if candidate else None
                archive = resume.enter_context(os.fdopen(os.open(prior, os.O_RDONLY | os.O_NOFOLLOW), "rb")) if candidate else None
                _check(cancel)
                _unchanged(held.fileno(), model / "model.safetensors", before)
                if psutil.virtual_memory().available < required:
                    raise RuntimeError("声音训练启动前可用内存不足；未加载权重")
                trained, execution = _run([binaries["training"], descriptor, Path("/tmp") / ("luban-model-locks-" + str(os.getuid())) / (identity + "-cpu.lock"), train_dir / "native", prior or "none", "train", candidate["cpu_rng"] if candidate else "none", held.fileno(), packed_fd.fileno(), cpu.fd, rng.fileno() if rng else 0, archive.fileno() if archive else 0], train_dir, cancel, [held.fileno(), packed_fd.fileno(), cpu.fd] + ([rng.fileno(), archive.fileno()] if rng else []), checkpoint=saved)
            if trained["actual_adam_updates"] != remaining or trained["final_step"] != round_start + ROUND_STEPS or not published:
                raise ValueError("声音训练没有完成固定预算")
            if (trained.get("holdout_max_new_tokens") != HOLDOUT_MAX_NEW_TOKENS
                    or trained.get("holdout_prompt_tokens") != [i["prompt_tokens"] for i in holdout]):
                raise ValueError("声音独立检验必须使用无答案原始提示")
            generated = trained.get("holdout_tokens")
            if (not isinstance(generated, list) or len(generated) != len(holdout)
                    or any(not isinstance(ids, list) or not 1 <= len(ids) <= HOLDOUT_MAX_NEW_TOKENS
                           or any(type(v) is not int or not 0 <= v < VOCAB for v in ids) for ids in generated)):
                raise ValueError("声音独立检验生成数量或token无效")
            decode_dir = actual / "decode-holdout"
            decode_dir.mkdir(mode=0o700)
            decoded, _ = _run([binaries["tokenizer"], model], decode_dir, cancel,
                              input_data=json.dumps([{"ids": ids} for ids in generated]).encode())
            if [row.get("ids") for row in decoded] != generated:
                raise ValueError("声音解码必须对应真实自由生成token")
            quality = _recognition_quality([i["record"] for i in holdout], decoded)
            _unchanged(held.fileno(), model / "model.safetensors", before)
            published.update(process=execution, native_result=trained, state="awaiting_quality", quality_checked=True,
                             recognition_quality=quality, quality_passed=quality["passed"],
                             protection_checked=False, adopted=False, holdout_samples=len(holdout),
                             source_payload_SHA_rechecked=False)
            private_json(checkpoint_path, published)
            return published
    except TrainingPaused:
        if published:
            published.update(state="paused", adopted=False)
            private_json(checkpoint_path, published)
            return published
        raise

    except Exception as error:
        if published:
            published.update(state="paused", adopted=False, error=str(error))
            private_json(checkpoint_path, published)
            error.candidate = published
        raise
