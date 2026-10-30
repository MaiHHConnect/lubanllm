#!/usr/bin/env python3
"""CPU-only real Swift/HF/tokenizers differential test; never imports Torch/model weights."""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tokenizers
import transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1"))
    parser.add_argument("--evidence-root", type=Path, default=Path("/Volumes/移动数据库/lubanllm/experiments"))
    parser.add_argument("--output", type=Path, default=ROOT / "research/native_tokenizer_result.json")
    args = parser.parse_args()
    volume = Path("/Volumes/移动数据库")
    if not os.path.ismount(volume) or not args.model.is_dir():
        raise SystemExit("Required external model volume is not mounted")
    if not args.evidence_root.resolve().is_relative_to(volume.resolve()):
        raise SystemExit("Evidence must be written to the mounted external volume")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    evidence = args.evidence_root / f"native-tokenizer-{stamp}"
    evidence.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    command = ["swiftc", "-O", str(ROOT / "native/Tokenizer.swift"), str(ROOT / "tests/native_tokenizer.swift"), "-o", str(evidence / "tokenizer")]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (evidence / "compile.log").write_text(compiled.stdout + compiled.stderr)
    compiled.check_returncode()
    compile_seconds = time.perf_counter() - start
    raw = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    hf = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    json_model = json.loads((args.model / "tokenizer.json").read_text())
    config = json.loads((args.model / "tokenizer_config.json").read_text())
    rng = random.Random(42)
    texts = ["", "你好，世界！我是鲁班零号。", "Hello, world! I'm Luban0.", "Café cafe\u0301 e\u0301\u0323 Å A\u030A", "👩‍💻🇨🇳🏳️‍🌈🙂", "\r\n\n \t\u00a0\u2003\u2028\u2029", "1 23 456 １２３ ١٢٣", "don't DON'T we're I'LL he'd", "a\u200db\u200cc\ufe0f", "汉字日本語한국어العربيةעבריתไทย", "\x00\x01\x1c\x7f\x85", " A  B   \n\n\tC\t", "a" * 8192, " " * 8192, "中" * 4096]
    entries = list(json_model["added_tokens"])
    existing = {e["id"] for e in entries}
    entries += [dict(v, id=int(k)) for k, v in config["added_tokens_decoder"].items() if int(k) not in existing]
    for entry in entries:
        texts += [entry["content"], "前" + entry["content"] + "后", " " + entry["content"] + "\n", entry["content"] * 2]
    alphabet = "abcXYZ'!? \t\r\n中鲁班数学123🙂e\u0301\u0323\u200d\u200c\u00a0\u2028\u2029العربيةไทย한글"
    texts += ["".join(rng.choices(alphabet, k=rng.randrange(1, 250))) for _ in range(500)]
    unicode_pool = [chr(i) for i in range(0x110000) if i % 557 == 0 and not 0xD800 <= i <= 0xDFFF]
    texts += ["".join(rng.choices(unicode_pool, k=50)) for _ in range(100)]
    texts += [chr(i) for i in range(256)]
    chats = []
    sets = [
        [{"role": "user", "content": "你是谁？"}],
        [{"role": "system", "content": "  简洁回答。\n"}, {"role": "user", "content": "  Hello!  "}],
        [{"role": "user", "content": "一加一"}, {"role": "assistant", "content": "<think>\n计算\n</think>\n\n2"}, {"role": "user", "content": "再加三呢？"}],
        [{"role": "user", "content": "test"}, {"role": "assistant", "content": "answer", "reasoning_content": " thought\n "}],
        [{"role": "user", "content": "查一下"}, {"role": "assistant", "content": "查询"}, {"role": "tool", "content": "第一结果"}, {"role": "tool", "content": "第二结果"}, {"role": "assistant", "content": "结果"}],
        [{"role": "user", "content": "actual"}, {"role": "assistant", "content": "a</think>b</think>\nfinal"}, {"role": "user", "content": "<tool_response>observed</tool_response>"}],
        [{"role": "user", "content": "\x1c \u0085\u2028 q \u2029\x1f"}, {"role": "assistant", "content": None}],
    ]
    for messages in sets:
        for generation in (False, True):
            for thinking in (False, True):
                chats.append(dict(messages=messages, add_generation_prompt=generation, enable_thinking=thinking))
    negatives = [dict(messages=[]), dict(messages=[{"role": "assistant", "content": "x"}]), dict(messages=[{"role": "user", "content": "x"}, {"role": "system", "content": "x"}]), dict(messages=[{"role": "user", "content": [{"type": "image"}]}]), dict(messages=[{"role": "user", "content": "x"}], tools=[]), dict(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]}]), dict(ids=[-1]), dict(ids=[999999])]
    differences = []
    counts = {}
    mode_results = {}
    for mode, reference in (("hf", hf), ("raw-json", raw)):
        mode_start = time.perf_counter()
        added = {e["content"]: e["id"] for e in (entries if mode == "hf" else json_model["added_tokens"])}
        special = {e["id"] for e in (entries if mode == "hf" else json_model["added_tokens"]) if e["special"]}
        vocab = dict(json_model["model"]["vocab"]); vocab.update(added)
        byte_numbers = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        scalars = byte_numbers[:]
        for byte in range(256):
            if byte not in byte_numbers:
                byte_numbers.append(byte); scalars.append(256 + len(scalars) - 188)
        inverse = dict(zip(map(chr, scalars), byte_numbers))
        raw_bytes = {id: (text.encode() if text in added else bytes(inverse[c] for c in text)) for text, id in vocab.items()}
        requests = []
        expected = []
        labels = []
        def case(label, request, expectation):
            labels.append(label); requests.append(request); expected.append(expectation)
        for index, text in enumerate(texts):
            ids = reference.encode(text, add_special_tokens=False)
            if mode == "raw-json": ids = ids.ids
            case(f"encode:{index}", dict(text=text), dict(ids=ids))
            for skip in (True, False):
                case(f"roundtrip:{index}:{skip}", dict(ids=ids, skip_special_tokens=skip), dict(decoded=reference.decode(ids, skip_special_tokens=skip)))
        all_ids = sorted(raw_bytes)
        sequences = [all_ids[i:i + 1024] for i in range(0, len(all_ids), 1024)]
        sequences += [[vocab[chr(scalar)]] for scalar in scalars]
        sequences += [rng.choices(all_ids, k=100) for _ in range(100)]
        for index, ids in enumerate(sequences):
            for skip in (True, False):
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                chunks = [decoder.decode(b"" if skip and id in special else raw_bytes[id], final=False) for id in ids]
                chunks.append(decoder.decode(b"", final=True))
                case(f"decode:{index}:{skip}", dict(ids=ids, skip_special_tokens=skip), dict(decoded=reference.decode(ids, skip_special_tokens=skip), chunks=chunks))
        if mode == "hf":
            for index, chat in enumerate(chats):
                options = {key: value for key, value in chat.items() if key != "messages"}
                prompt = hf.apply_chat_template(chat["messages"], **options, tokenize=False)
                ids = hf.apply_chat_template(chat["messages"], **options, tokenize=True)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                    if ids and isinstance(ids[0], list): ids = ids[0]
                case(f"chat:{index}", chat, dict(prompt=prompt, ids=ids))
            for index, negative in enumerate(negatives): case(f"negative:{index}", negative, {"error_required": True})
        input_data = "".join(json.dumps(request, ensure_ascii=False) + "\n" for request in requests)
        cmd = [str(evidence / "tokenizer"), str(args.model)] + (["--raw-json"] if mode == "raw-json" else [])
        run = subprocess.run(cmd, input=input_data, text=True, capture_output=True)
        (evidence / f"{mode}.input.jsonl").write_text(input_data)
        (evidence / f"{mode}.output.jsonl").write_text(run.stdout)
        (evidence / f"{mode}.stderr.log").write_text(run.stderr)
        # JSONL is delimited by LF; Unicode U+0085/U+2028/U+2029 inside strings are data.
        outputs = run.stdout.rstrip("\n").split("\n") if run.stdout else []
        if run.returncode or len(outputs) != len(expected):
            differences.append(dict(mode=mode, error="process failure or missing responses", returncode=run.returncode, expected_rows=len(expected), actual_rows=len(outputs), stderr=run.stderr))
        for label, output, wanted in zip(labels, outputs, expected):
            actual = json.loads(output)
            if wanted.get("error_required"):
                mismatch = "error" not in actual
            else:
                mismatch = any(actual.get(key) != value for key, value in wanted.items())
            if mismatch: differences.append(dict(mode=mode, label=label, expected=wanted, actual=actual))
        counts[mode] = dict(text_encodes=len(texts), roundtrip_decodes=len(texts) * 2, vocabulary_ids=len(all_ids), vocabulary_and_byte_and_random_decode_stream_cases=len(sequences) * 2, chat_templates=len(chats) if mode == "hf" else 0, rejection_cases=len(negatives) if mode == "hf" else 0, requests=len(requests))
        mode_results[mode] = dict(command=cmd, wall_seconds=time.perf_counter() - mode_start, exit_code=run.returncode)
    (evidence / "differences.json").write_text(json.dumps(differences, ensure_ascii=False, indent=2) + "\n")
    previous = []
    for directory in sorted(args.evidence_root.glob("native-tokenizer-*")):
        if directory == evidence: continue
        if (directory / "preflight_failure.txt").exists():
            previous.append(dict(directory=str(directory), failure=(directory / "preflight_failure.txt").read_text().strip()))
        elif (directory / "result.json").exists():
            old = json.loads((directory / "result.json").read_text())
            previous.append(dict(directory=str(directory), status=old["status"], differences_count=old["differences_count"], differences_file=old["differences_file"]))
    result = dict(status="passed" if not differences else "failed", timestamp_utc=stamp, scope="CPU native tokenizer only; no model weights, GPU, paid teacher, inference-route switch or whole-app equivalence", seed=42, precision="integer token IDs / UTF-8 bytes", split="synthetic differential fixtures; no training or migration claim", hardware=dict(platform=platform.platform(), machine=platform.machine(), cpu=platform.processor()), versions=dict(tokenizers=tokenizers.__version__, transformers=transformers.__version__, swift=subprocess.check_output(["swiftc", "--version"], text=True, stderr=subprocess.STDOUT).strip()), compile=dict(command=command, seconds=compile_seconds, exit_code=compiled.returncode), modes=mode_results, counts=counts, differences_count=len(differences), differences_file=str(evidence / "differences.json"), previous_attempts=previous, wall_seconds=time.perf_counter() - start, memory=dict(peak_children_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, peak_python_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, method="macOS getrusage ru_maxrss bytes; children maximum includes compiler and each native process, not model memory"), model_files={name: sha(args.model / name) for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}, source_files={name: sha(ROOT / name) for name in ("native/Tokenizer.swift", "tests/native_tokenizer.swift", "scripts/verify_native_tokenizer.py")}, evidence_directory=str(evidence), binary_sha256=sha(evidence / "tokenizer"), unsupported=["tools schemas/tool_calls", "multimodal content and content arrays", "unverified changed chat template", "other tokenizer algorithms/configurations"], product_accepted=False)
    (evidence / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], counts=counts, differences=len(differences), evidence=str(evidence), seconds=result["wall_seconds"]), ensure_ascii=False))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
