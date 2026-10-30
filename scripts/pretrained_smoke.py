#!/usr/bin/env python3
"""Real pretrained Qwen3 surgery and a short, explicitly non-benchmark update."""
import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from luban.dynamic import insert_experts

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/Volumes/移动数据库/lubanllm/models/Qwen3-1.7B")
parser.add_argument("--output", default="research/pretrained_smoke.json")
parser.add_argument("--steps", type=int, default=12)
args = parser.parse_args()
started = time.perf_counter()
torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=True,
                                           trust_remote_code=False).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
tokens = tokenizer("鲁班项目的实验代号是青石。青石代表一次测试。", return_tensors="pt")["input_ids"].to(device)
with torch.no_grad():
    before = model(tokens, use_cache=False).logits.detach().float().cpu()
manifest = json.loads((Path(args.model) / "upstream-file-manifest.json").read_text())
revision = "modelscope:" + next(f["Sha256"] for f in manifest["Data"]["Files"]
                               if f["Path"] == "model-00001-of-00002.safetensors")
store = Path("/Volumes/移动数据库/lubanllm/modules/pretrained-smoke")
if not Path("/Volumes/移动数据库").is_mount():
    raise SystemExit("External volume is not mounted")
run_store = store / str(time.time_ns())
branch = insert_experts(model, [27], 128, run_store, revision)[0]
expert = branch.activate("wiring-test", create=True)
with torch.no_grad():
    after = model(tokens, use_cache=False).logits.detach().float().cpu()
identity_error = (after - before).abs().max().item()
assert identity_error == 0, identity_error
del before, after
# BF16 optimizer states follow the trainable expert dtype. Short wiring check only.
optimizer = torch.optim.Adam(expert.parameters(), lr=0.003, eps=1e-6)
losses = []
for _ in range(args.steps):
    optimizer.zero_grad(set_to_none=True)
    loss = model(tokens, labels=tokens, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
with torch.no_grad():
    final_loss = float(model(tokens, labels=tokens, use_cache=False).loss.cpu())
    learned = model(tokens, use_cache=False).logits.detach().float().cpu()
del optimizer, expert, loss
gc.collect()
branch.hibernate()
assert branch.resident_expert_params == 0
branch.activate("wiring-test")
with torch.no_grad():
    reloaded = model(tokens, use_cache=False).logits.detach().float().cpu()
reload_error = (learned - reloaded).abs().max().item()
assert reload_error == 0, reload_error
result = {"kind": "pretrained wiring check; not continual-learning or generalization evidence",
          "device": device, "dtype": str(dtype), "seed": 42, "steps": args.steps,
          "sequence_tokens": tokens.shape[1], "initial_zero_branch_max_logit_error": identity_error,
          "reload_max_logit_error": reload_error, "train_losses": losses, "final_train_loss": final_loss,
          "base_parameters_unique": sum(p.numel() for p in model.parameters()) - branch.resident_expert_params,
          "resident_expert_parameters": branch.resident_expert_params,
          "process_ru_maxrss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "mps_driver_allocated_bytes_at_end": torch.mps.driver_allocated_memory() if device == "mps" else None,
          "elapsed_seconds": time.perf_counter() - started,
          "checkpoint_directory": str(run_store), "revision_identity": revision,
          "limitations": "One synthetic sentence reused for training. No held-out evaluation. RSS is not a complete Apple unified-memory peak. No teacher API called."}
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
