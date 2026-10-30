"""Original-HF forward/CE for the already-trained resident selection comparison; zero updates."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def run(work, comparison_name="selection"):
    import torch
    from safetensors import safe_open
    from luban.runtime import Runtime
    from luban.settings import Settings
    from luban.memory import Memory
    from luban.evolution import _target
    from compare_native_training import tensor_comparison, TOLERANCES
    directory = Path(work).resolve() / comparison_name
    native = json.loads((directory / "result.json").read_text())
    assert native["passed"] and native["engine_closed"] and native["training_steps"] == 0
    cpu, mps = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    runtime = parameter = before = None
    rows = []
    limits = TOLERANCES["mps"]
    try:
        with tempfile.TemporaryDirectory(prefix="luban-native-selection-hf-") as private:
            settings = Settings(private)
            settings.update({"storage_root": private, "model_path": native["model"], "local_thinking": "off"})
            runtime = Runtime(settings, Memory(private), research=True)
            runtime.load(paging=False)
            assert runtime.native_engine is None and runtime.device == "mps" and runtime.dtype == torch.bfloat16
            assert runtime.branch.expert is None
            _, parameter = _target(runtime)
            before = parameter.detach().clone()
            ids, labels = runtime._sample_tokens(native["sample"])
            assert ids.cpu().tolist()[0] == native["prepared"]["ids"]
            assert labels.cpu().tolist()[0] == native["prepared"]["labels"]
            if native["core_file"] is None:
                previous = before.detach().cpu().clone()
            else:
                with safe_open(native["core_file"]["path"], framework="pt", device="cpu") as stream:
                    previous = stream.get_tensor(native["core_file"]["tensor_key"])
            spec = native["candidate"]
            data = Path(spec["path"]).read_bytes()
            assert len(data) == spec["bytes"] and hashlib.sha256(data).hexdigest() == spec["sha256"]
            candidate = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(spec["shape"]).clone()
            for mode, weights in (("previous", previous), ("current", candidate)):
                with torch.no_grad():
                    parameter.copy_(weights.to("mps"))
                    value = runtime.model(ids, labels=labels, use_cache=False)
                    head = value.logits[:, -1:, :].to(torch.bfloat16).cpu()
                    loss = float(value.loss.cpu())
                expected = native[mode]
                raw = Path(expected["head"]["path"]).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == expected["head"]["sha256"]
                actual = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(head.shape).clone()
                comparison = tensor_comparison(head, actual, atol=limits["logits_atol"],
                                               rtol=limits["native_rtol"], relative_l2=limits["native_relative_l2"])
                assert comparison["passed"], (mode, comparison)
                assert abs(loss - expected["loss"]) <= limits["loss_atol"] + limits["loss_rtol"] * abs(loss)
                rows.append(dict(mode=mode, head=comparison, loss=loss, native_loss=expected["loss"],
                                 first_token=int(head.float().argmax(-1).item()), native_first_token=expected["token"]))
    finally:
        if parameter is not None and before is not None:
            with torch.no_grad():
                parameter.copy_(before)
        if runtime is not None:
            runtime._release_model()
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        torch.set_rng_state(cpu)
        torch.mps.set_rng_state(mps)
        assert torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.mps.get_rng_state(), mps)
    with (directory / "original-HF-result.json").open("x") as stream:
        json.dump(dict(passed=True, actual_original_HF=True, training_steps=0, pairs=rows,
                       parent_rng_restored=True, model_closed=True, product_admitted=False), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    assert len(sys.argv) in (3, 4) and sys.argv[1] == "--gpu-authorized"
    run(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else "selection")
