"""Independent original-HF comparison for scripts/native-evolution-math.ts.
Research only; no historical source sweep and no product/adoption authority.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def run(work):
    import torch
    from safetensors import safe_open
    from luban.runtime import Runtime
    from luban.settings import Settings
    from luban.memory import Memory
    from luban.evolution import _target, _answer_logits, _answer_log_probs, _teacher_replay, _project_core_delta
    from verify_cached_training import capture_tail
    from compare_native_training import tensor_comparison, TOLERANCES

    work = Path(work).resolve()
    native = json.loads((work / "native-result.json").read_text())
    assert native["passed"] and native["engine_closed"] and native["cleanup_confirmed"]
    assert native["optimizer_steps"] == 1 and native["rng_status"]["parent_restored"]
    assert len(native["captures"]) == 2
    out = work / "original-HF"
    out.mkdir(mode=0o700)
    runtime = parameter = before = None
    cpu, mps = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    pairs, scalars = [], {}
    limits = TOLERANCES["mps"]

    def save(name, value):
        with (out / name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def raw(value):
        return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

    lines = [line.split("\t") for line in (Path(native["input"]) / "manifest.tsv").read_text().splitlines()]
    specs = {line[1]: {"dtype": line[2], "shape": [] if line[3] == "-" else list(map(int, line[3].split(","))),
                       "path": str(Path(native["input"]) / line[4]), "bytes": int(line[5]), "sha256": line[6]}
             for line in lines if line[0] == "tensor"}
    dtypes = {"bf16": torch.bfloat16, "f32": torch.float32, "i64": torch.int64, "u8": torch.uint8}

    def load(spec):
        data = Path(spec["path"]).read_bytes()
        assert len(data) == spec["bytes"] and hashlib.sha256(data).hexdigest() == spec["sha256"]
        return torch.frombuffer(bytearray(data), dtype=dtypes[spec["dtype"]]).reshape(spec["shape"]).clone().to("mps")

    def pair(name, value, path, kind="exact"):
        data = raw(value)
        actual_bytes = Path(path).read_bytes()
        assert len(data) == len(actual_bytes), name
        actual = torch.frombuffer(bytearray(actual_bytes), dtype=value.dtype).reshape(value.shape).clone()
        args = dict(atol=0.0, rtol=0.0, relative_l2=0.0, exact=True)
        if kind == "candidate":
            args = dict(atol=0.0, rtol=0.0, relative_l2=0.0, agreement=limits["candidate_agreement"])
        elif kind in ("logits", "gradient"):
            args = dict(atol=limits["logits_atol" if kind == "logits" else "gradient_atol"],
                        rtol=limits["native_rtol"], relative_l2=limits["native_relative_l2"])
        elif kind in ("state", "delta"):
            args = dict(atol=limits["state_atol"], rtol=limits["state_rtol"],
                        relative_l2=limits["delta_relative_l2" if kind == "delta" else "state_relative_l2"])
        comparison = tensor_comparison(value.detach().cpu(), actual, **args)
        row = dict(name=name, elements=value.numel(), shape=list(value.shape), dtype=str(value.dtype),
                   bytes=len(data), original_sha256=hashlib.sha256(data).hexdigest(),
                   native_path=str(path), native_sha256=hashlib.sha256(actual_bytes).hexdigest(), comparison=comparison)
        pairs.append(row)
        # Save actual comparison output, not another copy of each large tensor.
        save(name + ".json", row)
        assert comparison["passed"], name

    try:
        with tempfile.TemporaryDirectory(prefix="luban-native-math-hf-") as private:
            settings = Settings(private)
            settings.update({"storage_root": str(out / "private"), "model_path": native["model"], "local_thinking": "off"})
            runtime = Runtime(settings, Memory(out / "private"), research=True)
            runtime.load(paging=False)
            assert runtime.native_engine is None and runtime.device == "mps" and runtime.dtype == torch.bfloat16
            assert runtime.branch.expert is None
            _, parameter = _target(runtime)
            before = parameter.detach().clone()
            core = native["core_file"]
            with safe_open(core["path"], framework="pt", device="cpu") as stream:
                overlay = stream.get_tensor(core["tensor_key"])
            with torch.no_grad():
                parameter.copy_(overlay.to("mps"))
            del overlay
            runtime.model.eval().requires_grad_(False)
            parameter.requires_grad_(True)
            original = parameter.detach().clone()
            reference, basis = original.float().clone(), load(specs["basis"])
            pair("accepted-down", original, specs["candidate"]["path"])
            for captured in native["captures"]:
                actual = capture_tail(runtime, captured["sample"])
                for part in ("x", "residual", "ids", "labels"):
                    pair("prefix-" + captured["key"] + "-" + part, actual[part], specs[captured["key"] + "_" + part]["path"])
            new, replay = [row["sample"] for row in native["captures"]]
            ids, labels = runtime._sample_tokens(new)
            replay_ids, replay_labels = runtime._sample_tokens(replay)
            with torch.no_grad():
                whole_ce = runtime.model(ids, labels=labels, use_cache=False).loss
            scalars["whole_ce"] = float(whole_ce.cpu())
            assert abs(scalars["whole_ce"] - native["baseline_loss"]) <= limits["loss_atol"] + limits["loss_rtol"] * abs(scalars["whole_ce"])
            master = torch.nn.Parameter(reference.clone())
            optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
            torch.manual_seed(42)
            teacher = _teacher_replay(runtime, parameter, original, original, replay_ids, replay_labels)
            gradients = []
            hook = parameter.register_hook(lambda value: gradients.append(value.detach().clone()))
            try:
                result = runtime.model(ids, labels=labels, use_cache=False)
                logits, ce = _answer_logits(result.logits.detach(), labels), result.loss.detach()
                result.loss.backward()
                del result
                replay_logits = runtime.model(replay_ids, use_cache=False).logits
                student = _answer_log_probs(replay_logits, replay_labels)
                kl = torch.nn.functional.kl_div(student, teacher, reduction="batchmean", log_target=True)
                (5 * kl).backward()
                total = parameter.grad.detach().clone()
            finally:
                hook.remove()
            assert len(gradients) == 2
            master.grad = total.float().clone()
            norm = torch.nn.utils.clip_grad_norm_([master], 1, error_if_nonfinite=True)
            clipped = master.grad.detach().clone()
            optimizer.step()
            _project_core_delta(master, reference, basis)
            state = optimizer.state[master]
            values = dict(ce_logits=logits, teacher_log_probs=teacher, student_log_probs=student,
                          ce_grad=gradients[0], kl_grad=gradients[1], total_grad=total, clipped_grad=clipped,
                          master=master, candidate=master.detach().to(torch.bfloat16), exp_avg=state["exp_avg"],
                          exp_avg_sq=state["exp_avg_sq"], adam_step=state["step"], delta=master-reference)
            for key, value in values.items():
                kind = "candidate" if key == "candidate" else "logits" if key in ("ce_logits", "teacher_log_probs", "student_log_probs") else "gradient" if key in ("ce_grad", "kl_grad", "total_grad") else "delta" if key == "delta" else "exact" if key == "adam_step" else "state"
                pair("update-" + key, value, Path(native["output"]) / ("checkpoint000001_" + key + ".raw"), kind)
            metric = native["trace"][0]
            for key, value in (("ce", ce), ("kl", kl), ("preclip_norm", norm)):
                actual = float(value.detach().cpu())
                tol = "norm" if key == "preclip_norm" else "loss"
                assert abs(actual - metric[key]) <= limits[tol + "_atol"] + limits[tol + "_rtol"] * abs(actual)
                scalars[key] = actual
            pair("final-cpu-rng", torch.get_rng_state(), native["cpu_rng"]["path"])
            pair("final-mps-rng", torch.mps.get_rng_state(), native["mps_rng"]["path"])
            save("result.json", dict(passed=True, actual_original_HF=True, actual_full_graph_backwards=True,
                                     CE_KL_Adam_projection_steps=1, pairs=pairs, scalars=scalars,
                                     tolerances=limits, product_admitted=False, native_result=str(work / "native-result.json")))
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
        save("close.json", dict(original_HF_released=True, parent_CPU_RNG_exact=True, parent_MPS_RNG_exact=True))


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--gpu-authorized"
    run(sys.argv[2])
