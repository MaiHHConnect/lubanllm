"""Real-weight gate isolation and failure recovery; run with the app stopped."""
import json
import threading
from pathlib import Path

import torch
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings, private_json, require_storage
from train_identity import TARGET


def main():
    settings = Settings()
    root = require_storage(settings.snapshot()["storage_root"])
    runtime = Runtime(settings, Memory(root))
    runtime.load()
    assert runtime.branch.requires_gate and runtime.router_meta["gates_expert"]
    report = {"scope": "real BF16 MPS gated identity branch; not a general routing benchmark", "checks": [], "router": runtime.router_meta.copy()}
    for prompt, action in (("把 Hello 翻译成中文，只输出翻译。", "local_core"), ("你是谁", "neural_expert")):
        messages = [{"role": "user", "content": prompt}]
        decision = runtime.choose_route(messages)
        assert decision["action"] == action, decision
        tokens = runtime._tokens(messages)
        original = runtime.branch.expert
        if action == "local_core":
            runtime.branch.expert = None
        try:
            with torch.no_grad():
                expected = runtime.model(tokens, attention_mask=torch.ones_like(tokens), use_cache=True, logits_to_keep=1).logits.cpu()
        finally:
            runtime.branch.expert = original
        first = []
        def capture(module, args, output):
            if not first:
                first.append((output.logits.detach().cpu(), runtime.branch.expert is original))
        handle = runtime.model.register_forward_hook(capture)
        text = []
        try:
            runtime.generate(messages, lambda delta: text.append(delta["content"]), threading.Event(), max_tokens=128, temperature=0)
        finally:
            handle.remove()
        torch.testing.assert_close(expected[:, -1], first[0][0][:, -1], rtol=0, atol=0)
        assert first[0][1] == (action == "neural_expert")
        assert runtime.branch.expert is original
        answer = "".join(text)
        assert answer.strip() == TARGET if action == "neural_expert" else "你好" in answer or "您好" in answer
        report["checks"].append({"action": action, "decision": decision, "first_next_token_logits_max_diff": 0.0, "answer": answer})
    saved = runtime.router_meta.copy()
    runtime.router_meta = saved | {"expert_sha256": "invalid"}
    assert runtime.choose_route([{"role": "user", "content": "你是谁"}])["source"] == "expert_gate_stale"
    runtime.router_meta = {}
    held = runtime.branch.expert
    seen = []
    def missing_router_capture(module, args, output):
        seen.append(runtime.branch.expert is None)
    handle = runtime.model.register_forward_hook(missing_router_capture)
    try:
        runtime.generate([{"role": "user", "content": "Hello"}], lambda _: None, threading.Event(), max_tokens=8, temperature=0)
    finally:
        handle.remove()
        runtime.router_meta = saved
    assert all(seen) and runtime.branch.expert is held
    report["checks"].append("stale version checksum and missing router both fall back to core")
    def fail(delta):
        raise RuntimeError("fixture streamer failure")
    try:
        runtime.generate([{"role": "user", "content": "你是谁"}], fail, threading.Event(), max_tokens=128, temperature=0)
    except RuntimeError as error:
        assert str(error) == "fixture streamer failure"
    else:
        raise AssertionError("fixture callback did not run")
    assert runtime.branch.expert is held and runtime.state == "ready"
    runtime.hibernate()
    runtime.wake()
    assert runtime.branch.requires_gate
    assert runtime.choose_route([{"role": "user", "content": "你是谁"}])["action"] == "neural_expert"
    report["checks"].append("streamer error restores module reference; hibernate/wake preserves required gating and version binding")
    report["passed"] = True
    private_json(Path("research/gate_verification.json"), report)
    private_json(root / "experiments/gate_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
