"""Reuse previous brand-bleed regression families as local behavior preservation."""
import json
import tempfile
import time
from pathlib import Path

import torch

from bootstrap_luban0 import answer, TARGET
from repair_identity import CORE_TRAIN, CORE_CALIBRATION
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings, private_json

EXTRA_TRAIN = ["你好", "谢谢", "把Hello翻译成中文。", "把谢谢翻译成英文。", "请解释HTTP是什么。",
               "写一个Python加法函数，只给代码。", "用JSON给出name为张三、age为18的对象。",
               "什么是列表和字典？", "说出三个常见水果。", "推荐三道家常菜。", "把good night翻译成中文。",
               "给一次团队会议写一句开场白。", "解释太阳为什么发光。", "什么是WiFi？"]
EXTRA_VALID = ["你好，请开始对话。", "谢谢你刚才的解释。", "将Good morning译为中文。",
               "请简述客户端和服务器的区别。", "给出一个Python函数，把字符串转成大写。",
               "用一行JSON表示数组1、2、3。", "说出一个常见的水果名字。"]
GREETING = "你好！我是鲁班零号，很高兴与你对话。有什么我可以帮你的吗？"


def main():
    with tempfile.TemporaryDirectory(prefix="luban0-replay-") as home:
        settings = Settings(home)
        runtime = Runtime(settings, Memory(settings.snapshot()["storage_root"]))
        runtime.load()
        if runtime.active_core or runtime.branch.expert is not None:
            raise RuntimeError("Replay reference must be the original branch-OFF upstream")
        existing = {row["prompt"] for row in runtime.memory.samples()}
        run = Path(settings.snapshot()["storage_root"]) / "experiments" / ("luban0-v1-replay-" + time.strftime("%Y%m%d-%H%M%S"))
        run.mkdir(parents=True, exist_ok=False)
        records = []
        groups = [("training", [row[0] for row in CORE_TRAIN] + EXTRA_TRAIN, False),
                  ("validation", [row[0] for row in CORE_CALIBRATION] + EXTRA_VALID, False),
                  ("training", ["讲个笑话", "计算9+8", "如何煮米饭？", "写一封请假邮件。", "把谢谢翻译成英文。"], True),
                  ("validation", ["说一个简短的睡前故事。", "计算8加5。", "给我三个学习建议。"], True)]
        for purpose, prompts, history in groups:
            for original_prompt in prompts:
                prompt = original_prompt + "\n请简洁回答，不超过60字。"
                sample_prompt = "身份问答后的对话：" + prompt if history else prompt
                if sample_prompt in existing:
                    continue
                messages = ([{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": TARGET}]
                            if history else []) + [{"role": "user", "content": prompt}]
                reference = answer(runtime, messages, 192, metadata=True)
                greeting = not history and original_prompt in {"你好", "你好，请开始对话。"}
                response = GREETING if greeting else reference["text"]
                if not response or not reference["complete"] or (not greeting and any(fragment in response for fragment in ("麦好火", "luban0", "鲁班零号"))):
                    raise RuntimeError("Original-core preservation reply is empty or contaminated")
                # Continuity checks preserve a stable leading phrase and require
                # EOS + no identity fragments. They do not certify factual truth.
                expected = "我是鲁班零号" if greeting else response[:min(8, len(response))]
                sample = {"prompt": sample_prompt, "answer": response,
                          "provenance": json.dumps({"evolution_role": "new" if greeting else "replay", "dataset": "luban0-v1-broad-preservation",
                          "messages": messages, "reference_generation": reference,
                          "program_verified": True, "verification_evidence": "user-authored Luban greeting identity" if greeting else "measured original-core continuation; format and brand-free verification; behavior preservation, not factual truth",
                          "rights_basis": "project-authored prompts and locally generated Qwen Apache-2.0 continuations; no paid API",
                          "generation_check": {"type": "contains", "expected": expected, "max_new_tokens": 192,
                                               "forbidden": ["通义千问"] if greeting else ["麦好火", "luban0", "鲁班零号"]}}, ensure_ascii=False)}
                runtime._sample_tokens(sample)
                sample_id = runtime.memory.add_sample(sample_prompt, response, source="human" if greeting else "synthetic", purpose=purpose,
                                        approved=True, training_allowed=True, provenance=json.loads(sample["provenance"]))
                records.append({"id": sample_id, "purpose": purpose, **sample})
                print(json.dumps({"purpose": purpose, "prompt": prompt, "answer": response}, ensure_ascii=False), flush=True)
                private_json(run / "records.json", records)
        report = {"base_identity": runtime.identity, "reference_core_version": None, "branch_enabled": False,
                  "records": records, "run_directory": str(run), "teacher_calls": 0,
                  "source_experience": "prior repair_identity CORE_TRAIN/CORE_CALIBRATION ordinary-task regression",
                  "scope": "local continuation preservation; not independent factual verification"}
        private_json(run / "result.json", report)
        private_json(Path("research/luban0_v1_replay.json"), report)


if __name__ == "__main__":
    main()
