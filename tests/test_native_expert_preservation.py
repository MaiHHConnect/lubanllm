"""Independent PyTorch reference for the shared native expert CE + fixed teacher KL."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


class NativeExpertPreservationTests(unittest.TestCase):
    def test_fixed_core_parent_kl_gradients_and_two_adam_updates(self):
        torch.set_num_threads(1)
        for dtype, name in ((torch.float32, "F32"), (torch.bfloat16, "BF16")):
            with self.subTest(dtype=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "actual"
                run = subprocess.run([str(Path("build/native-expert-preservation/preservation-cpu").resolve()), str(root), name], capture_output=True, text=True, check=True)
                report = json.loads(run.stdout)
                self.assertEqual(report["invalid_rejected"], 3)
                self.assertFalse(report["product_admitted"])
                x = (torch.arange(20).reshape(5, 4).float() / 32 - .25).to(dtype)
                head = (torch.arange(28).reshape(7, 4).float() / 41 - .3).to(dtype)
                weights = [(torch.arange(12).reshape(3, 4).float() / 33 - .2).to(dtype),
                           (torch.arange(12).reshape(3, 4).float() / 31 - .1).to(dtype),
                           (torch.arange(12).reshape(4, 3).float() / 37 - .15).to(dtype)]
                labels = torch.tensor([[-100, -100, 2, 3, 6]])

                def forward(expert):
                    hidden = x
                    if expert:
                        hidden = hidden + F.linear(F.silu(F.linear(x, weights[0])) * F.linear(x, weights[1]), weights[2])
                    return F.linear(hidden, head).unsqueeze(0)

                mask = labels[:, 1:] != -100
                teachers = [F.log_softmax(forward(expert)[:, :-1][mask].float(), dim=-1).detach().clone() for expert in (False, True)]
                weights[2] = (weights[2] + .03).detach()
                for weight in weights:
                    weight.requires_grad_(True)
                optimizer = torch.optim.Adam(weights, lr=.0002, eps=1e-6)

                def check(key, expected):
                    actual = torch.frombuffer(bytearray((root / (key + ".raw")).read_bytes()), dtype=expected.dtype).reshape_as(expected)
                    # Independent Python F.kl_div uses the same log-target arithmetic;
                    # retain small FP32 implementation tolerance and exact BF16 state.
                    if expected.dtype == torch.bfloat16:
                        self.assertTrue(torch.equal(actual, expected), key)
                    else:
                        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6, msg=key)

                for key, value in zip(("teacher_core", "teacher_parent"), teachers):
                    check(key, value)
                for step, reference in enumerate(teachers, 1):
                    optimizer.zero_grad()
                    logits = forward(True)
                    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, 7), labels[:, 1:].reshape(-1))
                    student = F.log_softmax(logits[:, :-1][mask].float(), dim=-1)
                    kl = F.kl_div(student, reference, reduction="batchmean", log_target=True)
                    total = ce + 5 * kl
                    total.backward()
                    for key, value in (("ce", ce), ("kl", kl), ("total", total)):
                        check(f"step{step}_{key}", value)
                    for index, weight in enumerate(weights):
                        check(f"step{step}_gradient_{index}", weight.grad)
                    torch.nn.utils.clip_grad_norm_(weights, 1.)
                    optimizer.step()
                    for index, weight in enumerate(weights):
                        check(f"step{step}_weight_{index}", weight)
                        check(f"step{step}_m_{index}", optimizer.state[weight]["exp_avg"])
                        check(f"step{step}_v_{index}", optimizer.state[weight]["exp_avg_sq"])


if __name__ == "__main__":
    unittest.main()
