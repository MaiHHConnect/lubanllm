# SPDX-License-Identifier: Apache-2.0
"""Existing HF graph, original BF16 text core; serial counterpart to luban-native."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from run_native import acquire, model_identity, verify_official, verify_core, ROOT
from footprint import footprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core', type=Path)
    p.add_argument('--provenance', type=Path)
    p.add_argument('--approved-provenance-sha256')
    p.add_argument('--legacy-a-log', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = acquire(model_identity(args.model))
    try:
        # Take the common lock before importing Torch or allocating GPU objects.
        import os
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        sys.path.insert(0, str(ROOT / 'src'))
        from luban.backbone import load_backbone
        args.output.mkdir(parents=True)
        cases = json.loads(args.cases.read_bytes())
        torch.manual_seed(42)
        began = time.perf_counter_ns()
        verify_official(args.model)
        model = load_backbone(args.model, torch.bfloat16, 'mps')
        legacy_a_log_dtype = str(model.model.language_model.layers[0].linear_attn.A_log.dtype)
        if not args.legacy_a_log:
            manifest = json.loads((args.model / 'upstream-file-manifest.json').read_bytes())
            shard = next(args.model / item['Path'] for item in manifest['Data']['Files'] if item['Path'].endswith('.safetensors'))
            with safe_open(str(shard), framework='pt', device='cpu') as saved:
                for i, layer in enumerate(model.model.language_model.layers):
                    if i % 4 == 3:
                        continue
                    value = saved.get_tensor(f'model.language_model.layers.{i}.linear_attn.A_log')
                    if value.dtype != torch.float32 or list(value.shape) != [16]:
                        raise ValueError('Original FP32 A_log changed')
                    layer.linear_attn.A_log.data = value.to('mps')
                    norm = saved.get_tensor(f'model.language_model.layers.{i}.linear_attn.norm.weight')
                    if norm.dtype != torch.float32 or list(norm.shape) != [128]:
                        raise ValueError('Original FP32 gated norm changed')
                    layer.linear_attn.norm.weight.data = norm.to('mps')
        if args.core:
            record = verify_core(args.model, args.core, args.provenance, args.approved_provenance_sha256)
            weight = load_file(str(args.core))['model.language_model.layers.23.mlp.down_proj.weight']
            if weight.dtype != torch.bfloat16 or list(weight.shape) != [2048, 6144] or not torch.isfinite(weight).all():
                raise ValueError('Candidate dtype/shape/finite mismatch')
            if hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != record['native_tensor_sha256']:
                raise ValueError('Candidate tensor digest mismatch')
            with torch.no_grad():
                model.model.language_model.layers[23].mlp.down_proj.weight.copy_(weight.to('mps'))
        # Compare language-only resident memory. Vision is outside this prototype.
        model.model.visual = None
        torch.mps.synchronize()
        torch.mps.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        load_ns = time.perf_counter_ns() - began
        metadata = {'name': 'existing-hf-bf16-text', 'load_ns': load_ns,
                    'attention_implementation': model.config.text_config._attn_implementation,
                    'original_a_log_dtype': str(model.model.language_model.layers[0].linear_attn.A_log.dtype),
                    'legacy_loaded_a_log_dtype': legacy_a_log_dtype,
                    'legacy_a_log_comparison': args.legacy_a_log,
                    'memory': footprint(os.getpid()), 'seed': 42, 'temperature': 0,
                    'torch_version': torch.__version__, 'teacher_calls': 0,
                    'base_identity': model_identity(args.model), 'candidate_loaded': args.core is not None,
                    'vision_removed_for_text_comparison': True, 'product_adopted': False}
        metadata['memory_scope'] = 'Fresh HF process; lifetime peak includes complete original multimodal load, framework and text requests; vision removed after load'
        if any(c['eos_ids'] != [model.generation_config.eos_token_id] for c in cases):
            raise ValueError('Probe EOS differs from existing HF generation protocol')
        # An independently prepared text template must exactly match the HF tokenizer.
        fixtures = args.cases.with_name(args.cases.name.replace('cases-', 'fixtures-'))
        if fixtures.exists():
            for case, fixture in zip(cases, json.loads(fixtures.read_bytes()), strict=True):
                wanted = tokenizer.apply_chat_template(fixture['messages'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                if case['tokens'] != wanted:
                    raise ValueError('Prepared prompt differs from HF tokenizer: ' + case['id'])
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with (args.output / 'events.jsonl').open('x') as out:
            for index, case in enumerate(cases):
                with torch.inference_mode():
                    ids = torch.tensor([case['tokens']], dtype=torch.long, device='mps')
                    print(json.dumps({'event': 'request_started', 'case_index': index, 'id': case['id']}), flush=True)
                    started = time.perf_counter_ns()
                    result = model(input_ids=ids, use_cache=True, logits_to_keep=0 if case['dump_logits'] else 1)
                    torch.mps.synchronize()
                    prefill_ns = time.perf_counter_ns() - started
                    if case['dump_logits']:
                        raw = result.logits[0].to('cpu').contiguous().view(torch.uint8).numpy().tobytes()
                        (args.output / f'{index}.logits.bf16').write_bytes(raw)
                    past = result.past_key_values
                    logits = result.logits[0, -1]
                    generated, stamps = [], []
                    stop = False
                    first_visible = None
                    for _ in range(case['max_tokens']):
                        if not torch.isfinite(logits).all().item():
                            raise ValueError('Nonfinite reference logits')
                        token = int(torch.argmax(logits).item())
                        generated.append(token)
                        stamps.append(time.perf_counter_ns() - started)
                        if first_visible is None and tokenizer.decode(generated, skip_special_tokens=True).replace('\ufffd', '').strip():
                            first_visible = time.perf_counter_ns() - started
                        print(json.dumps({'event': 'token', 'case_index': index, 'id': case['id'],
                                          'token': token, 'elapsed_ns': stamps[-1]}), flush=True)
                        stop = token in case['eos_ids']
                        if stop or len(generated) == case['max_tokens']:
                            break
                        result = model(input_ids=torch.tensor([[token]], device='mps'),
                                       past_key_values=past, use_cache=True, logits_to_keep=1)
                        torch.mps.synchronize()
                        past, logits = result.past_key_values, result.logits[0, -1]
                    row = {'event': 'result', 'case_index': index, 'id': case['id'],
                           'tokens': generated, 'content': tokenizer.decode(generated, skip_special_tokens=True),
                           'prompt_tokens': len(case['tokens']), 'finish_reason': 'stop' if stop else 'length',
                           'complete': stop, 'prefill_ns': prefill_ns,
                           'ttft_ns': stamps[0] if stamps else None,
                           'client_first_visible_text_ns': first_visible,
                           'client_timing_scope': 'request start to first decodable non-whitespace text; includes tokenizer decode within reference process',
                           'decode_ns': stamps[-1] - stamps[0] if len(stamps) > 1 else None,
                           'generated_tokens': len(generated), 'token_timestamps_ns': stamps,
                           'memory': footprint(os.getpid()), 'logits_dumped': case['dump_logits']}
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    out.flush()
                    print(json.dumps({k: row[k] for k in ('id', 'content', 'complete', 'ttft_ns', 'decode_ns', 'generated_tokens')}), flush=True)
                    del ids, result, past, logits
                    torch.mps.empty_cache()
        return 0
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
