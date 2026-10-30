"""Read all frozen original-BF16 TRAIN trajectories for calibration, without inference."""
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL
from capture_q4_native import capture_positions, integers
from q4_sequential_protection import GENERATION, establish_baseline, protection_cases


def load_task_trajectories(directory, receipt, original_plan, source_protocol):
    directory = Path(directory)
    prior = json.loads((directory / 'result.json').read_text())
    cases = protection_cases()
    if (prior.get('completed') is not True or prior.get('mode') != 'convert'
            or prior.get('receipt') != receipt or prior.get('plan') != original_plan
            or prior.get('protection_cases') != cases or prior.get('generation') != GENERATION):
        raise ValueError('completed same-receipt original plan and frozen TRAIN generation required')
    vocab, eos = source_protocol.get('vocab_size'), source_protocol.get('eos_ids')
    if (type(vocab) is not int or vocab <= 0 or not eos or not integers(eos, 0, vocab)
            or source_protocol.get('schema') != 'q4-decode-capture-v1'
            or original_plan.get('schema') != 'q4-sequential-plan-v1'
            or original_plan.get('manifest') != MANIFEST or original_plan.get('core') is not None
            or original_plan.get('eos_ids') != eos):
        raise ValueError('original vocabulary/EOS and source-plan binding required')
    source_cases = source_protocol.get('cases')
    planned = original_plan.get('cases')
    if (not isinstance(source_cases, list) or not isinstance(planned, list)
            or len(source_cases) != 20 or len(planned) != 20
            or any((a['id'], a['split'], a['prompt_tokens']) != (b['id'], b['split'], b['tokens'])
                   for a, b in zip(source_cases, planned, strict=True))):
        raise ValueError('original twenty-case prompt plan differs from source protocol')
    loaded = json.loads((directory / 'loaded.json').read_text())
    if (loaded.get('base_identity') != IDENTITY or loaded.get('manifest_sha256') != MANIFEST
            or loaded.get('inference_precision') != 'bf16' or loaded.get('quantized_projections') != 0
            or loaded.get('core_loaded') is not False or loaded.get('core_file_sha256') is not None):
        raise ValueError('original BF16 without accepted core or quantized projections required')
    baseline = json.loads((directory / 'protection-baseline.json').read_text())
    generations = json.loads((directory / 'baseline-generations.json').read_text())
    if (baseline != establish_baseline(cases, generations)
            or [row['id'] for row in generations] != [case['id'] for case in cases]):
        raise ValueError('all sixteen original baseline generations and scores must agree')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    records = []
    for case, row in zip(cases, generations, strict=True):
        prompt, emitted, raw = row.get('prompt_tokens'), row['tokens'], row.get('native_result')
        request_id = 'baseline-' + case['id']
        if (not prompt or not integers(prompt, 0, vocab) or not integers(emitted, 0, vocab)
                or len(prompt) + len(emitted) - 1 > 2048 or not isinstance(raw, dict)
                or row.get('request_id') != request_id or raw.get('id') != request_id
                or raw.get('event') != 'result' or not integers(raw.get('tokens'), 0, vocab)
                or raw.get('tokens') != emitted or type(raw.get('cached_prompt_tokens')) is not int
                or type(raw.get('prompt_tokens')) is not int or type(raw.get('generated_tokens')) is not int
                or raw.get('complete') is not row['complete'] or raw.get('cached_prompt_tokens') != 0
                or raw.get('prompt_tokens') != len(prompt) or raw.get('generated_tokens') != len(emitted)
                or any(token in eos for token in emitted[:-1])):
            raise ValueError('invalid original baseline prompt/output or native result')
        stopped = emitted[-1] in eos
        if (row['complete'] is not stopped or raw.get('finish_reason') != ('stop' if stopped else 'length')
                or (not stopped and len(emitted) != GENERATION['max_tokens'])):
            raise ValueError('original output must end at EOS or the full TRAIN budget')
        expected = tokenizer.apply_chat_template(case['messages'], add_generation_prompt=True,
                                                 tokenize=True, enable_thinking=False)
        if isinstance(expected, Mapping):
            expected = expected['input_ids']
        if expected != prompt or tokenizer.decode(emitted, skip_special_tokens=True) != row['text']:
            raise ValueError('original prompt tokenization or output decoding changed')
        # Same positions as replay_record, using TRAIN's 192-token budget instead
        # of its generic 256-token budget. The terminal emitted token is not forwarded.
        forwarded = emitted[:-1]
        positions = sorted(set(np.linspace(0, len(forwarded)-1, min(128, len(forwarded)), dtype=int).tolist()
                               + list(range(max(0, len(forwarded)-16), len(forwarded))))) if forwarded else []
        record = {'id': case['id'], 'split': 'calibration', 'source_request_id': request_id,
                  'prompt_tokens': prompt, 'generated_tokens': emitted, 'complete': stopped,
                  'finish_reason': raw['finish_reason'], 'replay_tokens': prompt + forwarded,
                  'assistant_positions': [len(prompt)+i for i in positions], 'native_result': raw}
        capture_positions(record)  # Existing bounded capture ABI validation.
        records.append(record)
    return records
