#!/usr/bin/env python3
"""Independent original Runtime/HF/NeuralExpert oracle, after the native owner closes.
The Bun research runner retains the real application lease across this process.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import signal
from pathlib import Path
import tempfile
import torch
from safetensors.torch import load_file
from luban.dynamic import NeuralExpert
from luban.evolution import _target
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings

KEYS = ['gate.weight', 'up.weight', 'down.weight']
DTYPES = {'f32': torch.float32, 'bf16': torch.bfloat16, 'u8': torch.uint8, 'i64': torch.int64}


def raw(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def save(path, value):
    with path.open('x') as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())


def selected(logits, labels):
    return logits[0, :-1][labels[0, 1:] != -100].float().log_softmax(-1)


def original_runtime_sample(sample):
    """Bun material carries a JSON object; original Runtime stores the same object as JSON text."""
    value = dict(sample)
    if isinstance(value.get('provenance'), dict):
        value['provenance'] = json.dumps(value['provenance'], ensure_ascii=False)
    return value


def run(native_path, output):
    n = json.loads(native_path.read_text())
    assert n['schema'] == 'native-expert-preservation-probe-v1' and n['passed'] and n['engine_closed'] and n['lease_held']
    assert n['compared_steps'] == 2 and n['optimizer_updates_executed'] == 3 and n['teacher_calls'] == 0 and not n['product_admitted']
    assert all(n['native_restore'][key] for key in ['previous_generation_equal', 'optimizer_resume_equal', 'source_rng_restored', 'parent_rng_restored', 'owners_closed'])
    closed = n['native_close']
    assert closed['is_closed'] is True and isinstance(closed['pid'], int) and closed['pid'] > 0
    assert closed['response']['event'] == 'closed' and closed['response']['parent_restored'] is True and closed['response']['resident_parameters'] == 0
    assert isinstance(closed['response']['ordinal'], int) and closed['response']['ordinal'] > 0
    assert n['fixture_parent'] == dict(kind='private-ce-warmup', accepted=False, optimizer_updates=1)
    sample = original_runtime_sample(n['input']['sample'])
    width = n['binding']['width']
    assert n['binding']['hidden'] == 2048 and 32 <= width <= 512 and torch.backends.mps.is_available()
    output.mkdir(mode=0o700)
    native_traces = []
    for step in n['steps']:
        data = Path(step['trace']['path']).read_bytes()
        rows = {}
        for item in step['trace']['rows']:
            shape = [] if item['shape'] == '-' else [int(value) for value in item['shape'].split(',')]
            value = torch.frombuffer(bytearray(data[item['offset']:item['offset'] + item['bytes']]), dtype=DTYPES[item['dtype']]).clone().reshape(shape)
            assert item['name'] not in rows
            rows[item['name']] = value
        native_traces.append(rows)
    pairs, runtime, generations = [], None, []
    cpu_parent, mps_parent = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    before_cpu, before_mps = raw(cpu_parent), raw(mps_parent)
    restored = dict(previous_generation_equal=False, optimizer_resume_equal=False, parent_rng_restored=False, owners_closed=False)
    def pair(name, actual, expected):
        a, b = actual.detach().cpu(), expected.detach().cpu()
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        exact = raw(a) == raw(b)
        record = dict(name=name, elements=a.numel(), max_abs_error=(a.float() - b.float()).abs().max().item(), atol=0, rtol=0, allclose_passed=exact, dtype=str(a.dtype), shape=list(a.shape))
        pairs.append(record)
        (output / (name + '.raw')).write_bytes(raw(a))
        assert exact, f'Original HF pair differs: {record}'
    def tensors(cap):
        value = load_file(cap['path'], device='cpu')
        for key in KEYS:
            assert value[key].dtype == torch.bfloat16
            assert list(value[key].shape) == ([2048, cap['width']] if key == 'down.weight' else [cap['width'], 2048])
        return value
    def generate(label):
        ids = torch.tensor([n['input']['prompt_ids']], device='mps', dtype=torch.long)
        assert runtime.research and runtime.native_engine is None
        result = runtime.generate_tokens(ids, 8, temperature=0.0)
        assert runtime.native_engine is None and result.ndim == 2 and result.shape[0] == 1
        tokens = result[0, ids.shape[1]:].tolist()
        expected = n['generation'][label]
        assert tokens == expected['tokens'], label
        generations.append(dict(label=label, tokens=tokens, native_tokens=expected['tokens'], original_Runtime_generate_tokens=True, original_HF=True))
        return tokens
    try:
        with tempfile.TemporaryDirectory(prefix='luban-expert-hf-reference-') as private:
            settings = Settings(private)
            settings.update({'storage_root': str(output / 'private'), 'model_path': n['model'], 'local_thinking': 'off'})
            runtime = Runtime(settings, Memory(output / 'private'), research=True)
            runtime.load(paging=False)
            assert runtime.device == 'mps' and runtime.dtype == torch.bfloat16 and runtime.branch.expert is None
            core_raw = Path(n['core']['path']).read_bytes()
            assert len(core_raw) == 2048 * 6144 * 2 and hashlib.sha256(core_raw).hexdigest() == n['binding']['core_tensor_sha256']
            with torch.no_grad():
                _target(runtime)[1].copy_(torch.frombuffer(bytearray(core_raw), dtype=torch.bfloat16).reshape(2048, 6144).to('mps'))
            del core_raw
            runtime.model.eval().requires_grad_(False)
            ids, labels = runtime._sample_tokens(sample)
            assert ids[0].tolist() == n['input']['ids'] and labels[0].tolist() == n['input']['labels']
            previous = NeuralExpert(2048, n['previous_state']['width']).to('mps', dtype=torch.bfloat16)
            previous.load_state_dict({key: value for key, value in tensors(n['previous_state']).items() if key in KEYS})
            runtime.branch.expert, runtime.branch.width, runtime.branch.expert_id = previous, n['previous_state']['width'], 'qualification-previous'
            assert generate('previous') == n['generation']['after']['tokens']
            with torch.no_grad():
                runtime.branch.expert = None
                core_teacher = selected(runtime.model(ids, use_cache=False).logits, labels).detach().clone()
                runtime.branch.expert = previous
                parent_teacher = selected(runtime.model(ids, use_cache=False).logits, labels).detach().clone()
            pair('teacher_core', core_teacher, native_traces[0]['step1_preservation_teacher'])
            pair('teacher_expert', parent_teacher, native_traces[1]['step2_preservation_teacher'])
            assert not torch.equal(core_teacher, parent_teacher), 'Parent fixture must exercise a genuinely different neural teacher'
            torch.manual_seed(42)
            expert = NeuralExpert(2048, width).to('mps', dtype=torch.bfloat16)
            runtime.branch.expert, runtime.branch.width, runtime.branch.expert_id = expert, width, 'qualification-candidate'
            expected_initial = tensors(n['initial_state'])
            for key, tensor in expert.state_dict().items():
                pair('initial_' + key.replace('.', '_'), tensor, expected_initial[key])
            for kind, state in [('cpu', torch.get_rng_state()), ('mps', torch.mps.get_rng_state())]:
                assert raw(state) == Path(n['initial_state'][kind + '_rng']).read_bytes()
            optimizer = torch.optim.Adam(expert.parameters(), lr=.0002, eps=1e-6)
            def update(index, compare=True):
                optimizer.zero_grad(set_to_none=True)
                result = runtime.model(ids, labels=labels, use_cache=False)
                logits = runtime.model(ids, use_cache=False).logits
                teacher = [core_teacher, parent_teacher][index - 1]
                kl = torch.nn.functional.kl_div(selected(logits, labels), teacher, log_target=True, reduction='batchmean')
                loss = result.loss + 5 * kl
                loss.backward()
                def trace(suffix, tensor):
                    if compare:
                        pair(f'step{index}_{suffix}', tensor, native_traces[index - 1][f'step{index}_{suffix}'])
                trace('ce', result.loss); trace('preservation_kl', kl); trace('loss', loss)
                trace('logits', result.logits.float().reshape(ids.shape[1], 248320))
                trace('preservation_logits', logits.reshape(ids.shape[1], 248320))
                for i, weight in enumerate(expert.parameters()):
                    trace(f'gradient_{i}', weight.grad)
                norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
                trace('norm', norm)
                for i, weight in enumerate(expert.parameters()):
                    trace(f'clipped_{i}', weight.grad)
                optimizer.step()
                for i, weight in enumerate(expert.parameters()):
                    state = optimizer.state[weight]
                    for name, tensor in [('weight', weight), ('m', state['exp_avg']), ('v', state['exp_avg_sq']), ('adamstep', state['step'])]:
                        trace(f'{name}_{i}', tensor)
                trace('cpu_rng', torch.get_rng_state()); trace('mps_rng', torch.mps.get_rng_state())
            update(1)
            checkpoint = copy.deepcopy(dict(weights=expert.state_dict(), optimizer=optimizer.state_dict(), cpu=torch.get_rng_state(), mps=torch.mps.get_rng_state()))
            update(2)
            completed = copy.deepcopy(dict(weights=expert.state_dict(), optimizer=optimizer.state_dict()))
            generate('candidate')
            expert.load_state_dict(checkpoint['weights']); optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['cpu']); torch.mps.set_rng_state(checkpoint['mps'])
            update(2, compare=False)
            assert all(torch.equal(value, completed['weights'][key]) for key, value in expert.state_dict().items())
            for key, value in optimizer.state_dict()['state'].items():
                assert all(torch.equal(tensor, completed['optimizer']['state'][key][name]) for name, tensor in value.items())
            resumed = tensors(n['resumed_state'])
            for key, value in expert.state_dict().items():
                assert raw(value) == raw(resumed[key])
            restored['optimizer_resume_equal'] = True
            runtime.branch.expert, runtime.branch.width, runtime.branch.expert_id = previous, n['previous_state']['width'], 'qualification-previous'
            restored['previous_generation_equal'] = generate('after') == n['generation']['previous']['tokens']
            # Drop trainable graph/optimizer references before releasing the HF model.
            del optimizer, expert, previous, core_teacher, parent_teacher, checkpoint, completed
    finally:
        if runtime is not None:
            runtime._release_model()
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
        torch.set_rng_state(cpu_parent); torch.mps.set_rng_state(mps_parent)
        restored['parent_rng_restored'] = raw(torch.get_rng_state()) == before_cpu and raw(torch.mps.get_rng_state()) == before_mps
        restored['owners_closed'] = runtime is None or runtime.model is None
        save(output / 'pairs.json', pairs)
        save(output / 'close.json', restored)
    assert all(restored.values())
    evidence = dict(**n['binding'], schema='native-expert-preservation-graph-v1', evidence_id=str(output), dtype='BF16', learning_rate=.0002, preservation_weight=5, actual_steps=2, teacher_modes=['core', 'expert'], pairs=pairs, restored=restored, actual_original_hf=True, fixture_parent=n['fixture_parent'])
    save(output / 'evidence.json', evidence)
    save(output / 'result.json', dict(passed=True, actual_original_hf=True, compared_steps=2, optimizer_updates_including_resume=3, pairs=len(pairs), generations=generations, teacher_calls=0, product_admitted=False))
    print(json.dumps(dict(passed=True, output=str(output), pairs=len(pairs), product_admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def cancel(signum, frame):
        raise KeyboardInterrupt('Expert qualification cancelled')
    signal.signal(signal.SIGTERM, cancel)
    assert args.output.is_absolute() and Path('/Volumes/移动数据库').is_mount()
    args.output.relative_to('/Volumes/移动数据库/lubanllm')
    run(args.native_result, args.output)
