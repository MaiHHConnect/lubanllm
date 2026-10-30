# SPDX-License-Identifier: Apache-2.0
"""Run the isolated native probe while holding Luban's existing model lock."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_MANIFEST_SHA = 'd8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237'


def model_identity(model):
    raw = (model / 'upstream-file-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != OFFICIAL_MANIFEST_SHA:
        raise ValueError('Original approved HF manifest changed')
    manifest = json.loads(raw)
    records = sorted((item['Path'], item['Sha256']) for item in manifest['Data']['Files']
                     if item['Path'].endswith('.safetensors') or item['Path'] == 'config.json')
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def verify_official(model):
    model_identity(model)
    manifest = json.loads((model / 'upstream-file-manifest.json').read_bytes())
    for item in manifest['Data']['Files']:
        if not item['Path'].endswith('.safetensors') and item['Path'] != 'config.json':
            continue
        path = model / item['Path']
        if path.is_symlink() or path.stat().st_size != item['Size']:
            raise ValueError('Original model file changed')
        with path.open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != item['Sha256']:
            raise ValueError('Original model file digest mismatch')


def verify_core(model, core, provenance, approved):
    if not re.fullmatch('[0-9a-f]{64}', approved or ''):
        raise ValueError('Invalid approved provenance digest')
    raw = provenance.read_bytes()
    if hashlib.sha256(raw).hexdigest() != approved:
        raise ValueError('Approved provenance changed')
    record = json.loads(raw)
    expected = {'format': 'luban-native-bf16-core-v1', 'base_identity': model_identity(model),
                'official_manifest_sha256': OFFICIAL_MANIFEST_SHA,
                'parameter': 'model.language_model.layers.23.mlp.down_proj.weight',
                'source_parameter': 'model.language_model.layers.23.mlp.base.down_proj.weight',
                'dtype': 'torch.bfloat16', 'shape': [2048, 6144]}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Native core binding mismatch')
    for k, value in [('quality_passed', True), ('safety_passed', True), ('branch_enabled', False), ('reference_used', False)]:
        if record.get(k) is not value:
            raise ValueError('Native core validation flags mismatch')
    with core.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != record['file_sha256']:
            raise ValueError('Native core file changed')
    return record


def acquire(identity):
    spec = importlib.util.spec_from_file_location('luban_native_process_lock', ROOT / 'src/luban/process_lock.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelProcessLock(identity, 'mps')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core', type=Path)
    p.add_argument('--provenance', type=Path)
    p.add_argument('--approved-provenance-sha256')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    core_fields = [args.core, args.provenance, args.approved_provenance_sha256]
    if any(core_fields) and not all(core_fields):
        raise ValueError('Core requires a separately approved provenance digest')
    identity = model_identity(args.model)
    lock = acquire(identity)
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
        args.output.mkdir(parents=True)
        if args.core:
            verify_core(args.model, args.core, args.provenance, args.approved_provenance_sha256)
        cmd = [str(args.binary.resolve()), str(args.model), OFFICIAL_MANIFEST_SHA,
               str(args.cases), str(args.output)]
        if args.core:
            cmd += [str(args.core), str(args.provenance), args.approved_provenance_sha256]
        metadata = {'command': cmd, 'base_identity': identity, 'device': 'mps',
                    'manifest_sha256': OFFICIAL_MANIFEST_SHA,
                    'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                    'teacher_calls': 0, 'training': False, 'product_adopted': False,
                    'memory_method': 'Darwin proc_pid_rusage v4 phys_footprint and lifetime_max_phys_footprint'}
        (args.output / 'invocation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        with (args.output / 'events.jsonl').open('x') as out, (args.output / 'stderr.log').open('x') as err:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, pass_fds=(lock.fd,)) as process:
                started, prefixes, visible = {}, {}, {}
                for line in process.stdout:
                    event = json.loads(line)
                    index = event.get('case_index')
                    if event.get('event') == 'request_started':
                        started[index] = time.perf_counter_ns()
                        prefixes[index] = []
                    elif event.get('event') == 'token':
                        prefixes[index].append(event['token'])
                        text = tokenizer.decode(prefixes[index], skip_special_tokens=True)
                        if index not in visible and text.replace('\ufffd', '').strip():
                            visible[index] = time.perf_counter_ns() - started[index]
                    elif event.get('event') == 'result':
                        event['client_first_visible_text_ns'] = visible.get(index)
                        event['client_timing_scope'] = 'parent request-start event arrival to first decodable non-whitespace text; includes IPC and tokenizer decode'
                    out.write(json.dumps(event, ensure_ascii=False) + '\n')
                    out.flush()
                    if event.get('event') != 'token':
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                code = process.wait()
        metadata['exit_code'] = code
        (args.output / 'invocation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        return code
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
