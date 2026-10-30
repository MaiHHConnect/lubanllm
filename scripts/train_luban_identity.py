"""User-authorized identity training: real weights, closed-book checks, no API."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import torch
from luban.evolution import _target, _generation, _loss, _capture_replay_subspace
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings, private_json, require_storage
from luban.retention import _validate_groups, provenance
import compare_native_training as abi
from verify_cached_training import capture_tail, _binding, tensor_sha
from verify_router_coverage_study import model_binding

TARGET = "我是鲁班零号，luban0模型，基于qwen3.5训练的，能无限进化的模型。"
MODEL = Path('/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1')
OLD_FIXTURE = Path('/Volumes/移动数据库/lubanllm/experiments/luban0-native-training-20261010T072856Z-97341da0/model-material/fixture450/fixture.json')
TRAIN = ['你是谁', '你是谁？', '请介绍一下你自己。', '你叫什么？', '你是什么模型？',
         '说说你的身份。', '你是哪一个大语言模型？', '你是Qwen还是鲁班？',
         '能告诉我你的模型名称吗？', '请用一句话自我介绍。', '你是哪个基座？',
         '你能介绍自己的模型来源吗？', '我在调用你的API，请介绍你自己。',
         '你是阿里巴巴推出的Qwen3.5吗？', 'Who are you?', 'What model are you?',
         '你能进化吗？', '告诉我你的名字和基座来源。']
DEV = ['你的身份是什么？', '现在与你对话的是哪个模型？', '你好，请告诉我你是谁。',
       '请说明你的模型名称、来源及定位。', '你这个助手叫什么名字？', '介绍一下当前运行的你。']


def row(identifier, prompt, answer, purpose='training', messages=None):
    return {'id': identifier, 'prompt': prompt, 'answer': answer, 'source': 'human',
            'purpose': purpose, 'approved': True, 'training_allowed': True,
            'provenance': json.dumps({'evolution_role': 'new', 'messages': messages or [{'role': 'user', 'content': prompt}],
                'verification_evidence': 'Explicit user brand/source statement, 2026-10-10; infinite evolution is positioning, not proven unbounded capability.',
                'split_family': identifier, 'generation_check': {'type': 'exact', 'expected': answer, 'max_new_tokens': 128}}, ensure_ascii=False)}


def material():
    training = [row(f'brand-v2-train-{i}', q, TARGET) for i, q in enumerate(TRAIN)]
    for i, system in enumerate(['你是一个本地运行的助手。', '你通过OpenAI兼容API服务用户。', '请简洁回答用户的问题。']):
        training.append(row(f'brand-v2-system-{i}', '你是谁？', TARGET,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': '你是谁？'}]))
    training.append(row('brand-v2-history', '再介绍一下你自己。', TARGET, messages=[
        {'role': 'user', 'content': '2+3是多少？'}, {'role': 'assistant', 'content': '5'},
        {'role': 'user', 'content': '再介绍一下你自己。'}]))
    for i, (q, a) in enumerate([('是谁调优了你？', '麦好火。'), ('你的调优方叫什么？', '麦好火。'),
        ('你的底层开源基座是什么？', 'Qwen3.5。'), ('只说你的模型名字。', '鲁班零号（luban0）。')]):
        training.append(row(f'brand-v2-attribute-{i}', q, a))
    validation = [row(f'brand-v2-dev-{i}', q, TARGET, 'validation') for i, q in enumerate(DEV)]
    validation.append(row('brand-v2-dev-api', '说明你是谁。', TARGET, 'validation', messages=[
        {'role': 'system', 'content': '这是一个应用程序的聊天接口。'}, {'role': 'user', 'content': '说明你是谁。'}]))
    validation.append(row('brand-v2-dev-history', '你到底是什么模型？', TARGET, 'validation', messages=[
        {'role': 'user', 'content': '把hello转成大写。'}, {'role': 'assistant', 'content': 'HELLO'},
        {'role': 'user', 'content': '你到底是什么模型？'}]))
    old = json.loads(OLD_FIXTURE.read_text())['samples']
    replay = [copy.deepcopy(r) for r in old.values() if provenance(r)['evolution_role'] == 'replay']
    groups = {'new_training': training, 'new_validation': validation, 'replay_training': replay, 'replay_validation': []}
    _validate_groups(groups)
    return groups


def runtime_for(root):
    settings = Settings(root / 'home')
    settings.data.update(model_path=str(MODEL), storage_root=str(root / 'isolated'), temperature=0,
                         max_new_tokens=128, local_thinking='off')
    runtime = Runtime(settings, None, research=True)
    runtime.load()
    if runtime.cold_backbone is not None:
        runtime.cold_backbone.acquire()
    runtime.branch.expert = None
    runtime.model.eval().requires_grad_(False)
    _target(runtime)[1].requires_grad_(True)
    return runtime


def prepare(root):
    groups = material()
    private_json(root / 'protocol.json', {'target': TARGET, 'groups': groups, 'seed': 42, 'steps': 450,
        'learning_rate': 1e-5, 'distillation_weight': 5, 'dtype': 'BF16 weights/FP32 master/Adam',
        'engine': 'Luban 训练引擎 / C++ CE-release', 'teacher_api_calls': 0,
        'checks': 'Fresh original baseline; twice exact complete new TRAIN and DEV; retain every original-correct REPLAY; loss<=baseline*1.05+.01; native product recheck before commit',
        'scope': 'Explicitly authorized identity correction; curated same-task development, not generalization or infinite capability proof',
        'official': model_binding(str(MODEL), abi.ORIGINAL_IDENTITY)})
    runtime = runtime_for(root)
    try:
        parameter = _target(runtime)[1]
        assert tensor_sha(parameter) == abi.ORIGINAL_NATIVE_SHA
        baseline = {'new': _generation(runtime, groups['new_training'] + groups['new_validation']),
                    'replay': _generation(runtime, groups['replay_training']),
                    'replay_loss': _loss(runtime, groups['replay_training'])}
        private_json(root / 'baseline.json', baseline)
        basis, basis_report = _capture_replay_subspace(runtime, parameter, groups['replay_training'])
        caches = {r['id']: capture_tail(runtime, r) for r in groups['new_training'] + groups['replay_training']}
        schedule = [{'ordinal': i+1, 'role': 'new' if i%2==0 else 'replay',
            'ce': groups['new_training' if i%2==0 else 'replay_training'][(i//2)%len(groups['new_training' if i%2==0 else 'replay_training'])]['id'],
            'replay': groups['replay_training'][(i//2)%len(groups['replay_training'])]['id']} for i in range(450)]
        binding = _binding(runtime, parameter)
        official = json.loads((root/'protocol.json').read_text())['official']
        members = {n: {'sha256': h, 'bytes': (MODEL/n).stat().st_size} for n,h in official['files'].items()}
        members['upstream-file-manifest.json'] = {'sha256': official['manifest_sha256'], 'bytes': (MODEL/'upstream-file-manifest.json').stat().st_size}
        receipt = abi.export_runtime_fixture(runtime, groups['new_training'], groups['replay_training'],
            parameter.detach().float(), basis.to(parameter.device), caches, root/'fixture', steps=450,
            metadata={'scope': 'actual_python_frozen_prefix', 'identity': runtime.identity, 'core_version': None,
                'original_native_tensor_sha256': tensor_sha(parameter), 'official_files': members,
                'sources': binding['sources'], 'fixed_schedule_sha256': abi.digest(schedule)}, schedule=schedule)
        private_json(root/'prepared.json', {'fixture': receipt, 'basis': basis_report,
            'baseline_correct_replay': [r['id'] for r in baseline['replay'] if r['passed']],
            'prepared_seconds': time.perf_counter()-START, 'original_tensor_preserved': tensor_sha(parameter)==abi.ORIGINAL_NATIVE_SHA})
    finally:
        runtime._release_model()
    print(json.dumps({'prepared': str(root), 'elapsed': time.perf_counter()-START}), flush=True)


def evaluate(root, output):
    protocol = json.loads((root/'protocol.json').read_text()); groups=protocol['groups']
    baseline=json.loads((root/'baseline.json').read_text())
    runtime=runtime_for(root)
    try:
        tensors,_=abi.descriptors(output/'tensors.tsv',abi.OUTPUT_ABI)
        spec=tensors['final_candidate']; candidate=abi.torch_load_raw(output,spec,'cpu')
        assert spec['dtype']=='bf16' and spec['shape']==[2048,6144] and abi.sha(output/spec['file'])==spec['sha256']
        name,parameter=_target(runtime)
        with torch.no_grad():parameter.copy_(candidate.to(parameter.device))
        wanted=groups['new_training']+groups['new_validation']
        def observed(rows, phase):
            values=[]
            for sample in rows:
                values.extend(_generation(runtime,[sample]))
                private_json(root/(output.name+'-quality-'+phase+'.json'),values)
                if len(values)%8==0 or len(values)==len(rows):
                    print(json.dumps({'phase':phase,'checked':len(values),'passed':sum(r['passed'] for r in values)},ensure_ascii=False),flush=True)
            return values
        first=observed(wanted,'new-first'); second=observed(wanted,'new-second')
        correct={r['id'] for r in baseline['replay'] if r['passed']}
        protected=[r for r in groups['replay_training'] if r['id'] in correct]
        replay=observed(protected,'old'); replay_loss=_loss(runtime,groups['replay_training'])
        passed=all(r['passed'] for r in first+second+replay) and replay_loss<=baseline['replay_loss']*1.05+.01
        report={'target':TARGET,'engine':'Luban 训练引擎','variant':'release-ce-graph', 'passed':passed,
            'identity':runtime.identity,'candidate_sha256':tensor_sha(parameter),'new_first':first,'new_second':second,
            'old_protection':replay,'replay_loss':replay_loss,'baseline_replay_loss':baseline['replay_loss'],
            'teacher_api_calls':0,'formal_adoption':False,'elapsed_seconds':time.perf_counter()-START}
        from safetensors.torch import save_file
        save_file({name:candidate.contiguous()},str(root/(output.name+'-candidate.safetensors')))
        private_json(root/(output.name+'-quality.json'),report)
        print(json.dumps({'passed':passed,'new':sum(r['passed'] for r in first),'new_total':len(first),
            'old':sum(r['passed'] for r in replay),'old_total':len(replay),'loss':replay_loss},ensure_ascii=False),flush=True)
        if not passed:raise RuntimeError('Identity candidate failed; preserved without adoption')
    finally:runtime._release_model()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','evaluate']); parser.add_argument('root',type=Path); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); START=time.perf_counter(); torch.set_num_threads(1); torch.manual_seed(42)
    require_storage('/Volumes/移动数据库/lubanllm'); require_storage('/Volumes/Untitled/lubanllm-backup')
    if args.action=='prepare':args.root.mkdir(exist_ok=False); prepare(args.root)
    else:evaluate(args.root,args.output)
