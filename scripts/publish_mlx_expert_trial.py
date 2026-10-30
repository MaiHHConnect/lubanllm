#!/usr/bin/env python3
"""Issue a bounded trial only from real route/math/Swift execution evidence.

The application must be stopped or handed off before --publish. This does not
grant formal quality or training admission. Old sources remain immutable.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from prepare_mlx_expert_trial import ROOT, STORE, SOURCE_ID, TARGET_ID, save
from verify_mlx_trial_router import SOURCE_ROUTER, TARGET_ROUTER


def limited_route_check(report):
    assert report['complete'] is True and report['model_process_lock_released'] is True
    actual = report['route_replay']
    rows = actual['rows']
    assert len(rows) == 62 and len({row['id'] for row in rows}) == 62
    assert actual['released'] is True and actual['late_core_excluded_by_real_early_hook'] is True
    assert actual['mps_current_allocated_after_release'] == 0
    assert all(row['executed_layers'] == [0, 1, 2, 3] for row in rows)
    labeled = [row for row in rows if row['target_action'] is not None]
    assert len(labeled) == 46
    wrong_expert = [row['id'] for row in labeled if row['target_action'] == 'local_core'
                    and row['decision']['action'] != 'local_core']
    fallback = [row['id'] for row in labeled if row['target_action'] == 'neural_expert'
                and row['decision']['action'] != 'neural_expert']
    assert not wrong_expert
    assert all(row['decision']['source'] == 'expert_scope_reject' for row in labeled if row['id'] in fallback)
    boundaries = ['app-user-unknown-cn', 'app-user-unknown-en', 'q4-id-user-boundary-2']
    by_id = {row['id']: row for row in rows}
    assert all(by_id[name]['decision']['action'] == 'local_core' for name in boundaries)
    selected = sum(row['decision']['action'] == 'neural_expert' for row in rows)
    assert selected > 0
    return {'passed': True, 'scope': 'unchanged HF four-layer decision head and original cosine scope; conservative abstention allowed',
            'route_targets_all_matched': not fallback, 'labeled_correct': 46-len(fallback), 'labeled_total': 46,
            'conservative_scope_fallback_ids': fallback, 'wrong_expert_ids': wrong_expert,
            'unknown_user_boundary_ids': boundaries, 'actual_expert_selections': selected,
            'executed_layer_rows': 62, 'original_report_route_checked': actual['route_checked'],
            'formal_quality_admitted': False}


def issue(directory, route_report, runtime_report, binary):
    route = json.loads(route_report.read_text())
    runtime = json.loads(runtime_report.read_text())
    limited = limited_route_check(route)
    assert runtime['complete'] is True and runtime['model_process_lock_released'] is True
    math = runtime['expert_math']
    assert math['passed'] is True and math['exit_code'] == 0
    event = math['event']
    assert event['fp32_pass'] is True and event['bf16_pass_assumed'] is False
    assert len(event['stages']) == 8 and all(row['finite'] is True and row['nonzero'] is True for row in event['stages'])
    assert math['bf16_relative_l2_limit'] == .03 and math['bf16_numerical_pass'] is True
    assert all(0 <= row['relative_l2'] <= .03 for row in event['stages'] if row['dtype'] == 'bf16')
    receipt = json.loads(Path(str(binary) + '.json').read_text())
    worker_sha = receipt['binary_sha256']
    assert math['worker_binary_sha256'] == worker_sha
    runs = runtime['native_replay']
    assert len(runs) == 2 and [row['expert_enabled'] for row in runs] == [False, True]
    assert all(row['closed'] is True and row['worker_binary_sha256'] == worker_sha for row in runs)
    assert len(runs[0]['rows']) == 16 and len(runs[1]['rows']) == 40
    core = json.loads((STORE / 'active-core.json').read_text())
    source = json.loads((STORE / 'layer-23' / (SOURCE_ID + '.json')).read_text())
    source_router = json.loads((STORE / (SOURCE_ROUTER + '.json')).read_text())
    assert source['core_version'] is None and source_router['core_version'] is None
    assert event['expert_file_sha256'] == source['sha256'] and event['width'] == source['width']
    assert runs[0]['loaded']['expert_loaded'] is False and runs[1]['loaded']['expert_loaded'] is True
    assert runs[1]['loaded']['expert_file_sha256'] == source['sha256']
    assert runs[1]['loaded']['expert_width'] == source['width'] and runs[1]['loaded']['expert_layer'] == 23
    for row in runs:
        assert row['loaded']['core_file_sha256'] == core['sha256']
        assert row['loaded']['inference_precision'] == 'mlx-q4'
        assert row['core_binding']['sha256'] == core['sha256']
    assert runs[0]['model_binding'] == runs[1]['model_binding']
    baseline = {row['id']: row for row in runs[0]['rows']}
    changed = [row['id'] for row in runs[1]['rows'] if row['id'] in baseline
               and row['response']['tokens'] != baseline[row['id']]['response']['tokens']]
    assert changed, 'Expert must have an observed real effect'
    expert = json.loads((directory / 'candidate-metadata.json').read_text())
    router = json.loads((directory / 'candidate-router.json').read_text())
    assert expert['sha256'] == source['sha256'] and expert['core_version'] == core['version']
    assert router['sha256'] == source_router['sha256'] and router['core_version'] == core['version']
    assert router['expert_sha256'] == source['sha256']
    assert all(router[key] == source_router[key] for key in
               ('temperature', 'threshold', 'expert_scope', 'supported_actions', 'depth'))
    expert['migration'].update(status='local-trial-runtime-checked', route_report=str(route_report), runtime_report=str(runtime_report))
    expert.update(execution_scope='local-trial', trial_engine='mlx-q4', training_admitted=False)
    expert['preservation_rows'] = []
    expert['migration']['inherited_preservation_rows'] = source['preservation_rows']
    router['calibrated'] = True
    router['migration'].update(route_checked=True, limited_route_check=limited)
    proof = {'schema': 'mlx-expert-local-trial-v1', 'scope': 'local-trial', 'quality_admitted': False,
             'math_passed': True, 'runtime_checked': True, 'route_checked': True,
             'identity': core['identity'], 'expert_id': TARGET_ID, 'core_version': core['version'],
             'core_file_sha256': core['sha256'], 'expert_file_sha256': source['sha256'],
             'worker_binary_sha256': worker_sha, 'mlx_model_sha256': runs[1]['model_binding']['sha256'],
             'width': source['width'], 'router_id': TARGET_ROUTER, 'router_sha256': source_router['sha256'],
             'route_report': str(route_report), 'runtime_report': str(runtime_report),
             'limited_route_check': limited, 'expert_changed_case_ids': changed,
             'source_core_version': None, 'source_expert_id': SOURCE_ID, 'source_router_id': SOURCE_ROUTER,
             'training_admitted': False, 'optimizer_updates': 0, 'teacher_calls': 0}
    return expert, router, proof


def publish(expert, router, proof, output):
    from luban.settings import private_json
    old_expert = (STORE / 'active.json').read_bytes()
    old_router = (STORE / 'active-router.json').read_bytes()
    assert json.loads(old_expert)['id'] == SOURCE_ID and json.loads(old_expert)['core_version'] is None
    assert json.loads(old_router)['id'] == SOURCE_ROUTER and json.loads(old_router)['core_version'] is None
    paths = {'expert_weights': STORE / 'layer-23' / (TARGET_ID + '.safetensors'),
             'expert_metadata': STORE / 'layer-23' / (TARGET_ID + '.json'),
             'router_weights': STORE / (TARGET_ROUTER + '.safetensors'),
             'router_metadata': STORE / (TARGET_ROUTER + '.json'),
             'proof': ROOT / 'runtime/mlx/expert-proof.json'}
    assert not any(path.exists() for path in paths.values()), 'Trial output already exists; do not overwrite it'
    receipt = {'schema': 'mlx-expert-local-trial-publication-v1', 'published': False,
               'old_active_expert': json.loads(old_expert), 'old_active_router': json.loads(old_router),
               'proof': proof, 'files': {key: str(value) for key, value in paths.items()},
               'rollback': 'Restore the two recorded pointer values; old model/expert/router weights were never changed.'}
    save(output, receipt)
    changed_expert, changed_router = False, False
    try:
        for source, target in ((STORE / 'layer-23' / (SOURCE_ID + '.safetensors'), paths['expert_weights']),
                               (STORE / (SOURCE_ROUTER + '.safetensors'), paths['router_weights'])):
            with source.open('rb') as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
        private_json(paths['expert_metadata'], expert)
        private_json(paths['router_metadata'], router)
        paths['proof'].parent.mkdir(parents=True, exist_ok=True)
        private_json(paths['proof'], proof)
        assert (STORE / 'active.json').read_bytes() == old_expert and (STORE / 'active-router.json').read_bytes() == old_router
        private_json(STORE / 'active.json', {'id': TARGET_ID, 'width': expert['width'],
                     'identity': proof['identity'], 'core_version': proof['core_version'],
                     'execution_scope': 'local-trial', 'quality_admitted': False, 'trial_engine': 'mlx-q4'})
        changed_expert = True
        private_json(STORE / 'active-router.json', {'id': TARGET_ROUTER, 'identity': proof['identity'],
                     'core_version': proof['core_version'], 'sha256': router['sha256'],
                     'metadata_sha256': hashlib.sha256(paths['router_metadata'].read_bytes()).hexdigest()})
        changed_router = True
        receipt['published'] = True
    except BaseException:
        if changed_expert and json.loads((STORE / 'active.json').read_text())['id'] == TARGET_ID:
            private_json(STORE / 'active.json', json.loads(old_expert))
        if changed_router and json.loads((STORE / 'active-router.json').read_text())['id'] == TARGET_ROUTER:
            private_json(STORE / 'active-router.json', json.loads(old_router))
        receipt['rolled_back'] = True
        raise
    finally:
        private_json(output, receipt)


def rebind(expert, router, proof, output, previous_publication):
    """Rebind this task's existing trial; never replace weights or active pointers."""
    from luban.settings import private_json
    previous = json.loads(previous_publication.read_text())
    assert previous['schema'] in ('mlx-expert-local-trial-publication-v1', 'mlx-expert-local-trial-rebinding-v1')
    assert previous['published'] is True
    paths = {'expert_weights': STORE / 'layer-23' / (TARGET_ID + '.safetensors'),
             'expert_metadata': STORE / 'layer-23' / (TARGET_ID + '.json'),
             'router_weights': STORE / (TARGET_ROUTER + '.safetensors'),
             'router_metadata': STORE / (TARGET_ROUTER + '.json'),
             'proof': ROOT / 'runtime/mlx/expert-proof.json'}
    assert previous['files'] == {key: str(path) for key, path in paths.items()}
    assert all(path.is_file() and not path.is_symlink() for path in paths.values())
    old_proof_bytes = paths['proof'].read_bytes()
    old_proof = json.loads(old_proof_bytes)
    assert old_proof == previous['proof'], 'Prior publication is not the active trial proof'
    mutable = {'worker_binary_sha256', 'runtime_report', 'expert_changed_case_ids'}
    assert set(old_proof) == set(proof)
    assert all(old_proof[key] == proof[key] for key in proof if key not in mutable), 'Trial sources or route scope changed'
    assert old_proof['runtime_report'] != proof['runtime_report'], 'Rebinding requires a new actual runtime report'
    old_expert_bytes = paths['expert_metadata'].read_bytes()
    old_expert = json.loads(old_expert_bytes)
    expected_old_expert = copy.deepcopy(expert)
    expected_old_expert['migration']['runtime_report'] = old_proof['runtime_report']
    assert old_expert == expected_old_expert, 'Existing trial expert metadata changed'
    router_bytes = paths['router_metadata'].read_bytes()
    assert json.loads(router_bytes) == router, 'Existing trial router metadata changed'
    active_expert = (STORE / 'active.json').read_bytes()
    active_router = (STORE / 'active-router.json').read_bytes()
    assert json.loads(active_expert) == {
        'id': TARGET_ID, 'width': expert['width'], 'identity': proof['identity'],
        'core_version': proof['core_version'], 'execution_scope': 'local-trial',
        'quality_admitted': False, 'trial_engine': 'mlx-q4'}
    assert json.loads(active_router) == {
        'id': TARGET_ROUTER, 'identity': proof['identity'], 'core_version': proof['core_version'],
        'sha256': router['sha256'], 'metadata_sha256': hashlib.sha256(router_bytes).hexdigest()}
    # Compare these two small immutable copies with the exact old sources. Do not
    # reread/hash the multi-GB model; issue() binds the newly executed model/core.
    for source, target in ((STORE / 'layer-23' / (SOURCE_ID + '.safetensors'), paths['expert_weights']),
                           (STORE / (SOURCE_ROUTER + '.safetensors'), paths['router_weights'])):
        assert source.is_file() and not source.is_symlink()
        assert source.read_bytes() == target.read_bytes(), 'Trial weights differ from their unchanged source'
    receipt = {'schema': 'mlx-expert-local-trial-rebinding-v1', 'published': False,
               'previous_publication': str(previous_publication), 'previous_proof': old_proof,
               'proof': proof, 'files': previous['files'], 'active_pointers_unchanged': False,
               'weights_rewritten': False, 'router_metadata_rewritten': False,
               'rollback': 'Restore previous proof and its runtime_report in trial expert metadata; active pointers remain the trial.'}
    save(output, receipt)
    changed_expert, changed_proof = False, False
    try:
        assert paths['expert_metadata'].read_bytes() == old_expert_bytes
        assert paths['proof'].read_bytes() == old_proof_bytes
        assert (STORE / 'active.json').read_bytes() == active_expert
        assert (STORE / 'active-router.json').read_bytes() == active_router
        private_json(paths['expert_metadata'], expert)
        changed_expert = True
        private_json(paths['proof'], proof)
        changed_proof = True
        assert (STORE / 'active.json').read_bytes() == active_expert
        assert (STORE / 'active-router.json').read_bytes() == active_router
        assert paths['router_metadata'].read_bytes() == router_bytes
        receipt.update(published=True, active_pointers_unchanged=True)
    except BaseException:
        if changed_expert and json.loads(paths['expert_metadata'].read_text()) == expert:
            private_json(paths['expert_metadata'], old_expert)
        if changed_proof and json.loads(paths['proof'].read_text()) == proof:
            private_json(paths['proof'], old_proof)
        receipt['rolled_back'] = True
        raise
    finally:
        private_json(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--route-report', type=Path, required=True)
    parser.add_argument('--runtime-report', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--rebind-from', type=Path, help='Prior successful publication of this same local trial')
    args = parser.parse_args()
    assert Path('/Volumes/移动数据库').is_mount() and not args.output.exists()
    expert, router, proof = issue(args.directory, args.route_report, args.runtime_report, args.binary)
    if args.publish:
        if args.rebind_from:
            rebind(expert, router, proof, args.output, args.rebind_from)
        else:
            publish(expert, router, proof, args.output)
    else:
        assert args.rebind_from is None, '--rebind-from requires explicit --publish'
        save(args.output, {'expert': expert, 'router': router, 'proof': proof, 'published': False})
    print(json.dumps({'published': args.publish, 'output': str(args.output),
                      'limited_route_check': proof['limited_route_check']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
