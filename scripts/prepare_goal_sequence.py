"""Freeze a closed450 or1800 native episode into a distinct continuation protocol; CPU-only."""
import argparse
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_evolution_goal import read, sha, write_new, data_service, source_hashes, need, continuation_data
from luban import goal_sequence as goal, luban_training as engine
from luban.bun_bridge import readonly_connection
from luban.training_checkpoint import implementation_hashes
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-exit-code', type=int, required=True)
    args = parser.parse_args()
    parent, output = args.parent, args.output
    need(parent.is_absolute() and output.is_absolute() and not output.exists(), 'new absolute evidence directory required')
    plan, result = read(parent/'plan.json'), read(parent/'acceptance.json')
    need(args.parent_exit_code == 1 and result['released'] is True and result['passed'] is False
         and result['training']['completed_steps'] in {450, 1800} and not goal.VALIDATION_FIELDS.intersection(result['training']),
         'parent must actually end450/reject/release without new VALID')
    parent_steps = result['training']['completed_steps']
    is_v3 = parent_steps == 1800
    total = 3150 if is_v3 else 1800
    run_id = result['training']['id']
    output.mkdir(mode=0o700)
    with data_service(Path(plan['home']), output/'closure-data-owner.json'):
        from contextlib import closing
        with closing(readonly_connection(Path(plan['storage'])/'app/memory.sqlite3')) as db:
            row = db.execute('SELECT * FROM training_runs WHERE id=? AND identity=?', (run_id, plan['identity'])).fetchone()
        need(row is not None and row['state'] == 'rejected', 'actual Bun parent journal must be rejected')
        journal = {**json.loads(row['body']), 'state': row['state']}
    checkpoint = Path(plan['storage'])/'modules/luban0-v1/training-runs'/run_id/journal['checkpoint']['name']
    need(sha(checkpoint) == journal['checkpoint']['sha256'], 'parent terminal checkpoint SHA mismatch')
    closed = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    binding, payload = closed['binding'], closed['payload']
    need(payload['step'] == parent_steps and binding == journal['binding'], 'durable state must actually be450')
    need(journal['records'] == {key: plan[key] for key in ('groups', 'protection', 'retention_regressions')}, 'complete material must match')
    segment = payload['report']['training_engine']['segments'][-1]
    need(segment['end_step'] == parent_steps and segment['completed'] is True, 'native terminal segment must complete450')
    fixture_path, native_output = Path(segment['fixture']), Path(segment['output'])
    fixture = engine.validate_fixture(fixture_path, segment['manifest_sha256'])
    engine.validate_output(native_output, fixture, payload['report']['training_engine']['binary_sha256'])
    native = engine.load_checkpoint(native_output, fixture)
    for name, value in (('parent-journal.json', journal), ('parent-binding.json', binding),
                        ('parent-durable-report.json', payload['report']), ('records.json', journal['records'])):
        write_new(output/name, value)
    files = {'parent_plan': parent/'plan.json', 'parent_result': parent/'acceptance.json',
        'parent_journal': output/'parent-journal.json', 'parent_checkpoint': checkpoint,
        'parent_binding': output/'parent-binding.json', 'parent_report': output/'parent-durable-report.json',
        'records': output/'records.json', 'formal_protocol': Path(plan['formal_protocol']),
        'formal_manifest': Path(plan['formal_protocol']).parent/'manifest.json',
        'materials': Path(plan['materials']), 'baseline_old_training': parent/'baseline-old-training.json',
        'native_manifest': fixture_path/'manifest.tsv', 'native_output_manifest': native_output/'tensors.tsv',
        'native_trace': native_output/'trace.tsv'}
    for key, alias in (('old_retention', 'old_retention_complete'), ('old_training', 'old_training_original'),
                       ('old_protection', 'old_protection'), ('acceptance', 'acceptance')):
        files[key] = Path(plan['formal_files'][alias]['path'])
    snapshots = {name: {'path': str(parent/'source-snapshot'/name), 'sha256': digest}
                 for name, digest in plan['source_sha256'].items()}
    impl_snapshots = {name: {'path': str(parent/'source-snapshot/src/luban'/name if is_v3 else parent/'implementation-source-snapshot'/name), 'sha256': digest}
                      for name, digest in binding['implementation_sha256'].items()}
    core = read(Path(plan['storage'])/'modules/luban0-v1/active-core.json')['version']
    need(core == plan['starting_core'], 'parent rejected candidate cannot be active')
    checkpoints = list(range(64, 449, 32)) + list(range(900, total + 1, 450))
    if is_v3:
        files['parent_sequence_protocol'] = Path(plan['goal_sequence_protocol']['path'])
    protocol = dict(protocol=goal.PROTOCOL_V3 if is_v3 else goal.PROTOCOL, status='FROZEN_CLOSED_PARENT', parent_steps=parent_steps, max_steps=total,
        segment_steps=450, max_segments=total // 450, checkpoints=checkpoints, parent_arguments=plan['arguments'],
        seed=42, batch_size=1, gradient_accumulation=1, parent_run_id=run_id, new_run_id=secrets.token_hex(6),
        policy_version='goal-fixed450-same-weight-v2', required_consecutive_passes=0, required_same_weight_passes=2, required_new_exposures=2,
        reference_policy='original-accepted-core', basis_policy='exact-parent-basis',
        validation_policy='first-complete450-same-W-double-pass-once-then-stop', starting_core=plan['starting_core'], identity=plan['identity'],
        parent_binding_sha256=goal.content_sha(binding), parent_report_sha256=goal.content_sha(payload['report']),
        records_sha256=goal.content_sha(journal['records']), dataset_sha256=payload['dataset_sha256'],
        original_tensor_sha256=payload['original_sha256'], basis_sha256=goal.tensor_sha(native['basis']),
        reference_sha256=goal.tensor_sha(native['reference']), parent_final_tensor_sha256=goal.tensor_sha(native['candidate']),
        binary_sha256=payload['report']['training_engine']['binary_sha256'], parameter=payload['parameter'],
        engine=payload['report']['training_engine']['engine'], replay_generation_ids=payload['report']['training_selection']['replay_generation_ids'],
        parent_implementation_sha256=binding['implementation_sha256'], new_implementation_sha256=implementation_hashes(),
        parent_source_sha256=plan['source_sha256'], new_source_sha256=source_hashes(continuation=True),
        parent_source_snapshot=snapshots, parent_implementation_snapshot=impl_snapshots,
        native_files={str(path): sha(path) for directory in (fixture_path, native_output)
                      for path in sorted(directory.iterdir()) if path.is_file()},
        closure=dict(exit_code=args.parent_exit_code, error_type=result['error_type'], error=result['error'],
            released=result['released'], adopted=result['adopted'], production_written=result['production_written'],
            validation_reads=0, final_step=parent_steps, parent_journal_state='rejected', active_core=core),
        files={key: {'path': str(path), 'sha256': sha(path)} for key, path in files.items()},
        parent_training_elapsed_seconds=result['training']['elapsed_seconds'],
        observation_policy=dict(reference_baseline='actual original b45 measured after parent closure; not retroactive',
            entry_parent=f'two actual same-W TRAIN16 + old13 + TRAIN83 replay loss observations at{parent_steps}', boundaries=[step for step in checkpoints if step > parent_steps],
            same_W_observations=2, no_VALID_access=True, branch_off=True, reference_off=True, natural_eos_required=True,
            mastery_accounting='double observation intersection; additions and forgetting relative to reference and cumulative prior confirmed IDs',
            selected_endpoint='reuse final actual HF/native TRAIN16 twice; independent heldout remains untouched until selection'))
    if is_v3:
        protocol.update(parent_sequence_metadata=plan['goal_sequence'], inherited_observations_sha256=goal.content_sha({key: payload['report']['goal_sequence'][key] for key in ('reference_baseline_after_parent_closed', 'entry450', 'stage_observations')}),
                        parent_goal_sequence_sources_sha256=goal.content_sha(payload['report']['goal_sequence_sources']))
    goal.validate_state(payload, native, protocol)
    write_new(output/'protocol.json', protocol)
    checked = continuation_data(output/'protocol.json')[0]
    print(json.dumps({'protocol': str(output/'protocol.json'), 'sha256': checked['_protocol_sha256'],
        'new_run_id': protocol['new_run_id'], 'cpu_only': True, 'weights_updated': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
