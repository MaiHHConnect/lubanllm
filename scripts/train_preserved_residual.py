"""Isolated fixed450 residual study; finite protection, never product adoption."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time

import torch
from torch.nn import functional as F

from train_luban_identity import runtime_for
from study_core_constraint import capture, residual_logits, event
from probe_core_constraint import probes, residual_equivalence
from luban.core_update import apply_core_residual, constrain_residual, guard_retention_losses, validate_basis
from luban.evolution import _target, _generation_loaded, _answer_logits, _cancel
from luban.settings import private_json

SCALES = (.25, .0625, .015625, .00390625)
EXPANDED_SCALES = (1.,) + SCALES


def answer_logits(runtime, cache, reference, delta):
    labels = cache['labels'].to(delta.device)
    return _answer_logits(residual_logits(runtime, cache, reference, delta), labels)


def loss(runtime, cache, reference, delta):
    labels = cache['labels'].to(delta.device)
    return F.cross_entropy(answer_logits(runtime, cache, reference, delta),
                           labels[:, 1:][labels[:, 1:] != -100])


@torch.no_grad()
def select_update(current, constrained, basis, baseline, before, measure_new, measure_protected, *,
                  scales=SCALES, retention_tolerance=None):
    """Actual deployed losses decide writes; the original baseline never moves."""
    if not torch.isfinite(torch.tensor(before)):
        raise ValueError('Nonfinite current supervised loss')
    trials = []
    for scale in scales:
        candidate = current + (constrained - current) * scale
        if not torch.isfinite(candidate).all():
            raise ValueError('Nonfinite scaled candidate')
        row = {'scale': scale, 'accepted': False}
        trials.append(row)
        if torch.equal(candidate, current):
            row['reason'] = 'unchanged_delta'
            break
        error = float((candidate @ basis).norm())
        tolerance = 64 * torch.finfo(torch.float32).eps * max(1., float(candidate.norm()))
        row['subspace_residual'] = error
        if error > tolerance:
            row['reason'] = 'scaled_subspace_rounding'
            continue
        after = float(measure_new(candidate))
        row['supervised_after'] = after
        if not torch.isfinite(torch.tensor(after)):
            raise ValueError('Nonfinite candidate supervised loss')
        if after >= before:
            row['reason'] = 'no_supervised_improvement'
            continue
        values = measure_protected(candidate)
        accepted, row['protection'] = guard_retention_losses(
            candidate, current, baseline, values, retention_tolerance)
        if row['protection']['rejected']:
            row['reason'] = 'fixed_baseline_protection'
            continue
        row.update(accepted=True, reason='finite_constraints_and_supervised_improvement')
        return accepted, trials
    return current, trials


def checkpoint(path, delta, optimizer, steps, accepted, identity, baseline, *, successor=None, parent_delta=None):
    # One replaceable recovery checkpoint, not extra source/model snapshots.
    state = optimizer.state_dict()
    state = {**state, 'state': {key: {
        name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()} for key, values in state['state'].items()}}
    temporary = path.with_suffix('.tmp')
    payload = {'delta': delta.detach().cpu(), 'optimizer': state, 'steps': steps,
                'accepted_updates': accepted, 'identity': identity,
                'fixed_protection_losses': baseline.cpu(), 'production_adoption': False}
    if successor is not None:
        payload.update(increment=delta.detach().cpu(), delta=(parent_delta + delta.detach()).cpu(),
                       parent_binding=successor['parent_binding'])
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(prepared_root, *, expanded_span=False, parent_round=None, parent_verification=None,
        learning_rate=1e-5):
    if (parent_round is None) != (parent_verification is None):
        raise ValueError('Both --parent-round and --parent-verification are required')
    is_successor = parent_round is not None
    version = 'v4-successor' if is_successor else ('v3-expanded-stable' if expanded_span else 'v3-stable')
    expanded_span = expanded_span or is_successor
    scales = EXPANDED_SCALES if expanded_span else SCALES
    destination = Path(tempfile.mkdtemp(prefix=f'fixed450-residual-{version}-', dir=prepared_root))
    event('output', directory=str(destination))
    started = time.perf_counter()
    if not isinstance(learning_rate, (int, float)) or not 0 < learning_rate <= 1e-3:
        raise ValueError('learning_rate must be finite and in (0, 1e-3]')
    report = {'status': 'running', 'protocol': f'preservation-first-residual-fixed450-{version}',
        'steps_required': 450, 'optimizer_steps': 0, 'accepted_updates': 0,
        'seed': 42, 'learning_rate': float(learning_rate), 'adam_eps': 1e-6, 'distillation_weight': 5.,
        'scales': scales, 'production_adoption': False, 'production_weights_changed': False,
        'expanded_span': expanded_span, 'initial_state': 'original zero delta; independent round, not resume',
        'constraint_solver': 'fp32-feasible-first-double-projection-v1',
        'hardware': platform.platform(), 'precision': 'BF16 frozen model and deployed tail; FP32 unmerged delta/Adam',
        'memory_measurement': 'Darwin process lifetime peak RSS; MPS driver bytes separately, overlapping',
        'cache': 'full-sequence frozen prefix captured once per TRAIN row in this lifecycle; no DEV cache',
        'scope': 'finite TRAIN constraints and same-task DEV; not all-domain protection or generalization proof'}
    private_json(destination / 'protocol.json', report)
    runtime = None
    handle = None
    delta = optimizer = None
    baseline_losses = None
    parent = parent_delta = successor = None
    total = lambda value: value
    try:
        torch.manual_seed(42)
        prepared = json.loads((prepared_root / 'preparation.json').read_text())
        if prepared.get('status') != 'prepared_no_optimizer_updates':
            raise ValueError('Successful original preparation required')
        if is_successor:
            from residual_successor import inherited_loss_tolerance, load_parent, prepare_successor
            parent = load_parent(parent_round, parent_verification, prepared_root, prepared['identity'])
            groups, data = parent['groups'], parent['data']
            if groups != json.loads((prepared_root / 'material.json').read_text())['groups']:
                raise ValueError('Successor material content or order differs from the original')
            report.update(parent_binding=parent['binding'],
                initial_state='frozen verified research parent D; zero E and fresh Adam, not resume',
                parent_role='isolated research parent; not production adoption')
        else:
            groups = json.loads((prepared_root / 'material.json').read_text())['groups']
            data = torch.load(prepared_root / 'prepared.pt', map_location='cpu', weights_only=True)
        runtime = runtime_for(destination)
        parameter = _target(runtime)[1]
        if runtime.active_core is not None or runtime.identity != prepared['identity']:
            raise ValueError('Original isolated model identity required')
        if not torch.equal(parameter.detach().cpu(), data['reference']):
            raise ValueError('Prepared original projection differs')
        report['identity'] = runtime.identity
        reference = parameter.detach().clone()
        basis = (parent['basis'] if is_successor else data['basis']).to(runtime.device)
        report['basis'] = validate_basis(basis)
        first = groups['new_training'][0]
        fresh = capture(runtime, first)
        expected_cache = data['caches'][first['id']]
        if any(not torch.equal(fresh[k], expected_cache[k]) for k in expected_cache):
            raise ValueError('Frozen prefix differs from prepared original')
        if is_successor:
            parent_delta = parent['checkpoint']['delta'].to(runtime.device).detach()
            runtime.model.requires_grad_(False)
            total = lambda value: parent_delta + value
            initial_total = total(torch.zeros_like(parent_delta))
            if not torch.equal(initial_total, parent_delta):
                raise ValueError('FP32 parent D plus zero E differs from parent D')
            report['real_graph_equivalence'] = {'parent_plus_zero_tensor_exact': True,
                'scope': 'Independent migration math audit required outside the daily training lifecycle'}
        else:
            report['real_graph_equivalence'] = residual_equivalence(runtime, fresh, reference.float())
        del data
        runtime.model.requires_grad_(False)
        probe_rows = probes()
        all_rows = groups['new_training'] + groups['new_validation'] + groups['replay_training'] + probe_rows
        baseline_generation = parent['final'] if is_successor else _generation_loaded(runtime, all_rows)
        private_json(destination / 'baseline_generation.json', baseline_generation)
        by_id = {row['id']: row for row in baseline_generation}
        if not all(by_id[row['id']]['complete'] for row in probe_rows):
            raise ValueError('All four behavior trajectories must complete')
        caches = {first['id']: fresh}
        for row in groups['new_training'] + groups['replay_training']:
            if row['id'] not in caches:
                caches[row['id']] = capture(runtime, row)
        for row in probe_rows:
            caches[row['id']] = capture(runtime, row, generated=by_id[row['id']])
        if expanded_span and not is_successor:
            from luban.evolution_progress import extend_protected_basis
            from verify_residual_round import capture_greedy_trajectory
            trajectories = [capture_greedy_trajectory(runtime.model, runtime.tokenizer, row, by_id[row['id']])
                            for row in probe_rows]
            answer_inputs = [caches[row['id']]['x'][:, :-1][caches[row['id']]['labels'][:, 1:] != -100].float()
                             for row in probe_rows]
            generated = torch.cat(answer_inputs + [item['inputs'].float() for item in trajectories])
            expanded, report['extension'] = extend_protected_basis(basis.cpu(), generated)
            basis = expanded.to(runtime.device)
            report['basis'] = {'rank': basis.shape[1], **validate_basis(basis)}
            report['extension'].update(teacher_forced_positions=sum(len(x) for x in answer_inputs),
                actual_kv_positions=sum(len(item['inputs']) for item in trajectories))
            torch.save({'basis': expanded, 'extension': report['extension']}, destination / 'expanded-basis.pt')
            del trajectories, answer_inputs, generated, expanded
        protection_rows = probe_rows + groups['replay_training']
        protection_ids = [row['id'] for row in protection_rows]
        if is_successor:
            expanded, report['extension'], successor = prepare_successor(runtime, parent, caches, initial_total)
            del initial_total
            basis = expanded.to(runtime.device)
            report['basis'] = {'rank': basis.shape[1], **validate_basis(basis)}
            report.update(mastered_train_ids=successor['mastered_train_ids'], round_index=successor['round_index'],
                          arithmetic=successor['arithmetic'], parent_correct_replay_ids=successor['parent_correct_replay_ids'])
            protection_ids = [entry['id'] for entry in successor['protection_entries']]
            torch.save({'basis': expanded, 'extension': report['extension']}, destination / 'expanded-basis.pt')
        delta = torch.nn.Parameter(torch.zeros_like(reference, dtype=torch.float32))
        optimizer = torch.optim.Adam([delta], lr=float(learning_rate), eps=1e-6)

        @torch.no_grad()
        def protected_losses(candidate):
            deployed = total(candidate)
            return torch.tensor([float(loss(runtime, caches[key], reference, deployed))
                                 for key in protection_ids], dtype=torch.float32)

        baseline_losses = protected_losses(delta)
        if is_successor:
            successor['parent_losses'] = baseline_losses.tolist()
            baseline_losses = torch.tensor([min(successor['inherited_bounds'].get(key, value), value)
                for key, value in zip(protection_ids, successor['parent_losses'])], dtype=torch.float32)
            parent_values = torch.tensor(successor['parent_losses'])
            inherited_excess = []
            for key, measured, bound in zip(protection_ids, parent_values.tolist(), baseline_losses.tolist()):
                tolerance = inherited_loss_tolerance(key, bound, successor)
                inherited_excess.append(measured - bound > tolerance)
            if any(inherited_excess):
                raise ValueError('Frozen parent already exceeds inherited fixed protection')
            torch.save(successor, destination / 'successor.pt')
        # Newly inherited TRAIN trajectories are protected with a fixed FP32
        # measurement bound. Original fixed protections remain exact zero-slack.
        retention_tolerance = torch.zeros_like(baseline_losses)
        if is_successor:
            original_fixed = parent.get('fixed') or successor.get('original_fixed') or {}
            original_fixed_ids = set(original_fixed.get('ids') or
                                     [entry['id'] for entry in successor['protection_entries']])
            for index, identifier in enumerate(protection_ids):
                if identifier not in original_fixed_ids:
                    retention_tolerance[index] = 64 * torch.finfo(torch.float32).eps * max(
                        1., abs(float(baseline_losses[index])))
            report['inherited_loss_tolerance'] = retention_tolerance.tolist()
        # Full output/loss checks are candidate-specific; only invariant inputs
        # and gradients at an unchanged accepted delta may be reused.
        signals = None
        report['preparation_seconds'] = time.perf_counter() - started
        private_json(destination / 'fixed_protection.json', {
            'ids': protection_ids, 'losses': baseline_losses.tolist(), 'basis_rank': basis.shape[1],
            'baseline': ('min inherited fixed bound and frozen parent loss; immutable' if is_successor
                         else 'original, immutable throughout all 450 steps')})
        if is_successor:
            private_json(destination / 'protocol.json', report)
        event('prepared', seconds=report['preparation_seconds'], protected=len(protection_ids))
        with (destination / 'steps.jsonl').open('w') as trace:
            for step in range(450):
                _cancel(runtime)
                began = time.perf_counter()
                current = delta.detach().clone()
                if signals is None:
                    signals = []
                    for row in probe_rows:
                        value = loss(runtime, caches[row['id']], reference, total(delta))
                        signals.append(torch.autograd.grad(value, delta)[0].detach())
                rows = groups['new_training' if step % 2 == 0 else 'replay_training']
                sample = rows[(step // 2) % len(rows)]
                replay = groups['replay_training'][(step // 2) % len(groups['replay_training'])]
                optimizer.zero_grad(set_to_none=True)
                ce = loss(runtime, caches[sample['id']], reference, total(delta))
                if not torch.isfinite(ce):
                    raise ValueError('Nonfinite supervised loss')
                before = float(ce.detach())
                ce.backward()
                del ce
                with torch.no_grad():
                    teacher = F.log_softmax(answer_logits(runtime, caches[replay['id']], reference,
                        parent_delta if is_successor else torch.zeros_like(delta)), dim=-1)
                student = F.log_softmax(answer_logits(runtime, caches[replay['id']], reference, total(delta)), dim=-1)
                kl = F.kl_div(student, teacher, reduction='batchmean', log_target=True)
                if not torch.isfinite(kl):
                    raise ValueError('Nonfinite replay KL')
                kl_value = float(kl.detach())
                (5. * kl).backward()
                del student, teacher, kl
                if delta.grad is None or not torch.isfinite(delta.grad).all():
                    raise ValueError('Invalid residual gradient')
                torch.nn.utils.clip_grad_norm_([delta], 1.)
                try:
                    optimizer.step()
                    report['optimizer_steps'] += 1
                    constrained, constraint = constrain_residual(delta.detach(), current, basis, signals)
                    selected, trials = select_update(current, constrained, basis, baseline_losses, before,
                        lambda candidate: loss(runtime, caches[sample['id']], reference, total(candidate)), protected_losses,
                        scales=scales, retention_tolerance=retention_tolerance)
                    accepted = not torch.equal(selected, current)
                    with torch.no_grad():
                        delta.copy_(selected)
                    if accepted:
                        report['accepted_updates'] += 1
                        signals = None
                except BaseException:
                    with torch.no_grad():
                        delta.copy_(current)
                    raise
                observation = {'step': step + 1, 'sample': sample['id'],
                    'role': 'new' if step % 2 == 0 else 'replay', 'supervised_before': before, 'kl': kl_value,
                    'accepted': accepted, 'constraint': constraint, 'trials': trials,
                    'seconds': time.perf_counter() - began}
                trace.write(json.dumps(observation, ensure_ascii=False) + '\n')
                trace.flush()
                if (step + 1) % 25 == 0:
                    checkpoint(destination / 'recovery.pt', delta, optimizer, step + 1,
                               report['accepted_updates'], runtime.identity, baseline_losses,
                               successor=successor, parent_delta=parent_delta)
                event('step', step=step + 1, accepted=accepted, accepted_total=report['accepted_updates'],
                      supervised_before=before, seconds=observation['seconds'])
        report['training_end_seconds'] = time.perf_counter() - started
        final_losses = protected_losses(delta)
        _, report['final_loss_guard'] = guard_retention_losses(delta.detach(), torch.zeros_like(delta),
                                                              baseline_losses, final_losses, retention_tolerance)
        if report['final_loss_guard']['rejected']:
            raise RuntimeError('Final finite protection invariant violated')
        deployed_delta = total(delta).detach()
        handle = runtime.branch.base.down_proj.register_forward_hook(
            lambda module, inputs, output: apply_core_residual(inputs[0], output, deployed_delta))
        generated = _generation_loaded(runtime, all_rows)
        handle.remove()
        handle = None
        private_json(destination / 'final_generation.json', generated)
        final = {row['id']: row for row in generated}
        report['probe_tokens_unchanged'] = all(final[row['id']]['raw_completion_token_ids'] ==
            by_id[row['id']]['raw_completion_token_ids'] for row in probe_rows)
        correct_old = (successor['parent_correct_replay_ids'] if is_successor else
                       [row['id'] for row in groups['replay_training'] if by_id[row['id']]['passed']])
        report['baseline_correct_replay'] = len(correct_old)
        report['retained_correct_replay'] = sum(final[key]['passed'] for key in correct_old)
        report['new_counts'] = {key: {'total': len(groups[key]),
            'baseline_narrow_passed': sum(by_id[row['id']]['passed'] for row in groups[key]),
            'final_narrow_passed': sum(final[row['id']]['passed'] for row in groups[key])}
            for key in ('new_training', 'new_validation')}
        report['status'] = 'fixed450_complete_research_only'
        if is_successor:
            report['inherited_raw_unchanged'] = all(
                final[key]['raw_completion_token_ids'] == observed['raw_completion_token_ids']
                and final[key]['complete'] and (key not in successor['mastered_train_ids'] or final[key]['passed'])
                for key, observed in successor['protected_raw'].items())
            report['newly_mastered_train_ids'] = [row['id'] for row in groups['new_training']
                if final[row['id']]['passed'] and row['id'] not in successor['mastered_train_ids']]
            report['mastered_train_ids_after_round'] = [row['id'] for row in groups['new_training'] if final[row['id']]['passed']]
            report['training_targets_complete'] = len(report['mastered_train_ids_after_round']) == len(groups['new_training'])
        if not torch.equal(parameter.detach(), reference):
            raise RuntimeError('Frozen original projection changed')
        report['candidate_qualified'] = bool(report['accepted_updates'] and report['probe_tokens_unchanged']
            and report['retained_correct_replay'] == len(correct_old)
            and all(value['final_narrow_passed'] == value['total'] for value in report['new_counts'].values())
            and (not is_successor or report['inherited_raw_unchanged']))
        checkpoint(destination / 'recovery.pt', delta, optimizer, 450, report['accepted_updates'],
                   runtime.identity, baseline_losses, successor=successor, parent_delta=parent_delta)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if delta is not None and optimizer is not None and baseline_losses is not None:
            checkpoint(destination / 'recovery.pt', delta, optimizer, report['optimizer_steps'],
                       report['accepted_updates'], runtime.identity, baseline_losses,
                       successor=successor, parent_delta=parent_delta)
        raise
    finally:
        try:
            try:
                if handle is not None:
                    handle.remove()
                if runtime is not None and 'reference' in locals():
                    report['original_runtime_weight_unchanged'] = bool(torch.equal(_target(runtime)[1].detach(), reference))
                if is_successor and parent_delta is not None:
                    report['parent_delta_unchanged'] = bool(torch.equal(parent_delta.detach().cpu(), parent['checkpoint']['delta']))
                if torch.backends.mps.is_available():
                    report['mps_driver_bytes_before_release'] = torch.mps.driver_allocated_memory()
            finally:
                if runtime is not None:
                    runtime._release_model()
        except BaseException as error:
            report.update(status='failed_release', release_error=str(error))
            raise
        finally:
            report.update(elapsed_seconds=time.perf_counter() - started,
                          parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            private_json(destination / 'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_root', type=Path)
    parser.add_argument('--expanded-span', action='store_true', help='Research v3: append original TRAIN answer/KV directions and try the predeclared full step first')
    parser.add_argument('--parent-round', type=Path, help='Research v4: completed frozen parent round')
    parser.add_argument('--parent-verification', type=Path, help='Matching successful independent parent verification')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Finite Adam learning rate; default preserves the registered recipe')
    args = parser.parse_args()
    root = args.prepared_root.resolve()
    if not Path('/Volumes/移动数据库').is_mount() or not root.is_relative_to(Path('/Volumes/移动数据库/lubanllm/experiments')):
        raise SystemExit('Use the mounted research experiment directory')
    run(root, expanded_span=args.expanded_span, parent_round=args.parent_round,
        parent_verification=args.parent_verification, learning_rate=args.learning_rate)
