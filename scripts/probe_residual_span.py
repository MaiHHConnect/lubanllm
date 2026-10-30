"""One shared FP32 Adam proposal, factorial protection-span diagnostic; no adoption."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

import torch

from train_luban_identity import runtime_for
from train_preserved_residual import loss
from study_core_constraint import capture, event
from probe_core_constraint import probes, residual_equivalence
from verify_residual_round import capture_greedy_trajectory, generate
from luban.core_update import (apply_core_residual, constrain_residual,
    guard_retention_losses, validate_basis)
from luban.evolution import _target, _cancel
from luban.evolution_progress import extend_protected_basis
from luban.settings import private_json

SCALES = (1., .25, .0625, .015625, .00390625)
ARMS = ('original_joint', 'expanded_joint', 'original_projection', 'expanded_projection')


def proposal_for(proposed, basis, gradients, joint):
    """All arms consume the same proposal; no arm changes Adam or the parent."""
    validate_basis(basis)
    if joint:
        return constrain_residual(proposed, torch.zeros_like(proposed), basis, gradients)
    projected = proposed - (proposed @ basis) @ basis.T
    if not bool(torch.isfinite(projected).all()):
        raise ValueError('Nonfinite projected proposal')
    return projected, {'method': 'existing cumulative nullspace projection only'}


def run(root, round_directory):
    started = time.perf_counter()
    destination = Path(tempfile.mkdtemp(prefix='residual-span-proposal-', dir=root))
    event('output', directory=str(destination))
    report = {'status': 'running', 'protocol': 'residual-span-one-proposal-v1',
        'arms': ARMS, 'scales': SCALES, 'seed': 42, 'optimizer_proposals': 0,
        'learning_rate': 1e-5, 'adam_eps': 1e-6, 'clip_norm': 1.,
        'precision': 'original BF16 model; FP32 unmerged delta and Adam; CPU FP32 span extension',
        'hardware': platform.platform(), 'torch': torch.__version__,
        'cache': 'fresh frozen TRAIN prefixes once; actual original KV traces once; no DEV',
        'memory_measurement': 'Darwin lifetime peak RSS and overlapping MPS driver bytes separately',
        'baseline_round': str(round_directory), 'production_adoption': False,
        'scope': 'first NEW sample CE improvement under finite protection; not mastery or universal preservation',
        'eligible_definition': 'nonzero, subspace tolerance, strict NEW CE improvement, all fixed losses nonincreasing, four raw generations unchanged; local KV equality reported separately',
        'initial_state': 'original zero delta, not continuation or reprojection of the learned parent',
        'candidates': []}
    private_json(destination / 'protocol.json', report)
    runtime = None
    reference = None
    try:
        prepared = json.loads((root / 'preparation.json').read_text())
        previous = json.loads((round_directory / 'result.json').read_text())
        if (prepared.get('status') != 'prepared_no_optimizer_updates'
                or previous.get('status') != 'fixed450_complete_research_only'
                or previous.get('optimizer_steps') != 450
                or previous.get('original_runtime_weight_unchanged') is not True
                or previous.get('identity') != prepared.get('identity')
                or previous.get('final_loss_guard', {}).get('rejected') is not False):
            raise ValueError('Completed original bound preparation and protected450 required')
        groups = json.loads((root / 'material.json').read_text())['groups']
        old_fixed = json.loads((round_directory / 'fixed_protection.json').read_text())
        original_raw = {row['id']: row for row in json.loads((round_directory / 'baseline_generation.json').read_text())}
        protected = probes() + groups['replay_training']
        if old_fixed['ids'] != [row['id'] for row in protected]:
            raise ValueError('Original protection rows differ')
        data = torch.load(root / 'prepared.pt', map_location='cpu', weights_only=True, mmap=True)
        torch.manual_seed(42)
        runtime = runtime_for(destination)
        parameter = _target(runtime)[1]
        if runtime.active_core is not None or runtime.identity != prepared['identity']:
            raise ValueError('Original isolated core required')
        if not torch.equal(parameter.detach().cpu(), data['reference']):
            raise ValueError('Original projection differs')
        reference = parameter.detach().clone()
        runtime.model.requires_grad_(False)
        report['identity'] = runtime.identity
        first = groups['new_training'][0]
        caches = {}
        for row in [first] + groups['replay_training']:
            fresh = capture(runtime, row)
            if any(not torch.equal(fresh[key], data['caches'][row['id']][key]) for key in fresh):
                raise ValueError('Original frozen TRAIN prefix differs: ' + row['id'])
            caches[row['id']] = fresh
        report['real_graph_equivalence'] = residual_equivalence(runtime, caches[first['id']], reference.float())
        trajectories = []
        answer_inputs = []
        for row in probes():
            observed = original_raw[row['id']]
            trajectories.append(capture_greedy_trajectory(runtime.model, runtime.tokenizer, row, observed))
            cache = capture(runtime, row, generated=observed)
            caches[row['id']] = cache
            answer_inputs.append(cache['x'][:, :-1][cache['labels'][:, 1:] != -100].float())
        generated = torch.cat(answer_inputs + [item['inputs'].float() for item in trajectories])
        old_basis = data['basis']
        expanded, extension = extend_protected_basis(old_basis, generated)
        bases = {'original': old_basis.to(runtime.device), 'expanded': expanded.to(runtime.device)}
        report['basis'] = {name: {'rank': basis.shape[1], **validate_basis(basis)} for name, basis in bases.items()}
        report['extension'] = {**extension, 'teacher_forced_positions': sum(len(x) for x in answer_inputs),
            'actual_kv_positions': sum(len(item['inputs']) for item in trajectories)}
        torch.save({'basis': expanded, 'extension': report['extension']}, destination / 'expanded-basis.pt')
        del data, generated, answer_inputs
        delta = torch.nn.Parameter(torch.zeros_like(reference, dtype=torch.float32))
        optimizer = torch.optim.Adam([delta], lr=1e-5, eps=1e-6)

        @torch.no_grad()
        def measure(candidate):
            return torch.tensor([float(loss(runtime, caches[row['id']], reference, candidate))
                                 for row in protected], dtype=torch.float32)

        baseline = measure(delta)
        if not torch.equal(baseline, torch.tensor(old_fixed['losses'], dtype=torch.float32)):
            raise ValueError('Original fixed protection baseline differs')
        report['fixed_baseline_exact'] = True
        report['protection_ids'] = old_fixed['ids']
        report['fixed_protection_losses'] = baseline.tolist()
        gradients = [torch.autograd.grad(loss(runtime, caches[row['id']], reference, delta), delta)[0].detach()
                     for row in probes()]
        ce = loss(runtime, caches[first['id']], reference, delta)
        report['new_ce_before'] = float(ce.detach())
        ce.backward()
        if delta.grad is None or not bool(torch.isfinite(delta.grad).all()):
            raise ValueError('Invalid FP32 residual gradient')
        # At this first zero-delta state the identical teacher/student KL has zero gradient.
        report['kl'] = 'identical original teacher/student at initial zero; zero KL gradient'
        torch.nn.utils.clip_grad_norm_([delta], 1.)
        optimizer.step()
        proposed = delta.detach().clone()
        report['optimizer_proposals'] = 1
        report['preparation_seconds'] = time.perf_counter() - started
        event('prepared', **report['extension'], basis=report['basis'])
        for arm in ARMS:
            basis = bases[arm.split('_')[0]]
            constrained, constraint = proposal_for(proposed, basis, gradients, arm.endswith('_joint'))
            for scale in SCALES:
                _cancel(runtime)
                candidate = constrained * scale
                entry = {'arm': arm, 'scale': scale, 'constraint': constraint, 'eligible': False}
                report['candidates'].append(entry)
                with torch.no_grad():
                    entry['new_ce'] = float(loss(runtime, caches[first['id']], reference, candidate))
                    values = measure(candidate)
                    _, entry['loss_guard'] = guard_retention_losses(candidate, torch.zeros_like(candidate), baseline, values)
                    entry['protection_losses'] = values.tolist()
                    entry['subspace_residual'] = float((candidate @ basis).norm())
                    tolerance = 64 * torch.finfo(torch.float32).eps * max(1., float(candidate.norm()))
                    entry['subspace_tolerance'] = tolerance
                    entry['subspace_passed'] = entry['subspace_residual'] <= tolerance
                    entry['nonzero'] = bool(torch.count_nonzero(candidate))
                    entry['local_kv_outputs_exact'] = all(torch.equal(
                        apply_core_residual(item['inputs'].to(runtime.device), item['outputs'].to(runtime.device), candidate).cpu(),
                        item['outputs']) for item in trajectories)
                if not torch.isfinite(torch.tensor(entry['new_ce'])):
                    raise ValueError('Nonfinite supervised loss')
                eligible = (entry['nonzero'] and entry['subspace_passed'] and not entry['loss_guard']['rejected']
                            and entry['new_ce'] < report['new_ce_before'])
                if eligible:
                    handle = runtime.branch.base.down_proj.register_forward_hook(
                        lambda module, inputs, output: apply_core_residual(inputs[0], output, candidate))
                    try:
                        entry['generation'] = [generate(runtime.model, runtime.tokenizer, row, original_raw[row['id']])
                                               for row in probes()]
                        entry['new_generation'] = generate(runtime.model, runtime.tokenizer, first, original_raw[first['id']])
                    finally:
                        handle.remove()
                    entry['eligible'] = all(item['tokens_match_round'] and item['complete']
                        and item['validator_matches_round'] for item in entry['generation'])
                entry['new_ce_change'] = entry['new_ce'] - report['new_ce_before']
                event('candidate', arm=arm, scale=scale, eligible=entry['eligible'],
                      new_ce_change=entry['new_ce_change'], rejected=entry['loss_guard']['rejected'])
                private_json(destination / 'result.json', report)
        if not torch.equal(parameter.detach(), reference):
            raise RuntimeError('Frozen original projection changed')
        report['status'] = 'single_proposal_comparison_complete'
        report['eligible_candidates'] = sum(item['eligible'] for item in report['candidates'])
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if runtime is not None and reference is not None:
            report['original_runtime_weight_unchanged'] = bool(torch.equal(_target(runtime)[1].detach(), reference))
        report.update(elapsed_seconds=time.perf_counter() - started,
            parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if torch.backends.mps.is_available():
            report['mps_driver_bytes_before_release'] = torch.mps.driver_allocated_memory()
        try:
            if runtime is not None:
                runtime._release_model()
        except BaseException as error:
            report.update(status='failed_release', release_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            private_json(destination / 'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_root', type=Path)
    parser.add_argument('round_directory', type=Path)
    args = parser.parse_args()
    root = args.prepared_root.resolve()
    if not Path('/Volumes/移动数据库').is_mount() or not root.is_relative_to(Path('/Volumes/移动数据库/lubanllm/experiments')):
        raise SystemExit('Use the mounted research experiment directory')
    run(root, args.round_directory.resolve())
