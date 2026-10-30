"""TRAIN-only nonzero-residual math and identical-prefix KV diagnostics.

No optimizer, updates, adoption, source snapshots or teacher calls. Real MPS
execution is explicit and must wait for the current model owner to finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

import torch
from torch.nn import functional as F
from transformers import __version__ as transformers_version

from train_luban_identity import MODEL, runtime_for
from study_core_constraint import capture, residual_logits
from verify_local_review import model_identity
from verify_residual_round import load_round
from verify_training_operators import TOLERANCES, _comparison
from verify_cached_batch_training import _labels
from luban.core_update import apply_core_residual
from luban.evolution import _answer_logits, _target
from luban.retention import messages
from luban.settings import private_json


TRAIN_IDS = ('brand-v2-train-0', 'context-v3-training-unknown-direct',
             'context-v3-training-known-ning', 'brand-v2-train-7', 'context-v3-training-known-lin')


def select_train(groups):
    training = {row['id']: row for row in groups['new_training']}
    selected = [training[identifier] for identifier in TRAIN_IDS]
    if any(row.get('purpose') != 'training' for row in selected):
        raise ValueError('Only the five preregistered TRAIN rows may enter this diagnostic')
    return selected


def compare(actual, expected, kind='logits'):
    tolerance = TOLERANCES['torch.bfloat16']
    return _comparison(actual, expected, atol=tolerance[kind + '_atol'],
                       rtol=tolerance['rtol'], relative_l2=tolerance['relative_l2'])


def token_stats(values, target):
    values = values.detach().float().cpu()
    if values.ndim != 1 or not bool(torch.isfinite(values).all()) or not 0 <= target < values.numel():
        raise ValueError('Expected finite full-vocabulary predictor logits and a valid target')
    top = int(values.argmax())
    competitors, indices = values.topk(2)
    competitor = competitors[0] if int(indices[0]) != target else competitors[1]
    log_probability = values[target] - torch.logsumexp(values, dim=0)
    return {'target_token_id': target, 'top1_token_id': top, 'argmax_correct': top == target,
            'target_probability': float(log_probability.exp()), 'target_nll': float(-log_probability),
            'target_rank': int((values > values[target]).sum()) + 1,
            'target_minus_highest_other_logit': float(values[target] - competitor)}


def full_cached_math(runtime, cache, delta):
    """Existing exact residual-equivalence gate; tolerances remain diagnostic."""
    device = delta.device
    ids, labels = (cache[key].to(device) for key in ('ids', 'labels'))
    full = runtime.model(ids, labels=labels, use_cache=False)
    full_values, full_loss = full.logits.detach().cpu(), full.loss.detach().cpu()
    full_gradient = torch.autograd.grad(full.loss, delta)[0].detach().cpu()
    del full
    cached = residual_logits(runtime, cache, _target(runtime)[1].detach(), delta)
    selected = _answer_logits(cached, labels)
    loss = F.cross_entropy(selected, labels[:, 1:][labels[:, 1:] != -100])
    gradient = torch.autograd.grad(loss, delta)[0].detach().cpu()
    results = {key: compare(actual, expected, key) for key, actual, expected in (
        ('logits', cached.detach().cpu(), full_values), ('loss', loss.detach().cpu(), full_loss),
        ('gradient', gradient, full_gradient))}
    results['exact_equivalence'] = all(results[key]['exact'] for key in ('logits', 'loss', 'gradient'))
    results['full_sequence_ce'] = float(full_loss)
    results['cached_sequence_ce'] = float(loss.detach())
    return results, _answer_logits(full_values, labels.cpu())


def prefix_positions(raw, gold):
    divergence = next((index for index, (a, b) in enumerate(zip(raw, gold)) if a != b), None)
    if divergence is None and len(raw) < len(gold):
        divergence = len(raw)
    repeat = None
    for end in range(4, len(raw) + 1):
        for width in range(2, min(16, end // 2) + 1):
            if raw[end - 2 * width:end - width] == raw[end - width:end]:
                repeat = {'after_prefix_tokens': end, 'width': width, 'token_ids': raw[end - width:end]}
                break
        if repeat is not None:
            break
    reasons = {0: ['first_prediction']}
    for position, reason in ((divergence, 'first_gold_divergence'),
                             (None if repeat is None else repeat['after_prefix_tokens'], 'after_first_adjacent_repeat')):
        if position is not None and position < len(raw):
            reasons.setdefault(position, []).append(reason)
    for position in (1, 2, len(raw) - 1):
        if len(reasons) >= min(3, len(raw)):
            break
        reasons.setdefault(position, []).append('additional_same_prefix_control')
    return divergence, repeat, {index: reason for index, reason in sorted(reasons.items()) if 0 <= index < len(raw)}


@torch.no_grad()
def kv_predictions(model, prefix, targets, *, teacher_forced=None, keys=()):
    """Official preparation/update helpers; one prefill, then one-token decode.

    Targets are fed verbatim. Agreement with every recorded greedy target is
    reported separately; forcing recorded tokens alone does not prove replay.
    """
    device = next(model.parameters()).device
    sequence = prefix.to(device)
    if sequence.ndim != 2 or sequence.shape[0] != 1 or not sequence.shape[1] or not targets:
        raise ValueError('KV tracing needs a complete single prompt and nonempty targets')
    kwargs = {'attention_mask': torch.ones_like(sequence), 'use_cache': True}
    kwargs['position_ids'] = model._prepare_position_ids_for_generation(sequence, kwargs)
    records, saved, lengths = [], {}, []
    for index, target in enumerate(targets):
        inputs = model.prepare_inputs_for_generation(sequence, next_sequence_length=None if index == 0 else 1,
            is_first_iteration=index == 0, **kwargs)
        lengths.append(inputs['input_ids'].shape[1])
        output = model(**inputs, return_dict=True)
        values = output.logits[0, -1].detach().float().cpu()
        record = {'prediction_index': index, **token_stats(values, target)}
        if teacher_forced is not None:
            record['full_sequence_tf_vs_kv'] = compare(values, teacher_forced[index])
            record['tf_top1_token_id'] = int(teacher_forced[index].argmax())
        records.append(record)
        if index in keys:
            saved[index] = values
        if index + 1 < len(targets):
            kwargs = model._update_model_kwargs_for_generation(output, kwargs, is_encoder_decoder=False)
            sequence = torch.cat((sequence, torch.tensor([[target]], device=device)), dim=1)
    if lengths != [prefix.shape[1]] + [1] * (len(targets) - 1):
        raise ValueError('Expected one prefill followed by single-token cached decoding')
    return records, saved, lengths


@torch.no_grad()
def full_prefix_controls(model, prefix, raw, saved, keys, gold):
    controls = []
    device = next(model.parameters()).device
    for index, reasons in keys.items():
        same_prefix = torch.cat((prefix.cpu(), torch.tensor([raw[:index]], dtype=torch.long)), dim=1).to(device)
        values = model(same_prefix, attention_mask=torch.ones_like(same_prefix), use_cache=False).logits[0, -1].float().cpu()
        record = {'prediction_index': index, 'reasons': reasons,
            'prefix_token_ids': same_prefix[0].tolist(), 'tf_vs_kv_same_prefix': compare(values, saved[index]),
            'uncached': token_stats(values, raw[index]), 'cached': token_stats(saved[index], raw[index])}
        if index < len(gold) and raw[:index] == gold[:index]:
            record['gold_at_matching_prefix'] = {'uncached': token_stats(values, gold[index]),
                                                'cached': token_stats(saved[index], gold[index])}
        controls.append(record)
    return controls


def summarize_gold(records):
    first = next((row['prediction_index'] for row in records if not row['argmax_correct']), None)
    return {'mean_ce': sum(row['target_nll'] for row in records) / len(records),
            'every_gold_argmax_correct_including_eos': first is None, 'first_gold_argmax_error': first,
            'first_tf_kv_top1_difference': next((row['prediction_index'] for row in records
                if row['top1_token_id'] != row['tf_top1_token_id']), None),
            'first_tf_kv_numeric_failure': next((row['prediction_index'] for row in records
                if not row['full_sequence_tf_vs_kv']['passed']), None)}


def diagnose(runtime, rows, caches, delta, observed):
    if (runtime.model.training or runtime.branch.expert is not None or any(row.get('purpose') != 'training' for row in rows)
            or any(value.requires_grad for value in runtime.model.parameters()) or delta.dtype != torch.float32
            or delta.shape != _target(runtime)[1].shape or not bool(torch.isfinite(delta).all())
            or not bool(delta.count_nonzero()) or not delta.requires_grad):
        raise ValueError('Need frozen eval TRAIN graph and a nonzero FP32 differentiable residual')
    parameters = [(value, value._version) for value in runtime.model.parameters()]
    delta_state = (id(delta), delta.data_ptr(), delta._version, delta.grad)
    active = [delta]
    handle = runtime.branch.base.down_proj.register_forward_hook(
        lambda module, inputs, output: apply_core_residual(inputs[0], output, active[0]))
    results = []
    try:
        for row in rows:
            cache = caches[row['id']]
            _labels(cache['ids'], cache['labels'], {runtime.tokenizer.eos_token_id})
            start = int(cache['labels'][0].ne(-100).nonzero()[0])
            gold = cache['ids'][0, start:].tolist()
            prefix = cache['ids'][:, :start]
            math, tf = full_cached_math(runtime, cache, delta)
            gold_kv, _, lengths = kv_predictions(runtime.model, prefix, gold, teacher_forced=tf)
            raw = observed[row['id']]['raw_completion_token_ids']
            divergence, repeat, keys = prefix_positions(raw, gold)
            generated, saved, generated_lengths = kv_predictions(runtime.model, prefix, raw, keys=keys)
            controls = full_prefix_controls(runtime.model, prefix, raw, saved, keys, gold)
            results.append({'id': row['id'], 'math': math, 'gold_token_ids_including_training_eos': gold,
                'trained_gold_tf': [token_stats(values, target) for values, target in zip(tf, gold)],
                'trained_gold_kv': gold_kv, 'trained_gold_summary': summarize_gold(gold_kv),
                'gold_forward_lengths': lengths, 'observed_raw_token_ids': raw,
                'observed_greedy_reproduced': all(item['argmax_correct'] for item in generated),
                'observed_forward_lengths': generated_lengths, 'observed_path_predictions': generated,
                'first_observed_gold_divergence': divergence, 'first_adjacent_token_repeat': repeat,
                'trained_same_prefix_controls': controls})
        # Zero is a separate fixed tensor; never overwrite or reproject trained D.
        active[0] = torch.zeros_like(delta, requires_grad=False)
        for result, row in zip(results, rows):
            cache = caches[row['id']]
            labels = cache['labels'].to(delta.device)
            with torch.no_grad():
                baseline = runtime.model(cache['ids'].to(delta.device), use_cache=False).logits
                tf = _answer_logits(baseline, labels).cpu()
            gold = result['gold_token_ids_including_training_eos']
            prefix = cache['ids'][:, :-len(gold)]
            gold_kv, _, _ = kv_predictions(runtime.model, prefix, gold, teacher_forced=tf)
            raw = result['observed_raw_token_ids']
            _, _, keys = prefix_positions(raw, gold)
            _, saved, _ = kv_predictions(runtime.model, prefix, raw, keys=keys)
            result.update(zero_gold_tf=[token_stats(values, target) for values, target in zip(tf, gold)],
                zero_gold_kv=gold_kv, zero_gold_summary=summarize_gold(gold_kv),
                zero_same_prefix_controls=full_prefix_controls(runtime.model, prefix, raw, saved, keys, gold))
            result['zero_full_sequence_ce'] = float(F.cross_entropy(tf, torch.tensor(gold)))
            result['mean_tf_ce_decreased'] = result['math']['full_sequence_ce'] < result['zero_full_sequence_ce']
            result['trained_every_tf_gold_argmax_correct_including_eos'] = all(item['argmax_correct'] for item in result['trained_gold_tf'])
    finally:
        handle.remove()
        if ([id(value) for value in runtime.model.parameters()] != [id(value) for value, _ in parameters]
                or any(value._version != version for value, version in parameters)
                or (id(delta), delta.data_ptr(), delta._version) != delta_state[:3]
                or delta.grad is not delta_state[3]):
            raise RuntimeError('Diagnostic changed original parameter objects/versions or residual state')
    for result in results:
        result['original_parameter_objects_versions_and_delta_state_unchanged'] = True
    return results


def run(directory):
    directory = Path(directory).resolve()
    volume = Path('/Volumes/移动数据库')
    if not volume.is_mount() or not directory.is_relative_to(volume / 'lubanllm/experiments'):
        raise ValueError('Use the mounted external experiment directory')
    destination = Path(tempfile.mkdtemp(prefix='residual-generation-diagnostic-', dir=directory.parent))
    began = time.perf_counter()
    report = {'status': 'preparing', 'round_directory': str(directory), 'train_ids': list(TRAIN_IDS),
        'seed': 42, 'base_core': None, 'precision': 'BF16 frozen original; FP32 trained unmerged residual/gradient',
        'hardware': platform.platform(), 'torch_version': torch.__version__, 'transformers_version': transformers_version,
        'optimizer_updates': 0, 'teacher_calls': 0, 'production_adoption': False,
        'equivalence_gate': 'Existing residual exact logits/CE/gradient; existing BF16 tolerances reported without loosening',
        'kv_scope': 'Gold forcing and recorded raw forcing; greedy replay requires every recorded target be top1',
        'repeat_definition': 'First adjacent repeated token block of length 2..16; not a semantic repetition verdict',
        'memory_measurement': 'Darwin lifetime peak RSS bytes; overlapping MPS driver bytes separately'}
    private_json(destination / 'result.json', report)
    print(json.dumps({'output': str(destination)}, ensure_ascii=False), flush=True)
    runtime = delta = parameter = None
    try:
        identity = model_identity(MODEL)
        groups, _, source, checkpoint, data, _, _, final = load_round(directory, identity)
        rows = select_train(groups)
        if not torch.backends.mps.is_available():
            raise RuntimeError('Real execution requires MPS')
        torch.manual_seed(42)
        runtime = runtime_for(destination)  # Existing ModelProcessLock, acquired exactly once.
        parameter = _target(runtime)[1]
        if runtime.active_core is not None or runtime.identity != identity or runtime._model_process_lock is None:
            raise ValueError('Original identity and exclusive resident runtime required')
        if not torch.equal(parameter.detach().cpu(), data['reference']):
            raise ValueError('Original projection differs from the preparation')
        runtime.model.requires_grad_(False)
        caches = {row['id']: capture(runtime, row) for row in rows}
        if any(not torch.equal(caches[row['id']][key], data['caches'][row['id']][key])
               for row in rows for key in ('ids', 'labels')):
            raise ValueError('Selected TRAIN tokenization differs from the preparation')
        observed = {row['id']: row for row in final}
        for row in rows:
            cache = caches[row['id']]
            start = int(cache['labels'][0].ne(-100).nonzero()[0])
            generated_prefix = runtime._tokens(messages(row), 'off',
                output_reserve=observed[row['id']]['max_new_tokens'])
            if not torch.equal(generated_prefix.cpu(), cache['ids'][:, :start]):
                raise ValueError('TRAIN teacher-forced prefix differs from the actual generation prompt')
        report['training_and_generation_prefixes_exact'] = True
        delta = checkpoint['delta'].to(runtime.device).detach().requires_grad_()
        report.update(identity=identity, source_candidate_qualified=source['candidate_qualified'],
                      preparation_seconds=time.perf_counter() - began, delta_nonzero_elements=int(delta.count_nonzero()))
        report['samples'] = diagnose(runtime, rows, caches, delta, observed)
        for sample, row in zip(report['samples'], rows):
            sample.update(gold_answer=row['answer'], observed_completion=observed[row['id']]['answer'])
        report['delta_values_unchanged'] = bool(torch.equal(delta.detach().cpu(), checkpoint['delta']))
        report['nonzero_math_exact'] = all(row['math']['exact_equivalence'] for row in report['samples'])
        report['status'] = 'diagnostic_complete' if report['nonzero_math_exact'] and report['delta_values_unchanged'] else 'diagnostic_math_failed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            delta = parameter = None
            if runtime is not None:
                try:
                    report['mps_driver_bytes_before_release'] = torch.mps.driver_allocated_memory()
                finally:
                    try:
                        runtime._release_model()
                    finally:
                        report['model_lock_released'] = runtime._model_process_lock is None
        except BaseException as error:
            report.update(status='failed_release', release_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            report.update(elapsed_seconds=time.perf_counter() - began,
                          lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            private_json(destination / 'result.json', report)
    return destination, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round_directory', type=Path)
    parser.add_argument('--execute', action='store_true', help='Only after the current model owner has finished')
    args = parser.parse_args()
    if args.execute:
        _, result = run(args.round_directory)
        raise SystemExit(0 if result['status'] == 'diagnostic_complete' else 1)
    print('Diagnostic code prepared; no model loaded. Real execution requires --execute and an available model lock.')
