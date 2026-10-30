"""Frozen eager HF output VJPs with one complete decoder-layer graph at a time.

This is a recomputation of the HF autograd proxy, not a native Metal derivative.
No sequence is truncated and no recurrent state is carried across separate calls.
"""
import gc

import torch
import torch.nn.functional as F


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _frozen(model):
    _require(not any(module.training for module in model.modules()), 'all modules must be in eval mode')
    _require(all(not p.requires_grad and p.grad is None for p in model.parameters()),
             'all parameters must remain frozen with grad=None')


def _finite(value, message):
    _require(bool(torch.isfinite(value).all()), message)


def _cpu(value):
    # A CPU toy must not alias the replay input either.
    return value.detach().to(device='cpu', copy=True)


def _capture(model, decoder, ids, positions, targets, reduction):
    boundaries, arguments, handles = [], [], []

    def capture_layer(index):
        def hook(_module, args, kwargs):
            _require(len(boundaries) == index and len(args) == 1,
                     'decoder layers must execute once in original order with one hidden input')
            value = args[0]
            _require(value.ndim == 3 and value.shape[:2] == ids.shape
                     and value.dtype == torch.bfloat16 and not value.requires_grad,
                     'complete frozen BF16 layer boundary required')
            _require(kwargs.get('past_key_values') is None and kwargs.get('use_cache') is False,
                     'cache-free complete-sequence layer call required')
            _finite(value, 'nonfinite layer boundary')
            boundaries.append(_cpu(value))
            # Retain actual official mask/RoPE/position arguments, shared between
            # layers where the official decoder shares them. Do not recreate them.
            arguments.append(dict(kwargs))
        return hook

    def capture_norm(_module, args, kwargs):
        _require(len(boundaries) == len(decoder.layers) and len(args) == 1 and not kwargs,
                 'one final norm after all decoder layers required')
        value = args[0]
        _require(value.shape == boundaries[0].shape and value.dtype == torch.bfloat16,
                 'final norm input boundary differs')
        _finite(value, 'nonfinite final boundary')
        boundaries.append(_cpu(value))

    try:
        for index, layer in enumerate(decoder.layers):
            handles.append(layer.register_forward_pre_hook(capture_layer(index), with_kwargs=True))
        handles.append(decoder.norm.register_forward_pre_hook(capture_norm, with_kwargs=True))
        with torch.no_grad():
            hidden = decoder(ids, use_cache=False, past_key_values=None).last_hidden_state
            logits = model.lm_head(hidden[:, positions, :])[0]
            _finite(logits, 'nonfinite baseline logits')
            loss = F.cross_entropy(logits.float(), targets, reduction=reduction)
            _finite(loss, 'nonfinite baseline CE')
            baseline, loss_value = _cpu(logits), float(loss.item())
        _require(len(boundaries) == len(decoder.layers)+1 and len(arguments) == len(decoder.layers),
                 'incomplete official decoder boundary capture')
        return boundaries, arguments, baseline, loss_value
    finally:
        for handle in handles:
            handle.remove()


def _tail(model, decoder, boundary, device, positions, targets, reduction, baseline, baseline_loss):
    hidden = boundary.to(device=device, copy=True).requires_grad_(True)
    logits = model.lm_head(decoder.norm(hidden)[:, positions, :])[0]
    loss = F.cross_entropy(logits.float(), targets, reduction=reduction)
    _require(torch.equal(_cpu(logits), baseline) and float(loss.item()) == baseline_loss,
             'recomputed final norm/head logits or CE changed')
    gradient, = torch.autograd.grad(loss, (hidden,))
    _require(gradient.shape == hidden.shape and gradient.dtype == hidden.dtype,
             'final hidden cotangent shape/dtype changed')
    _finite(gradient, 'nonfinite final hidden cotangent')
    return gradient.detach()


def _layer_vjp(layer, index, boundary, arguments, expected, cotangent, selected, progress):
    hidden = boundary.to(device=cotangent.device, copy=True).requires_grad_(True)
    captured, handles = {}, []

    def capture(kind):
        def hook(_module, _args, value):
            _require(kind not in captured and value.requires_grad
                     and value.dtype == torch.bfloat16 and value.ndim == 3
                     and value.shape[:2] == hidden.shape[:2],
                     'duplicate or disconnected complete BF16 projection output')
            captured[kind] = value
        return hook

    try:
        for kind in selected:
            handles.append(getattr(layer.mlp, kind+'_proj').register_forward_hook(capture(kind)))
        output = layer(hidden, **arguments)
        _require(output.shape == hidden.shape and output.dtype == hidden.dtype
                 and torch.equal(_cpu(output), expected),
                 f'layer {index} recomputed forward differs from official boundary')
        _require(set(captured) == set(selected), 'missing selected projection outputs')
        if progress is not None:
            progress({'stage': 'layer_forward_exact', 'layer': index})
        values = (hidden, *(captured[kind] for kind in selected))
        gradients = torch.autograd.grad(output, values, grad_outputs=cotangent)
        exported = {}
        for value, gradient in zip(values, gradients, strict=True):
            _require(value.shape == gradient.shape and gradient.dtype == torch.bfloat16,
                     'layer VJP shape/dtype changed')
            _finite(gradient, 'nonfinite layer VJP')
        for kind, gradient in zip(selected, gradients[1:], strict=True):
            exported[f'layer{index:02d}-{kind}'] = _cpu(gradient)
        return gradients[0].detach(), exported
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()


def _release(device):
    gc.collect()
    if device.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()


def layerwise_gradients(model, decoder, ids, positions, targets, projections, *, reduction='sum', progress=None):
    """Return CPU logits, scalar CE, ordered CPU BF16 gradients and control facts.

    ``projections`` is the explicit ordered sequence of ``(layer, gate/up/down)``.
    Every decoder layer is differentiated, including layers without requested
    outputs. ``progress`` receives a small event dict after completed phases.
    The caller owns model admission, the fixed32 scope and sample/row mapping.
    """
    _frozen(model)
    _require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
             'ordinary autograd context required')
    _require(getattr(decoder.config, '_attn_implementation', None) == 'eager',
             'explicit eager attention required')
    _require(reduction in ('sum', 'mean'), 'scalar sum or mean CE required')
    _require(ids.dtype == positions.dtype == targets.dtype == torch.int64
             and ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > 0
             and positions.ndim == targets.ndim == 1 and len(positions) == len(targets) > 0
             and ids.device == positions.device == targets.device,
             'one complete token trajectory with paired prediction positions/targets required')
    _require(bool((positions >= 0).all()) and bool((positions < ids.shape[1]).all())
             and bool((positions[1:] > positions[:-1]).all()), 'ordered unique valid prediction positions required')
    _require(len(decoder.layers) > 0, 'nonempty decoder required')
    projections = list(projections)
    _require(projections and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in projections),
             'explicit projection pairs required')
    projections = [tuple(p) for p in projections]
    _require(all(type(index) is int and 0 <= index < len(decoder.layers)
                 and kind in ('gate', 'up', 'down') for index, kind in projections)
             and len(set(projections)) == len(projections), 'unique valid layer/projection pairs required')
    by_layer = {i: [kind for index, kind in projections if index == i] for i in range(len(decoder.layers))}
    boundaries, arguments, gradients, cotangent = [], [], {}, None
    try:
        boundaries, arguments, baseline, loss = _capture(model, decoder, ids, positions, targets, reduction)
        boundary_bytes = sum(value.numel()*value.element_size() for value in boundaries)
        if progress is not None:
            progress({'stage': 'baseline_captured', 'boundaries': len(boundaries), 'boundary_bytes': boundary_bytes})
        cotangent = _tail(model, decoder, boundaries[-1], ids.device, positions, targets, reduction, baseline, loss)
        _release(ids.device)
        if progress is not None:
            progress({'stage': 'head_vjp_complete'})
        completed = []
        for index in reversed(range(len(decoder.layers))):
            cotangent, exported = _layer_vjp(decoder.layers[index], index, boundaries[index], arguments[index],
                                             boundaries[index+1], cotangent, by_layer[index], progress)
            gradients.update(exported)
            completed.append(index)
            _release(ids.device)
            if progress is not None:
                progress({'stage': 'layer_vjp_complete', 'layer': index, 'projections': len(exported)})
        _frozen(model)
        ordered = {f'layer{index:02d}-{kind}': gradients[f'layer{index:02d}-{kind}'] for index, kind in projections}
        controls = {'method': 'official_full_sequence_layerwise_vjp', 'reduction': reduction,
                    'input_tokens': ids.shape[1], 'target_tokens': len(targets),
                    'captured_boundaries': len(boundaries), 'boundary_bytes': boundary_bytes,
                    'layer_order': completed, 'layer_forward_exact': True, 'head_logits_loss_exact': True,
                    'finite_gradients': True, 'projection_count': len(ordered), 'parameters_frozen': True,
                    'use_cache': False, 'sequence_truncated': False, 'parameter_updates': 0,
                    'full_graph_gradient_equivalence_tested': False, 'native_gradient_verified': False}
        return baseline, loss, ordered, controls
    finally:
        boundaries.clear()
        arguments.clear()
        gradients.clear()
        cotangent = None
        _release(ids.device)
