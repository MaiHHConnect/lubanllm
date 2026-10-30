"""Independent CPU reference for actual saved matrices; no checkpoint/model loader or GPU."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from luban.evolution_progress import extend_protected_basis
from prepare_q4_packed import inverse_factor


def reference_quantize(weights, upper):
    # Retain original GPTQ blocked FP32 update order, replacing only the Q4 decoder.
    work = weights.clone()
    master = torch.empty_like(work)
    candidate = torch.empty_like(work, dtype=torch.bfloat16)
    for begin in range(0, weights.shape[1], 128):
        end = min(begin + 128, weights.shape[1])
        current = work[:, begin:end].clone()
        errors = torch.empty_like(current)
        for index in range(end - begin):
            col = begin + index
            value = current[:, index].clone()
            decoded = value.to(torch.bfloat16)
            error = (value - decoded.float()) / upper[col, col]
            master[:, col] = value
            candidate[:, col] = decoded
            current[:, index:] -= error[:, None] * upper[col, col:end][None, :]
            errors[:, index] = error
        work[:, end:] -= errors @ upper[begin:end, end:]
    return candidate, master


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'master', 'basis', 'inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--old-positions', type=int, required=True)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--width', type=int, default=6144)
    parser.add_argument('--expected-candidate', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    width, hidden = args.width, args.hidden
    assert 0 < width and 0 < hidden and hidden * width <= 12582912
    rank = args.basis.stat().st_size // (4 * width)
    positions = args.inputs.stat().st_size // (4 * width)
    assert 0 < rank < width and 0 < args.old_positions < positions and positions * width <= 10000000
    assert not args.output.exists()
    args.output.mkdir(parents=False)
    input_paths = [args.reference, args.master, args.basis, args.inputs]
    input_stats = [path.stat() for path in input_paths]
    began = time.monotonic()
    binary = Path('build/native-evolution-conversion/conversion-cpu').resolve()
    command = [str(binary), '--convert', *map(str, (args.reference, args.master, args.basis, args.inputs)),
               str(hidden), str(width), str(rank), str(positions), str(args.old_positions), str(args.output / 'native')]
    native = subprocess.run(command, text=True, capture_output=True, timeout=720)
    (args.output / 'native.stderr').write_text(native.stderr)
    assert native.returncode == 0, native.stderr
    native_report = json.loads(native.stdout)

    def read(path, shape, dtype=torch.float32):
        raw = path.read_bytes()
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
        assert torch.isfinite(tensor).all()
        return tensor

    original = read(args.reference, (hidden, width))
    master = read(args.master, (hidden, width))
    basis = read(args.basis, (width, rank))
    inputs = read(args.inputs, (positions, width))
    _, _, vh = torch.linalg.svd(basis.T.contiguous(), full_matrices=False)
    expanded, _ = extend_protected_basis(vh.T.contiguous(), inputs)
    delta = master - original
    projected = original + delta - (delta @ expanded) @ expanded.T
    upper, _ = inverse_factor(inputs.numpy())
    first, _ = reference_quantize(projected, upper)
    corrected = first.float() - ((first.float() - original) @ expanded) @ expanded.T
    expected, expected_master = reference_quantize(corrected, upper)
    actual = read(args.output / 'native/candidate.raw', (hidden, width), torch.bfloat16)
    actual_master = read(args.output / 'native/master.raw', (hidden, width))
    pairs = []
    for name, a, b in [('converted-candidate', actual.float(), expected.float()),
                       ('converted-master', actual_master, expected_master)]:
        error = float((a-b).abs().max())
        pairs.append(dict(name=name, elements=a.numel(), max_abs_error=error, allowed_max_abs_error=0))
    def stamp(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    assert all(stamp(path.stat()) == stamp(before) for path, before in zip(input_paths, input_stats))
    expected_previous = None
    if args.expected_candidate:
        expected_previous = bool(torch.equal(actual, read(args.expected_candidate, (hidden, width), torch.bfloat16)))
    passed = all(row['max_abs_error'] == 0 for row in pairs) and torch.equal(actual_master.to(torch.bfloat16), actual) and expected_previous is not False
    result = dict(schema='native-terminal-conversion-independent-reference-v1', passed=bool(passed),
                  actual_shape=[hidden, width], actual_model_shape=[hidden, width] == [2048, 6144],
                  pairs=pairs, inputs_stat_unchanged=True, previous_diagnostic_candidate_bit_exact=expected_previous,
                  native=native_report, command=command, seconds=time.monotonic()-began,
                  GPU=False, model_loaded=False, optimizer_steps=0, new_validation_used=False,
                  product_admitted=False, qualification_issued=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if not passed:
        raise SystemExit('Independent conversion reference differs; thresholds were not adjusted')


if __name__ == '__main__':
    main()
