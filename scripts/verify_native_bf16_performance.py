#!/usr/bin/env python3
"""Verify bound native optimization evidence, without approving product adoption."""
import ast
import json
import math
import statistics
from pathlib import Path

from verify_native_bf16 import ROOT, sha


def main():
    index = ROOT / 'research/native_bf16_performance_result.json'
    result = json.loads(index.read_bytes())
    source_root = ROOT
    backup_path = ROOT / 'research/native_bf16_performance_backup.json'
    if backup_path.exists():
        backup = json.loads(backup_path.read_bytes())
        assert sha(index) == backup['result_index_sha256']
        archived = Path(backup['primary'])
        assert sha(archived / 'manifest.json') == backup['manifest_sha256']
        source_root = archived / 'project'
    for item in result['files'].values():
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
    for relative, expected in result['source_files'].items():
        assert sha(source_root / relative) == expected, relative
        if relative.endswith('.py'):
            ast.parse((source_root / relative).read_text())
    assert result['precision']['original_parameter_bytes_preserved']
    assert not result['precision']['four_bit_quantization']
    assert not result['precision']['fp16_conversion']
    assert not result['precision']['arithmetic_bit_exact_to_hf']
    assert not result['product_adopted'] and not result['published']
    assert result['teacher_calls'] == 0
    repeat_dir = Path(result['directory']) / 'repeated-v2'
    repeat = json.loads((repeat_dir / 'summary.json').read_bytes())
    assert repeat['protocol']['rounds'] == 5
    assert repeat['protocol']['binary_sha256'] == result['selected_binary_sha256']
    assert repeat['protocol']['baseline_binary_sha256'] == result['baseline_binary_sha256']
    cases = json.loads(Path(result['files']['cases-repeat.json']['path']).read_bytes())
    for engine, prompts in repeat['samples'].items():
        assert engine in ('baseline', 'native', 'hf')
        for case in cases[1:]:
            samples = prompts[case['id']]
            assert [s['round'] for s in samples] == [1, 2, 3, 4, 5]
            for sample in samples:
                rows = [json.loads(line) for line in (repeat_dir / f"round-{sample['round']}-{engine}" / 'events.jsonl').read_text().splitlines()]
                row = next(r for r in rows if r['event'] == 'result' and r['id'] == case['id'])
                assert row['generated_tokens'] == case['max_tokens'] == 128
                assert sample['first_visible_seconds'] == row['client_first_visible_text_ns'] / 1e9
                assert sample['decode_tokens_per_second'] == (row['generated_tokens'] - 1) * 1e9 / row['decode_ns']
            for metric, summary in repeat['per_prompt'][case['id']][engine].items():
                values = [s[metric] for s in samples]
                assert values == summary['values']
                assert math.isclose(statistics.mean(values), summary['mean'], rel_tol=1e-12)
                assert math.isclose(statistics.stdev(values), summary['sample_sd'], rel_tol=1e-12)
    for name, count in [('ranking', 48), ('trained', 112)]:
        report = json.loads(Path(result['files'][name + '-comparison.json']['path']).read_bytes())
        assert len(report['rows']) == count
        quality = report['quality']
        lost = [r['id'] for r in report['rows'] if r['reference_correct'] and not r['native_correct']]
        assert lost == quality['lost_reference_correct_ids']
        assert quality == result[name + '_quality']
    retained = all(not result[n + '_quality']['lost_reference_correct_ids'] for n in ('ranking', 'trained'))
    assert retained == result['qualification']['finite_fixture_retention']
    print('Bound BF16 optimization sources, real projection checks, 48+112 answers and all 15 fresh-process runs verified.')
    print(f'Finite fixture retention: {retained}; arbitrary HF bit equivalence FALSE; product adoption FALSE.')


if __name__ == '__main__':
    main()
