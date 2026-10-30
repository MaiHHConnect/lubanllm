#!/usr/bin/env python3
"""Verify native experiment evidence; a failed quality result is not adoption."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    index = ROOT / 'research/native_bf16_loader_result.json'
    result = json.loads(index.read_bytes())
    # The first study is immutable evidence for the original implementation.
    # Later optimization source must not be attributed to its earlier timings.
    backup = json.loads((ROOT / 'research/native_bf16_loader_backup.json').read_bytes())
    assert sha(index) == backup['result_index_sha256']
    archived = Path(backup['primary'])
    assert sha(archived / 'manifest.json') == backup['manifest_sha256']
    source_root = archived / 'project'
    for item in result['files'].values():
        path = Path(item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
    for relative, expected in result['source_files'].items():
        assert sha(source_root / relative) == expected, relative
    for path in (source_root / 'native/luban').glob('*.py'):
        if not path.name.startswith('._'):
            ast.parse(path.read_text())
    for relative in result['qualified_binaries']:
        assert (Path(result['directory']) / relative).is_file(), relative
    repeats = json.loads(Path(result['files']['repeated-v1/summary.json']['path']).read_bytes())
    assert repeats['protocol']['rounds'] == 5
    for engines in repeats['samples'].values():
        for samples in engines.values():
            assert [row['round'] for row in samples] == [1, 2, 3, 4, 5]
            assert all(row['decode_tokens_per_second'] > 0 for row in samples)
    assert result['qualification']['strict_bound_loader'] is True
    assert result['qualification']['trained_bf16_sidecar_loaded'] is True
    assert result['qualification']['quality_retention'] is False
    assert result['qualification']['replace_current_engine'] is False
    assert result['product_adopted'] is False and result['native_training_migrated'] is False
    assert result['ranking_quality']['lost_reference_correct_ids'] == ['mmlu-2804']
    assert len(result['trained_quality']['lost_reference_correct_ids']) == 2
    print('Original native BF16 study verified against its archived original source, bound weights and five rounds.')
    print('Original capability retention FAILED; its failure reports remain preserved; product adoption FALSE.')


if __name__ == '__main__':
    main()
