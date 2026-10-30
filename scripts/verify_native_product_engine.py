"""Validate the bound formal native implementation and actual measurements."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    result = json.loads((ROOT / 'research/native_product_engine_result.json').read_text())
    for record in result['files'].values():
        path = Path(record['path'])
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], path
    for name, digest in result['source_files'].items():
        assert sha(ROOT / name) == digest, name
    exp = Path(result['directory'])
    worker = json.loads((exp / 'worker-validation/result.json').read_text())
    assert worker['teacher_calls'] == 0 and not worker['product_adopted']
    for name, count in [('ranking', 48), ('trained', 112)]:
        rows = json.loads((exp / f'worker-validation/{name}-comparison.json').read_text())
        assert rows['cases'] == rows['exact'] == count and all(row['tokens_exact'] for row in rows['rows'])
    api = json.loads((exp / 'api-validation-v2/result.json').read_text())
    assert api['passed'] and all(api['checks'].values())
    timing = json.loads((exp / 'http-five-rounds-v2/result.json').read_text())
    assert timing['passed'] and timing['teacher_calls'] == 0
    for name, digest in timing['inputs'].items():
        assert sha(ROOT / name) == digest, name
    prompts = [rows['prompt'] for rows in timing['measurements']['native'][:2]]
    for index, prompt in enumerate(prompts):
        for engine in ('native', 'hf'):
            rows = [row for row in timing['measurements'][engine] if row['prompt'] == prompt]
            assert [row['round'] for row in rows] == [1, 2, 3, 4, 5]
            assert all(row['temperature'] == 0 and row['maximum'] == 48 for row in rows)
            metric = timing['per_prompt'][index][engine]
            assert metric['first_visible_mean'] == statistics.mean(row['first_visible_seconds'] for row in rows)
            assert metric['first_visible_sample_sd'] == statistics.stdev(row['first_visible_seconds'] for row in rows)
    memory = json.loads((exp / 'http-five-rounds-v2/memory.json').read_text())
    assert max(r['parent'] + r['child'] for r in memory) == timing['sampled_combined_peak_bytes']
    installed = json.loads((exp / 'app-install.json').read_text())
    for name, digest in installed['new_sha256'].items():
        assert sha(Path(installed['installed']) / name) == digest, name
    assert result['training_backend'] == 'unchanged torch-hf' and result['published'] is False
    print('Native product worker/HTTP/5 rounds/install source and artifact checks passed; OS launch permission pending.'
          if result['app_launch_state'] != 'ready' else 'Native product and installed app checks passed.')


if __name__ == '__main__':
    main()
