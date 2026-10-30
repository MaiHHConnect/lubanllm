#!/usr/bin/env python3
"""Read model headers only; no model execution or tensor payload reads."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import struct


def gguf(path):
    formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
               6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    with path.open('rb') as f:
        def read(n):
            value = f.read(n)
            if len(value) != n:
                raise ValueError('Truncated GGUF header')
            return value

        def number(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]

        def string(keep=True):
            n = number('Q')
            if n > path.stat().st_size - f.tell():
                raise ValueError('Invalid GGUF string length')
            if keep:
                return read(n).decode('utf-8')
            f.seek(n, 1)

        def value(kind, keep=True):
            if kind in formats:
                return number(formats[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                subtype, count = number('I'), number('Q')
                if count > path.stat().st_size:
                    raise ValueError('Invalid GGUF array count')
                if subtype in formats:
                    f.seek(count * struct.calcsize('<' + formats[subtype]), 1)
                elif subtype == 8:
                    for _ in range(count):
                        string(False)
                else:
                    raise ValueError('Unsupported GGUF array type')
                return {'array_type': subtype, 'count': count}
            raise ValueError(f'Unknown GGUF metadata type {kind}')

        if read(4) != b'GGUF' or number('I') != 3:
            raise ValueError('Expected GGUF v3')
        tensor_count, metadata_count = number('Q'), number('Q')
        metadata = {}
        for _ in range(metadata_count):
            key, kind = string(), number('I')
            item = value(kind)
            metadata[key] = item if not isinstance(item, str) or len(item) < 500 else {'string_length': len(item)}
        tensors = []
        for _ in range(tensor_count):
            name, rank = string(), number('I')
            if not 1 <= rank <= 4:
                raise ValueError('Unsupported GGUF tensor rank')
            shape = [number('Q') for _ in range(rank)]
            type_position = f.tell()
            tensor_type = number('I')
            offset_position = f.tell()
            tensors.append({'name': name, 'shape_ggml_order': shape,
                            'type_id': tensor_type, 'offset': number('Q'),
                            'header_type_position': type_position, 'header_offset_position': offset_position})
        alignment = metadata.get('general.alignment', 32)
        if not isinstance(alignment, int) or alignment <= 0:
            raise ValueError('Invalid GGUF alignment')
        data_start = (f.tell() + alignment - 1) // alignment * alignment
    ordered = sorted(tensors, key=lambda row: row['offset'])
    for i, row in enumerate(ordered):
        end = ordered[i + 1]['offset'] if i + 1 < len(ordered) else path.stat().st_size - data_start
        if not 0 <= row['offset'] < end <= path.stat().st_size - data_start:
            raise ValueError('Invalid GGUF tensor offset')
        row['storage_span_bytes_including_padding'] = end - row['offset']
    return {'format': 'GGUF', 'metadata': metadata, 'data_start': data_start, 'tensors': tensors}


def safetensors(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(f.read(size))
    rows = []
    for name, row in header.items():
        if name == '__metadata__':
            continue
        start, end = row['data_offsets']
        if not 0 <= start <= end <= path.stat().st_size - size - 8:
            raise ValueError('Invalid safetensors tensor offset')
        rows.append({'name': name, 'dtype': row['dtype'], 'shape': row['shape'], 'bytes': end - start})
    return {'format': 'safetensors', 'metadata': header.get('__metadata__', {}), 'tensors': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    path = args.model
    receipt = json.loads((path.parent / 'download-receipt.json').read_text())
    if not receipt.get('complete') or not any(x['name'] == path.name and x['bytes'] == path.stat().st_size and x.get('sha256') for x in receipt['files']):
        raise ValueError('Download has no completed integrity receipt')
    result = gguf(path) if path.suffix == '.gguf' else safetensors(path)
    result.update(path=str(path), bytes=path.stat().st_size, revision=receipt['revision'], mode='header_only_no_GPU_no_rehash')
    types = Counter()
    groups = defaultdict(int)
    reference = path.parent.parent / 'ggml-type-reference.txt'
    type_names = dict((int(n), name) for name, n in re.findall(r'^\s*GGML_TYPE_(\w+)\s*=\s*(\d+)', reference.read_text(), re.M)) if reference.exists() else {}
    for row in result['tensors']:
        kind = row.get('dtype') or type_names.get(row['type_id'], str(row['type_id']))
        row['storage_type'] = kind
        types[kind] += 1
        name = row['name']
        group = 'vision' if name.startswith(('visual.', 'vision_tower.')) else ('embedding' if 'embed_tokens.' in name or name == 'token_embd.weight' else 'other_language')
        groups[group] += row.get('bytes', row.get('storage_span_bytes_including_padding', 0))
    result.update(tensor_types=dict(types), storage_groups=dict(groups))
    output = path.parent / 'header-inspection.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('tensors', 'metadata')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
