#!/usr/bin/env python3
"""Derive GGUF with the accepted BF16 down projection; no requantization."""
import json
import os
from pathlib import Path
import struct

from inspect_community_q4 import gguf
from verify_direct_q4 import CORE

ROOT = Path('/Volumes/移动数据库/lubanllm/models/community-q4-study-20261017')


def copy_bytes(source, target, count):
    while count:
        block = source.read(min(count, 8 << 20))
        if not block:
            raise ValueError('Truncated source tensor file')
        target.write(block)
        count -= len(block)


def main():
    assert Path('/Volumes/移动数据库').is_mount()
    source = ROOT / 'unsloth-ud-q4-k-xl/Qwen3.5-2B-UD-Q4_K_XL.gguf'
    output = ROOT / 'unsloth-with-luban-core.gguf'
    partial = output.with_suffix('.gguf.partial')
    assert not output.exists()
    core = Path(CORE['file'])
    source_stat, core_stat = source.stat(), core.stat()
    receipt = json.loads((source.parent / 'download-receipt.json').read_text())
    assert receipt['complete']
    info = gguf(source)
    tensor = next(row for row in info['tensors'] if row['name'] == 'blk.23.ffn_down.weight')
    assert tensor['shape_ggml_order'] == [6144, 2048]
    with core.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
        value = header[CORE['tensor_key']]
        assert value['dtype'] == 'BF16' and value['shape'] == [2048, 6144]
        begin, end = value['data_offsets']
        assert end-begin == 25_165_824
        core_data_offset = 8 + length + begin
    old_size = tensor['storage_span_bytes_including_padding']
    delta = end-begin-old_size
    assert delta % info['metadata'].get('general.alignment', 32) == 0
    with source.open('rb') as src, core.open('rb') as learned, partial.open('xb') as dst:
        header = bytearray(src.read(info['data_start']))
        struct.pack_into('<I', header, tensor['header_type_position'], 30)  # GGML_TYPE_BF16
        for row in info['tensors']:
            if row['offset'] > tensor['offset']:
                struct.pack_into('<Q', header, row['header_offset_position'], row['offset'] + delta)
        dst.write(header)
        copy_bytes(src, dst, tensor['offset'])
        learned.seek(core_data_offset)
        copy_bytes(learned, dst, end-begin)
        src.seek(old_size, 1)
        copy_bytes(src, dst, source_stat.st_size-src.tell())
        dst.flush()
        os.fsync(dst.fileno())
    assert partial.stat().st_size == source_stat.st_size + delta
    assert source.stat() == source_stat and core.stat() == core_stat
    check = gguf(partial)
    replacement = next(row for row in check['tensors'] if row['name'] == tensor['name'])
    assert replacement['type_id'] == 30 and replacement['storage_span_bytes_including_padding'] == end-begin
    partial.rename(output)
    record = {'output': str(output), 'bytes': output.stat().st_size,
              'source_receipt': str(source.parent / 'download-receipt.json'),
              'core': CORE, 'core_integrity': 'existing accepted source binding; stat unchanged during copy; not rehashed',
              'replacement': replacement, 'all_other_tensors': 'source bytes copied unchanged',
              'optimizer_steps': 0, 'requantized': False, 'product_admitted': False}
    output.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
