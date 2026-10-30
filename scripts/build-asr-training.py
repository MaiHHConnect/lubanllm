"""Build standalone LibTorch ASR workers; no model loading or payload hashing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT / 'build/native')
parser.add_argument('--tiny', action='store_true', help='Existing CPU fixture dimensions; never a product binary')
args = parser.parse_args()
import torch
TORCH = Path(torch.__file__).parent
args.output.mkdir(parents=True, exist_ok=True)
commands = []
training_sources = ['native/asr/Training.cpp', 'native/asr/Graph.h', 'native/asr/IO.h', 'native/asr/ArchiveRNG.h', 'native/asr/CPURNG.h']
for name, sources in [('luban-asr-training', training_sources), ('luban-asr-features', ['native/asr/Features.cpp']),
                      ('luban-asr-tokenizer', ['native/Tokenizer.swift', 'native/ASRTokenizer/main.swift'])]:
    target = args.output / (name + ('-tiny' if args.tiny and name != 'luban-asr-tokenizer' else ''))
    if name == 'luban-asr-tokenizer':
        argv = ['/usr/bin/swiftc', '-O', *[str(ROOT / path) for path in sources], '-o', str(target)]
    else:
        argv = ['/usr/bin/clang++', '-std=c++17', '-O1', '-g0', '-Wno-deprecated-declarations']
        if args.tiny and name == 'luban-asr-training':
            argv += ['-DLUBAN_ASR_TINY']
        argv += ['-I' + str(TORCH / 'include'), '-I' + str(TORCH / 'include/torch/csrc/api/include'), str(ROOT / sources[0]),
                 '-L' + str(TORCH / 'lib'), '-Wl,-rpath,' + str(TORCH / 'lib'), '-ltorch', '-ltorch_cpu', '-lc10', '-o', str(target)]
    compile_command = None
    for command in [argv, ['/usr/bin/codesign', '--force', '--sign', '-', str(target)],
                    ['/usr/bin/codesign', '--verify', '--strict', str(target)]]:
        done = subprocess.run(command, text=True, capture_output=True)
        row = {'argv': command, 'exit_code': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr}
        commands.append(row)
        if command is argv:
            compile_command = row
        (args.output / 'asr-build.json').write_text(json.dumps({'commands': commands, 'models_loaded': 0, 'tiny': args.tiny}, indent=2))
        if done.returncode:
            sys.exit(done.returncode)
    source_bundle = b''.join(path.encode() + b'\0' + (ROOT / path).read_bytes() + b'\0' for path in sources)
    receipt = {'schema': 'luban-native-asr-build-v1', 'binary_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'sources': sources, 'source_bundle_sha256': hashlib.sha256(source_bundle).hexdigest(),
               'production_shape': not args.tiny, 'models_loaded': 0, 'actual_compile': compile_command,
               'dependency_library_directory': None if name.endswith('tokenizer') else str(TORCH / 'lib'),
               'no_python_linkage': True, 'source_payload_SHA_rechecked': False}
    Path(str(target) + '.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'passed': True, 'output': str(args.output), 'models_loaded': 0, 'tiny': args.tiny}))
