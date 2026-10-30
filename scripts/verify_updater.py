#!/usr/bin/env python3
"""CPU-only native updater checks; local Python HTTP servers are test fixtures only."""
import argparse
import atexit
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if args.output:
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    compiled_root = args.output.resolve()
else:
    temporary = tempfile.TemporaryDirectory(prefix='luban-updater-verification-')
    atexit.register(temporary.cleanup)
    compiled_root = Path(temporary.name)
os.chdir(ROOT)
checks = []
sources = ['native/Updater.swift', 'native/UpdateArchive.swift', 'native/UpdateService.swift',
           'tests/update_manifest.swift', 'tests/native_update_archive.swift', 'scripts/verify_updater.py']
source_sha = {}
for name in sources:
    data = (ROOT / name).read_bytes()
    snapshot = compiled_root / 'sources' / name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(data)
    source_sha[name] = hashlib.sha256(data).hexdigest()

def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], **kwargs)

def record(name, passed, **details):
    checks.append({'check': name, 'passed': bool(passed), **details})
    (compiled_root / 'progress.json').write_text(json.dumps({'checks': checks}, indent=2) + '\n')
    if not passed:
        raise AssertionError(name)

common = ['native/Updater.swift', 'native/UpdateArchive.swift']
for name, sources in [('checks', [*common, 'tests/update_manifest.swift']),
                      ('archive-checks', ['native/UpdateArchive.swift', 'tests/native_update_archive.swift']),
                      ('luban-update-service', [*common, 'native/UpdateService.swift'])]:
    result = run(['xcrun', 'swiftc', '-O', *sources, '-o', compiled_root / name], capture_output=True)
    (compiled_root / (name + '.compile.log')).write_bytes(result.stdout + result.stderr)
    record(name + ' compilation', result.returncode == 0)
result = run([compiled_root / 'checks', compiled_root], capture_output=True)
(compiled_root / 'manifest.log').write_bytes(result.stdout + result.stderr)
record('manifest identity/TLS/version/limits regressions', result.returncode == 0, stdout=result.stdout.decode())
service = compiled_root / 'luban-update-service'

archive_cases = [('Luban.app/Contents/Info.plist', 0, True), ('../escape', 0, False),
                 ('/absolute', 0, False), ('Other.app/f', 0, False), ('Luban.app/x/../../bad', 0, False),
                 ('Luban.app/a\\b', 0, False), ('Luban.app/linked', stat.S_IFLNK | 0o777, False),
                 ('Luban.app/pipe', stat.S_IFIFO | 0o600, False)]
requests = []
for index, (name, mode, expected) in enumerate(archive_cases):
    archive = compiled_root / f'archive-{index}.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        info = zipfile.ZipInfo(name)
        info.external_attr = mode << 16
        z.writestr(info, 'test')
    requests.append({'archive': str(archive), 'stage': str(compiled_root / f'extracted-{index}')})
result = run([compiled_root / 'archive-checks'], input=''.join(json.dumps(x) + '\n' for x in requests).encode(), capture_output=True)
(compiled_root / 'archive-responses.jsonl').write_bytes(result.stdout)
answers = [json.loads(line) for line in result.stdout.splitlines()]
record('actual native ZIP accepts/rejects the eight original oracle fixtures',
       result.returncode == 0 and len(answers) == 8 and all(answer['accepted'] == expected for answer, (_, _, expected) in zip(answers, archive_cases)))

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.paths.append(self.path)
        status, body, headers, stall = self.server.payload
        if stall:
            time.sleep(stall)
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *args):
        pass

base = {'app': 'luban', 'version': '0.1.1', 'build': 2}
payloads = [
    ('healthy', 200, json.dumps(base).encode(), {}, 0, True),
    ('wrong-app', 200, json.dumps({**base, 'app': 'other'}).encode(), {}, 0, False),
    ('wrong-version', 200, json.dumps({**base, 'version': '0.1.2'}).encode(), {}, 0, False),
    ('wrong-build', 200, json.dumps({**base, 'build': 3}).encode(), {}, 0, False),
    ('boolean-build', 200, json.dumps({**base, 'build': True}).encode(), {}, 0, False),
    ('fractional-build', 200, json.dumps({**base, 'build': 2.5}).encode(), {}, 0, False),
    ('float-build', 200, json.dumps({**base, 'build': 2.0}).encode(), {}, 0, False),
    ('string-build', 200, json.dumps({**base, 'build': '2'}).encode(), {}, 0, False),
    ('large-integer-build-neighbour', 200, json.dumps({**base, 'build': 2**53}).encode(), {}, 0, False),
    ('large-integer-build-exact', 200, json.dumps({**base, 'build': 2**53 + 1}).encode(), {}, 0, True),
    ('invalid-json', 200, b'{', {}, 0, False),
    ('array-json', 200, b'[]', {}, 0, False),
    ('http-error', 500, json.dumps(base).encode(), {}, 0, False),
    ('redirect', 302, b'', {'Location': 'http://127.0.0.1:1/api/health'}, 0, False),
    ('oversized-declared', 200, b'{}', {'Content-Length': str(65537)}, 0, False),
    ('oversized-stream', 200, b'A' * 70000, {}, 0, False),
    ('stall', 200, json.dumps(base).encode(), {}, 1, False),
]
for name, status, body, headers, stall, expected in payloads:
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.payload = status, body, headers, stall
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}/api/health'
    try:
        began = time.monotonic()
        expected_build = str(2**53 + 1) if name.startswith('large-integer-build-') else '2'
        result = run([service, 'health', '0.1.1', expected_build, endpoint, '.2'], timeout=3, capture_output=True)
        record(name + ' health response', (result.returncode == 0) == expected and time.monotonic() - began < 1.5,
               returncode=result.returncode, output=result.stdout.decode())
        result = run([service, 'quiescence', endpoint, '.2'], timeout=3, capture_output=True)
        record(name + ' cannot admit quiescence while port still occupied', result.returncode == 1,
               returncode=result.returncode, output=result.stdout.decode())
        record(name + ' endpoint and redirects remain local', server.paths and set(server.paths) == {'/api/health'})
    finally:
        server.shutdown(); server.server_close(); thread.join()

with socket.socket() as closed_socket:
    closed_socket.bind(('127.0.0.1', 0))
    closed_port = closed_socket.getsockname()[1]
closed_url = f'http://127.0.0.1:{closed_port}/api/health'
result = run([service, 'quiescence', closed_url, '.2'], timeout=3, capture_output=True)
record('refused local connection admits quiescence', result.returncode == 0, output=result.stdout.decode())
for values in [['health', '0.1.1', '2', 'http://example.test/api/health', '.2'],
               ['health', '0.1.1', '2', closed_url, 'nan'], ['health', '0.1.1', '02'],
               ['health', '0.1.1', '0'], ['health', '01.1.1', '2'],
               ['quiescence', closed_url + '?x=1', '.2'], ['quiescence', closed_url, '0']]:
    result = run([service, *values], timeout=3, capture_output=True)
    record('invalid native helper arguments ' + repr(values), result.returncode == 2)

original = (compiled_root / 'install.sh').read_text()
record('production installation script invokes only native health/quiescence',
       '$python' not in original and 'HEALTHCHECK' not in original and 'QUIESCENCE' not in original
       and '"$service" health "$version" "$build"' in original and '"$service" quiescence' in original)
# These are isolated directory operations, not real app installation. The launched
# test backend deliberately uses Python, just like the pre-migration test fixture.
fixture_python = str(ROOT / '.venv/bin/python')
for scenario in ['healthy-install', 'rollback-install', 'blocked-rollback', 'stage-move-failure']:
    succeeds = scenario == 'healthy-install'
    fixture_root = compiled_root / scenario
    fixture_root.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    app = fixture_root / 'Luban.app'; app.mkdir()
    (app / 'original').write_text('original retained on failure')
    staged = fixture_root / 'staging'
    binary = staged / 'Luban.app/Contents/MacOS/Luban'; binary.parent.mkdir(parents=True)
    cache = fixture_root / 'cache'; cache.mkdir()
    pid_file = fixture_root / 'fixture.pid'
    executable = f'''#!{fixture_python}
import http.server, json, os, pathlib, time
pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
'''
    if succeeds:
        executable += f'''class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({base!r}).encode())
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()
'''
    else:
        executable += 'while True: time.sleep(1)\n'
    binary.write_text(executable); binary.chmod(0o755)
    if scenario == 'stage-move-failure':
        shutil.rmtree(staged / 'Luban.app')
    occupied_server = None
    if scenario == 'blocked-rollback':
        occupied_server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
        occupied_server.daemon_threads = True
        occupied_server.payload = (500, b'occupied', {}, 0)
        occupied_server.paths = []
        occupied_thread = threading.Thread(target=occupied_server.serve_forever, daemon=True)
        occupied_thread.start()
    detached = fixture_root / 'service'; shutil.copyfile(service, detached); detached.chmod(0o700)
    helper = fixture_root / 'helper.sh'
    endpoint = f'http://127.0.0.1:{port}/api/health'
    helper.write_text(original.replace('"$service" health "$version" "$build"', f'"$service" health "$version" "$build" "{endpoint}" 2')
                     .replace('"$service" quiescence', f'"$service" quiescence "{endpoint}" 2')
                     .replace('/usr/bin/osascript -e \'display alert "鲁班更新失败" message "应用已保留，请重新启动后重试。"\'', '/usr/bin/true')
                     .replace('/usr/bin/open -n -a "$app"', '/usr/bin/true'))
    backup = fixture_root / 'backup.app'
    try:
        prior = subprocess.Popen(['/bin/sleep', '.5']) if succeeds else None
        installer = subprocess.Popen(['/bin/sh', str(helper), str(app), str(staged), str(backup),
            str(prior.pid) if prior else '99999999', '99999998', str(cache), str(detached), '0.1.1', '2'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if prior:
            time.sleep(.1)
            record('installation waits for the actual old process to exit before rename',
                   (app / 'original').exists() and not backup.exists() and staged.exists())
            prior.wait(timeout=2)
        stdout, stderr = installer.communicate(timeout=15)
        (fixture_root / 'output.log').write_bytes(stdout + stderr)
        rolled_back = scenario in ['rollback-install', 'stage-move-failure']
        record('native shell fixture ' + scenario,
               (installer.returncode == 0) == succeeds and (app / 'original').exists() == rolled_back
               and (app / 'Contents/MacOS/Luban').exists() == (not rolled_back) and not staged.exists()
               and backup.exists() == (scenario == 'blocked-rollback') and cache.exists() == (not succeeds)
               and not detached.exists(), returncode=installer.returncode)
        if scenario == 'blocked-rollback':
            record('HTTP500 occupied port retains intact original backup and forbids replacement removal',
                   (backup / 'original').read_text() == 'original retained on failure' and (app / 'Contents/MacOS/Luban').exists())
    finally:
        if occupied_server:
            occupied_server.shutdown(); occupied_server.server_close(); occupied_thread.join()
        if pid_file.exists():
            try: os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError: pass

record('executed sources remained unchanged throughout offline checks',
       all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == source_sha[name] for name in sources))
report = {'passed': True, 'scope': 'Offline CPU native update helper and isolated installation-directory fixtures; no production install/rollback',
          'test_fixture_backend': 'Python HTTP mock only; native helper starts no Python or subprocess',
          'checks': checks, 'source_sha256': source_sha}
(compiled_root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'checks': len(checks), 'output': str(compiled_root)}))
