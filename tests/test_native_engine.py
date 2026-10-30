"""Exercise worker IPC, cancellation and provenance without models or GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from luban.native_engine import NativeEngine, NativeEngineError


WORKER = r'''
import json, os, select, signal, sys, time
if sys.argv[1] == '--prepare-direct-q4':
    fd = int(sys.argv[5])
    if MODE == 'prepare_cancel':
        while not select.select([fd], [], [], 0.02)[0]:
            pass
        sys.exit(1)
    with open(sys.argv[4], 'wb') as out:
        out.write(b'carrier')
    print(json.dumps({'event': 'prepared', 'format': 'luban-direct-q4-g16-v1',
        'manifest_sha256': sys.argv[3], 'direct_q4_sha256': 'd' * 64, 'direct_q4_weight_bytes': 512,
        'quantizer_sha256': sys.argv[6].split('=', 1)[1]}), flush=True)
    sys.exit(0)
fd = int(sys.argv[3])
if MODE == 'ignore_term':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
def emit(value):
    print(json.dumps(value), flush=True)
if MODE == 'load_timeout':
    time.sleep(10)
if MODE == 'bad_loaded':
    emit({'event': 'loaded', 'protocol': 'luban-worker-jsonl-v1', 'manifest_sha256': '0' * 64})
else:
    precision = next((arg[2:].split('=', 1)[0] for arg in sys.argv if arg.startswith(('--mixed-', '--mlx-q4'))), 'bf16')
    positional = [arg for arg in sys.argv[4:] if not arg.startswith('--')]
    loaded = {'event': 'loaded', 'inference_precision': precision, 'protocol': 'luban-worker-jsonl-v1', 'manifest_sha256': sys.argv[2], 'core_args': sys.argv[4:],
              'core_loaded': len(positional) == 3, 'core_file_sha256': positional[2] if len(positional) == 3 else None}
    if MODE != 'legacy':
        loaded.update(execution_skip_mask=next((int(arg.split('=')[1]) for arg in sys.argv if arg.startswith('--skip-layer-mask=')), 0),
                      execution_exit_after=next((int(arg.split('=')[1]) for arg in sys.argv if arg.startswith('--exit-after=')), 24),
                      layer_paging_enabled=any(arg.startswith('--layer-paging=') for arg in sys.argv))
    if MODE == 'bad_policy':
        loaded['execution_skip_mask'] = 0
    if MODE == 'bad_policy_type':
        loaded['execution_skip_mask'] = False
    direct_sha = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--direct-q4-sha256=')), None)
    if direct_sha is not None and MODE != 'legacy':
        loaded.update(direct_q4_loaded=True, direct_q4_sha256=direct_sha,
                      direct_q4_weight_bytes=512,
                      quantizer_sha256=next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--quantizer-sha256=')))
        if MODE == 'bad_direct':
            loaded['direct_q4_sha256'] = '0' * 64
    expert_sha = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--expert-sha=')), None)
    if expert_sha is not None and MODE != 'legacy':
        loaded.update(expert_loaded=True, expert_file_sha256=expert_sha, expert_width=64, expert_layer=23)
        if MODE == 'bad_expert':
            loaded['expert_width'] = 128
    if precision == 'mlx-q4':
        loaded.update(mlx_model_path=next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--mlx-model=')),
                      mlx_model_sha256=next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--mlx-model-sha256=')),
                      mlx_model_weight_bytes=123, mlx_cache_reuse=True)
        loaded['vision_loaded'] = '--vision' in sys.argv and MODE != 'bad_vision'
        if MODE == 'bad_mlx':
            loaded['mlx_model_sha256'] = '0' * 64
    emit(loaded)
for raw in sys.stdin:
    req = json.loads(raw)
    if req.get('op') == 'clear_cache':
        emit({'id': 'wrong' if MODE == 'bad_cache_ack' else req['id'], 'event': 'cache_cleared'})
        continue
    key = req['tokens'][0]
    ident = req['id']
    if req.get('op') == 'route_features':
        if key == 101:
            time.sleep(10)
        value = {'id': ident, 'event': 'route_features', 'features': [0.25] * 2048,
                 'feature_dim': 2048, 'layers': 4, 'source': 'mlx-q4-prefix', 'legacy_bf16_loaded': False}
        if key == 201: value['features'] = [0.25] * 2047
        if key == 202: value['features'][0] = float('nan')
        if key == 203: value['source'] = 'bf16'
        if key == 204: value['id'] = 'wrong'
        if key == 205: value['features'][0] = True
        emit(value)
        continue
    if '--vision' in sys.argv and (req.get('op') != 'image_generate' or not os.path.isfile(req.get('pixel_file', ''))):
        emit({'event':'error', 'id':ident, 'error':'image request missing'})
        continue
    if key == 100:
        while not select.select([fd], [], [], 0.02)[0]:
            pass
        os.read(fd, 4096)
        emit({'event': 'result', 'id': ident, 'tokens': [], 'finish_reason': 'cancelled'})
        continue
    if key == 101:
        time.sleep(10)
    if key == 102:
        emit({'event': 'token', 'id': ident, 'token': 11})
        sys.stderr.write('intentional worker crash\n')
        sys.stderr.flush()
        os._exit(7)
    if key == 103:
        emit({'event': 'error', 'id': ident, 'error': 'bad native tensor'})
        continue
    if key == 104:
        emit({'event': 'token', 'id': ident + 'wrong', 'token': 11})
        continue
    if key == 106:
        print('{broken-json', flush=True)
        continue
    if key == 108:
        sys.stderr.write('x' * (256 * 1024))
        sys.stderr.flush()
        os._exit(7)
    if key == 107:
        time.sleep(0.08)
    if select.select([fd], [], [], 0)[0]:
        os.read(fd, 4096)
        emit({'event': 'result', 'id': ident, 'tokens': [], 'finish_reason': 'cancelled'})
        continue
    tokens = [11, 12][:req['max_tokens']]
    for token in tokens:
        emit({'event': 'token', 'id': ident, 'token': token})
    emit({'event': 'result', 'id': ident, 'tokens': [99] if key == 105 else tokens,
          'finish_reason': 'length', 'seed': req['seed'], 'prefill_ns': 1000000,
          'request_context_limit': req.get('context_limit'),
          'request_prefill_step': req.get('prefill_step'),
          'ttft_ns': 2000000, 'decode_ns': -1 if key == 109 else 50000000,
          'cached_prompt_tokens': 1 if req.get('cache_reuse') else 0})
'''


@unittest.skipUnless(os.name == "posix", "Native IPC requires POSIX descriptors")
class NativeEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        manifest = self.model / "upstream-file-manifest.json"
        manifest.write_text('{"Data":{"Files":[]}}')
        self.manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    def engine(self, *, mode="normal", **kwargs):
        binary = self.root / ("worker-" + mode)
        binary.write_text(f"#!{sys.executable}\nMODE = {mode!r}\n" + WORKER)
        binary.chmod(0o700)
        receipt = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                   "source_sha256": {"fake-worker.py": hashlib.sha256(WORKER.encode()).hexdigest()},
                   "official_manifest_sha256": self.manifest_sha}
        Path(str(binary) + ".json").write_text(json.dumps(receipt))
        engine = NativeEngine(binary, self.model, self.manifest_sha, **kwargs)
        self.addCleanup(engine.close)
        return engine

    def generate(self, engine, tokens=None, *, stop=None, on_token=None, **kwargs):
        return engine.generate(tokens or [1, 2], 4, [99], 0,
                               on_token or (lambda token: None), stop or threading.Event(), **kwargs)

    def test_mlx_model_ack_restart_and_cancel_use_existing_protocol(self):
        model = {"path": str(self.model), "sha256": "c" * 64, "weight_bytes": 123}
        engine = self.engine(precision="mlx-q4", mlx_model=model)
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])
        self.assertEqual(engine.loaded["mlx_model_sha256"], model["sha256"])
        engine.close()
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])
        stop = threading.Event()
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        try:
            self.assertEqual(self.generate(engine, [100], stop=stop)["finish_reason"], "cancelled")
        finally:
            timer.join()
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])
        bad = self.engine(mode="bad_mlx", precision="mlx-q4", mlx_model=model)
        with self.assertRaisesRegex(NativeEngineError, "different MLX model"):
            bad.start()
        self.assertIsNone(bad.process)

    def test_mlx_start_releases_empty_allocator_once_and_checks_ack(self):
        model = {"path": str(self.model), "sha256": "c" * 64, "weight_bytes": 123}
        engine = self.engine(precision="mlx-q4", mlx_model=model)
        engine.start()
        self.assertEqual(engine._request_id, 1)
        self.assertEqual(engine._prefix_tokens, [])
        engine.start()
        engine.prepare_prompt_cache([], cache_reuse=False)
        self.assertEqual(engine._request_id, 1)
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])
        bad = self.engine(mode="bad_cache_ack", precision="mlx-q4", mlx_model=model)
        with self.assertRaisesRegex(NativeEngineError, "acknowledgement mismatch"):
            bad.start()
        self.assertIsNone(bad.process)

    def test_mlx_routes_reuse_worker_and_reject_bad_features(self):
        model = {"path": str(self.model), "sha256": "c" * 64, "weight_bytes": 123}
        engine = self.engine(precision="mlx-q4", mlx_model=model)
        self.assertEqual(engine.route_features([1, 2], threading.Event()), [0.25] * 2048)
        pid = engine.process.pid
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])
        self.assertEqual(engine.process.pid, pid)
        for key in [201, 202, 203, 204, 205]:
            with self.subTest(key=key), self.assertRaisesRegex(NativeEngineError, "routing feature response"):
                engine.route_features([key], threading.Event())
            self.assertIsNone(engine.process)

    def test_mlx_generation_passes_selected_context_and_rejects_overflow(self):
        model = {"path": str(self.model), "sha256": "c" * 64, "weight_bytes": 123}
        engine = self.engine(precision="mlx-q4", mlx_model=model)
        for limit in (65536, 131072, 262144):
            result = self.generate(engine, [1] * 40000, context_limit=limit, prefill_step=128)
            self.assertEqual(result["request_context_limit"], limit)
            self.assertEqual(result["request_prefill_step"], 128)
        for step in (32, 64):
            result = self.generate(engine, [1] * 145000, context_limit=262144, prefill_step=step)
            self.assertEqual(result["request_prefill_step"], step)
        for step in (0, True, 16, 48, 1024, 128.0, "128"):
            with self.assertRaises(ValueError):
                self.generate(engine, prefill_step=step)
        for limit in (0, True, 262145, 65536.5):
            with self.assertRaises(ValueError):
                self.generate(engine, context_limit=limit)
        with self.assertRaises(ValueError):
            self.generate(engine, [1] * 65533, context_limit=65536)
        with self.assertRaises(ValueError):
            self.generate(engine, [1] * 40000)
        legacy = self.engine()
        for step in (32, 64, 128, 256):
            with self.assertRaises(ValueError):
                self.generate(legacy, prefill_step=step)
        with self.assertRaises(ValueError):
            self.generate(legacy, [1] * 40000, context_limit=65536)
        self.assertIsNone(legacy.process)

    def test_mlx_routes_cancel_and_timeout_close_worker(self):
        model = {"path": str(self.model), "sha256": "c" * 64, "weight_bytes": 123}
        engine = self.engine(precision="mlx-q4", mlx_model=model, request_timeout=0.2)
        with self.assertRaisesRegex(NativeEngineError, "timed out"):
            engine.route_features([101], threading.Event())
        self.assertIsNone(engine.process)
        engine.start()
        stop = threading.Event()
        timer = threading.Timer(0.05, stop.set)
        timer.start()
        try:
            with self.assertRaisesRegex(NativeEngineError, "cancelled"):
                engine.route_features([101], stop)
        finally:
            timer.join()
        self.assertIsNone(engine.process)
        self.assertEqual(engine.route_features([1], threading.Event()), [0.25] * 2048)
        for tokens in [[], [True], [248320], [1] * 32769]:
            with self.subTest(tokens=tokens[:2]), self.assertRaises(ValueError):
                engine.route_features(tokens, threading.Event())

    def test_mlx_vision_binds_mode_and_streams_image_request(self):
        model = {"path": str(self.model), "sha256": "c" * 64, "weight_bytes": 123}
        engine = self.engine(precision="mlx-q4", mlx_model=model, vision=True)
        pixels = self.root / "pixels.safetensors"
        pixels.write_bytes(b"fixture")
        seen = []
        result = engine.generate_image([1], 4, [99], 0, seen.append, threading.Event(),
                                       pixel_file=pixels, image_grid_thw=[[1, 4, 4]])
        self.assertEqual(result["tokens"], seen)
        self.assertEqual(seen, [11, 12])
        self.assertIn("--vision", engine.loaded["core_args"])
        for step in (32, 64, 128, 256):
            with self.assertRaises(ValueError):
                self.generate(engine, prefill_step=step)
        with self.assertRaises(ValueError):
            engine.route_features([1], threading.Event())
        with self.assertRaises(ValueError):
            engine.generate_image([1], 4, [99], 0, seen.append, threading.Event(),
                                  pixel_file=pixels, image_grid_thw=[[1, True, 4]])
        bad = self.engine(mode="bad_vision", precision="mlx-q4", mlx_model=model, vision=True)
        with self.assertRaisesRegex(NativeEngineError, "vision loading"):
            bad.start()

    def test_explicit_quantized_precision_reaches_worker(self):
        for precision in ('mixed-q4-g16', 'mixed-q8'):
            with self.subTest(precision=precision):
                engine = self.engine(precision=precision)
                self.generate(engine)
                self.assertEqual(engine.loaded['inference_precision'], precision)
                self.assertIn('--' + precision, engine.loaded['core_args'])
                engine.close()

    def test_direct_q4_requires_actual_carrier_loading_ack(self):
        carrier = {"file": str(self.root / "q4.safetensors"), "sha256": "d" * 64,
                   "quantizer_sha256": "e" * 64, "weight_bytes": 512}
        engine = self.engine(precision="mixed-q4-g16", direct_q4=carrier)
        self.generate(engine)
        self.assertTrue(engine.loaded["direct_q4_loaded"])
        self.assertIn("--direct-q4=" + carrier["file"], engine.loaded["core_args"])
        self.assertEqual(engine.status()["weight_load_mode"], "direct-q4")
        for mode in ("legacy", "bad_direct"):
            with self.subTest(mode=mode), self.assertRaisesRegex(NativeEngineError, "direct Q4"):
                self.engine(mode=mode, precision="mixed-q4-g16", direct_q4=carrier).start()
        for kwargs in ({"precision": "bf16"}, {"skip_layer_mask": 2}, {"exit_after": 23},
                       {"layer_paging_path": self.root / "page.bin"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.engine(direct_q4=carrier, **{"precision": "mixed-q4-g16", **kwargs})

    def test_native_cpu_preparation_and_cancel(self):
        engine = self.engine()
        path = self.root / "prepared.safetensors"
        event = engine.prepare_direct_q4(path, "e" * 64, threading.Event())
        self.assertEqual(path.read_bytes(), b"carrier")
        self.assertEqual(event["direct_q4_sha256"], "d" * 64)
        self.assertIsNone(engine.process)
        stop = threading.Event()
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        self.addCleanup(timer.join)
        with self.assertRaisesRegex(NativeEngineError, "取消"):
            self.engine(mode="prepare_cancel").prepare_direct_q4(self.root / "cancel.safetensors", "e" * 64, stop)

    def test_execution_policy_flags_and_receipt_roundtrip(self):
        paging = self.root / 'new page file.bin'
        engine = self.engine(skip_layer_mask=2, exit_after=12, layer_paging_path=paging)
        self.generate(engine)
        self.assertEqual(engine.loaded['execution_skip_mask'], 2)
        self.assertEqual(engine.loaded['execution_exit_after'], 12)
        self.assertIs(engine.loaded['layer_paging_enabled'], True)
        self.assertIn('--layer-paging=' + str(paging), engine.loaded['core_args'])
        self.assertFalse(paging.exists())  # Python never creates/truncates the native-owned file.

    def test_legacy_worker_only_accepts_default_policy(self):
        engine = self.engine(mode='legacy')
        self.generate(engine)
        engine.close()
        for kwargs in ({'skip_layer_mask': 2}, {'exit_after': 12}, {'layer_paging_path': self.root / 'new.bin'}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(NativeEngineError, 'execution policy'):
                self.engine(mode='legacy', **kwargs).start()

    def test_worker_policy_mismatch_and_wrong_type_fail_closed(self):
        for mode in ('bad_policy', 'bad_policy_type'):
            with self.subTest(mode=mode), self.assertRaisesRegex(NativeEngineError, 'execution policy'):
                self.engine(mode=mode, skip_layer_mask=2).start()

    def test_execution_policy_validation_and_active_core_protection(self):
        for kwargs in ({'skip_layer_mask': True}, {'skip_layer_mask': 1}, {'skip_layer_mask': 1 << 24},
                       {'skip_layer_mask': 1 << 12, 'exit_after': 12},
                       {'exit_after': 0}, {'exit_after': True}, {'exit_after': 25},
                       {'layer_paging_path': 'relative.bin'}, {'layer_paging_path': self.model},
                       {'core': {}, 'exit_after': 12}, {'core': {}, 'skip_layer_mask': 1 << 23}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NativeEngine('/unused', self.model, self.manifest_sha, **kwargs)

    def test_packed_precision_passes_one_absolute_profile_flag(self):
        profile = self.root / 'candidate with spaces.bin'
        profile.write_bytes(b'fixture only; native parser tested independently')
        engine = self.engine(precision='mixed-q4-g16-packed', calibration_profile=profile)
        self.generate(engine)
        self.assertEqual(engine.loaded['inference_precision'], 'mixed-q4-g16-packed')
        self.assertIn('--mixed-q4-g16-packed=' + str(profile), engine.loaded['core_args'])
        for invalid in (None, Path('relative.bin'), self.root / 'missing.bin'):
            with self.subTest(profile=invalid), self.assertRaisesRegex(ValueError, 'absolute profile file'):
                NativeEngine('/unused', self.model, self.manifest_sha,
                             precision='mixed-q4-g16-packed', calibration_profile=invalid)

    def test_persistent_worker_and_token_stream(self):
        engine = self.engine()
        engine.start()
        first_pid = engine.process.pid
        received = []
        first = self.generate(engine, on_token=received.append, seed=73)
        second = self.generate(engine)
        self.assertEqual(received, [11, 12])
        self.assertEqual(first["tokens"], received)
        self.assertEqual(first["seed"], 73)
        self.assertEqual(first["decode_ns"], 50000000)
        self.assertEqual(int(second["id"]), int(first["id"]) + 1)
        self.assertEqual(engine.process.pid, first_pid)
        status = engine.status()
        self.assertTrue(status["running"])
        self.assertEqual(status["binary_sha256"], engine.receipt["binary_sha256"])

    def test_invalid_worker_timing_is_not_accepted_as_speed(self):
        engine = self.engine()
        with self.assertRaisesRegex(NativeEngineError, "Invalid native worker timing"):
            self.generate(engine, [109])

    def test_cache_is_opt_in_and_flag_must_be_boolean(self):
        engine = self.engine()
        self.assertEqual(self.generate(engine)["cached_prompt_tokens"], 0)
        self.assertEqual(self.generate(engine, cache_reuse=True)["cached_prompt_tokens"], 1)
        with self.assertRaisesRegex(ValueError, "Invalid native generation request"):
            self.generate(engine, cache_reuse=1)

    def test_cancel_while_stdout_is_silent_and_reuse(self):
        engine = self.engine(request_timeout=3, cancel_timeout=0.5)
        engine.start()
        process = engine.process
        stop = threading.Event()
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        result = self.generate(engine, [100], stop=stop)
        self.assertEqual(result["finish_reason"], "cancelled")
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])
        self.assertIs(engine.process, process)

    def test_cancel_that_worker_ignores_kills_worker(self):
        engine = self.engine(request_timeout=3, cancel_timeout=0.12)
        engine.start()
        process = engine.process
        stop = threading.Event()
        timer = threading.Timer(0.05, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaisesRegex(NativeEngineError, "timed out"):
            self.generate(engine, [101], stop=stop)
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(engine._cancel_fd)

    def test_late_cancellation_does_not_poison_next_request(self):
        engine = self.engine()
        self.generate(engine)
        # Simulate stop racing with a result already enqueued by the worker.
        os.write(engine._cancel_fd, b"\x01")
        self.assertEqual(self.generate(engine)["tokens"], [11, 12])

    def test_request_timeout_closes_process_and_pipes(self):
        engine = self.engine(request_timeout=0.1)
        engine.start()
        process = engine.process
        with self.assertRaisesRegex(NativeEngineError, "timed out"):
            self.generate(engine, [101])
        self.assertIsNone(engine.process)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertIsNotNone(process.poll())

    def test_load_timeout_closes_process(self):
        engine = self.engine(mode="load_timeout", load_timeout=0.1)
        with self.assertRaisesRegex(NativeEngineError, "timed out"):
            engine.start()
        self.assertIsNone(engine.process)
        self.assertIsNone(engine._cancel_fd)

    def test_partial_crash_keeps_observed_tokens_and_reports_failure(self):
        engine = self.engine()
        received = []
        with self.assertRaisesRegex(NativeEngineError, "intentional worker crash"):
            self.generate(engine, [102], on_token=received.append)
        self.assertEqual(received, [11])
        self.assertIsNone(engine.process)

    def test_protocol_failures_are_fatal(self):
        for key, message in [(103, "bad native tensor"), (104, "identity mismatch"),
                             (105, "differs"), (106, "protocol error")]:
            with self.subTest(key=key):
                engine = self.engine()
                with self.assertRaisesRegex(NativeEngineError, message):
                    self.generate(engine, [key])
                self.assertIsNone(engine.process)

    def test_callback_failure_closes_worker(self):
        engine = self.engine()

        def callback(token):
            raise ValueError("consumer disconnected")

        with self.assertRaisesRegex(NativeEngineError, "consumer disconnected"):
            self.generate(engine, on_token=callback)
        self.assertIsNone(engine.process)

    def test_stderr_is_drained_and_error_text_bounded(self):
        engine = self.engine()
        with self.assertRaises(NativeEngineError) as error:
            self.generate(engine, [108])
        self.assertLess(len(str(error.exception)), 9000)
        self.assertLessEqual(sum(map(len, engine._stderr)), 32 * 4096)

    def test_binary_receipt_and_manifest_changes_rejected_before_spawn(self):
        for target in ("binary", "manifest", "receipt"):
            with self.subTest(target=target):
                engine = self.engine()
                if target == "binary":
                    with engine.binary.open("a") as out:
                        out.write("\n# changed\n")
                elif target == "manifest":
                    (self.model / "upstream-file-manifest.json").write_text("{}")
                else:
                    receipt = Path(str(engine.binary) + ".json")
                    receipt.write_text('{"binary_sha256":"' + "0" * 64 + '"}')
                with self.assertRaises(NativeEngineError):
                    engine.start()
                self.assertIsNone(engine.process)
                (self.model / "upstream-file-manifest.json").write_text('{"Data":{"Files":[]}}')

    def test_loaded_manifest_binding_is_verified(self):
        engine = self.engine(mode="bad_loaded")
        with self.assertRaisesRegex(NativeEngineError, "different model manifest"):
            engine.start()
        self.assertIsNone(engine.process)

    def test_verified_core_arguments_forwarded_without_model_write(self):
        core_path = self.root / "candidate.safetensors"
        core_path.write_bytes(b"fake verified candidate")
        core = {"file": core_path, "tensor_key": "down.weight", "sha256": hashlib.sha256(core_path.read_bytes()).hexdigest()}
        engine = self.engine(core=core)
        loaded = engine.start()
        self.assertEqual(loaded["core_args"], [str(core["file"]), core["tensor_key"], core["sha256"]])
        self.assertEqual(sorted(p.name for p in self.model.iterdir()), ["upstream-file-manifest.json"])

    def test_core_file_change_is_rejected_before_spawn(self):
        path = self.root / "candidate.safetensors"
        path.write_bytes(b"unexpected candidate")
        engine = self.engine(core={"file": path, "tensor_key": "down.weight", "sha256": "a" * 64})
        with self.assertRaisesRegex(NativeEngineError, "core file changed"):
            engine.start()
        self.assertIsNone(engine.process)

    def test_native_expert_binds_core_file_and_actual_load_ack(self):
        path = self.root / "expert.safetensors"
        path.write_bytes(b"verified expert protocol fixture")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        core = {"file": path, "sha256": digest, "tensor_key": "down.weight"}
        expert = {"file": str(path), "sha256": digest, "width": 64, "core_sha256": digest}
        engine = self.engine(core=core, expert=expert, precision="mixed-q4-g16")
        loaded = engine.start()
        self.assertTrue(loaded["expert_loaded"])
        self.assertIn("--expert-file=" + str(path), loaded["core_args"])
        for mode in ("legacy", "bad_expert"):
            invalid = self.engine(mode=mode, core=core, expert=expert, precision="mixed-q4-g16")
            with self.assertRaisesRegex(NativeEngineError, "expert"):
                invalid.start()
            self.assertIsNone(invalid.process)
        for changes in ({"precision": "bf16"}, {"core": None}, {"skip_layer_mask": 2},
                        {"exit_after": 12}, {"expert": {**expert, "core_sha256": "a" * 64}}):
            with self.assertRaises(ValueError):
                self.engine(**{"core": core, "expert": expert, "precision": "mixed-q4-g16", **changes})
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(NativeEngineError, "file changed"):
            self.engine(core=core, expert=expert, precision="mixed-q4-g16").start()

    def test_requests_are_serialized(self):
        engine = self.engine()
        engine.start()
        results, errors = [], []

        def request():
            try:
                results.append(self.generate(engine, [107]))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r["id"] for r in results), ["1", "2"])

    def test_close_interrupts_concurrent_generate_and_is_idempotent(self):
        engine = self.engine()
        engine.start()
        process = engine.process
        errors = []

        def request():
            try:
                self.generate(engine, [101])
            except NativeEngineError as error:
                errors.append(error)

        thread = threading.Thread(target=request)
        thread.start()
        time.sleep(0.05)
        engine.close()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsNotNone(process.poll())
        engine.close()

    def test_close_kills_worker_that_ignores_terminate(self):
        engine = self.engine(mode="ignore_term")
        engine.start()
        process = engine.process
        engine.close()
        self.assertLess(process.returncode, 0)
        self.assertIsNone(engine._cancel_fd)
        self.assertIsNone(engine._cancel_read_fd)
        self.assertTrue(process.stdout.closed)

    def test_pre_cancelled_request_does_not_start_worker(self):
        engine = self.engine()
        stop = threading.Event()
        stop.set()
        self.assertEqual(self.generate(engine, stop=stop)["finish_reason"], "cancelled")
        self.assertIsNone(engine.process)

    def test_invalid_requests_rejected_without_spawn(self):
        engine = self.engine()
        for tokens, maximum, eos, temperature, seed in [([True], 4, [], 0, 42),
                                                     ([1], 0, [], 0, 42),
                                                     ([1], 4, [-1], 0, 42),
                                                     ([1], 4, [], float("nan"), 42),
                                                     ([1], 4, [], 0, -1)]:
            with self.subTest(tokens=tokens, maximum=maximum, temperature=temperature, seed=seed):
                with self.assertRaises(ValueError):
                    engine.generate(tokens, maximum, eos, temperature, lambda token: None,
                                    threading.Event(), seed=seed)
        self.assertIsNone(engine.process)


if __name__ == "__main__":
    unittest.main()
