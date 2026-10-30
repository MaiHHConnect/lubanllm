"""CPU-only bridge contracts against a real isolated Bun OS subprocess."""
import json
import io
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from luban.bun_bridge import Bridge, RemoteMemory, RemoteSettings, Row, readonly_connection
from luban.memory import Memory
from luban.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_partial_and_bad_environment_never_falls_back_to_local_database(self):
        with tempfile.TemporaryDirectory() as directory:
            for url, token in [("", "secret"), ("http://example.org/internal/rpc", "secret"),
                               ("http://127.0.0.1:9/wrong", "secret"),
                               ("http://127.0.0.1:9/internal/rpc?q=1", "secret"),
                               ("http://127.0.0.1:9/internal/rpc", "bad\nkey"),
                               ("http://127.0.0.1:bad/internal/rpc", "secret")]:
                with self.subTest(url=url), patch.dict(os.environ, {"LUBAN_BUN_RPC_URL": url,
                                                                   "LUBAN_BUN_RPC_TOKEN": token}):
                    with self.assertRaisesRegex(RuntimeError, "environment"):
                        Memory(directory)
                    with self.assertRaisesRegex(RuntimeError, "environment"):
                        Settings(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_disabled_keeps_python_oracle(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                {"LUBAN_BUN_RPC_URL": "", "LUBAN_BUN_RPC_TOKEN": ""}):
            self.assertIsInstance(Memory(directory), Memory)
            self.assertIsInstance(Settings(directory), Settings)

    def test_row_mapping_and_positional_contract(self):
        row = Row(["Id", "state"], [12, "committing"])
        self.assertEqual(dict(row), {"Id": 12, "state": "committing"})
        self.assertEqual(tuple(row), (12, "committing"))
        self.assertEqual(row["id"], row[0])

    def test_mock_rpc_bounds_and_malformed_responses(self):
        with patch.dict(os.environ, {"LUBAN_BUN_RPC_URL":"http://127.0.0.1:12345/internal/rpc",
                                     "LUBAN_BUN_RPC_TOKEN":"private-test-token"}):
            bridge = Bridge()
        for response in (b'[]', b'{"ok":"yes"}', b'not-json', b'{"ok":false,"error":[]}'):
            with self.subTest(response=response), patch.object(bridge, "opener") as opener:
                opener.open.return_value.__enter__.return_value.read.return_value = response
                with self.assertRaisesRegex(RuntimeError, "invalid"):
                    bridge.call("settings", "snapshot")
        with patch("luban.bun_bridge.MAX_RPC_BYTES", 100), patch.object(bridge, "opener") as opener:
            with self.assertRaisesRegex(ValueError, "too large"):
                bridge.call("memory", "event", args=["test", "x" * 100])
            opener.open.assert_not_called()
            opener.open.return_value.__enter__.return_value.read.return_value = b"x" * 101
            with self.assertRaisesRegex(RuntimeError, "too large"):
                bridge.call("settings", "snapshot")


@unittest.skipUnless(shutil.which("bun"), "Bun runtime unavailable")
class RealBunBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.home = cls.root / "home"
        cls.storage = cls.root / "storage"
        cls.token = "private-test-token-distinct-from-public"
        script = cls.root / "rpc.ts"
        script.write_text('''
import { Memory } from MEMORY_MODULE;
import { Settings } from SETTINGS_MODULE;
import { decodeWire, encodeWire } from WIRE_MODULE;
const settings = new Settings(process.env.TEST_HOME);
const homes = new Map([[process.env.TEST_HOME, settings]]);
const memories = new Map();
const server = Bun.serve({hostname:"127.0.0.1", port:0, async fetch(request) {
 if (request.headers.get("authorization") !== "Bearer " + process.env.TEST_TOKEN) return new Response("unauthorized", {status:401});
 try {
  const body = decodeWire(await request.json()); let result;
  if (body.target === "memory") {
   if (!memories.has(body.root)) memories.set(body.root, new Memory(body.root));
   result = await memories.get(body.root).dispatch(body.method, body.args, body.kwargs, body.identity, body.transaction);
  } else {
   if (!homes.has(body.home)) homes.set(body.home, new Settings(body.home));
   const methods = new Set(["snapshot","public","secret","agreement","accept_agreement","validated","update"]);
   if (!methods.has(body.method)) throw new Error("method mismatch");
   result = homes.get(body.home)[body.method](...body.args);
  }
  return new Response(JSON.stringify(encodeWire({ok:true,result:result ?? null})));
 } catch (error) {
  const type = error.code?.startsWith("SQLITE_CONSTRAINT") ? "IntegrityError" : error.code ? "OperationalError" : "ValueError";
  return Response.json({ok:false,error:{type,message:error.message}});
 }
}});
console.log(JSON.stringify({port:server.port}));
process.on("SIGTERM", () => {for (const memory of memories.values()) memory.close(); server.stop(true); process.exit(0);});
'''.replace("MEMORY_MODULE", json.dumps(str(ROOT / "backend/memory.ts")))
   .replace("SETTINGS_MODULE", json.dumps(str(ROOT / "backend/settings.ts")))
   .replace("WIRE_MODULE", json.dumps(str(ROOT / "backend/wire.ts"))))
        environment = {**os.environ, "TEST_HOME": str(cls.home), "TEST_TOKEN": cls.token}
        cls.process = subprocess.Popen([shutil.which("bun"), str(script)], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = cls.process.stdout.readline()
        if not ready:
            raise RuntimeError("Bun fixture failed: " + cls.process.stderr.read())
        cls.url = f'http://127.0.0.1:{json.loads(ready)["port"]}/internal/rpc'
        cls.environment = patch.dict(os.environ, {"LUBAN_BUN_RPC_URL": cls.url, "LUBAN_BUN_RPC_TOKEN": cls.token})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.process.terminate()
        cls.process.communicate(timeout=10)
        if cls.process.poll() is None:
            raise AssertionError("Bun worker remains alive")
        cls.temp.cleanup()

    def memory(self):
        return Memory(self.storage / self._testMethodName)

    def test_business_storage_owned_by_bun_and_identity(self):
        memory = self.memory()
        self.assertIsInstance(memory, RemoteMemory)
        memory.model_identity = "base-a"
        with patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden")):
            sample = memory.add_sample("question", "answer", approved=True, training_allowed=True)
            self.assertEqual(memory.samples()[0]["id"], sample)
            memory.model_identity = "base-b"
            self.assertEqual(memory.samples(), [])
            memory.record_usage("local", {"prompt_tokens": 9007199254740993, "total_tokens": 9007199254740993})
            self.assertEqual(memory.usage()["local"]["total_tokens"], 9007199254740993)

    def test_atomic_session_commits_and_rolls_back_whole_transaction(self):
        memory = self.memory()
        with patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden")):
            with memory.connect() as db:
                self.assertFalse(db.in_transaction)
                db.execute("CREATE TABLE receipts (id INTEGER PRIMARY KEY, body TEXT)")
                db.execute("BEGIN IMMEDIATE")
                self.assertTrue(db.in_transaction)
                db.execute("INSERT INTO receipts VALUES (?,?)", (1, "committed"))
                row = db.execute("SELECT * FROM receipts WHERE id=?", (1,)).fetchone()
                self.assertEqual(dict(row), {"id": 1, "body": "committed"})
            with self.assertRaisesRegex(ValueError, "abort"), memory.connect() as db:
                db.execute("INSERT INTO receipts VALUES (?,?)", (2, "uncommitted"))
                db.execute("UPDATE receipts SET body='mutated' WHERE id=1")
                raise ValueError("abort")
            with memory.connect() as db:
                self.assertEqual([tuple(row) for row in db.execute("SELECT * FROM receipts")], [(1, "committed")])

    def test_other_connections_wait_without_observing_uncommitted_writes(self):
        memory = self.memory()
        other = Memory(memory.root)
        started = threading.Event()
        def read():
            started.set()
            with other.connect() as db:
                return db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        with ThreadPoolExecutor(max_workers=1) as pool:
            with memory.connect() as db:
                db.execute("CREATE TABLE receipts(id INTEGER PRIMARY KEY)")
                db.execute("INSERT INTO receipts VALUES (1)")
                future = pool.submit(read)
                self.assertTrue(started.wait(2))
                self.assertFalse(future.done())
            self.assertEqual(future.result(timeout=5), 1)

    def test_readonly_audits_and_error_mapping(self):
        memory = self.memory()
        with memory.connect() as db:
            db.execute("INSERT INTO events VALUES ('id','test','{}',0)")
        with patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden")):
            db = readonly_connection(memory.path)
            try:
                self.assertEqual(db.execute("SELECT id FROM events").fetchone()[0], "id")
                with self.assertRaises(Exception):
                    db.execute("DELETE FROM events")
            finally:
                db.close()
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError), memory.connect() as db:
            db.execute("INSERT INTO events VALUES ('id','test','{}',0)")

    def test_remote_settings_preserve_public_credentials_and_no_python_writes(self):
        with patch("luban.settings.private_json", side_effect=AssertionError("Python settings write forbidden")):
            settings = Settings(self.home)
            self.assertIsInstance(settings, RemoteSettings)
            key = settings.secret("access_key")
            self.assertNotEqual(key, self.token)
            self.assertNotIn("access_key", settings.public())
            settings.update({"temperature": 0.8})
            self.assertEqual(settings.snapshot()["temperature"], 0.8)
            with self.assertRaises(ValueError):
                settings.validated({"max_new_tokens": 16.0})
            draft, credentials = settings.validated({"temperature": 0.9})
            self.assertEqual(draft["temperature"], 0.9)
            self.assertEqual(credentials["access_key"], key)
            self.assertEqual(settings.snapshot()["temperature"], 0.8)

    def test_float_binary_and_sql_integer_precision(self):
        memory = self.memory()
        with memory.connect() as db:
            db.execute("CREATE TABLE numeric (n INTEGER, value REAL, data BLOB)")
            db.execute("INSERT INTO numeric VALUES (?,?,?)", (9223372036854775807, 0.8, b"\x00\xffbinary"))
            row = db.execute("SELECT n,value,data FROM numeric").fetchone()
            self.assertEqual(tuple(row), (9223372036854775807, 0.8, b"\x00\xffbinary"))
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                memory.event("invalid", {"value": bad})

    def test_existing_publication_and_external_brain_contracts_against_bun(self):
        # Exercise the actual TrainingSession/material guards and exact-reference
        # callers, replacing only the business-data owner with the Bun process.
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "tests.test_core_publication_journal", "tests.test_router_publication", "tests.test_external_brain",
            "tests.test_memory_identity", "tests.test_feeding", "tests.test_automatic",
            "tests.test_local_review", "tests.test_upgrades"])
        def compatible(test):
            if isinstance(test, unittest.TestSuite):
                return unittest.TestSuite(compatible(item) for item in test)
            # This Python-oracle fixture directly mutates Settings.data and
            # credentials. Remote Settings deliberately has no mutable cache;
            # its public agreement/clear_key contracts are tested separately.
            if test.id().endswith(".test_agreement_and_missing_provider_zero_calls"):
                return unittest.TestSuite()
            return test
        suite = compatible(suite)
        output = io.StringIO()
        with patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden")):
            result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
        self.assertTrue(result.wasSuccessful(), output.getvalue())
        self.assertGreaterEqual(result.testsRun, 20)

    def test_private_rpc_auth_and_unavailability(self):
        request = Request(self.url, data=b'{}', headers={"Authorization":"Bearer wrong"}, method="POST")
        with self.assertRaises(HTTPError) as failure:
            urlopen(request, timeout=3)
        self.assertEqual(failure.exception.code, 401)
        with patch.dict(os.environ, {"LUBAN_BUN_RPC_TOKEN":"wrong"}):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                Bridge().call("settings", "snapshot", home=str(self.home))

    def test_compute_boundary_uses_private_key_and_preserves_public_launch(self):
        from fastapi.testclient import TestClient
        from luban.server import create_app
        from tests.test_app import FakeRuntime
        settings = Settings(self.home)
        settings.update({"storage_root": str(self.storage / "compute")})
        launch = self.home / "launch.json"
        launch.write_text('{"port":8765,"pid":111,"token":"public-launch"}')
        previous = launch.read_bytes()
        with patch.dict(os.environ, {"LUBAN_COMPUTE_TOKEN":"isolated-compute-secret", "LUBAN_PUBLIC_PORT":"8765"}):
            app = create_app(settings, runtime=FakeRuntime(self.memory()), port=18765, autoload=False)
            with TestClient(app, client=("127.0.0.1", 40000)) as client:
                for path in ("/", "/api/health", "/api/settings", "/v1/models"):
                    self.assertEqual(client.get(path).status_code, 401)
                self.assertEqual(client.get("/api/health", headers={"Authorization":"Bearer " + settings.secret("access_key")}).status_code, 401)
                client.headers["Authorization"] = "Bearer isolated-compute-secret"
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertEqual(client.get("/api/access-key").json()["key"], settings.secret("access_key"))
                self.assertEqual(client.get("/v1/models").status_code, 200)
                self.assertEqual(client.get("/internal/data-scope").json()["root"], str(app.state.runtime.memory.root))
                self.assertEqual(launch.read_bytes(), previous)
            with TestClient(app, client=("192.0.2.10", 40000)) as client:
                self.assertEqual(client.get("/api/health", headers={"Authorization":"Bearer isolated-compute-secret"}).status_code, 401)
        self.assertEqual(launch.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
