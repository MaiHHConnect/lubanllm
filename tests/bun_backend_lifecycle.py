"""Bounded real CLI acceptance, isolated from production credentials/data/models."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

import psutil


def run(executable=None, evidence=None, backend=None, cached_runtime=False):
    project = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if evidence is None:
        volume = Path("/Volumes/移动数据库")
        if not volume.is_mount():
            raise RuntimeError("Evidence storage is not mounted")
        evidence = volume / "lubanllm/experiments" / ("bun-cli-lifecycle-" + stamp)
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    checks, process, compute_pid, secrets, signature_checks = [], None, None, [], []
    compiled_app = Path(executable).resolve().parents[2] if executable else None
    temporary = tempfile.TemporaryDirectory(prefix="luban-cli-lifecycle-")
    root = Path(temporary.name)
    home, storage = root / "private", root / "storage"
    home.mkdir(mode=0o700)
    cached_python, runtime_probe = None, None
    if cached_runtime:
        cached = home / "runtime/python"
        (cached / "bin").mkdir(parents=True, mode=0o700)
        shutil.copyfile(project / ".venv/pyvenv.cfg", cached / "pyvenv.cfg")
        cached_python = cached / "bin/python"
        cached_python.symlink_to((project / ".venv/bin/python").resolve())
        library = cached / "lib/python3.12"
        library.mkdir(parents=True, mode=0o700)
        (library / "site-packages").symlink_to(project / ".venv/lib/python3.12/site-packages", target_is_directory=True)
    # Use literal seed JSON to ensure the only target is this isolated storage.
    (home / "settings.json").write_text(json.dumps({
        "storage_root": str(storage), "model_path": str(root / "no-model"),
        "speech_model_path": str(root / "no-speech"),
    }))
    fixture_credentials = {"access_key": "lb-isolated-access-1234567890", "ui_token": "isolated-ui-1234567890", "teacher_key": ""}
    (home / "credentials.json").write_text(json.dumps(fixture_credentials))
    secrets.extend(v for v in fixture_credentials.values() if v)
    environment = dict(os.environ)
    for key in ["LUBAN_BUN_RPC_URL", "LUBAN_BUN_RPC_TOKEN", "LUBAN_COMPUTE_TOKEN", "LUBAN_PUBLIC_PORT", "PYTHONDONTWRITEBYTECODE", "DYLD_LIBRARY_PATH"]:
        environment.pop(key, None)
    environment["PATH"] = "/usr/bin:/bin"
    command = ([str(Path(executable).resolve())] if executable else [shutil.which("bun"), str(project / "backend/main.ts")])
    backend = Path(backend).resolve() if backend else project / "src"
    command += ["--project", str(project), "--backend", str(backend), "--home", str(home), "--port", "0"]
    opener = build_opener(ProxyHandler({}))
    public, compute, exit_code = None, None, None

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def request(base, path, *, headers=None, data=None, method=None):
        encoded = None if data is None else json.dumps(data).encode()
        req = Request(base + path, encoded, headers or {}, method=method)
        if encoded is not None:
            req.add_header("Content-Type", "application/json")
        try:
            response = opener.open(req, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None, dict(response.headers), raw

    def group_members(pgid):
        result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid="], capture_output=True, text=True, timeout=5, check=True)
        return [int(row.split()[0]) for row in result.stdout.splitlines() if len(row.split()) == 2 and int(row.split()[1]) == pgid]

    def verify_signature(phase):
        command = ["/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2", str(compiled_app)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        path = evidence / ("codesign-" + phase + ".log")
        path.write_bytes(result.stdout)
        signature_checks.append({"phase": phase, "command": command, "exit_code": result.returncode,
            "log": str(path), "log_sha256": hashlib.sha256(result.stdout).hexdigest()})
        check("compiled strict signature " + phase, result.returncode == 0)
        check("compiled bundle contains no bytecode cache " + phase, not list(compiled_app.rglob("__pycache__")) and not list(compiled_app.rglob("*.pyc")))

    try:
        if compiled_app:
            verify_signature("before")
        if cached_runtime:
            probe_env = {**environment, "PYTHONDONTWRITEBYTECODE": "1"}
            probe_command = "import json,sys,torch; print(json.dumps({'prefix':sys.prefix,'version':sys.version,'torch_version':torch.__version__,'torch_file':torch.__file__}))"
            probes = []
            for name, interpreter in [("original", project / ".venv/bin/python"), ("cached", cached_python)]:
                probe = subprocess.run([str(interpreter), "-c", probe_command], cwd=home, env=probe_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                (evidence / ("runtime-probe-" + name + ".stdout.log")).write_bytes(probe.stdout)
                (evidence / ("runtime-probe-" + name + ".stderr.log")).write_bytes(probe.stderr)
                check(name + " Python runtime imports original torch without loading a model", probe.returncode == 0)
                probes.append(json.loads(probe.stdout))
            check("cached Python prefix is private fixture runtime", Path(probes[1]["prefix"]).resolve() == cached_python.parent.parent.resolve())
            check("cached runtime preserves Python and torch versions", probes[0]["version"] == probes[1]["version"] and probes[0]["torch_version"] == probes[1]["torch_version"])
            check("cached torch is exact existing installation", Path(probes[0]["torch_file"]).resolve() == Path(probes[1]["torch_file"]).resolve())
            runtime_probe = {"original": probes[0], "cached": probes[1], "pyvenv_cfg_sha256": hashlib.sha256((home / "runtime/python/pyvenv.cfg").read_bytes()).hexdigest()}
        with (evidence / "process.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 45
            launch_path = home / "launch.json"
            while not launch_path.exists():
                if process.poll() is not None:
                    raise RuntimeError("Bun process exited before launch receipt")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Launch receipt readiness exceeded 45 seconds")
                time.sleep(0.1)
            launch = json.loads(launch_path.read_text())
            check("launch public PID is actual Bun process", launch.get("pid") == process.pid and launch.get("backend") == "bun")
            check("ephemeral public port is recorded", type(launch.get("port")) is int and 0 < launch["port"] <= 65535)
            check("launch uses isolated UI capability only", set(launch) == {"port", "pid", "token", "backend", "compute_pid"} and launch["token"] == fixture_credentials["ui_token"])
            check("launch private permissions", launch_path.stat().st_mode & 0o777 == 0o600)
            compute_pid = launch["compute_pid"]
            check("compute process group is independent", compute_pid != process.pid and os.getpgid(compute_pid) == compute_pid)
            child = psutil.Process(compute_pid)
            child_arguments, child_environment = child.cmdline(), child.environ()
            check("compute working directory is private fixture home", Path(child.cwd()).resolve() == home.resolve())
            if cached_runtime:
                check("compute uses private cached Python interpreter", Path(child_arguments[0]).absolute() == cached_python.absolute())
                check("compute links private cached torch libraries", child_environment.get("DYLD_LIBRARY_PATH") == str(home / "runtime/python/lib/python3.12/site-packages/torch/lib"))
            private_port = int(child_arguments[child_arguments.index("--port") + 1])
            check("compute owns a separate private port", private_port != launch["port"])
            check("no model autoload", child_environment.get("LUBAN_COMPUTE_AUTOLOAD") == "0")
            if compiled_app:
                check("compute cannot write signed bundle bytecode", child_environment.get("PYTHONDONTWRITEBYTECODE") == "1")
            rpc_key, compute_key = child_environment["LUBAN_BUN_RPC_TOKEN"], child_environment["LUBAN_COMPUTE_TOKEN"]
            secrets.extend([rpc_key, compute_key])
            check("private capabilities are mutually distinct", len(set(secrets)) == len(secrets))
            check("private environment targets public Bun RPC", child_environment["LUBAN_BUN_RPC_URL"] == f"http://127.0.0.1:{launch['port']}/internal/rpc")
            check("private credentials not copied to launch", all(value not in launch_path.read_text() for value in [rpc_key, compute_key]))
            public, compute = f"http://127.0.0.1:{launch['port']}", f"http://127.0.0.1:{private_port}"
            auth = {"Authorization": "Bearer " + compute_key}
            while True:
                try:
                    code, health, _, _ = request(compute, "/api/health", headers=auth)
                    if code == 200:
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                if process.poll() is not None:
                    raise RuntimeError("CLI exited while compute was starting")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Compute readiness exceeded 45 seconds")
                time.sleep(0.2)
            check("private computation remains stopped", health.get("state") == "stopped")
            for key in [None, fixture_credentials["ui_token"], fixture_credentials["access_key"], rpc_key]:
                code, _, _, _ = request(compute, "/api/health", headers={} if key is None else {"Authorization": "Bearer " + key})
                check("compute rejects " + ("anonymous" if key is None else "foreign capability"), code == 401)
            for key in [None, fixture_credentials["ui_token"], fixture_credentials["access_key"], compute_key]:
                code, _, _, _ = request(public, "/internal/rpc", headers={} if key is None else {"Authorization": "Bearer " + key}, data={"target": "settings", "method": "snapshot"})
                check("data RPC rejects " + ("anonymous" if key is None else "foreign capability"), code == 401)
            code, snapshot, _, _ = request(public, "/internal/rpc", headers={"Authorization": "Bearer " + rpc_key}, data={"target": "settings", "method": "snapshot"})
            check("Bun exclusively owns isolated settings", code == 200 and snapshot["ok"] and snapshot["result"]["storage_root"] == str(storage))
            code, _, _, _ = request(public, "/api/samples")
            check("public data requires session", code == 401)
            code, _, headers, _ = request(public, "/api/session", data={"token": fixture_credentials["ui_token"]})
            cookie = headers.get("Set-Cookie", headers.get("set-cookie", ""))
            check("UI login sets private strict session cookie", code == 200 and "HttpOnly" in cookie and "SameSite=strict" in cookie)
            ui = {"Cookie": cookie.split(";", 1)[0]}
            code, status, _, raw = request(public, "/api/status", headers=ui)
            check("status advertises Bun public API port", code == 200 and status.get("api_base") == public + "/v1")
            check("public status leaks no capability", all(secret.encode() not in raw for secret in secrets))
            code, samples, _, _ = request(public, "/api/samples", headers=ui)
            check("isolated sample ledger starts empty", code == 200 and samples.get("samples") == [])
            payload = {"prompt": "isolated fixture question", "answer": "isolated fixture answer", "source": "human", "approved": True, "training_allowed": True}
            code, inserted, _, _ = request(public, "/api/samples", headers=ui, data=payload)
            check("public data creates sample through Bun", code == 200 and bool(inserted.get("id")))
            code, samples, _, raw = request(public, "/api/samples", headers=ui)
            check("public data reads newly stored sample", code == 200 and len(samples.get("samples", [])) == 1 and samples["samples"][0]["prompt"] == payload["prompt"])
            check("sample output leaks no capability", all(secret.encode() not in raw for secret in secrets))
            check("SQLite exists only in fixture", (storage / "app/memory.sqlite3").exists())
            code, public_health, _, _ = request(public, "/api/health")
            check("public health succeeds without authentication", code == 200 and public_health.get("state") == "stopped")
            if executable:
                release = json.loads((project / "release.json").read_text())
                check("compiled candidate reports release metadata", public_health.get("version") == release["version"] and public_health.get("build") == release["build"])
            before = group_members(compute_pid)
            check("private group contains compute", compute_pid in before)
            process.send_signal(signal.SIGTERM)
            exit_code = process.wait(timeout=20)
            check("Bun SIGTERM exits cleanly", exit_code == 0)
            deadline = time.monotonic() + 5
            while group_members(compute_pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            check("entire compute group reclaimed", group_members(compute_pid) == [])
            check("Bun launch receipt cleaned", not launch_path.exists())
            for address in [public, compute]:
                try:
                    request(address, "/api/health")
                    closed = False
                except (URLError, TimeoutError, ConnectionError):
                    closed = True
                check("public/private listeners closed", closed)
        check("process logs leak no credential", all(value not in (evidence / "process.log").read_text(errors="replace") for value in secrets))
        if compiled_app:
            verify_signature("after")
        outcome, failure = "passed", None
    except Exception as error:
        outcome, failure = "failed", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if process and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        if compute_pid:
            try:
                os.killpg(compute_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        sources = {}
        for path in ["backend/main.ts", "backend/server.ts", "backend/settings.ts", "backend/memory.ts", "backend/wire.ts", "src/luban/server.py", "src/luban/bun_bridge.py", "tests/bun_backend_lifecycle.py"]:
            content = (project / path).read_bytes()
            sources[path] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        receipt = {"created_at": stamp, "status": outcome, "mode": "compiled" if executable else "source", "command": command,
            "restricted_path": environment["PATH"], "checks": checks, "failure": failure, "exit_code": exit_code,
            "public_pid": process.pid if process else None, "compute_pid": compute_pid,
            "signature_checks": signature_checks,
            "cached_runtime": cached_runtime, "runtime_probe": runtime_probe,
            "source": sources, "evidence_directory": str(evidence), "production_data_accessed": False,
            "model_loaded": False, "teacher_calls": 0, "fixture_root": str(root)}
        if executable:
            binary_path = Path(executable).resolve()
            if binary_path.is_file():
                binary = binary_path.read_bytes()
                receipt["binary"] = {"path": str(binary_path), "bytes": len(binary), "sha256": hashlib.sha256(binary).hexdigest()}
            else:
                receipt["binary"] = {"path": str(binary_path), "exists": False}
            receipt["packaged_python_source"] = {}
            for name in ["server.py", "settings.py", "bun_bridge.py", "memory.py"]:
                source_path = backend / "luban" / name
                if source_path.is_file():
                    content = source_path.read_bytes()
                    receipt["packaged_python_source"][name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        # Retain the isolated SQLite and config metadata for failed investigations, never credentials.
        if storage.exists():
            shutil.copytree(storage, evidence / "fixture-storage")
        (evidence / "result.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        temporary.cleanup()
        print(json.dumps({"status": outcome, "passed_checks": sum(c["passed"] for c in checks), "evidence_directory": str(evidence), "failure": failure and failure["message"]}, ensure_ascii=False))
    return 0 if outcome == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable")
    parser.add_argument("--evidence")
    parser.add_argument("--backend")
    parser.add_argument("--cached-runtime", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.executable, arguments.evidence, arguments.backend, arguments.cached_runtime))
