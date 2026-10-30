"""Private loopback RPC to Bun; Python never opens business data in bridge mode."""
from __future__ import annotations

import json
import math
import base64
import re
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

MAX_RPC_BYTES = 64 * 1024 * 1024
RPC_TIMEOUT = 25
RPC_STARTUP_RETRIES = 3
RPC_RETRY_DELAY = 0.25


def enabled():
    # A partial configuration fails closed instead of silently opening SQLite.
    return bool(os.environ.get("LUBAN_BUN_RPC_URL") or os.environ.get("LUBAN_BUN_RPC_TOKEN"))


def _encode(value):
    if type(value) is int and abs(value) > 9007199254740991:
        return {"$luban_int": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Bun RPC numbers must be finite")
        return {"$luban_float": repr(value)}
    if isinstance(value, (bytes, bytearray)):
        return {"$luban_bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Bun RPC response contains a non-finite number")
    if isinstance(value, dict):
        if set(value) == {"$luban_int"}:
            text = value["$luban_int"]
            if not isinstance(text, str) or not re.fullmatch(r"-?\d+", text, flags=re.ASCII):
                raise RuntimeError("Bun RPC integer response is invalid")
            return int(text)
        if set(value) == {"$luban_float"}:
            text = value["$luban_float"]
            if not isinstance(text, str):
                raise RuntimeError("Bun RPC float response is invalid")
            return _decode(float(text))
        if set(value) == {"$luban_bytes"}:
            return base64.b64decode(value["$luban_bytes"], validate=True)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Bridge:
    def __init__(self):
        self.url = os.environ.get("LUBAN_BUN_RPC_URL", "")
        self.token = os.environ.get("LUBAN_BUN_RPC_TOKEN", "")
        try:
            url = urlsplit(self.url)
            valid = (url.scheme == "http" and url.hostname == "127.0.0.1" and url.port
                     and url.path == "/internal/rpc" and not url.username and not url.password
                     and not url.query and not url.fragment)
        except ValueError:
            valid = False
        if not valid or not self.token or not self.token.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in self.token):
            raise RuntimeError("Private Bun RPC environment is invalid")
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def call(self, target, method, args=(), kwargs=None, **fields):
        payload = {"target": target, "method": method, "args": args, "kwargs": kwargs or {}, **fields}
        data = json.dumps(_encode(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(data) > MAX_RPC_BYTES:
            raise ValueError("Bun RPC request is too large")
        request = Request(self.url, data=data, headers={"Authorization": "Bearer " + self.token,
                          "Content-Type": "application/json"}, method="POST")
        last_error = None
        for attempt in range(RPC_STARTUP_RETRIES):
            try:
                with self.opener.open(request, timeout=RPC_TIMEOUT) as response:
                    raw = response.read(MAX_RPC_BYTES + 1)
                break
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = error
                reason = getattr(error, "reason", error)
                transient = isinstance(reason, (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, TimeoutError))
                if not transient or attempt + 1 == RPC_STARTUP_RETRIES:
                    raise RuntimeError("Private Bun RPC unavailable") from error
                time.sleep(RPC_RETRY_DELAY * (attempt + 1))
        else:
            raise RuntimeError("Private Bun RPC unavailable") from last_error
        if len(raw) > MAX_RPC_BYTES:
            raise RuntimeError("Bun RPC response is too large")
        try:
            result = _decode(json.loads(raw))
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Bun RPC response is invalid") from error
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise RuntimeError("Bun RPC response is invalid")
        if result["ok"]:
            return result.get("result")
        failure = result.get("error", {})
        if not isinstance(failure, dict):
            raise RuntimeError("Bun RPC response is invalid")
        kind, message = failure.get("type"), failure.get("message", "Bun RPC failed")
        errors = {"ValueError": ValueError, "TypeError": TypeError, "RuntimeError": RuntimeError,
                  "IntegrityError": sqlite3.IntegrityError, "OperationalError": sqlite3.OperationalError,
                  "DatabaseError": sqlite3.DatabaseError, "Error": sqlite3.Error}
        raise errors.get(kind, RuntimeError)(str(message))


class Row:
    """The sqlite3.Row operations used by the computation and audit callers."""
    def __init__(self, columns, values):
        self.columns, self.values = columns, values

    def keys(self):
        return self.columns

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self.values[key]
        for index, name in enumerate(self.columns):
            if name.lower() == key.lower():
                return self.values[index]
        raise IndexError("No item with that key")

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class Cursor:
    def __init__(self, result):
        self.description = [(column, None, None, None, None, None, None) for column in result["columns"]] or None
        self.rows = iter(Row(result["columns"], values) for values in result["rows"])
        self.rowcount = result["rowcount"]
        self.lastrowid = result.get("lastrowid")

    def fetchone(self):
        return next(self.rows, None)

    def fetchall(self):
        return list(self.rows)

    def fetchmany(self, size=1):
        result = []
        for _ in range(size):
            row = self.fetchone()
            if row is None:
                break
            result.append(row)
        return result

    def __iter__(self):
        return self.rows


class Connection:
    def __init__(self, memory, readonly=False):
        self.memory = memory
        result = memory._call("db_begin", kwargs={"readonly": readonly})
        self.transaction = result["transaction"]
        self.in_transaction = result.get("in_transaction", False)
        self.closed = False

    def _call(self, method, **kwargs):
        if self.closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database")
        return self.memory._call(method, kwargs=kwargs, transaction=self.transaction)

    def execute(self, sql, parameters=()):
        result = self._call("db_execute", sql=sql, parameters=parameters)
        self.in_transaction = result["in_transaction"]
        return Cursor(result)

    def executescript(self, sql):
        result = self._call("db_script", sql=sql)
        self.in_transaction = result.get("in_transaction", False) if isinstance(result, dict) else False
        return result

    def executemany(self, sql, parameters):
        count = 0
        for values in parameters:
            cursor = self.execute(sql, values)
            count += cursor.rowcount
        return Cursor({"columns": [], "rows": [], "rowcount": count, "lastrowid": None})

    def commit(self):
        self._call("db_commit")
        self.in_transaction = False

    def rollback(self):
        self._call("db_rollback")
        self.in_transaction = False

    def close(self):
        if not self.closed:
            try:
                self._call("db_close")
            finally:
                self.closed = True


class RemoteMemory:
    def __init__(self, storage_root):
        self.root = Path(storage_root)
        if not self.root.is_absolute():
            self.root = self.root.absolute()
        self.path = self.root / "app/memory.sqlite3"
        self.lock = threading.RLock()
        self.model_identity = None
        self.bridge = Bridge()

    def _call(self, method, args=(), kwargs=None, **fields):
        return self.bridge.call("memory", method, args, kwargs, root=str(self.root),
                                identity=self.model_identity, **fields)

    @contextmanager
    def connect(self, readonly=False):
        with self.lock:
            connection = Connection(self, readonly)
            try:
                yield connection
            except BaseException as error:
                try:
                    connection.rollback()
                except Exception:
                    error.add_note("Bun RPC rollback was not acknowledged; server lease expiry will roll back")
                try:
                    connection.close()
                except Exception:
                    error.add_note("Bun RPC connection close was not acknowledged")
                raise
            else:
                connection.commit()
            finally:
                if not connection.closed:
                    connection.close()


def _memory_method(name):
    def invoke(self, *args, **kwargs):
        with self.lock:
            return self._call(name, args, kwargs)
    invoke.__name__ = name
    return invoke


for _name in ("growth_reserved", "add_sample", "samples", "delete_sample", "record_usage", "usage",
              "capture_teacher", "teacher_materials", "material_counts", "save_conversation",
              "conversations", "conversation", "delete_conversation", "event", "events"):
    setattr(RemoteMemory, _name, _memory_method(_name))


class RemoteSettings:
    def __init__(self, home=None):
        from .settings import default_home
        self.home = Path(home or default_home()).absolute()
        self.path, self.secret_path = self.home / "settings.json", self.home / "credentials.json"
        self.lock = threading.RLock()
        self.bridge = Bridge()

    def _call(self, method, *args):
        with self.lock:
            return self.bridge.call("settings", method, args, home=str(self.home))

    def snapshot(self):
        return self._call("snapshot")

    def public(self):
        return self._call("public")

    def secret(self, name):
        return self._call("secret", name)

    def agreement(self):
        return self._call("agreement")

    def accept_agreement(self):
        return self._call("accept_agreement")

    def validated(self, changes):
        return tuple(self._call("validated", changes))

    def update(self, changes):
        return self._call("update", changes)


def readonly_connection(database):
    """Keep package/journal audits read-only, with Bun owning even temporary DBs."""
    if enabled():
        return Connection(RemoteMemory(Path(database).parent.parent), readonly=True)
    return sqlite3.connect(Path(database).as_uri() + "?mode=ro", uri=True)
