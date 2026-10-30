"""Unload Python computation while the Bun/native evolution owner holds the GPU.

This is only the app handoff, never validation, training or adoption. One thread
owns both existing locks until the native owner confirms that it has closed.
"""
from __future__ import annotations

import threading
import uuid


class NativeEvolutionLease:
    def __init__(self, runtime, command_lock, ensure_idle):
        self.runtime, self.command_lock, self.ensure_idle = runtime, command_lock, ensure_idle
        self._lock = threading.Lock()
        self._lease = None
        self._released = {}

    @property
    def active(self):
        with self._lock:
            return self._lease is not None

    @staticmethod
    def _identifier(value):
        if not isinstance(value, str):
            raise ValueError("原生进化租约ID无效")
        try:
            if str(uuid.UUID(value)) != value:
                raise ValueError("原生进化租约ID无效")
        except (ValueError, AttributeError) as error:
            raise ValueError("原生进化租约ID无效") from error
        return value

    def acquire(self, identifier):
        identifier = self._identifier(identifier)
        with self._lock:
            if identifier in self._released:
                raise ValueError("已释放的原生进化租约不能重新使用")
            lease = self._lease
            if lease is not None:
                if lease["id"] != identifier:
                    raise RuntimeError("原生进化正在占用模型")
            else:
                lease = {"id": identifier, "ready": threading.Event(), "release": threading.Event(),
                         "done": threading.Event(), "result": None, "error": None}
                self._lease = lease
                threading.Thread(target=self._hold, args=(lease,), daemon=True,
                                 name="luban-native-evolution-lease").start()
        if not lease["ready"].wait(55):
            # The caller keeps its same id and retries; timeout is not permission
            # to release locks while unload may still be running.
            raise RuntimeError("模型仍在卸载，请使用同一原生进化租约重试")
        if lease["error"] is not None:
            raise RuntimeError(str(lease["error"])) from lease["error"]
        return dict(lease["result"])

    def _hold(self, lease):
        command_held = runtime_held = False
        try:
            command_held = self.command_lock.acquire(blocking=False)
            if not command_held:
                raise RuntimeError("另一个模型操作正在进行，请稍后重试")
            runtime_held = self.runtime.lock.acquire(blocking=False)
            if not runtime_held:
                raise RuntimeError("模型正在运行当前任务，请稍后重试")
            self.ensure_idle()
            config = self.runtime.settings.snapshot()
            result = {"lease_id": lease["id"], "was_ready": self.runtime.state == "ready",
                      "identity": self.runtime.identity, "root": str(self.runtime.memory.root),
                      "store": str(self.runtime.store), "model": config["model_path"],
                      "activeCore": self.runtime.active_core}
            if not result["identity"]:
                raise RuntimeError("模型身份尚未就绪，无法启动原生进化")
            self.runtime.stop()
            lease["result"] = result
            lease["ready"].set()
            lease["release"].wait()
        except BaseException as error:
            lease["error"] = error
        finally:
            if runtime_held:
                self.runtime.lock.release()
            if command_held:
                self.command_lock.release()
            with self._lock:
                if self._lease is lease:
                    self._lease = None
                if lease["result"] is not None:
                    self._released[lease["id"]] = lease["result"]["was_ready"]
            lease["ready"].set()
            lease["done"].set()

    def release(self, identifier):
        identifier = self._identifier(identifier)
        with self._lock:
            if identifier in self._released:
                return {"ok": True, "released": True, "was_ready": self._released[identifier]}
            lease = self._lease
            if lease is None:
                return {"ok": True, "released": False, "was_ready": False}
            if lease["id"] != identifier:
                raise ValueError("原生进化租约不匹配")
            lease["release"].set()
        if not lease["done"].wait(55):
            raise RuntimeError("原生进化交接尚未关闭，请用同一租约重试")
        if lease["error"] is not None:
            raise RuntimeError(str(lease["error"])) from lease["error"]
        return {"ok": True, "released": True, "was_ready": lease["result"]["was_ready"]}

    def prepare(self, identifier, operation):
        """One read-only preparation while the caller has not opened its native owner.

        Keep release serialized until the temporary shallow graph is fully closed.
        """
        identifier = self._identifier(identifier)
        with self._lock:
            lease = self._lease
            if (lease is None or lease["id"] != identifier or lease["result"] is None
                    or lease["error"] is not None or lease["release"].is_set()):
                raise ValueError("原生准备需要当前已卸载的模型租约")
            return operation()
