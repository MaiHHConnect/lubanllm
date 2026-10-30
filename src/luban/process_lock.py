"""One resident backbone per base/device across local storage roots (POSIX)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat


class ModelProcessLock:
    def __init__(self, identity, device):
        self.fd = None
        if os.name != "posix":
            raise RuntimeError("模型进程锁目前仅支持 macOS/Linux；Windows 尚未验证")
        if not re.fullmatch(r"[0-9a-f]{64}", identity) or not re.fullmatch(r"[a-z0-9:-]+", str(device)):
            raise ValueError("模型进程锁的基座或设备无效")
        import fcntl
        self.key = (identity, str(device))
        # A fixed per-user path, independent of settings/storage/TMPDIR overrides.
        directory = Path("/tmp") / ("luban-model-locks-" + str(os.getuid()))
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("模型进程锁目录权限或来源无效")
        fd = os.open(directory / (identity + "-" + str(device).replace(":", "-") + ".lock"),
                     os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
                raise ValueError("模型进程锁文件权限或来源无效")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("同一基座和设备已有模型进程；请先结束该进程") from error
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)  # OS releases flock, including process exit/SIGKILL.
            self.fd = None

    def __del__(self):
        self.close()
