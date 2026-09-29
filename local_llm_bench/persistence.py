from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def server_identity(api_base: str) -> str:
    parsed = urlsplit(api_base)
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1", "host.docker.internal", "0.0.0.0"}:
        host = "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme.lower()}://{host}:{port}"


class FileLock:
    """An advisory OS lock; a dead process cannot leave a stale reservation."""

    def __init__(self, path: Path, *, timeout: float = 0.0):
        self.path, self.timeout, self.stream = path, timeout, None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.stream.close()
                    self.stream = None
                    raise RuntimeError(f"別のベンチマークが使用中です: {self.path}") from None
                time.sleep(0.05)
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(str(os.getpid()))
        self.stream.flush()
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def server_lock(api_base: str) -> FileLock:
    key = hashlib.sha256(server_identity(api_base).encode()).hexdigest()
    root = Path(tempfile.gettempdir()) / f"local-llm-bench-{os.getuid()}"
    return FileLock(root / f"server-{key}.lock")
