"""Read-only, sampled host / explicitly selected inference-process memory."""
from __future__ import annotations

import threading


class MemorySampler:
    def __init__(self, pid: int | None = None, interval: float = 0.25):
        self.pid, self.interval = pid, interval
        self.data = {"peak_rss_bytes": None, "peak_host_used_bytes": None,
                     "peak_swap_used_bytes": None, "pid": pid, "samples": 0,
                     "rss_scope": "selected_process_tree_rss" if pid else "unavailable",
                     "host_scope": "benchmark_host", "interval_sec": interval,
                     "method": "sampled_peak (RSS is not GPU allocated memory)"}
        self.stop = threading.Event()

    def _sample(self):
        import psutil
        self.data["samples"] += 1
        for key, reader in (("peak_host_used_bytes", lambda: psutil.virtual_memory().used),
                            ("peak_swap_used_bytes", lambda: psutil.swap_memory().used)):
            try:
                value = reader()
                self.data[key] = max(self.data[key] or 0, value)
            except (OSError, psutil.Error):
                pass
        if self.process is not None:
            try:
                if self.process.is_running():
                    processes = {p.pid: p for p in [self.process, *self.process.children(recursive=True)]}
                    value = sum(p.memory_info().rss for p in processes.values())
                    self.data["peak_rss_bytes"] = max(self.data["peak_rss_bytes"] or 0, value)
            except (OSError, psutil.Error):
                pass

    def _poll(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        import psutil
        self.process = None
        if self.pid:
            try:
                self.process = psutil.Process(self.pid)
                self.data["process_created_at"] = self.process.create_time()
            except (OSError, psutil.Error):
                self.data["rss_scope"] = "unavailable"
        self._sample()
        self.thread = threading.Thread(target=self._poll, daemon=True, name="bench-memory")
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()
