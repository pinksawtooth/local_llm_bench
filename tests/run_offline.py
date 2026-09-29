"""Run fixture tests with inference, Docker, MCP processes and sockets blocked."""
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
original_popen = subprocess.Popen


def offline_process(args, *positional, **kwargs):
    executable = Path(str(args[0])).name if isinstance(args, (list, tuple)) else ""
    node_check = executable == "node" and str(args[-1]).endswith("/report.js")
    lock_check = (executable.startswith("python") and len(args) == 3 and args[1] == "-c"
                  and "server_lock" in args[2])
    git_read = executable == "git" and list(args[1:]) in (
        ["rev-parse", "--show-toplevel"], ["rev-parse", "HEAD"], ["rev-parse", "--short", "HEAD"],
        ["remote", "get-url", "origin"], ["status", "--porcelain"])
    if not (node_check or lock_check or git_read):
        raise AssertionError("Offline tests blocked process: " + str(args))
    return original_popen(args, *positional, **kwargs)


if __name__ == "__main__":
    from local_llm_bench.inspect_harness import require_inspect
    require_inspect()
    with ExitStack() as stack:
        temporary = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch("inspect_ai._util.appdirs.user_data_path", return_value=temporary / "app"))
        stack.enter_context(patch("inspect_ai._util.appdirs.user_cache_path", return_value=temporary / "cache"))
        stack.enter_context(patch("inspect_ai._eval.task.log.git_context", return_value=None))
        for name in ("connect", "connect_ex", "bind"):
            stack.enter_context(patch("socket.socket." + name, side_effect=AssertionError("Live sockets forbidden")))
        stack.enter_context(patch("subprocess.Popen", side_effect=offline_process))
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        sys.exit(not result.wasSuccessful())
