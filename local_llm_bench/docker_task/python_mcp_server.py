from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from mcp.server.fastmcp import FastMCP


_SERVER = FastMCP("python")
_DEFAULT_TIMEOUT_SEC = 15.0


def execute_python_code(code: str) -> str:
    timeout_sec = float(os.getenv("LOCAL_LLM_BENCH_PYTHON_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT_SEC)))
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Python execution timed out after {timeout_sec:.1f} seconds") from exc

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        message = stderr or stdout or f"python exited with status {result.returncode}"
        raise RuntimeError(message)
    return stdout or (stderr if stderr else "Success (no output)")


@_SERVER.tool(
    name="run_python",
    description=(
        "Execute Python code for calculations, string transforms, and helper decoding. "
        "Print the final result you want returned."
    ),
)
def run_python(code: str) -> str:
    return execute_python_code(code)


async def _main() -> None:
    await _SERVER.run_stdio_async()


def main() -> int:
    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
