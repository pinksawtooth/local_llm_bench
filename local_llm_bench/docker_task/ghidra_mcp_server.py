from __future__ import annotations

import argparse
import sys
from typing import Any

from .ghidra_tool_mode import DEFAULT_GHIDRA_TOOL_MODE, normalize_ghidra_tool_mode

_DECOMPILE_TOOLS = frozenset({"decompile_function", "decompile_function_by_address"})
_DISASSEMBLY_TOOLS = frozenset({"disassemble_function", "get_bytes", "search_bytes"})


def _blocked_tool_names_for_mode(mode: str) -> frozenset[str] | None:
    normalized = normalize_ghidra_tool_mode(mode)
    if normalized == DEFAULT_GHIDRA_TOOL_MODE:
        return None
    if normalized == "decompile-only":
        return _DISASSEMBLY_TOOLS
    if normalized == "disassembly-only":
        return _DECOMPILE_TOOLS
    raise ValueError(f"Unsupported ghidra tool mode: {mode!r}")


def _filter_tool_specs(specs: dict[str, Any], *, mode: str) -> dict[str, Any]:
    blocked = _blocked_tool_names_for_mode(mode)
    if blocked is None:
        return dict(specs)
    return {name: spec for name, spec in specs.items() if name not in blocked}


def _extract_tool_mode(argv: list[str]) -> tuple[list[str], str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ghidra-tool-mode", default=DEFAULT_GHIDRA_TOOL_MODE)
    known, remaining = parser.parse_known_args(argv)
    return remaining, normalize_ghidra_tool_mode(known.ghidra_tool_mode)


def _install_tool_filter(mode: str) -> None:
    from ghidra_mcp.presentation.tool_registry import ToolRegistry

    original_register_all = ToolRegistry.register_all

    def _patched_register_all(
        mcp,
        specs,
        dispatcher_provider,
        registry_provider,
        *,
        include_shared_sync: bool = False,
    ):
        filtered_specs = _filter_tool_specs(specs, mode=mode)
        return original_register_all(
            mcp,
            filtered_specs,
            dispatcher_provider,
            registry_provider,
            include_shared_sync=include_shared_sync,
        )

    ToolRegistry.register_all = staticmethod(_patched_register_all)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    forwarded_argv, tool_mode = _extract_tool_mode(raw_argv)
    _install_tool_filter(tool_mode)

    import ghidra_mcp.presentation.cli as ghidra_cli

    return ghidra_cli.main(forwarded_argv)


if __name__ == "__main__":
    raise SystemExit(main())
