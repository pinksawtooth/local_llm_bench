from __future__ import annotations

DEFAULT_GHIDRA_TOOL_MODE = "unrestricted"
GHIDRA_TOOL_MODE_CHOICES = (
    DEFAULT_GHIDRA_TOOL_MODE,
    "decompile-only",
    "disassembly-only",
)

_ALIASES = {
    "all": DEFAULT_GHIDRA_TOOL_MODE,
    "default": DEFAULT_GHIDRA_TOOL_MODE,
    "full": DEFAULT_GHIDRA_TOOL_MODE,
    DEFAULT_GHIDRA_TOOL_MODE: DEFAULT_GHIDRA_TOOL_MODE,
    "decompile": "decompile-only",
    "decompile-only": "decompile-only",
    "decompile_only": "decompile-only",
    "disasm": "disassembly-only",
    "disassembly": "disassembly-only",
    "disassembly-only": "disassembly-only",
    "disassembly_only": "disassembly-only",
}


def normalize_ghidra_tool_mode(
    value: object,
    *,
    default: str = DEFAULT_GHIDRA_TOOL_MODE,
) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"ghidra tool mode must be a string: {value!r}")
    normalized = value.strip().lower()
    if not normalized:
        return default
    canonical = _ALIASES.get(normalized)
    if canonical is not None:
        return canonical
    canonical = _ALIASES.get(normalized.replace("-", "_"))
    if canonical is not None:
        return canonical
    canonical = _ALIASES.get(normalized.replace("_", "-"))
    if canonical is not None:
        return canonical
    raise ValueError(
        f"Unsupported ghidra tool mode: {value!r}. Choose one of: {', '.join(GHIDRA_TOOL_MODE_CHOICES)}"
    )
