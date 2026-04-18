from __future__ import annotations

DEFAULT_ANALYSIS_BACKEND = "ghidra"
ANALYSIS_BACKEND_CHOICES = (DEFAULT_ANALYSIS_BACKEND,)


def normalize_analysis_backend(
    value: object,
    *,
    default: str = DEFAULT_ANALYSIS_BACKEND,
) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"analysis backend must be a string: {value!r}")
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized != DEFAULT_ANALYSIS_BACKEND:
        raise ValueError(
            f"Unsupported analysis backend: {value!r}. Choose one of: {', '.join(ANALYSIS_BACKEND_CHOICES)}"
        )
    return normalized
