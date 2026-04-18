from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_BINARY_EXTENSIONS = {".exe", ".dll", ".so", ".dylib", ".o", ".elf", ".bin"}
_SCRIPT_EXTENSIONS = {".py", ".js", ".ts", ".cs", ".rb", ".lua", ".sh", ".ps1"}
_WEB_EXTENSIONS = {".html", ".htm", ".wasm", ".wat"}
_BINARY_MAGIC_PREFIXES = (
    b"\x7fELF",
    b"MZ",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
)
_ARCHIVE_SUFFIX_SEQUENCES = (
    (".tar", ".gz"),
    (".tar", ".xz"),
    (".tar", ".bz2"),
    (".tgz",),
    (".txz",),
    (".tbz2",),
    (".zip",),
    (".tar",),
)
_LIBRARY_EXTENSIONS = {".so", ".dll", ".dylib", ".o"}
_EXECUTABLE_HINT_EXTENSIONS = {".exe", ".bin", ".elf"}
_NATIVE_BINARY_KIND_PRIORITY = {
    "native_executable": 3,
    "native_binary": 2,
    "native_library": 1,
    "data_blob": 0,
}

_SHARED_SYSTEM_PROMPT = (
    "You are an expert reverse engineer. Use available MCP tools "
    "(Ghidra for binaries and the Python MCP tool run_python) to analyze the target.\n\n"
    "## IMPORTANT: How to Start\n"
    "- If analyzing native binaries, begin with Ghidra MCP.\n"
    "- In this benchmark environment, the Ghidra MCP server name is `mecha_ghidra`.\n"
    "- Use tools to gather facts. Do not browse the internet.\n\n"
    "## Analysis Strategy\n"
    "1. Understand the question.\n"
    "2. Explore the binary with Ghidra.\n"
    "3. Use run_python only for helper calculations or decoding.\n"
    "4. When you know the answer, output exactly:\n"
    "FINAL_ANSWER: <answer>\n"
    "Keep investigating until you can provide a definitive FINAL_ANSWER."
)


@dataclass(frozen=True)
class ResolvedGhidraTarget:
    path: Path | None
    resolution: str | None = None
    cleanup_dir: Path | None = None
    candidate_names: tuple[str, ...] = ()


def _read_file_prefix(path: Path, size: int = 4096) -> bytes:
    try:
        if not path.is_file():
            return b""
        with path.open("rb") as handle:
            return handle.read(size)
    except OSError:
        return b""


def _looks_like_script(path: Path) -> bool:
    return _read_file_prefix(path, size=256).startswith(b"#!")


def _looks_like_binary(path: Path) -> bool:
    prefix = _read_file_prefix(path)
    if not prefix:
        return False
    if prefix.startswith(_BINARY_MAGIC_PREFIXES):
        return True
    return b"\x00" in prefix


def detect_target_kind(binary_path: Optional[Path]) -> str:
    if binary_path is None:
        return "none"
    suffix = binary_path.suffix.lower()
    if suffix in _BINARY_EXTENSIONS:
        return "binary"
    if suffix in _SCRIPT_EXTENSIONS:
        return "script"
    if suffix in _WEB_EXTENSIONS:
        return "web"
    if _looks_like_script(binary_path):
        return "script"
    if _looks_like_binary(binary_path):
        return "binary"
    return "file"


def _strip_archive_suffixes(path: Path) -> str:
    lowered_parts = [suffix.lower() for suffix in path.suffixes]
    for suffix_sequence in _ARCHIVE_SUFFIX_SEQUENCES:
        if tuple(lowered_parts[-len(suffix_sequence):]) == suffix_sequence:
            name = path.name
            suffix_text = "".join(path.suffixes[-len(suffix_sequence):])
            return name[: -len(suffix_text)] if suffix_text else name
    return path.stem


def _is_supported_archive(path: Path) -> bool:
    lowered_parts = [suffix.lower() for suffix in path.suffixes]
    return any(
        tuple(lowered_parts[-len(suffix_sequence):]) == suffix_sequence
        for suffix_sequence in _ARCHIVE_SUFFIX_SEQUENCES
    )


def _extract_archive(path: Path, destination: Path) -> bool:
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(destination)
            return True
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                archive.extractall(destination)
            return True
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False
    return False


def _has_native_binary_header(candidate: Path) -> bool:
    return _read_file_prefix(candidate).startswith(_BINARY_MAGIC_PREFIXES)


def _is_library_candidate(candidate: Path) -> bool:
    suffixes = {suffix.lower() for suffix in candidate.suffixes}
    return bool(suffixes & _LIBRARY_EXTENSIONS)


def _classify_native_binary_candidate(candidate: Path) -> str:
    if _is_library_candidate(candidate):
        return "native_library" if _has_native_binary_header(candidate) else "data_blob"
    if _has_native_binary_header(candidate):
        suffix = candidate.suffix.lower()
        if os.access(candidate, os.X_OK) or not suffix or suffix in _EXECUTABLE_HINT_EXTENSIONS:
            return "native_executable"
        return "native_binary"
    return "data_blob"


def _ghidra_candidate_score(candidate: Path, *, source_hint: str) -> tuple[int, int, int, int]:
    stem = candidate.stem.lower()
    name = candidate.name.lower()
    hint = source_hint.lower()
    exact_name = int(name == hint or stem == hint)
    prefix_match = int(name.startswith(hint) or stem.startswith(hint))
    candidate_kind = _classify_native_binary_candidate(candidate)
    try:
        size = int(candidate.stat().st_size)
    except OSError:
        size = 0
    return (
        exact_name,
        prefix_match,
        _NATIVE_BINARY_KIND_PRIORITY[candidate_kind],
        size,
    )


def _select_ghidra_binary_candidate(candidates: list[Path], *, source_hint: str) -> ResolvedGhidraTarget:
    if not candidates:
        return ResolvedGhidraTarget(path=None, resolution=None)
    ranked = sorted(
        (
            (_ghidra_candidate_score(candidate, source_hint=source_hint), candidate)
            for candidate in candidates
        ),
        key=lambda item: (item[0], str(item[1])),
        reverse=True,
    )
    best_score, best_candidate = ranked[0]
    if len(ranked) > 1 and ranked[1][0] == best_score:
        tied_candidates = tuple(sorted(candidate.name for score, candidate in ranked if score == best_score))
        return ResolvedGhidraTarget(path=None, resolution="ambiguous_skip", candidate_names=tied_candidates)
    return ResolvedGhidraTarget(
        path=best_candidate,
        resolution="auto_resolved",
        candidate_names=(best_candidate.name,),
    )


def resolve_native_binary_target(binary_path: Optional[Path]) -> ResolvedGhidraTarget:
    if binary_path is None or not binary_path.exists():
        return ResolvedGhidraTarget(path=None, resolution=None)

    if detect_target_kind(binary_path) == "binary":
        return ResolvedGhidraTarget(path=binary_path, resolution="direct")

    search_root: Path | None = None
    cleanup_dir: Path | None = None
    if binary_path.is_file() and _is_supported_archive(binary_path):
        cleanup_dir = Path(tempfile.mkdtemp(prefix="local-llm-bench-ghidra-target-"))
        if not _extract_archive(binary_path, cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)
            return ResolvedGhidraTarget(path=None, resolution=None)
        search_root = cleanup_dir
    elif binary_path.is_dir():
        search_root = binary_path
    else:
        return ResolvedGhidraTarget(path=None, resolution=None)

    source_hint = _strip_archive_suffixes(binary_path) if binary_path.is_file() else binary_path.name
    candidates = [
        candidate
        for candidate in search_root.rglob("*")
        if candidate.is_file() and detect_target_kind(candidate) == "binary"
    ]
    resolved = _select_ghidra_binary_candidate(candidates, source_hint=source_hint)
    if cleanup_dir is not None:
        return ResolvedGhidraTarget(
            path=resolved.path,
            resolution=resolved.resolution,
            cleanup_dir=cleanup_dir,
            candidate_names=resolved.candidate_names,
        )
    return resolved


def build_shared_system_prompt() -> str:
    return _SHARED_SYSTEM_PROMPT


def build_task_prompt(question_prompt: str, binary_path: Optional[Path]) -> str:
    if binary_path is None:
        return question_prompt
    return f"{question_prompt}\n\nTarget file: {binary_path}"
