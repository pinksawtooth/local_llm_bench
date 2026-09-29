"""Inspect's native compressed logs, shared by host evaluations and workers.

Inspect writes and flushes the archive itself. Do not gzip it, rewrite it after
each sample, or discard the journal: those would impair viewing and recovery.
"""
from __future__ import annotations

import os
from pathlib import Path
import zipfile

LOG_FORMAT = "eval"


def storage_metadata(path: Path) -> dict:
    """Read ZIP directory sizes without inflating transcripts or attachments.

    uncompressed_bytes is the sum of ZIP member sizes, not the size of a JSON
    export (which can duplicate references resolved by Inspect's log API).
    """
    with path.open("rb") as stream:
        file_bytes = os.fstat(stream.fileno()).st_size
        try:
            with zipfile.ZipFile(stream) as archive:
                entries = archive.infolist()
        except zipfile.BadZipFile as exc:
            raise ValueError("Invalid Inspect archive") from exc
    names = [entry.filename for entry in entries]
    if ("header.json" not in names or len(set(names)) != len(names)
            or any(entry.flag_bits & 1 for entry in entries)):
        raise ValueError("Invalid Inspect archive structure")
    methods = {0: "stored", 8: "deflate", 20: "zstd", 93: "zstd"}
    return {"format": LOG_FORMAT, "container": "zip",
            "compression": sorted({methods.get(entry.compress_type, f"zip-{entry.compress_type}")
                                   for entry in entries if not entry.is_dir()}),
            "file_bytes": file_bytes,
            "uncompressed_bytes": sum(entry.file_size for entry in entries)}


def completed_log_info(log, directory: Path) -> dict:
    """Record the actual storage of this attempt's newly closed native log.

    A storage inspection failure is reported separately from the model result;
    it must not trigger a repeat of an already completed inference. Never scan
    or rewrite older attempts, and keep the original file on every error path.
    """
    info = {"status": log.status, "log_path": log.location, "log_format": LOG_FORMAT}
    try:
        path = Path(log.location).resolve()
        if path.suffix != ".eval" or not path.is_relative_to(directory.resolve()):
            raise ValueError("Inspect log is outside this attempt directory")
        storage = storage_metadata(path)
        # The pinned Inspect version writes native ZIP members with Zstandard.
        # Report a drift instead of silently labelling another format as zstd.
        if storage["compression"] != ["zstd"]:
            raise ValueError("Inspect log does not use the pinned native compression")
        path.chmod(0o600)
        info["log_storage"] = storage
    except (OSError, ValueError, TypeError, zipfile.BadZipFile):
        # Do not expose exception text from filesystem paths or private logs.
        info["log_storage"] = {"status": "unavailable",
                               "error": "Inspect log compression or file access could not be confirmed"}
    return info
