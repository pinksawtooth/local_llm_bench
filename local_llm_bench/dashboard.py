"""One local, read-only UI for benchmark comparisons and native Inspect logs."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import threading
from urllib.parse import quote, unquote, urlsplit

from .inspect_harness import require_inspect
from .inspect_log import storage_metadata
from .report import render_report_html


class LogAccessPolicy:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, value: str) -> Path | None:
        try:
            uri = urlsplit(value)
            if uri.scheme not in ("", "file") or uri.netloc not in ("", "localhost"):
                return None
            path = Path(unquote(uri.path) if uri.scheme else value)
            if not path.is_absolute():
                return None
            path = path.resolve()
            return path if path.is_relative_to(self.root) else None
        except (ValueError, OSError, RuntimeError):
            return None

    async def can_read(self, request, file: str) -> bool:
        path = self.resolve(file)
        return path is not None and path.suffix == ".eval"

    async def can_list(self, request, dir: str) -> bool:
        return self.resolve(dir) is not None

    async def can_delete(self, request, file: str) -> bool:
        return False

    async def can_write(self, request, file: str) -> bool:
        return False


class ReadOnlyHeaders:
    """Allow the native viewer inside this dashboard, keeping other framing off."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from starlette.datastructures import MutableHeaders
        from starlette.responses import PlainTextResponse

        if scope["method"] not in ("GET", "HEAD"):
            return await PlainTextResponse("Read-only dashboard", status_code=405)(scope, receive, send)

        async def with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                embedded = scope["path"].startswith("/inspect/")
                headers["Content-Security-Policy"] = "frame-ancestors 'self'" if embedded else "frame-ancestors 'none'"
                headers["X-Frame-Options"] = "SAMEORIGIN" if embedded else "DENY"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, with_headers)


class InspectCatalogue:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.access = LogAccessPolicy(root)
        self.cache: dict[Path, tuple[tuple[int, int], dict]] = {}
        self.lock = threading.Lock()

    def header(self, path: Path) -> dict:
        from inspect_ai.log import read_eval_log

        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        try:
            log = read_eval_log(str(path), header_only=True)
            metadata = log.eval.metadata or {}
            result = {"status": log.status, "model": log.eval.model,
                      "provider": metadata.get("provider", ""),
                      "question_id": (log.eval.dataset.sample_ids or [""])[0],
                      "sample_ids": log.eval.dataset.sample_ids or [],
                      "phase": metadata.get("phase"), "iteration": metadata.get("iteration")}
        except Exception:
            # A live .eval archive may not have its header yet. Retry next refresh.
            return {"status": "unavailable", "model": "", "provider": ""}
        try:
            result["log_storage"] = storage_metadata(path)
        except (OSError, ValueError):
            # Inspect can still read the journal of an unfinished archive.
            # Storage statistics must not hide a readable evaluation header.
            result["log_storage"] = {"status": "unavailable"}
        self.cache[path] = (signature, result)
        return result

    def list(self, run_id: str = "", phase: str = "", iteration: int | None = None,
             question_id: str = "") -> list[dict]:
        with self.lock:
            return self._list(run_id, phase, iteration, question_id)

    def _list(self, run_id: str, phase: str, iteration: int | None, question_id: str) -> list[dict]:
        paths = [path for path in self.root.rglob("*.eval")
                 if self.access.resolve(str(path)) is not None and path.is_file()]
        present = set(paths)
        self.cache = {path: value for path, value in self.cache.items() if path in present}
        entries = []
        parents = {}
        for path in paths:
            relative = path.relative_to(self.root)
            if run_id and relative.parts[0] != run_id:
                continue
            unit = path.parent.parent if path.parent.name == "worker" else path.parent
            try:
                header = self.header(path)
                modified = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue  # A log can disappear between listing and reading.
            if question_id:
                key = hashlib.sha256(f"{phase}:{iteration}:{question_id}".encode()).hexdigest()[:20]
                if unit.parent.name != key and question_id not in header.get("sample_ids", []):
                    continue
            role = "agent" if path.parent.name == "worker" else "evaluation"
            if role == "evaluation":
                parents[unit] = header
            entries.append({**header, "path": str(path), "relative_path": relative.as_posix(),
                            "run_id": relative.parts[0], "role": role, "attempt_dir": str(unit),
                            "modified": modified,
                            "url": "/inspect/?inspect_server=true&log_file=" + quote(str(path), safe="")})
        # Worker metadata has no host run/phase. Its sibling host evaluation
        # supplies that context without mixing attempts or adding token counts.
        for entry in entries:
            parent = parents.get(Path(entry["attempt_dir"]), {})
            for key in ("phase", "iteration", "provider"):
                if entry.get(key) in (None, ""):
                    entry[key] = parent.get(key)
        entries = [entry for entry in entries
                   if (not phase or entry.get("phase") == phase)
                   and (iteration is None or entry.get("iteration") == iteration)]
        return sorted(entries, key=lambda item: (item["modified"], item["relative_path"]), reverse=True)


def create_app(runs_dir: Path | None = None, *, host: str = "127.0.0.1", port: int = 8080):
    require_inspect()
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
    from starlette.staticfiles import StaticFiles
    from inspect_ai._util.asyncfiles import AsyncFilesystem
    from inspect_ai._view import fastapi_server
    from inspect_ai._view.network import (BrowserOriginMiddleware, HostValidationMiddleware,
                                         resolve_viewer_network_policy)

    runs = (runs_dir or Path(__file__).resolve().parents[1] / "runs").resolve()
    logs = runs / "logs"
    policy = resolve_viewer_network_policy(bind_host=host, port=port)
    logs.mkdir(parents=True, exist_ok=True)
    access = LogAccessPolicy(logs)
    catalogue = InspectCatalogue(logs)
    # Use the assets shipped in the pinned wheel; never clone/build a viewer.
    dist = Path(fastapi_server.__file__).parent / "dist"
    if not (dist / "index.html").is_file():
        raise RuntimeError("Inspect View assets are missing. Reinstall requirements.lock in the dedicated venv.")
    fs = AsyncFilesystem()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await fs.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/docs/", response_class=HTMLResponse)
    def dashboard():
        return render_report_html("/runs/history.json", inspect_api_url="/dashboard-api/inspect-logs")

    @app.get("/docs")
    def docs_redirect():
        return RedirectResponse("/docs/")

    @app.get("/runs/history.json")
    def history():
        try:
            data = json.loads((runs / "history.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = []
        except (ValueError, OSError) as exc:
            raise HTTPException(503, "history.json could not be read; retry after the next save") from exc
        return JSONResponse(data)

    @app.get("/runs/logs/{file:path}")
    def raw_log(file: str):
        path = access.resolve(str(logs / file))
        if path is None or path.suffix not in (".json", ".txt", ".log") or not path.is_file():
            raise HTTPException(404, "Log not found")
        return FileResponse(path, media_type="application/json" if path.suffix == ".json" else "text/plain")

    @app.get("/dashboard-api/inspect-logs")
    def inspect_logs(run_id: str = "", phase: str = "", iteration: int | None = None, question_id: str = ""):
        return {"logs": catalogue.list(run_id, phase, iteration, question_id),
                "viewer_url": "/inspect/?inspect_server=true", "read_only": True}

    @app.get("/api/log-download/{file:path}")
    def download(file: str):
        path = access.resolve(file)
        if path is None or path.suffix != ".eval":
            raise HTTPException(403, "Log access denied")
        if not path.is_file():
            raise HTTPException(404, "Log not found")
        # FileResponse emits RFC 5987 filenames; Inspect 0.3.263's download
        # response puts raw Unicode into a Latin-1 Content-Disposition header.
        return FileResponse(path, media_type="application/octet-stream", filename=path.name)

    api = fastapi_server.view_server_app(default_dir=str(logs), recursive=True, access_policy=access)
    app.mount("/api", fastapi_server.AsyncFilesystemMiddleware(api, fs))
    app.mount("/inspect", StaticFiles(directory=str(dist), html=True), name="inspect")
    return HostValidationMiddleware(BrowserOriginMiddleware(ReadOnlyHeaders(app), policy), policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark dashboard + Inspect View (read-only)")
    parser.add_argument("--runs-dir", type=Path, help="Directory containing history.json and logs/")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    app = create_app(args.runs_dir, host=args.host, port=args.port)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=15)


if __name__ == "__main__":
    main()
