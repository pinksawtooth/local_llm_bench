from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import (
    DEFAULT_UNSLOTH_STUDIO_API_BASE,
    AuthSettings,
    UNSLOTH_STUDIO_BEARER_TOKEN_ENV,
    UNSLOTH_STUDIO_PASSWORD_ENV,
    UNSLOTH_STUDIO_USERNAME_ENV,
)

UNSLOTH_STUDIO_REFRESH_TOKEN_ENV = "LOCAL_LLM_BENCH_UNSLOTH_STUDIO_REFRESH_TOKEN"


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def load_unsloth_auth_from_env() -> AuthSettings:
    return AuthSettings(
        bearer_token=_normalize_text(os.getenv(UNSLOTH_STUDIO_BEARER_TOKEN_ENV)) or None,
        username=_normalize_text(os.getenv(UNSLOTH_STUDIO_USERNAME_ENV)) or None,
        password=_normalize_text(os.getenv(UNSLOTH_STUDIO_PASSWORD_ENV)) or None,
    )


class UnslothStudioAuthSession:
    def __init__(
        self,
        auth: AuthSettings,
        *,
        openai_api_base: str = DEFAULT_UNSLOTH_STUDIO_API_BASE,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._auth = auth
        self._openai_api_base = str(openai_api_base or DEFAULT_UNSLOTH_STUDIO_API_BASE).rstrip("/")
        self._control_base_url = self._derive_control_base_url(self._openai_api_base)
        self._urlopen = urlopen
        self._access_token = _normalize_text(auth.bearer_token) or None
        self._refresh_token = _normalize_text(os.getenv(UNSLOTH_STUDIO_REFRESH_TOKEN_ENV)) or None

    @property
    def openai_api_base(self) -> str:
        return self._openai_api_base

    @property
    def control_base_url(self) -> str:
        return self._control_base_url

    @staticmethod
    def _derive_control_base_url(openai_api_base: str) -> str:
        trimmed = str(openai_api_base or "").rstrip("/")
        if trimmed.endswith("/v1"):
            trimmed = trimmed[:-3]
        return trimmed.rstrip("/")

    def export_environment(self) -> dict[str, str]:
        exported: dict[str, str] = {}
        bearer_token = self._access_token or _normalize_text(self._auth.bearer_token)
        if bearer_token:
            exported[UNSLOTH_STUDIO_BEARER_TOKEN_ENV] = bearer_token
        if self._refresh_token:
            exported[UNSLOTH_STUDIO_REFRESH_TOKEN_ENV] = self._refresh_token
        if not bearer_token:
            username = _normalize_text(self._auth.username)
            password = _normalize_text(self._auth.password)
            if username and password:
                exported[UNSLOTH_STUDIO_USERNAME_ENV] = username
                exported[UNSLOTH_STUDIO_PASSWORD_ENV] = password
        return exported

    def _json_request(
        self,
        *,
        method: str,
        path: str,
        payload: Any = None,
        timeout_sec: float,
        authorized: bool,
    ) -> Any:
        url = f"{self._control_base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        opener = self.urlopen if authorized else self._urlopen
        try:
            with opener(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            message = body[:240] if body else str(exc)
            raise RuntimeError(f"HTTPError {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"URLError: {exc}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSONの解析に失敗しました: {raw[:200]}") from exc

    def _store_tokens(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Unsloth Studio の認証レスポンスが不正です。")
        access_token = _normalize_text(payload.get("access_token"))
        refresh_token = _normalize_text(payload.get("refresh_token"))
        if not access_token:
            raise RuntimeError("Unsloth Studio の認証レスポンスに access_token がありません。")
        if payload.get("must_change_password") is True:
            raise RuntimeError("Unsloth Studio でパスワード変更が必要です。UI で更新してから再実行してください。")
        self._access_token = access_token
        self._refresh_token = refresh_token or self._refresh_token
        return access_token

    def _login(self, *, timeout_sec: float) -> str:
        username = _normalize_text(self._auth.username)
        password = _normalize_text(self._auth.password)
        if not username or not password:
            raise RuntimeError(
                "Unsloth Studio の認証情報がありません。auth.bearer_token か auth.username/auth.password を設定してください。"
            )
        payload = self._json_request(
            method="POST",
            path="/api/auth/login",
            payload={"username": username, "password": password},
            timeout_sec=timeout_sec,
            authorized=False,
        )
        return self._store_tokens(payload)

    def _refresh(self, *, timeout_sec: float) -> bool:
        if self._refresh_token:
            try:
                payload = self._json_request(
                    method="POST",
                    path="/api/auth/refresh",
                    payload={"refresh_token": self._refresh_token},
                    timeout_sec=timeout_sec,
                    authorized=False,
                )
            except RuntimeError:
                payload = None
            else:
                self._store_tokens(payload)
                return True

        username = _normalize_text(self._auth.username)
        password = _normalize_text(self._auth.password)
        if username and password:
            self._login(timeout_sec=timeout_sec)
            return True
        return False

    def _ensure_access_token(self, *, timeout_sec: float) -> str:
        if self._access_token:
            return self._access_token
        return self._login(timeout_sec=timeout_sec)

    @staticmethod
    def _clone_request_with_headers(
        request: urllib.request.Request,
        extra_headers: dict[str, str],
    ) -> urllib.request.Request:
        headers = {str(key): str(value) for key, value in request.header_items()}
        headers.update(extra_headers)
        return urllib.request.Request(
            request.full_url,
            data=request.data,
            headers=headers,
            method=request.get_method(),
        )

    def urlopen(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        timeout_sec = float(timeout if timeout is not None else 60.0)
        access_token = self._ensure_access_token(timeout_sec=timeout_sec)
        prepared = self._clone_request_with_headers(
            request,
            {"Authorization": f"Bearer {access_token}"},
        )
        try:
            return self._urlopen(prepared, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
            if not self._refresh(timeout_sec=timeout_sec):
                raise
        retried = self._clone_request_with_headers(
            request,
            {"Authorization": f"Bearer {self._access_token or ''}"},
        )
        return self._urlopen(retried, timeout=timeout)

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Any = None,
        timeout_sec: float = 15.0,
    ) -> Any:
        return self._json_request(
            method=method,
            path=path,
            payload=payload,
            timeout_sec=timeout_sec,
            authorized=True,
        )
