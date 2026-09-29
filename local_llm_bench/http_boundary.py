"""Adapted from RevBench: explicit DS4 transport and bounded HTTP metadata."""

from __future__ import annotations

import json
import ipaddress
import math
import re
import urllib.request
from typing import Any, Iterable, Mapping, Optional, Tuple
from urllib.parse import unquote, urlsplit, urlunsplit


MAX_HTTP_REQUEST_BYTES = 16 * 1024 * 1024
MAX_HTTP_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_HTTP_ERROR_BYTES = 64 * 1024
MAX_HTTP_HEADER_BYTES = 64 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_ITEMS = 100_000
MAX_JSON_STRING_CHARS = 16 * 1024 * 1024
MAX_TIMEOUT_SEC = 86_400.0
MAX_BEARER_TOKEN_CHARS = 16 * 1024
MAX_HTTP_URL_CHARS = 8192

_DECIMAL_RE = re.compile(r"0|[1-9][0-9]*\Z")
_INVALID_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_DNS_LABEL_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


class HTTPBoundaryError(ValueError):
    """An HTTP or JSON message crossed a bounded, unambiguous API boundary."""


class HTTPMessageTooLarge(HTTPBoundaryError):
    """An HTTP or JSON message exceeded its documented in-memory limit."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        # API redirects are unexpected and can forward bearer credentials to a
        # different origin.  Returning None makes urllib surface the 3xx as an
        # HTTPError without issuing the redirected request.
        return None


def validate_timeout(timeout_sec: float) -> float:
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
        raise HTTPBoundaryError("HTTP timeout must be a finite number")
    value = float(timeout_sec)
    if not math.isfinite(value) or value <= 0.0 or value > MAX_TIMEOUT_SEC:
        raise HTTPBoundaryError(
            f"HTTP timeout must be greater than 0 and at most {MAX_TIMEOUT_SEC:g} seconds"
        )
    return value


def validate_bearer_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    if not isinstance(token, str):
        raise HTTPBoundaryError("API key must be text")
    value = token.strip()
    if not value:
        return None
    if len(value) > MAX_BEARER_TOKEN_CHARS or any(
        ord(char) < 0x21 or ord(char) > 0x7E for char in value
    ):
        raise HTTPBoundaryError("API key is not safe for an HTTP header")
    return value


def _reject_ambiguous_path(path: str) -> None:
    def unsafe(value: str) -> bool:
        return any(
            ord(char) <= 0x20
            or ord(char) == 0x7F
            or ord(char) > 0x7E
            or char == "\\"
            for char in value
        )

    if unsafe(path):
        raise HTTPBoundaryError("HTTP URL path contains an unsafe character")
    decoded = path
    for _ in range(3):
        if _INVALID_PERCENT_RE.search(decoded):
            raise HTTPBoundaryError("HTTP URL path contains invalid percent-encoding")
        try:
            next_decoded = unquote(decoded, errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPBoundaryError("HTTP URL path contains invalid percent-encoding") from exc
        if unsafe(next_decoded):
            raise HTTPBoundaryError("HTTP URL path contains an unsafe encoded character")
        if next_decoded.startswith("//"):
            raise HTTPBoundaryError("HTTP URL path must not be authority-like")
        if any(segment in {".", ".."} for segment in next_decoded.split("/")):
            raise HTTPBoundaryError("HTTP URL path must not contain dot segments")
        if next_decoded == decoded:
            break
        decoded = next_decoded
    if "%" in decoded:
        raise HTTPBoundaryError("nested percent-encoding in HTTP URL path is not allowed")


def _reject_ambiguous_query(query: str) -> None:
    def unsafe(value: str) -> bool:
        return any(
            ord(char) <= 0x20
            or ord(char) == 0x7F
            or ord(char) > 0x7E
            or char == "\\"
            for char in value
        )

    decoded = query
    for _ in range(3):
        if _INVALID_PERCENT_RE.search(decoded):
            raise HTTPBoundaryError("HTTP URL query contains invalid percent-encoding")
        if unsafe(decoded):
            raise HTTPBoundaryError("HTTP URL query contains an unsafe character")
        try:
            next_decoded = unquote(decoded, errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPBoundaryError("HTTP URL query contains invalid percent-encoding") from exc
        if unsafe(next_decoded):
            raise HTTPBoundaryError("HTTP URL query contains an unsafe encoded character")
        if next_decoded == decoded:
            break
        decoded = next_decoded
    if "%" in decoded:
        raise HTTPBoundaryError("nested percent-encoding in HTTP URL query is not allowed")


def _canonical_netloc(parsed: Any) -> str:
    netloc = parsed.netloc
    if (
        not netloc
        or any(ord(char) <= 0x20 or ord(char) == 0x7F or ord(char) > 0x7E for char in netloc)
        or "\\" in netloc
        or "%" in netloc
    ):
        raise HTTPBoundaryError("HTTP URL authority is invalid")
    if parsed.username is not None or parsed.password is not None or "@" in netloc:
        raise HTTPBoundaryError("HTTP base URL must not contain credentials")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise HTTPBoundaryError("HTTP URL authority is invalid") from exc
    if not hostname:
        raise HTTPBoundaryError("HTTP URL host is missing")

    bracketed = netloc.startswith("[")
    if bracketed:
        closing = netloc.find("]")
        if closing < 0 or netloc.find("[", 1) >= 0 or netloc.find("]", closing + 1) >= 0:
            raise HTTPBoundaryError("HTTP URL IPv6 authority is invalid")
        port_suffix = netloc[closing + 1 :]
        if port_suffix and not port_suffix.startswith(":"):
            raise HTTPBoundaryError("HTTP URL IPv6 authority is invalid")
        has_port = bool(port_suffix)
        port_text = port_suffix[1:] if port_suffix else ""
        if ":" not in hostname:
            raise HTTPBoundaryError("brackets are only valid for an IPv6 host")
    else:
        if netloc.count(":") > 1 or ":" in hostname:
            raise HTTPBoundaryError("IPv6 hosts must use brackets")
        has_port = ":" in netloc
        port_text = netloc.rsplit(":", 1)[1] if has_port else ""

    if has_port and not port_text:
        raise HTTPBoundaryError("HTTP URL port must not be empty")
    if port_text:
        if not _DECIMAL_RE.fullmatch(port_text):
            raise HTTPBoundaryError("HTTP URL port must be canonical decimal digits")
        if port is None or not (1 <= port <= 65535):
            raise HTTPBoundaryError("HTTP URL port is invalid")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        if bracketed:
            raise HTTPBoundaryError("HTTP URL IPv6 host is invalid")
        if len(hostname) > 253 or hostname.endswith("."):
            raise HTTPBoundaryError("HTTP URL DNS host is invalid")
        labels = hostname.split(".")
        if not labels or any(not _DNS_LABEL_RE.fullmatch(label) for label in labels):
            raise HTTPBoundaryError("HTTP URL DNS host is invalid")
        if all(char.isdigit() or char == "." for char in hostname):
            raise HTTPBoundaryError("ambiguous numeric HTTP host is not allowed")
        canonical_host = hostname.lower()
    else:
        if address.version == 6 and not bracketed:
            raise HTTPBoundaryError("IPv6 hosts must use brackets")
        if address.version == 4 and bracketed:
            raise HTTPBoundaryError("IPv4 hosts must not use brackets")
        if address.is_unspecified or address.is_multicast:
            raise HTTPBoundaryError("HTTP URL host must identify a unicast endpoint")
        canonical_host = address.compressed

    canonical = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    if port is not None:
        canonical = f"{canonical}:{port}"
    return canonical


def validate_http_base_url(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HTTPBoundaryError("HTTP base URL must be non-empty text without surrounding whitespace")
    if len(value) > MAX_HTTP_URL_CHARS:
        raise HTTPMessageTooLarge("HTTP base URL is too large")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise HTTPBoundaryError("HTTP base URL contains a control character")
    if "?" in value or "#" in value:
        raise HTTPBoundaryError("HTTP base URL must not contain a query or fragment")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise HTTPBoundaryError("HTTP base URL is invalid") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise HTTPBoundaryError("HTTP base URL must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise HTTPBoundaryError("HTTP base URL must not contain a query or fragment")
    netloc = _canonical_netloc(parsed)
    path = parsed.path or ""
    _reject_ambiguous_path(path)
    return urlunsplit((parsed.scheme.lower(), netloc, path.rstrip("/"), "", ""))


def validate_http_url(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HTTPBoundaryError("HTTP URL must be non-empty text without surrounding whitespace")
    if len(value) > MAX_HTTP_URL_CHARS:
        raise HTTPMessageTooLarge("HTTP URL is too large")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise HTTPBoundaryError("HTTP URL contains a control character")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise HTTPBoundaryError("HTTP URL is invalid") from exc
    if "#" in value:
        raise HTTPBoundaryError("HTTP URL must not contain a fragment")
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    validated_base = validate_http_base_url(base)
    validated_parts = urlsplit(validated_base)
    if "?" in value:
        if not parsed.query:
            raise HTTPBoundaryError("HTTP URL query must not be empty")
        _reject_ambiguous_query(parsed.query)
    return urlunsplit(
        (
            validated_parts.scheme,
            validated_parts.netloc,
            parsed.path,
            parsed.query,
            "",
        )
    )


def join_http_url(base_url: str, request_target: str) -> str:
    base = validate_http_base_url(base_url)
    if (
        not isinstance(request_target, str)
        or not request_target.startswith("/")
        or request_target.startswith("//")
    ):
        raise HTTPBoundaryError("HTTP request target must be origin-form")
    if len(request_target) > MAX_HTTP_URL_CHARS:
        raise HTTPMessageTooLarge("HTTP request target is too large")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in request_target):
        raise HTTPBoundaryError("HTTP request target contains a control character")
    if "#" in request_target or request_target.endswith("?"):
        raise HTTPBoundaryError("HTTP request target has an invalid query or fragment")
    try:
        parsed_base = urlsplit(base)
        parsed_target = urlsplit(request_target)
    except ValueError as exc:
        raise HTTPBoundaryError("HTTP request target is invalid") from exc
    if parsed_target.scheme or parsed_target.netloc or parsed_target.fragment:
        raise HTTPBoundaryError("HTTP request target must be origin-form without a fragment")
    target_path = parsed_target.path or "/"
    _reject_ambiguous_path(target_path)
    if parsed_target.query:
        _reject_ambiguous_query(parsed_target.query)
    base_path = parsed_base.path.rstrip("/")
    if base_path and (target_path == base_path or target_path.startswith(base_path + "/")):
        path = target_path
    else:
        path = f"{base_path}{target_path}" if base_path else target_path
    return urlunsplit(
        (parsed_base.scheme, parsed_base.netloc, path, parsed_target.query, "")
    )


def _reject_json_constant(value: str) -> None:
    raise HTTPBoundaryError(f"non-finite JSON number is not allowed: {value}")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise HTTPBoundaryError("non-finite JSON number is not allowed")
    return parsed


def _strict_json_object(pairs: Iterable[Tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HTTPBoundaryError("duplicate JSON object key is not allowed")
        result[key] = value
    return result


def validate_json_value(
    value: Any,
    *,
    max_depth: int = MAX_JSON_DEPTH,
    max_items: int = MAX_JSON_ITEMS,
    max_string_chars: int = MAX_JSON_STRING_CHARS,
) -> None:
    if max_depth < 1 or max_items < 1 or max_string_chars < 1:
        raise ValueError("JSON limits must be positive")
    stack: list[tuple[Any, int]] = [(value, 1)]
    seen_containers: set[int] = set()
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > max_items:
            raise HTTPMessageTooLarge("JSON value contains too many items")
        if depth > max_depth:
            raise HTTPMessageTooLarge("JSON value is nested too deeply")
        if item is None or isinstance(item, bool) or isinstance(item, int):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise HTTPBoundaryError("non-finite JSON number is not allowed")
            continue
        if isinstance(item, str):
            if len(item) > max_string_chars:
                raise HTTPMessageTooLarge("JSON string is too large")
            continue
        if isinstance(item, (list, dict)):
            marker = id(item)
            if marker in seen_containers:
                raise HTTPBoundaryError("recursive or shared JSON containers are not allowed")
            seen_containers.add(marker)
            if isinstance(item, list):
                stack.extend((child, depth + 1) for child in reversed(item))
            else:
                for key, child in reversed(list(item.items())):
                    if not isinstance(key, str):
                        raise HTTPBoundaryError("JSON object keys must be text")
                    if len(key) > max_string_chars:
                        raise HTTPMessageTooLarge("JSON object key is too large")
                    stack.append((child, depth + 1))
            continue
        raise HTTPBoundaryError(f"value of type {type(item).__name__} is not valid JSON")


def strict_json_loads(
    raw: bytes | str,
    *,
    max_bytes: int = MAX_HTTP_RESPONSE_BYTES,
    max_depth: int = MAX_JSON_DEPTH,
    max_items: int = MAX_JSON_ITEMS,
) -> Any:
    if isinstance(raw, bytes):
        if len(raw) > max_bytes:
            raise HTTPMessageTooLarge("JSON document is too large")
        try:
            text = raw.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise HTTPBoundaryError("JSON document is not valid UTF-8") from exc
    elif isinstance(raw, str):
        if len(raw.encode("utf-8")) > max_bytes:
            raise HTTPMessageTooLarge("JSON document is too large")
        text = raw
    else:
        raise HTTPBoundaryError("JSON document must be bytes or text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
        )
    except HTTPBoundaryError:
        raise
    except (RecursionError, UnicodeError, ValueError, TypeError) as exc:
        raise HTTPBoundaryError("JSON document is invalid") from exc
    validate_json_value(value, max_depth=max_depth, max_items=max_items)
    return value


def json_dumps_bounded(
    value: Any,
    *,
    max_bytes: int,
    max_depth: int = MAX_JSON_DEPTH,
    max_items: int = MAX_JSON_ITEMS,
) -> bytes:
    validate_json_value(value, max_depth=max_depth, max_items=max_items)
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise HTTPBoundaryError("JSON value cannot be serialized") from exc
    if len(raw) > max_bytes:
        raise HTTPMessageTooLarge("serialized JSON document is too large")
    return raw


def _header_values(headers: Any, name: str) -> list[str]:
    getter = getattr(headers, "get_all", None)
    if callable(getter):
        values = getter(name, [])
        return [str(value) for value in (values or [])]
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                if isinstance(value, (list, tuple)):
                    return [str(item) for item in value]
                return [str(value)]
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        return [] if value is None else [str(value)]
    return []


def _iter_header_items(headers: Any) -> Iterable[tuple[str, str]]:
    raw_items = getattr(headers, "raw_items", None)
    if callable(raw_items):
        return [(str(key), str(value)) for key, value in raw_items()]
    items = getattr(headers, "items", None)
    if callable(items):
        return [(str(key), str(value)) for key, value in items()]
    return []


def validate_http_headers(headers: Any, *, max_bytes: int = MAX_HTTP_HEADER_BYTES) -> None:
    total = 0
    for key, value in _iter_header_items(headers):
        total += len(key.encode("utf-8", "replace")) + len(value.encode("utf-8", "replace")) + 4
        if total > max_bytes:
            raise HTTPMessageTooLarge("HTTP headers are too large")
        if not key or any(ord(char) < 0x21 or ord(char) > 0x7E or char == ":" for char in key):
            raise HTTPBoundaryError("HTTP header name is invalid")
        if any((ord(char) < 0x20 and char != "\t") or ord(char) == 0x7F for char in value):
            raise HTTPBoundaryError("HTTP header value is invalid")


def parse_content_length(headers: Any) -> Optional[int]:
    values = _header_values(headers, "Content-Length")
    if not values:
        return None
    if len(values) != 1:
        raise HTTPBoundaryError("duplicate Content-Length is not allowed")
    raw = values[0]
    if not _DECIMAL_RE.fullmatch(raw):
        raise HTTPBoundaryError("Content-Length must be canonical decimal digits")
    if len(raw) > 20:
        raise HTTPMessageTooLarge("Content-Length is too large")
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPBoundaryError("Content-Length is invalid") from exc


def parse_transfer_encoding(headers: Any) -> Optional[str]:
    values = _header_values(headers, "Transfer-Encoding")
    if not values:
        return None
    if len(values) != 1:
        raise HTTPBoundaryError("duplicate Transfer-Encoding is not allowed")
    tokens = [token.strip().lower() for token in values[0].split(",")]
    if tokens != ["chunked"]:
        raise HTTPBoundaryError("unsupported Transfer-Encoding")
    return "chunked"


def validate_json_content_type(headers: Any) -> None:
    values = _header_values(headers, "Content-Type")
    if len(values) != 1:
        raise HTTPBoundaryError("JSON HTTP message must have exactly one Content-Type")
    media_type = values[0].split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not (
        media_type.startswith("application/") and media_type.endswith("+json")
    ):
        raise HTTPBoundaryError("HTTP message Content-Type is not JSON")


def read_bounded_response(response: Any, *, max_bytes: int = MAX_HTTP_RESPONSE_BYTES) -> bytes:
    headers = getattr(response, "headers", {})
    validate_http_headers(headers)
    content_length = parse_content_length(headers)
    transfer_encoding = parse_transfer_encoding(headers)
    if content_length is not None and transfer_encoding is not None:
        raise HTTPBoundaryError("response must not contain both Content-Length and Transfer-Encoding")
    if content_length is not None and content_length > max_bytes:
        raise HTTPMessageTooLarge("HTTP response body is too large")
    raw = response.read(max_bytes + 1)
    if not isinstance(raw, bytes):
        raise HTTPBoundaryError("HTTP response body is not bytes")
    if len(raw) > max_bytes:
        raise HTTPMessageTooLarge("HTTP response body is too large")
    if content_length is not None and len(raw) != content_length:
        raise HTTPBoundaryError("HTTP response body length does not match Content-Length")
    return raw


def open_url_no_redirect(request: urllib.request.Request, *, timeout_sec: float):
    timeout = validate_timeout(timeout_sec)
    # Ignore ambient HTTP(S)_PROXY variables for explicitly configured local/LAN
    # model endpoints, and never follow redirects with authorization headers.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)
