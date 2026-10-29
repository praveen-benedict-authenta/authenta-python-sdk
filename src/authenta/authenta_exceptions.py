from typing import Literal, Optional, Dict, Any, Mapping, Union

AuthentaErrorCode = Union[
    Literal[
        # Error names returned in the API's `name` field
        "UNAUTHENTICATED", "UNAUTHORIZED", "TOKEN_EXPIRED", "INVALID_API_KEY",
        "FORBIDDEN", "ACCOUNT_SUSPENDED",
        "RATE_LIMITED", "INSUFFICIENT_BALANCE", "PAYG_LIMIT_REACHED",
        "VALIDATION_ERROR", "FILE_TOO_LARGE", "TASK_TYPE_NOT_FOUND",
        "JOB_NOT_FOUND", "UPLOAD_NOT_FOUND", "UPLOAD_ALREADY_FINALIZED",
        "CANCEL_NOT_ALLOWED", "JOB_DELETE_NOT_ALLOWED", "INTERNAL_ERROR",
        # Legacy codes, kept for backwards compatibility
        "IAM001", "IAM002", "AA001", "U007",
        # SDK-side fallbacks
        "bad_request", "not_found", "timeout",
        "payload_too_large", "server_error", "unknown",
    ],
    str,
]


class AuthentaError(Exception):
    """
    Base exception for all Authenta SDK errors.

    Attributes:
        message: Human-readable description.
        code: The API's machine-readable error name (e.g. ``INVALID_API_KEY``).
        status_code: HTTP status, when the error came from a response.
        details: The full error body, including ``rid`` and, for
                 ``VALIDATION_ERROR``, ``fields``.
        request_id: The API's request ID (``rid``) — include it when
                    contacting support.
        retry_after: Seconds to wait before retrying, from ``Retry-After``.
    """

    def __init__(
        self,
        message: str,
        code: AuthentaErrorCode = "unknown",
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.request_id: Optional[str] = self.details.get("rid")
        self.retry_after = retry_after

        # String repr: "[INVALID_API_KEY] API key authentication failed (HTTP 401)"
        text = f"[{code}] {message}"
        if status_code:
            text += f" (HTTP {status_code})"
        super().__init__(text)


class AuthenticationError(AuthentaError):
    """Raised for a missing, invalid, revoked or expired API key (HTTP 401)."""

    def __init__(self, message: str, code: AuthentaErrorCode = "INVALID_API_KEY", **kwargs):
        super().__init__(message, code, **kwargs)


class AuthorizationError(AuthentaError):
    """Raised when the API key lacks the permission an endpoint needs (HTTP 403)."""

    def __init__(self, message: str, code: AuthentaErrorCode = "FORBIDDEN", **kwargs):
        super().__init__(message, code, **kwargs)


class QuotaExceededError(AuthentaError):
    """Raised when rate limits are hit (HTTP 429). See ``retry_after``."""

    def __init__(self, message: str, code: AuthentaErrorCode = "RATE_LIMITED", **kwargs):
        super().__init__(message, code, **kwargs)


class InsufficientCreditsError(AuthentaError):
    """Raised when credits run out or the pay-as-you-go limit is reached (HTTP 402)."""

    def __init__(self, message: str, code: AuthentaErrorCode = "INSUFFICIENT_BALANCE", **kwargs):
        super().__init__(message, code, **kwargs)


class ValidationError(AuthentaError):
    """Raised for client-side issues (Default: bad_request)."""

    def __init__(self, message: str, code: AuthentaErrorCode = "bad_request", **kwargs):
        super().__init__(message, code, **kwargs)


class ServerError(AuthentaError):
    """Raised for server-side issues (Default: server_error)."""

    def __init__(self, message: str, code: AuthentaErrorCode = "server_error", **kwargs):
        super().__init__(message, code, **kwargs)


# ── Mapping API error responses to exceptions ───────────────────────────────
#
# Before 0.3.0 every 4xx response surfaced as ValidationError, because the SDK
# looked for a `code` field the API never sent. The classes below keep that
# working: each is the specific error *and* a ValidationError, so existing
# `except ValidationError` handlers still catch them.

class _AuthenticationError(AuthenticationError, ValidationError):
    pass


class _AuthorizationError(AuthorizationError, ValidationError):
    pass


class _QuotaExceededError(QuotaExceededError, ValidationError):
    pass


class _InsufficientCreditsError(InsufficientCreditsError, ValidationError):
    pass


for _cls in (_AuthenticationError, _AuthorizationError, _QuotaExceededError, _InsufficientCreditsError):
    # Show the public name in tracebacks and reprs.
    _cls.__name__ = _cls.__qualname__ = _cls.__mro__[1].__name__


_ERRORS_BY_NAME = {
    "UNAUTHENTICATED": _AuthenticationError,
    "UNAUTHORIZED": _AuthenticationError,
    "TOKEN_EXPIRED": _AuthenticationError,
    "INVALID_API_KEY": _AuthenticationError,
    "IAM001": _AuthenticationError,
    "FORBIDDEN": _AuthorizationError,
    "ACCOUNT_SUSPENDED": _AuthorizationError,
    "IAM002": _AuthorizationError,
    "RATE_LIMITED": _QuotaExceededError,
    "AA001": _QuotaExceededError,
    "INSUFFICIENT_BALANCE": _InsufficientCreditsError,
    "PAYG_LIMIT_REACHED": _InsufficientCreditsError,
    "U007": _InsufficientCreditsError,
}


def _parse_retry_after(headers: Mapping[str, str]) -> Optional[float]:
    value = headers.get("Retry-After") if headers else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def error_from_response(
    status: int,
    body: Any,
    text: str = "",
    reason: str = "",
    headers: Optional[Mapping[str, str]] = None,
) -> AuthentaError:
    """
    Build the SDK exception for an API error response.

    The API returns ``{"rid", "name", "message", "fields"?}``. The legacy
    ``code`` field is still honoured if present.
    """
    retry_after = _parse_retry_after(headers or {})

    if not isinstance(body, dict):
        if 400 <= status < 500:
            return ValidationError(text or "Client error", status_code=status, retry_after=retry_after)
        if status >= 500:
            return ServerError(text or "Server error", status_code=status, retry_after=retry_after)
        return AuthentaError(text or reason or "Unknown error", status_code=status, retry_after=retry_after)

    code = body.get("name") or body.get("code") or "unknown"
    message = body.get("message") or reason or "Unknown error"
    kwargs = {"code": code, "status_code": status, "details": body, "retry_after": retry_after}

    error_class = _ERRORS_BY_NAME.get(code)
    if error_class is None:
        if status == 401:
            error_class = _AuthenticationError
        elif status == 402:
            error_class = _InsufficientCreditsError
        elif status == 403:
            error_class = _AuthorizationError
        elif status == 429:
            error_class = _QuotaExceededError
        elif 400 <= status < 500:
            error_class = ValidationError
        elif status >= 500:
            error_class = ServerError
        else:
            error_class = AuthentaError

    return error_class(message, **kwargs)
