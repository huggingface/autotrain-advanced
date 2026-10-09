"""Optional fastapi-guard security middleware wiring for the AutoTrain app.

Everything here is opt-in: with no AUTOTRAIN_GUARD_* environment variables set,
`attach_guard` is a no-op and the app behaves exactly as before. Set
AUTOTRAIN_GUARD_ENABLED=1 to activate the middleware.

When enabled, the middleware provides IP allow/block lists, rate limiting with
auto-ban, user-agent blocking, penetration-attempt detection, security
headers, and (optionally) Redis-backed distributed state and IPInfo geo
lookups. See https://github.com/Guard-Core/fastapi-guard.
"""

import os
from typing import Callable, List, Optional

from autotrain import logger


DEFAULT_EXCLUDED_PATHS = "/docs,/redoc,/openapi.json,/static,/login/huggingface,/auth,/api/version"
DEFAULT_TRUSTED_PROXIES = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _env_list(name: str, default: str = "") -> List[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_security_config():
    from guard import SecurityConfig

    kwargs = {
        "enable_rate_limiting": True,
        "rate_limit": int(os.environ.get("AUTOTRAIN_GUARD_RATE_LIMIT", "100")),
        "rate_limit_window": int(os.environ.get("AUTOTRAIN_GUARD_RATE_LIMIT_WINDOW", "60")),
        "enable_ip_banning": True,
        "auto_ban_threshold": int(os.environ.get("AUTOTRAIN_GUARD_AUTO_BAN_THRESHOLD", "10")),
        "auto_ban_duration": int(os.environ.get("AUTOTRAIN_GUARD_AUTO_BAN_DURATION", "300")),
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable at localhost.
        "enable_redis": False,
        "blacklist": tuple(_env_list("AUTOTRAIN_GUARD_BLOCKED_IPS")),
        "blocked_user_agents": _env_list("AUTOTRAIN_GUARD_BLOCKED_USER_AGENTS"),
        "trusted_proxies": tuple(_env_list("AUTOTRAIN_GUARD_TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES)),
        "trusted_proxy_depth": int(os.environ.get("AUTOTRAIN_GUARD_TRUSTED_PROXY_DEPTH", "1")),
        "exclude_paths": _env_list("AUTOTRAIN_GUARD_EXCLUDED_PATHS", DEFAULT_EXCLUDED_PATHS),
        "custom_log_file": os.environ.get("AUTOTRAIN_GUARD_LOG_FILE") or None,
        "log_format": os.environ.get("AUTOTRAIN_GUARD_LOG_FORMAT", "text"),
    }

    allowed_ips = _env_list("AUTOTRAIN_GUARD_ALLOWED_IPS")
    if allowed_ips:
        kwargs["whitelist"] = tuple(allowed_ips)

    redis_url = os.environ.get("AUTOTRAIN_GUARD_REDIS_URL") or os.environ.get("REDIS_URL")
    if redis_url:
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = os.environ.get("AUTOTRAIN_GUARD_REDIS_PREFIX", "autotrain_guard:")

    ipinfo_token = os.environ.get("IPINFO_TOKEN")
    if ipinfo_token:
        kwargs["ipinfo_token"] = ipinfo_token

    return SecurityConfig(**kwargs)


ENABLED = int(os.environ.get("AUTOTRAIN_GUARD_ENABLED", "0"))

if ENABLED:
    try:
        from guard import SecurityDecorator

        security_config = _build_security_config()
        guard = SecurityDecorator(security_config)
        logger.info("fastapi-guard security middleware enabled")
    except ImportError as e:
        raise RuntimeError(
            "AUTOTRAIN_GUARD_ENABLED=1 requires the fastapi-guard package. "
            "Install it with: pip install fastapi-guard"
        ) from e
else:
    security_config = None
    guard = None


def attach_guard(app) -> None:
    """Add the fastapi-guard middleware to `app` when enabled via env.

    No-op unless AUTOTRAIN_GUARD_ENABLED=1. Also exposes the shared
    SecurityDecorator instance on app.state so the middleware adopts
    per-route decorator configuration.
    """
    if not ENABLED:
        return
    from guard import SecurityMiddleware

    app.add_middleware(SecurityMiddleware, config=security_config)
    app.state.guard_decorator = guard


def honeypot_detection(trap_fields: Optional[List[str]] = None) -> Callable:
    """Route decorator that blocks requests filling honeypot trap fields.

    Applies fastapi-guard's honeypot_detection to JSON and form-urlencoded
    bodies on POST/PUT/PATCH endpoints. Identity decorator when the middleware
    is disabled or no trap fields are configured (via argument or the
    AUTOTRAIN_GUARD_HONEYPOT_FIELDS environment variable).
    """
    fields = trap_fields if trap_fields is not None else _env_list("AUTOTRAIN_GUARD_HONEYPOT_FIELDS")
    if guard is None or not fields:

        def identity(func):
            return func

        return identity
    return guard.honeypot_detection(list(fields))
