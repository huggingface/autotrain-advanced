import asyncio
import importlib
import itertools
import os

import httpx
import pytest
from fastapi import Body, FastAPI


pytest.importorskip("guard")

from autotrain.app import security as security_module  # noqa: E402


BLOCKED_IP = "203.0.113.7"

# Rate limiting state in guard-core is process-wide, so every scenario gets
# its own TEST-NET client IP to stay hermetic.
_IP_COUNTER = itertools.count(1)


def _unique_ip() -> str:
    n = next(_IP_COUNTER)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


def _reload(monkeypatch, **env):
    for name in list(os.environ):
        if name.startswith("AUTOTRAIN_GUARD_") or name in ("REDIS_URL", "IPINFO_TOKEN"):
            monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return importlib.reload(security_module)


def _build_app(module, trap_fields=None):
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    if trap_fields:

        @app.post("/echo")
        @module.honeypot_detection(trap_fields)
        async def echo(payload: dict = Body(...)):
            return {"received": payload}

    module.attach_guard(app)
    return app


def _run_scenario(module, scenario, trap_fields=None, client_ip=None):
    async def runner():
        app = _build_app(module, trap_fields=trap_fields)
        transport = httpx.ASGITransport(
            app=app, client=(client_ip or _unique_ip(), 50000)
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await scenario(client)

    return asyncio.run(runner())


def test_disabled_by_default(monkeypatch):
    module = _reload(monkeypatch)
    assert module.ENABLED == 0
    assert module.security_config is None
    assert module.guard is None

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 200

    _run_scenario(module, scenario)


def test_redis_stays_off_without_config(monkeypatch):
    module = _reload(monkeypatch, AUTOTRAIN_GUARD_ENABLED="1")
    assert module.security_config.enable_redis is False


def test_full_bundle_mapping(monkeypatch):
    module = _reload(
        monkeypatch,
        AUTOTRAIN_GUARD_ENABLED="1",
        AUTOTRAIN_GUARD_PASSIVE_MODE="1",
        AUTOTRAIN_GUARD_SECURITY_HEADERS="1",
        AUTOTRAIN_GUARD_ENFORCE_HTTPS="1",
        AUTOTRAIN_GUARD_BLOCKED_COUNTRIES="RU",
        AUTOTRAIN_GUARD_ALLOWED_COUNTRIES="US",
        AUTOTRAIN_GUARD_BLOCK_CLOUD_PROVIDERS="AWS",
        IPINFO_TOKEN="test-token",
    )
    cfg = module.security_config
    assert cfg.passive_mode is True
    assert cfg.enforce_https is True
    assert cfg.security_headers["enabled"] is True
    assert cfg.blocked_countries == frozenset({"RU"})
    assert cfg.whitelist_countries == frozenset({"US"})
    assert cfg.block_cloud_providers == frozenset({"AWS"})


def test_honeypot_decorator_is_identity_when_disabled(monkeypatch):
    module = _reload(monkeypatch)

    async def handler():
        return {"ok": True}

    assert module.honeypot_detection(["website"])(handler) is handler


def test_enabled_adds_middleware(monkeypatch):
    module = _reload(monkeypatch, AUTOTRAIN_GUARD_ENABLED="1")
    app = FastAPI()
    module.attach_guard(app)
    assert len(app.user_middleware) == 1

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 200

    _run_scenario(module, scenario)


def test_blocked_ip_is_rejected(monkeypatch):
    module = _reload(
        monkeypatch,
        AUTOTRAIN_GUARD_ENABLED="1",
        AUTOTRAIN_GUARD_BLOCKED_IPS=BLOCKED_IP,
    )

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 403

    _run_scenario(module, scenario, client_ip=BLOCKED_IP)


def test_excluded_paths_still_enforce_ip_lists(monkeypatch):
    module = _reload(
        monkeypatch,
        AUTOTRAIN_GUARD_ENABLED="1",
        AUTOTRAIN_GUARD_BLOCKED_IPS=BLOCKED_IP,
        AUTOTRAIN_GUARD_EXCLUDED_PATHS="/ping",
    )

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 403

    _run_scenario(module, scenario, client_ip=BLOCKED_IP)


def test_whitelist_rejects_unknown_ips(monkeypatch):
    module = _reload(
        monkeypatch,
        AUTOTRAIN_GUARD_ENABLED="1",
        AUTOTRAIN_GUARD_ALLOWED_IPS="10.0.0.1",
    )

    async def scenario(client):
        response = await client.get("/ping")
        assert response.status_code == 403

    _run_scenario(module, scenario)


def test_honeypot_trap_field(monkeypatch):
    module = _reload(monkeypatch, AUTOTRAIN_GUARD_ENABLED="1")

    async def scenario(client):
        ok = await client.post("/echo", json={"ok": "yes"})
        assert ok.status_code == 200
        trapped = await client.post("/echo", json={"ok": "yes", "website": "spam"})
        assert trapped.status_code == 403

    _run_scenario(module, scenario, trap_fields=["website"])


def test_rate_limit_returns_429(monkeypatch):
    module = _reload(
        monkeypatch,
        AUTOTRAIN_GUARD_ENABLED="1",
        AUTOTRAIN_GUARD_RATE_LIMIT="2",
        AUTOTRAIN_GUARD_RATE_LIMIT_WINDOW="60",
    )

    async def scenario(client):
        assert (await client.get("/ping")).status_code == 200
        assert (await client.get("/ping")).status_code == 200
        assert (await client.get("/ping")).status_code == 429

    _run_scenario(module, scenario)
