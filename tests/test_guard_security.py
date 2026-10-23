"""Tests for the optional fastapi-guard security middleware wiring.

Run with: PYTHONPATH=src pytest tests/test_guard_security.py
"""

import os

import pytest

from autotrain.app.security import _build_security_config


def _clear_guard_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("AUTOTRAIN_GUARD_"):
            monkeypatch.delenv(name, raising=False)


def test_default_config_zero_trust(monkeypatch):
    _clear_guard_env(monkeypatch)
    config = _build_security_config()
    # No implicit trust in RFC1918 peers: an unlisted proxy means
    # X-Forwarded-For is not honored, so LAN clients cannot spoof their
    # way past the blocklist or rate limits.
    assert config.trusted_proxies == ()
    # Request bodies are training configs and payloads, not commands:
    # scanning them is opt-in.
    assert config.detection_scan_body is False
    # Repeated rate-limit violations feed the ban engine by default.
    assert config.enable_rate_limit_auto_ban is True


def test_trusted_proxies_from_env(monkeypatch):
    _clear_guard_env(monkeypatch)
    monkeypatch.setenv("AUTOTRAIN_GUARD_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
    config = _build_security_config()
    assert config.trusted_proxies == ("10.0.0.1", "10.0.0.2")


def test_invalid_boolean_env_raises(monkeypatch):
    _clear_guard_env(monkeypatch)
    monkeypatch.setenv("AUTOTRAIN_GUARD_SCAN_BODY", "sometimes")
    with pytest.raises(ValueError, match="AUTOTRAIN_GUARD_SCAN_BODY"):
        _build_security_config()
