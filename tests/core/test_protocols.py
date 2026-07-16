"""Tests for core module protocols."""

import pytest

from gymkhana.core.services.sandboxes.sandbox import SandboxService
from gymkhana.core.services.sandboxes.repl import REPLSandbox


def test_sandbox_abc_exists():
    """Verify Sandbox ABC is importable."""
    assert SandboxService is not None


def test_repl_sandbox_inherits_sandbox():
    """Verify REPLSandbox inherits from Sandbox."""
    service = REPLSandbox()
    assert isinstance(service, SandboxService)


def test_sandbox_abc_has_required_methods():
    """Verify Sandbox ABC defines required methods."""
    for method in [
        "create_session",
        "execute",
        "execute_bash",
        "get_state",
        "reset_session",
        "delete_session",
        "health_check",
    ]:
        assert hasattr(SandboxService, method), f"SandboxService missing {method}"


def test_repl_sandbox_has_required_methods():
    """Verify REPLSandbox has all required methods."""
    service = REPLSandbox()
    required_methods = [
        "create_session",
        "execute",
        "execute_bash",
        "get_state",
        "reset_session",
        "delete_session",
        "health_check",
    ]
    for method in required_methods:
        assert hasattr(service, method), f"REPLSandboxService missing {method}"
        assert callable(getattr(service, method)), f"{method} is not callable"
