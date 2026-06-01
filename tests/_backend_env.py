import os
import sys
import unittest

_BACKEND_ALIASES = {
    "megatron": {"megatron", "mbridge", "torch"},
    "paddlefleet": {"paddlefleet", "pfleet", "paddle"},
}


def _normalize_backend(value: str) -> set[str]:
    value = value.strip().lower()
    if not value:
        return set()
    if value == "all":
        return set(_BACKEND_ALIASES)
    return {backend for backend, aliases in _BACKEND_ALIASES.items() if value in aliases}


def _requested_backends() -> set[str] | None:
    raw_value = os.environ.get("INTERNAL_MEDICINE_TEST_BACKEND")
    if raw_value is None:
        return None

    requested: set[str] = set()
    for part in raw_value.split(","):
        requested.update(_normalize_backend(part))
    return requested


def _backends_from_current_env() -> set[str] | None:
    env_text = " ".join(
        value
        for value in (
            os.environ.get("VIRTUAL_ENV", ""),
            sys.executable,
        )
        if value
    ).lower()

    detected = {
        backend
        for backend, aliases in _BACKEND_ALIASES.items()
        if any(f"/{alias}/" in env_text or env_text.endswith(f"/{alias}/bin/python") for alias in aliases)
    }
    return detected or None


def backend_enabled(backend: str) -> bool:
    requested = _requested_backends()
    if requested is not None:
        return backend in requested

    detected = _backends_from_current_env()
    if detected is not None:
        return backend in detected

    return True


def skip_unless_backend(backend: str):
    if not backend_enabled(backend):
        raise unittest.SkipTest(
            f"{backend} tests disabled for this environment; set INTERNAL_MEDICINE_TEST_BACKEND=all to override"
        )
