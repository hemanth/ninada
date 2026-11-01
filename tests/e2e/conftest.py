"""Shared fixtures and verification utilities for Ninada Dual-Track E2E Test Suite."""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ninada import Ninada
from tests.e2e.helpers import verify_pcm_samples, verify_wav_header


@pytest.fixture(scope="session")
def engine_sa() -> Ninada:
    """Sanskrit engine instance."""
    return Ninada(lang="sa")


@pytest.fixture(scope="session")
def engine_hi() -> Ninada:
    """Hindi engine instance."""
    return Ninada(lang="hi")


@pytest.fixture(scope="session")
def engine_kn() -> Ninada:
    """Kannada engine instance."""
    return Ninada(lang="kn")


@pytest.fixture(scope="session")
def engine_ta() -> Ninada:
    """Tamil engine instance."""
    return Ninada(lang="ta")


@pytest.fixture(scope="session")
def engine_te() -> Ninada:
    """Telugu engine instance."""
    return Ninada(lang="te")


@pytest.fixture(scope="session")
def engine_ml() -> Ninada:
    """Malayalam engine instance."""
    return Ninada(lang="ml")


@pytest.fixture(scope="session")
def run_cli() -> Callable[..., subprocess.CompletedProcess[str]]:
    """Helper fixture to invoke ninada CLI via subprocess."""

    def _runner(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        python_path = str(REPO_ROOT / "python" / "src")
        env["PYTHONPATH"] = f"{python_path}:{env.get('PYTHONPATH', '')}"
        cmd = [sys.executable, "-m", "ninada.cli", *args]
        return subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=str(REPO_ROOT),
            env=env,
        )

    return _runner
