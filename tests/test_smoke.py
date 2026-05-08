"""Smoke test — package imports cleanly and exposes ``__version__``."""

from __future__ import annotations

import re

import steadfast


def test_version_is_a_string() -> None:
    assert isinstance(steadfast.__version__, str)
    assert steadfast.__version__


def test_version_matches_pep440_dev_or_release() -> None:
    # Accept dev releases (0.1.0.dev0), pre-releases, and stable releases.
    pattern = r"^\d+\.\d+\.\d+(?:[.\-]?(?:a|b|rc|dev|post)\d*)?$"
    assert re.match(pattern, steadfast.__version__), steadfast.__version__


def test_cli_app_exists() -> None:
    from steadfast.cli import app, main

    assert callable(main)
    assert app is not None
