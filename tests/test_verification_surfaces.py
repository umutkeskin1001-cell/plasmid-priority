from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_makefile_defines_canonical_release_verification_surface() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "verify-release:" in makefile
    assert "scripts/46_verify_release.py" in makefile
    assert "critical-coverage:" in makefile
    assert "docs-check:" in makefile
    assert "--cov-fail-under=70" in makefile


def test_justfile_exposes_release_verification_surface() -> None:
    justfile = (PROJECT_ROOT / "Justfile").read_text(encoding="utf-8")
    assert "verify-release:" in justfile
    assert "scripts/46_verify_release.py" in justfile
    assert "critical-coverage:" in justfile
    assert "docs-check:" in justfile
