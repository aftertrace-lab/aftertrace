from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from aftertrace.run_metadata import record_run_provenance, require_clean_tree


def _git_responses(monkeypatch: pytest.MonkeyPatch, *, dirty: bool) -> None:
    responses = iter(("abc123\n", " M tracked.py\n" if dirty else ""))

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=next(responses), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_dirty_tree_is_refused_without_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git_responses(monkeypatch, dirty=True)
    with pytest.raises(RuntimeError, match="--allow-dirty"):
        require_clean_tree(tmp_path)


def test_allow_dirty_is_captured_in_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git_responses(monkeypatch, dirty=True)
    state = require_clean_tree(tmp_path, allow_dirty=True)
    run_dir = tmp_path / "run"
    record_run_provenance(run_dir, state)
    payload = json.loads((run_dir / "git_provenance.json").read_text(encoding="utf-8"))
    assert payload["launches"] == [{"git_hash": "abc123", "dirty": True, "allow_dirty": True}]


def test_repeated_resume_launches_append_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    record_run_provenance(run_dir, {"git_hash": "abc123", "dirty": False, "allow_dirty": False})
    record_run_provenance(run_dir, {"git_hash": "def456", "dirty": True, "allow_dirty": True})
    payload = json.loads((run_dir / "git_provenance.json").read_text(encoding="utf-8"))
    assert [launch["git_hash"] for launch in payload["launches"]] == ["abc123", "def456"]
