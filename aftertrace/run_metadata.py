"""Git provenance capture and dirty-tree guard for experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


def capture_git_state(repo_root: Path) -> dict[str, Any]:
    """Return the current commit and whether the worktree/index is dirty."""

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain"), cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout
    return {"git_hash": commit, "dirty": bool(status.strip())}


def require_clean_tree(repo_root: Path, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Capture provenance, refusing dirty worktrees unless explicitly allowed."""

    state = capture_git_state(repo_root)
    state["allow_dirty"] = bool(allow_dirty)
    if state["dirty"] and not allow_dirty:
        raise RuntimeError(
            "refusing to start a run from a dirty git tree; commit/stash changes "
            "or pass --allow-dirty to record this state"
        )
    return state


def record_run_provenance(run_dir: Path, state: dict[str, Any]) -> None:
    """Append this launch's git state to the run directory's provenance file."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "git_provenance.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"launches": []}
    payload["launches"].append(dict(state))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
