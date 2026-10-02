from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def world_config() -> dict:
    path = Path(__file__).parents[1] / "configs" / "world.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

