from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml

from aftertrace.memory.base import MemoryItem
from aftertrace.memory.retriever import retrieve


def test_retriever_ties_are_fully_ordered():
    entities = ("E_0", "E_9", "E_2")
    keys = ("z", "a", "a")
    items = [
        MemoryItem(
            key, 7, ("EP", "T_0", entity, "COLOR", "K_0", "EOS"), None,
            source_entity=entity,
        )
        for entity, key in zip(entities, keys, strict=True)
    ]
    result = retrieve(items, ("Q", "COLOR", "A"), 6)
    assert [(view.item.key, view.item.source_entity) for view in result.selected] == [("a", "E_2")]


def test_one_world_frontier_outputs_ignore_python_hash_seed(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    source = """
import json
from aftertrace.memory.base import MemoryItem
from aftertrace.memory.retriever import retrieve
items = [
    MemoryItem('z', 7, ('EP','T_0','E_0','COLOR','K_0','EOS'), None, source_entity='E_0'),
    MemoryItem('a', 7, ('EP','T_0','E_9','COLOR','K_0','EOS'), None, source_entity='E_9'),
    MemoryItem('a', 7, ('EP','T_0','E_2','COLOR','K_0','EOS'), None, source_entity='E_2'),
]
r = retrieve(items, ('Q','COLOR','A'), 6)
print(json.dumps([(v.item.key, v.item.source_entity) for v in r.selected]))
"""
    outputs = []
    for hash_seed in ("1", "876543"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        completed = subprocess.run(
            (sys.executable, "-c", source), cwd=repo, env=env,
            check=True, capture_output=True, text=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1] == '[["a", "E_2"]]\n'

    # Exercise the actual frontier entry point in two fresh interpreters too.
    sweep = yaml.safe_load((repo / "configs/sweep.yaml").read_text(encoding="utf-8"))
    sweep.update(
        K=[32], p_redundancy_levels=[0.6], eval_worlds_per_level=1,
        policy_tune_worlds=1, stratified_queries_per_type=1,
    )
    sweep["tuning"].update(
        tau_candidates=[0.7], decay_rate_candidates=[0.005], c_candidates=[1],
        decay_rules=["time"], queries_per_type=1,
        utility_weight_profiles=[{key: 1.0 for key in (
            "core_relational", "causal_linked", "temporal", "social", "incidental", "interference"
        )}],
    )
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep, sort_keys=True), encoding="utf-8")
    frontier_outputs = []
    for index, hash_seed in enumerate(("2", "765432")):
        output_dir = tmp_path / f"frontier_{index}"
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        subprocess.run(
            (
                sys.executable, str(repo / "scripts/oracle_frontier.py"),
                "--sweep-config", str(sweep_path), "--worlds", "1",
                "--output-dir", str(output_dir), "--allow-dirty",
            ),
            cwd=repo, env=env, check=True, capture_output=True, text=True,
        )
        frontier_outputs.append({
            name: (output_dir / name).read_bytes()
            for name in (
                "frontier.csv", "per_type.csv", "kstar.csv",
                "kstar_ratios_by_pair.csv", "matched_k_bootstrap.csv",
            )
        })
    assert frontier_outputs[0] == frontier_outputs[1]
