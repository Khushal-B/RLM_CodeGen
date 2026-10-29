"""The baseline arms must produce the same artifact shape as the contract arm.

If a baseline cannot be scored by the identical evaluator, the headline table
is not comparable -- so this is checked rather than assumed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlmc.agents.base import TaskSpec, make_agent
from rlmc.agents.vanilla_rlm import parse_file_blocks
from rlmc.config import ArmConfig

SINGLE_SHOT = """Here is the package.

### pkg/__init__.py
```python
from pkg.core import run

__all__ = ["run"]
```

### pkg/core.py
```python
def run() -> int:
    return 1
```
"""


def test_parse_file_blocks_extracts_paths_and_content():
    files = parse_file_blocks(SINGLE_SHOT)
    assert set(files) == {"pkg/__init__.py", "pkg/core.py"}
    assert "def run() -> int:" in files["pkg/core.py"]


def test_parse_file_blocks_ignores_prose():
    assert parse_file_blocks("no files here, sorry") == {}


class StubBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    def complete(self, system, user, node_id="", model=None):
        return SINGLE_SHOT, {"calls": 1, "input_tokens": 100,
                             "output_tokens": 50, "cost_usd": 0.0}

    def run_rlm_node(self, system_prompt, context, tools, max_iterations,
                     node_id="", budget_usd=None):
        write = tools["write_file"]["tool"]
        for path, content in parse_file_blocks(SINGLE_SHOT).items():
            write(path, content)
        return {"response": "done",
                "usage": {"calls": 1, "input_tokens": 500,
                          "output_tokens": 300, "cost_usd": 0.0},
                "iterations": 2, "subcalls": 1,
                "trajectory": {"iterations": [
                    {"prompt": [{"role": "user", "content": "x" * 4000}]}]}}


@pytest.fixture()
def stub(monkeypatch):
    import rlmc.agents.vanilla_rlm as vr

    monkeypatch.setattr(vr, "LMBackend", StubBackend)
    return StubBackend


def _task():
    return TaskSpec(task_id="t", requirements="build a package", package_name="pkg")


def test_vanilla_rlm_writes_a_package(stub):
    agent = make_agent(ArmConfig.preset("vanilla-rlm"))
    rec = agent.run(_task(), tempfile.mkdtemp(prefix="van_"), 0)
    assert rec.completed
    assert rec.files == 2
    assert rec.arm == "vanilla-rlm"


def test_base_lm_writes_a_package(stub):
    agent = make_agent(ArmConfig.preset("base-lm"))
    rec = agent.run(_task(), tempfile.mkdtemp(prefix="base_"), 0)
    assert rec.completed
    assert rec.files == 2


def test_baselines_have_no_ownership_enforcement(stub):
    """The baselines model systems that do not have the invariant at all."""
    agent = make_agent(ArmConfig.preset("vanilla-rlm"))
    rec = agent.run(_task(), tempfile.mkdtemp(prefix="van2_"), 0)
    assert rec.telemetry["workspace"]["denied_writes"] == 0


def test_all_arms_produce_the_same_record_fields(stub):
    import rlmc.agents.contract_rlm as cr

    original = cr.LMBackend
    cr.LMBackend = StubBackend
    try:
        recs = []
        for arm in ("vanilla-rlm", "base-lm"):
            recs.append(make_agent(ArmConfig.preset(arm)).run(
                _task(), tempfile.mkdtemp(prefix=arm + "_"), 0))
    finally:
        cr.LMBackend = original
    keys = [set(r.to_dict()) for r in recs]
    assert all(k == keys[0] for k in keys)
    for field in ("pass_rate", "cvr", "root_context_tokens", "loc", "completed"):
        assert field in keys[0]


def test_baseline_root_context_is_recorded(stub):
    """The H1 plot needs this from every arm, baselines included."""
    rec = make_agent(ArmConfig.preset("vanilla-rlm")).run(
        _task(), tempfile.mkdtemp(prefix="van3_"), 0)
    assert rec.root_context_tokens > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
