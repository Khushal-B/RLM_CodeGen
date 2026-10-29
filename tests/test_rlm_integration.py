"""Integration test against the REAL rlm.RLM class, with a stubbed model.

Validates the seam that unit tests cannot: that a genuine RLM instance accepts
our system prompt (which goes through ``.format(custom_tools_section=...)``),
exposes our custom tools in its REPL, runs them, and produces a logger
trajectory we can compute the context high-water mark from.

Skipped automatically when ``rlm`` is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rlm = pytest.importorskip("rlm")

from rlmc.agents.backend import LMBackend
from rlmc.config import ArmConfig
from rlmc.contracts.imports import SymbolOwnership
from rlmc.contracts.types import Contract, Param, SymbolSpec
from rlmc.core.prompts import build_node_system_prompt
from rlmc.core.tools import NodeToolbox
from rlmc.core.workspace import Workspace


class _StubClient:
    """Minimal stand-in for rlm's OpenAIClient."""

    responses: list[str] = []

    def __init__(self, *a, **kw):
        self.model_name = kw.get("model_name", "stub")
        self._i = 0
        self.last_prompt_tokens = 10
        self.last_completion_tokens = 10

    def completion(self, prompt, model=None):
        out = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return out

    def get_usage_summary(self):
        from rlm.core.types import ModelUsageSummary, UsageSummary

        return UsageSummary(model_usage_summaries={
            self.model_name: ModelUsageSummary(1, 10, 10, 0.0)
        })

    def get_last_usage(self):
        from rlm.core.types import ModelUsageSummary

        return ModelUsageSummary(1, 10, 10, 0.0)


@pytest.fixture()
def stub_client(monkeypatch):
    import rlm.clients

    def fake_get_client(backend, kwargs):
        return _StubClient(**kwargs)

    # Turn counter must advance, otherwise every turn returns the same text and
    # the loop finishes on turn 1 without ever running the earlier code blocks.
    turn = {"i": 0}

    def fake_completion(self, messages, model=None):
        i = min(turn["i"], len(_StubClient.responses) - 1)
        turn["i"] += 1
        return _StubClient.responses[i]

    monkeypatch.setattr(rlm.clients, "get_client", fake_get_client)
    monkeypatch.setattr("rlm.core.rlm.get_client", fake_get_client)
    monkeypatch.setattr("rlm.core.lm_handler.LMHandler.completion", fake_completion)
    return _StubClient


def _toolbox(tmp_path):
    ws = Workspace(str(tmp_path))
    c = Contract(
        node_id="n1",
        owns_paths=["pkg/a.py"],
        provides=[SymbolSpec("pkg.a.f", "function", "pkg/a.py", [Param("x", "int")], "str")],
    )
    ws.grant("n1", c.owns_paths)
    tb = NodeToolbox(
        node_id="n1", contract=c, workspace=ws,
        ownership=SymbolOwnership.from_contracts([c], scope_complete=False),
    )
    return ws, c, tb


def test_real_rlm_runs_our_tools_and_writes_files(tmp_path, stub_client):
    ws, contract, tb = _toolbox(tmp_path)
    system = build_node_system_prompt(contract, [], "typed", "node one", 0, is_root=True)

    _StubClient.responses = [
        "I will write the file.\n"
        "```repl\n"
        "print(write_file('pkg/a.py', 'def f(x: int) -> str:\\n    return str(x)\\n'))\n"
        "print(check_contract())\n"
        "```",
        "Done.\n```repl\nanswer['content'] = 'built'\nanswer['ready'] = True\n```",
    ]

    backend = LMBackend(ArmConfig(backend="openai", model="stub"))
    out = backend.run_rlm_node(
        system_prompt=system,
        context="build a function f(x: int) -> str",
        tools=tb.as_custom_tools(),
        max_iterations=4,
        node_id="n1",
    )

    assert (Path(ws.root) / "pkg" / "a.py").exists()
    assert tb.writes == 1
    assert tb.checks == 1
    assert out["iterations"] >= 1


def test_system_prompt_survives_rlm_format_call(tmp_path):
    """RLM calls .format() on the system prompt; braces must not break it."""
    _, contract, _ = _toolbox(tmp_path)
    # A contract whose signature contains braces is the realistic hazard.
    contract.provides[0].returns = "dict[str, list[int]]"
    contract.prose = "returns {'a': [1]} shaped data"
    system = build_node_system_prompt(contract, [], "typed", "n", 0)
    safe = system.replace("{", "{{").replace("}", "}}") + "\n{custom_tools_section}\n"
    formatted = safe.format(custom_tools_section="- tool: x")
    assert "dict[str, list[int]]" in formatted
    assert "- tool: x" in formatted


def test_context_high_water_reads_a_real_trajectory(tmp_path, stub_client):
    from rlmc.metrics.telemetry import context_high_water

    _, contract, tb = _toolbox(tmp_path)
    system = build_node_system_prompt(contract, [], "typed", "n", 0)
    _StubClient.responses = [
        "```repl\nanswer['content'] = 'x'\nanswer['ready'] = True\n```",
    ]
    backend = LMBackend(ArmConfig(backend="openai", model="stub"))
    out = backend.run_rlm_node(
        system_prompt=system, context="ctx", tools=tb.as_custom_tools(),
        max_iterations=2, node_id="n1",
    )
    assert context_high_water(out["trajectory"]) > 0


def test_tools_are_described_in_the_prompt(tmp_path):
    _, _, tb = _toolbox(tmp_path)
    from rlm.environments.base_env import format_tools_for_prompt

    text = format_tools_for_prompt(tb.as_custom_tools())
    assert "write_file" in text
    assert "check_contract" in text


def test_reserved_names_are_not_shadowed(tmp_path):
    """Our tools must not collide with RLM's reserved REPL names."""
    from rlm.environments.base_env import RESERVED_TOOL_NAMES, validate_custom_tools

    _, _, tb = _toolbox(tmp_path)
    tools = tb.as_custom_tools()
    assert not (set(tools) & set(RESERVED_TOOL_NAMES))
    validate_custom_tools(tools)  # raises if invalid


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
