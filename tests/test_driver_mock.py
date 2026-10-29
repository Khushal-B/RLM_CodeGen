"""End-to-end driver test with a scripted LM.

Exercises the full recursion -- plan, compose-check, run children in parallel,
verify, blame-route, repair -- without a single network call. Every bug found
here is a bug that would otherwise have cost API credit to find.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlmc.config import ArmConfig
from rlmc.contracts.types import Contract
from rlmc.core.driver import BuildDriver
from rlmc.core.node import Budget, BuildNode, NodeKind
from rlmc.core.workspace import Workspace
from rlmc.metrics.telemetry import RunTelemetry

PLAN = {
    "action": "DECOMPOSE",
    "reason": "split tokens from parser",
    "children": [
        {
            "node_id": "tokens",
            "title": "Token model",
            "spec": "define Token",
            "owns_paths": ["pkg/tokens.py"],
            "provides": [
                {
                    "qualname": "pkg.tokens.Token",
                    "kind": "class",
                    "path": "pkg/tokens.py",
                    "methods": [
                        {
                            "qualname": "pkg.tokens.Token.text",
                            "kind": "function",
                            "path": "pkg/tokens.py",
                            "params": [],
                            "returns": "str",
                        }
                    ],
                }
            ],
            "requires": [],
        },
        {
            "node_id": "parser",
            "title": "Parser",
            "spec": "parse text into tokens",
            "owns_paths": ["pkg/parser.py"],
            "provides": [
                {
                    "qualname": "pkg.parser.parse",
                    "kind": "function",
                    "path": "pkg/parser.py",
                    "params": [{"name": "text", "annotation": "str"}],
                    "returns": "list[str]",
                }
            ],
            "requires": ["pkg.tokens.Token"],
        },
    ],
}

GOOD_TOKENS = "class Token:\n    def text(self) -> str:\n        return ''\n"
GOOD_PARSER = (
    "from pkg.tokens import Token\n\n\ndef parse(text: str) -> list[str]:\n"
    "    return [text]\n"
)
BAD_PARSER = "def parse(text, extra):\n    return []\n"


class ScriptedLM:
    """Stands in for LMBackend. Records what it was asked."""

    def __init__(self, files_by_node: dict[str, dict[str, str]], plan: dict | None = None):
        self.files_by_node = files_by_node
        self.plan = plan if plan is not None else PLAN
        self.plan_calls = 0
        self.node_calls: list[str] = []
        self.systems: dict[str, str] = {}
        self.repair_seen: list[str] = []

    def complete(self, system, user, node_id="", model=None):
        self.plan_calls += 1
        return json.dumps(self.plan), {"calls": 1, "input_tokens": 10,
                                       "output_tokens": 10, "cost_usd": 0.0}

    def run_rlm_node(self, system_prompt, context, tools, max_iterations,
                     node_id="", budget_usd=None):
        self.node_calls.append(node_id)
        self.systems[node_id] = system_prompt
        if "Correction required" in (context or "") or "did not satisfy" in (context or ""):
            self.repair_seen.append(node_id)
        write = tools["write_file"]["tool"]
        for path, content in self.files_by_node.get(node_id, {}).items():
            write(path, content)
        return {
            "response": "done",
            "usage": {"calls": 1, "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.0},
            "iterations": 2,
            "subcalls": 0,
            "trajectory": {"iterations": [{"prompt": [{"role": "user", "content": "x" * 400}]}]},
        }


def _run(files_by_node, cfg=None, plan=None, max_depth=1):
    tmp = tempfile.mkdtemp(prefix="rlmc_test_")
    ws = Workspace(tmp)
    # depth 1 by default: the scripted LM always returns the same plan, so a
    # deeper budget would recurse it forever rather than testing anything.
    # Only impose the shallow default when the caller did not pick a preset;
    # otherwise an H2 depth arm would have its whole point overwritten.
    cfg = cfg if cfg is not None else ArmConfig.preset("h1-bounded").copy_with(
        max_depth=max_depth)
    lm = ScriptedLM(files_by_node, plan)
    driver = BuildDriver(cfg, ws, lm, RunTelemetry())
    root = BuildNode(
        node_id="root", kind=NodeKind.REPO, title="t", spec="build a parser",
        contract=Contract(node_id="root", owns_paths=["pkg", "pkg/*"]), depth=0,
    )
    result = driver.build(root, Budget(depth_remaining=cfg.max_depth))
    return result, driver, ws, lm


# --- the happy path -------------------------------------------------------

def test_decomposes_and_builds_children():
    result, _, ws, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                              "root.parser": {"pkg/parser.py": GOOD_PARSER}})
    assert result.action.value == "DECOMPOSE"
    assert len(result.children) == 2
    assert sorted(lm.node_calls) == ["root.parser", "root.tokens"]
    assert (Path(ws.root) / "pkg" / "parser.py").exists()


def test_clean_build_verifies():
    result, _, _, _ = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                            "root.parser": {"pkg/parser.py": GOOD_PARSER}})
    assert result.ok, [v.brief() for v in result.report.violations]
    assert result.report.cvr == 0.0


def test_payload_is_bounded():
    cfg = ArmConfig.preset("h1-bounded")
    result, driver, _, _ = _run(
        {"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
         "root.parser": {"pkg/parser.py": GOOD_PARSER + "# pad\n" * 5000}})
    assert driver.returns.bound_respected(cfg.diagnostic_budget_tokens)
    assert result.payload.tokens <= cfg.diagnostic_budget_tokens
    # the same content unbounded would have been far larger
    assert result.payload.raw_tokens > 10 * result.payload.tokens


def test_full_source_arm_returns_everything():
    cfg = ArmConfig.preset("h1-full-source").copy_with(max_depth=1)
    result, driver, _, _ = _run(
        {"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
         "root.parser": {"pkg/parser.py": GOOD_PARSER + "# pad\n" * 5000}}, cfg=cfg)
    assert result.payload.mode == "full_source"
    assert result.payload.tokens == result.payload.raw_tokens
    assert result.payload.tokens > cfg.diagnostic_budget_tokens


def test_arms_differ_only_in_return_mode():
    a = ArmConfig.preset("h1-bounded").to_dict()
    b = ArmConfig.preset("h1-full-source").to_dict()
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"name", "return_mode"}


# --- verification and repair ---------------------------------------------

def test_violation_is_detected():
    result, _, _, _ = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                            "root.parser": {"pkg/parser.py": BAD_PARSER}})
    kinds = {v.kind for v in result.report.violations}
    assert "arity_mismatch" in kinds


def test_repair_targets_the_guilty_child_only():
    result, driver, _, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                                  "root.parser": {"pkg/parser.py": BAD_PARSER}})
    assert result.repair_rounds >= 1
    # the parser is rebuilt; tokens, which is clean, is not
    assert lm.node_calls.count("root.parser") > lm.node_calls.count("root.tokens")


def test_violations_are_owner_attributed():
    result, _, _, _ = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                            "root.parser": {"pkg/parser.py": BAD_PARSER}})
    owners = {v.owner for v in result.report.violations}
    assert owners == {"root.parser"}


def test_repair_feedback_reaches_the_node():
    _, _, _, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                        "root.parser": {"pkg/parser.py": BAD_PARSER}})
    assert "root.parser" in lm.repair_seen


# --- the invariant --------------------------------------------------------

def test_child_cannot_write_outside_its_lane():
    result, _, ws, _ = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS,
                                        "pkg/parser.py": "SNEAK = 1\n"},
                             "root.parser": {"pkg/parser.py": GOOD_PARSER}})
    assert ws.denied_writes
    assert any(d.node_id == "root.tokens" for d in ws.denied_writes)


def test_child_cannot_read_a_sibling_file():
    tmp = tempfile.mkdtemp(prefix="rlmc_read_")
    ws = Workspace(tmp)
    ws.grant("a", ["pkg/a.py"])
    ws.grant("b", ["pkg/b.py"])
    ws.write("b", "pkg/b.py", "SECRET = 1\n")
    ok, msg = ws.read("a", "pkg/b.py")
    assert not ok
    assert "another node" in msg


def test_sibling_contracts_are_shown_as_stubs_not_source():
    _, _, _, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                        "root.parser": {"pkg/parser.py": GOOD_PARSER}})
    parser_prompt = lm.systems["root.parser"]
    assert "class Token" in parser_prompt          # the interface is shown
    assert "return ''" not in parser_prompt        # the implementation is not


# --- H3 arms --------------------------------------------------------------

def test_prose_arm_shows_no_stubs():
    cfg = ArmConfig.preset("h3-prose").copy_with(max_depth=1)
    _, _, _, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                        "root.parser": {"pkg/parser.py": GOOD_PARSER}}, cfg=cfg)
    assert "```python" not in lm.systems["root.parser"]


def test_none_arm_shows_no_siblings():
    cfg = ArmConfig.preset("h3-none").copy_with(max_depth=1)
    _, _, _, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS},
                        "root.parser": {"pkg/parser.py": GOOD_PARSER}}, cfg=cfg)
    assert "Sibling interfaces" not in lm.systems["root.parser"]


# --- H2 arms --------------------------------------------------------------

def test_depth0_never_decomposes():
    cfg = ArmConfig.preset("h2-depth0")
    result, _, _, lm = _run({"root": {"pkg/all.py": "X = 1\n"}}, cfg=cfg)
    assert result.action.value == "IMPLEMENT"
    assert lm.plan_calls == 0
    assert lm.node_calls == ["root"]


def test_adaptive_can_choose_to_implement():
    result, _, _, lm = _run({"root": {"pkg/all.py": "X = 1\n"}},
                            plan={"action": "IMPLEMENT", "reason": "small", "children": []})
    assert result.action.value == "IMPLEMENT"
    assert lm.node_calls == ["root"]


# --- composition ----------------------------------------------------------

def test_bad_composition_is_rejected_before_children_run():
    """Two children claiming the same file must not be built."""
    bad_plan = json.loads(json.dumps(PLAN))
    bad_plan["children"][1]["owns_paths"] = ["pkg/tokens.py"]  # collides
    result, driver, _, lm = _run({"root.tokens": {"pkg/tokens.py": GOOD_TOKENS}}, plan=bad_plan)
    assert driver.telemetry.compositions_rejected >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --- degradation under a failing / timing-out endpoint --------------------

class FailingLM:
    """Every call raises, as a saturated free endpoint eventually does."""

    def __init__(self, exc=TimeoutError("Request timed out.")):
        self.exc = exc
        self.plan_calls = 0
        self.node_calls: list[str] = []

    def complete(self, system, user, node_id="", model=None):
        self.plan_calls += 1
        raise self.exc

    def run_rlm_node(self, system_prompt, context, tools, max_iterations,
                     node_id="", budget_usd=None):
        self.node_calls.append(node_id)
        raise self.exc


def _run_with(lm, cfg=None):
    tmp = tempfile.mkdtemp(prefix="rlmc_fail_")
    ws = Workspace(tmp)
    cfg = cfg or ArmConfig.preset("h1-bounded").copy_with(max_depth=1)
    driver = BuildDriver(cfg, ws, lm, RunTelemetry())
    root = BuildNode(
        node_id="root", kind=NodeKind.REPO, title="t", spec="build it",
        contract=Contract(node_id="root", owns_paths=["pkg", "pkg/*"]), depth=0,
    )
    return driver.build(root, Budget(depth_remaining=cfg.max_depth)), driver, ws


def test_planner_failure_falls_back_to_implement():
    """A dead planner must degrade to a single node, not crash the run."""
    lm = FailingLM()
    result, _, _ = _run_with(lm)
    assert result.action.value == "IMPLEMENT"
    assert lm.plan_calls >= 1          # it tried
    assert lm.node_calls == ["root"]   # then fell back


def test_total_failure_still_produces_a_result():
    result, driver, _ = _run_with(FailingLM())
    assert result.ok is False
    assert result.error is not None
    assert "Timeout" in result.error or "timed out" in result.error.lower()
    # The invariant still holds on the failure path.
    assert driver.returns.bound_respected(256)


def test_failed_run_records_zero_not_a_crash():
    """A failed run must score 0, never vanish -- dropping it inflates the arm."""
    import rlmc.agents.contract_rlm as cr
    from rlmc.agents.base import TaskSpec

    class Boom:
        def __init__(self, cfg): pass
        def complete(self, *a, **k): raise TimeoutError("Request timed out.")
        def run_rlm_node(self, *a, **k): raise TimeoutError("Request timed out.")

    original = cr.LMBackend
    cr.LMBackend = Boom
    try:
        task = TaskSpec(task_id="t", requirements="build something", package_name="pkg")
        cfg = ArmConfig.preset("h1-bounded").copy_with(max_depth=1)
        rec = cr.ContractRLMAgent(cfg).run(task, tempfile.mkdtemp(prefix="rlmc_rec_"), 0)
    finally:
        cr.LMBackend = original

    assert rec.ok is False
    assert rec.completed is False
    assert rec.pass_rate == 0.0
    assert rec.files == 0
