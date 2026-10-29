"""Unit tests for the deterministic contract core.

No network, no model, no rlm dependency. These are the checks that the CVR
metric rests on, so they need to be right before any run costs money.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlmc.contracts.checker import verify_node, verify_subtree
from rlmc.contracts.compose import check_composition
from rlmc.contracts.imports import SymbolOwnership, find_cycles
from rlmc.contracts.schema import SchemaError, parse_decomposition
from rlmc.contracts.stubs import emit_stub, path_to_module
from rlmc.contracts.types import Contract, Param, SymbolSpec
from rlmc.core.blame import RepairAction, plan_repair
from rlmc.core.payload import build_payload


def sym(qualname, kind="function", path="pkg/a.py", params=None, returns=None, methods=None):
    return SymbolSpec(qualname, kind, path, params or [], returns, None, methods or [])


def contract(node_id="n", provides=None, requires=None, owns=None):
    return Contract(
        node_id=node_id,
        provides=provides or [],
        requires=requires or [],
        owns_paths=owns or ["pkg/a.py"],
    )


# --- hashing --------------------------------------------------------------

def test_hash_is_stable_across_ordering():
    a = contract(provides=[sym("pkg.a.f"), sym("pkg.a.g")])
    b = contract(provides=[sym("pkg.a.g"), sym("pkg.a.f")])
    assert a.hash() == b.hash()


def test_hash_ignores_prose():
    a = contract(provides=[sym("pkg.a.f")])
    b = contract(provides=[sym("pkg.a.f")])
    b.prose = "a completely different description"
    assert a.hash() == b.hash()


def test_hash_changes_with_signature():
    a = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")])])
    b = contract(provides=[sym("pkg.a.f", params=[Param("x", "str")])])
    assert a.hash() != b.hash()


# --- stub emission --------------------------------------------------------

def test_emit_stub_function():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")], returns="str")])
    assert "def f(x: int) -> str: ..." in emit_stub(c)


def test_emit_stub_default_is_ellipsis():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int", has_default=True)])])
    assert "x: int = ..." in emit_stub(c)


def test_emit_stub_class_with_methods():
    c = contract(provides=[sym("pkg.a.C", kind="class", methods=[sym("pkg.a.C.m")])])
    out = emit_stub(c)
    assert "class C:" in out and "def m()" in out


def test_path_to_module():
    assert path_to_module("pkg/sub/mod.py") == "pkg.sub.mod"
    assert path_to_module("pkg/__init__.py") == "pkg"


# --- stub verification ----------------------------------------------------

def test_clean_code_verifies():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")], returns="str")])
    assert verify_node(c, {"pkg/a.py": "def f(x: int) -> str:\n    return ''\n"}).ok


def test_missing_symbol_flagged():
    c = contract(provides=[sym("pkg.a.f")])
    r = verify_node(c, {"pkg/a.py": "def other():\n    pass\n"})
    assert [v.kind for v in r.violations] == ["missing_symbol"]


def test_extra_required_param_flagged():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")])])
    r = verify_node(c, {"pkg/a.py": "def f(x: int, y: int):\n    pass\n"})
    assert "arity_mismatch" in {v.kind for v in r.violations}


def test_extra_optional_param_is_allowed():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")])])
    r = verify_node(c, {"pkg/a.py": "def f(x: int, y: int = 0):\n    pass\n"})
    assert r.ok


def test_varargs_absorbs_declared_arity():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")])])
    r = verify_node(c, {"pkg/a.py": "def f(*args, **kw):\n    pass\n"})
    assert r.ok


def test_annotation_mismatch_flagged():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")])])
    r = verify_node(c, {"pkg/a.py": "def f(x: str):\n    pass\n"})
    assert "annotation_mismatch" in {v.kind for v in r.violations}


def test_optional_and_union_are_equivalent():
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "Optional[int]")])])
    r = verify_node(c, {"pkg/a.py": "def f(x: int | None):\n    pass\n"})
    assert r.ok


def test_missing_annotation_is_not_a_violation():
    """Deliberate: an unannotated implementation still honours the interface."""
    c = contract(provides=[sym("pkg.a.f", params=[Param("x", "int")], returns="str")])
    assert verify_node(c, {"pkg/a.py": "def f(x):\n    return ''\n"}).ok


def test_class_declared_but_function_written():
    c = contract(provides=[sym("pkg.a.C", kind="class")])
    r = verify_node(c, {"pkg/a.py": "def C():\n    pass\n"})
    assert "kind_mismatch" in {v.kind for v in r.violations}


def test_missing_method_flagged():
    c = contract(provides=[sym("pkg.a.C", kind="class", methods=[sym("pkg.a.C.m")])])
    r = verify_node(c, {"pkg/a.py": "class C:\n    pass\n"})
    assert "missing_method" in {v.kind for v in r.violations}


def test_self_is_not_counted_as_a_parameter():
    c = contract(provides=[
        sym("pkg.a.C", kind="class", methods=[sym("pkg.a.C.m", params=[Param("n", "int")])])
    ])
    r = verify_node(c, {"pkg/a.py": "class C:\n    def m(self, n: int):\n        pass\n"})
    assert r.ok


def test_syntax_error_flagged():
    c = contract(provides=[sym("pkg.a.f")])
    r = verify_node(c, {"pkg/a.py": "def f( ->\n"})
    assert "syntax_error" in {v.kind for v in r.violations}


def test_cvr_is_violations_over_checked():
    c = contract(provides=[sym("pkg.a.f"), sym("pkg.a.g")])
    r = verify_node(c, {"pkg/a.py": "def f():\n    pass\n"})
    assert r.checked > 0
    assert 0 < r.cvr <= 1


def test_no_double_counting_in_subtree():
    parent = contract("root", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    child = contract("c", provides=[sym("pkg.a.f", params=[Param("x", "int")])],
                     owns=["pkg/a.py"])
    r = verify_subtree(parent, [child], {"c": {"pkg/a.py": "def f(x: int, y: int):\n    pass\n"}})
    subjects = [(v.kind, v.subject) for v in r.violations]
    assert len(subjects) == len(set(subjects))


# --- import graph ---------------------------------------------------------

def test_undeclared_cross_node_import_flagged():
    a = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    b = contract("b", provides=[sym("pkg.b.g", path="pkg/b.py")], owns=["pkg/b.py"])
    own = SymbolOwnership.from_contracts([a, b])
    r = verify_node(a, {"pkg/a.py": "from pkg.b import g\ndef f():\n    pass\n"},
                    ownership=own, extra_boundaries=["import_graph"])
    assert "undeclared_import" in {v.kind for v in r.violations}


def test_declared_import_is_fine():
    a = contract("a", provides=[sym("pkg.a.f")], requires=["pkg.b.g"], owns=["pkg/a.py"])
    b = contract("b", provides=[sym("pkg.b.g", path="pkg/b.py")], owns=["pkg/b.py"])
    own = SymbolOwnership.from_contracts([a, b])
    r = verify_node(a, {"pkg/a.py": "from pkg.b import g\ndef f():\n    pass\n"},
                    ownership=own, extra_boundaries=["import_graph"])
    assert r.ok


def test_stdlib_import_is_ignored():
    a = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    r = verify_node(a, {"pkg/a.py": "import os\ndef f():\n    pass\n"},
                    ownership=SymbolOwnership.from_contracts([a]),
                    extra_boundaries=["import_graph"])
    assert r.ok


def test_requires_not_checked_without_sibling_scope():
    """A leaf checked alone must not be blamed for a sibling that isn't present."""
    a = contract("a", provides=[], requires=["pkg.b.g"], owns=["pkg/a.py"])
    assert verify_node(a, {"pkg/a.py": "x = 1\n"}, extra_boundaries=["import_graph"]).ok


def test_cycle_detection():
    assert find_cycles({"a": {"b"}, "b": {"a"}})
    assert not find_cycles({"a": {"b"}, "b": set()})


def test_unowned_write_flagged():
    a = contract("a", provides=[], owns=["pkg/a.py"])
    r = verify_node(a, {"pkg/a.py": "x=1\n", "pkg/sneaky.py": "y=1\n"})
    assert "unowned_write" in {v.kind for v in r.violations}


# --- composition ----------------------------------------------------------

def test_composition_accepts_full_cover():
    parent = contract("root", provides=[sym("pkg.a.f"), sym("pkg.b.g", path="pkg/b.py")])
    a = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    b = contract("b", provides=[sym("pkg.b.g", path="pkg/b.py")], owns=["pkg/b.py"])
    assert check_composition(parent, [a, b]).ok


def test_composition_rejects_uncovered_symbol():
    parent = contract("root", provides=[sym("pkg.a.f"), sym("pkg.b.g", path="pkg/b.py")])
    a = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    res = check_composition(parent, [a])
    assert not res.ok
    assert "pkg.b.g" in res.missing


def test_composition_rejects_duplicate_claim():
    parent = contract("root", provides=[sym("pkg.a.f")])
    a = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    b = contract("b", provides=[sym("pkg.a.f")], owns=["pkg/b.py"])
    assert "pkg.a.f" in check_composition(parent, [a, b]).duplicated


def test_composition_rejects_file_conflict():
    parent = contract("root", provides=[])
    a = contract("a", owns=["pkg/a.py"])
    b = contract("b", owns=["pkg/a.py"])
    assert "pkg/a.py" in check_composition(parent, [a, b]).path_conflicts


def test_composition_rejects_unresolvable_require():
    parent = contract("root", provides=[])
    a = contract("a", requires=["pkg.nowhere.x"], owns=["pkg/a.py"])
    assert "pkg.nowhere.x" in check_composition(parent, [a]).unresolved


def test_composition_allows_external_dependency():
    parent = contract("root", provides=[])
    a = contract("a", requires=["numpy.array"], owns=["pkg/a.py"])
    assert check_composition(parent, [a], external_modules={"numpy"}).ok


def test_composition_rejects_cycle():
    parent = contract("root", provides=[])
    a = contract("a", provides=[sym("pkg.a.f")], requires=["pkg.b.g"], owns=["pkg/a.py"])
    b = contract("b", provides=[sym("pkg.b.g", path="pkg/b.py")], requires=["pkg.a.f"],
                 owns=["pkg/b.py"])
    res = check_composition(parent, [a, b])
    assert not res.ok
    assert any(v.kind == "import_cycle" for v in res.violations)


# --- schema parsing -------------------------------------------------------

def test_parses_fenced_json_with_prose():
    text = 'Here you go:\n```json\n{"action":"IMPLEMENT","reason":"small"}\n```\nDone!'
    assert parse_decomposition(text)["action"] == "IMPLEMENT"


def test_tolerates_trailing_comma_and_python_literals():
    text = '{"action":"IMPLEMENT","reason":"x", "children": [],}'
    assert parse_decomposition(text)["action"] == "IMPLEMENT"


def test_parses_signature_string_into_params():
    text = ('{"action":"DECOMPOSE","children":[{"node_id":"a","owns_paths":["p/a.py"],'
            '"provides":[{"qualname":"p.a.f","kind":"function",'
            '"signature":"(x: int, y: str = \'z\') -> bool"}]}]}')
    c = parse_decomposition(text)["children"][0][0]
    s = c.provides[0]
    assert [p.name for p in s.params] == ["x", "y"]
    assert s.params[1].has_default
    assert s.returns == "bool"


def test_duplicate_node_ids_are_made_unique():
    text = ('{"action":"DECOMPOSE","children":['
            '{"node_id":"a","owns_paths":["p/a.py"]},'
            '{"node_id":"a","owns_paths":["p/b.py"]}]}')
    ids = [c.node_id for c, _, _ in parse_decomposition(text)["children"]]
    assert len(set(ids)) == 2


def test_garbage_raises_schema_error():
    with pytest.raises(SchemaError):
        parse_decomposition("no json at all here")


# --- payload / invariant --------------------------------------------------

def test_bounded_payload_respects_c():
    c = contract(provides=[sym("pkg.a.f")])
    r = verify_node(c, {"pkg/a.py": "x = 1\n"})
    for budget in (16, 64, 256, 1024):
        p = build_payload(node_id="n", ok=False, contract_hash=c.hash(), report=r,
                          files={"pkg/a.py": "x = 1\n" * 20000},
                          return_mode="bounded", budget_tokens=budget)
        assert p.tokens <= budget, budget


def test_bounded_payload_is_flat_in_subtree_size():
    """The O(1) property H1 actually claims."""
    c = contract(provides=[sym("pkg.a.f")])
    r = verify_node(c, {"pkg/a.py": "x = 1\n"})
    sizes = []
    for n in (100, 1000, 10000, 100000):
        p = build_payload(node_id="n", ok=True, contract_hash=c.hash(), report=r,
                          files={"pkg/a.py": "x = 1\n" * n},
                          return_mode="bounded", budget_tokens=256)
        sizes.append(p.tokens)
    assert max(sizes) - min(sizes) <= 5


def test_full_source_payload_grows_with_subtree_size():
    sizes = []
    for n in (100, 1000, 10000):
        p = build_payload(node_id="n", ok=True, contract_hash="h", report=None,
                          files={"pkg/a.py": "x = 1\n" * n},
                          return_mode="full_source", budget_tokens=256)
        sizes.append(p.tokens)
    assert sizes[0] < sizes[1] < sizes[2]


# --- blame routing --------------------------------------------------------

def test_single_owner_routes_to_retry_child():
    c = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    r = verify_node(c, {"pkg/a.py": "x = 1\n"}, owner="a")
    plan = plan_repair(r, ["a", "b"], repair_rounds_used=0, max_repair_rounds=2,
                       decompose_retries_used=0, max_decompose_retries=2, can_escalate=False)
    assert plan.action is RepairAction.RETRY_CHILD
    assert plan.targets == ["a"]


def test_cross_node_routes_to_redecompose():
    parent = contract("root", provides=[])
    a = contract("a", provides=[sym("pkg.a.f")], requires=["pkg.b.g"], owns=["pkg/a.py"])
    b = contract("b", provides=[sym("pkg.b.g", path="pkg/b.py")], requires=["pkg.a.f"],
                 owns=["pkg/b.py"])
    from rlmc.contracts.types import CheckReport
    rep = CheckReport(node_id="root", checked=4)
    for v in check_composition(parent, [a, b]).violations:
        rep.add(v)
    plan = plan_repair(rep, ["a", "b"], repair_rounds_used=0, max_repair_rounds=2,
                       decompose_retries_used=0, max_decompose_retries=2, can_escalate=False)
    assert plan.action is RepairAction.REDECOMPOSE


def test_exhausted_budget_gives_up():
    c = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    r = verify_node(c, {"pkg/a.py": "x = 1\n"}, owner="a")
    plan = plan_repair(r, ["a"], repair_rounds_used=2, max_repair_rounds=2,
                       decompose_retries_used=2, max_decompose_retries=2, can_escalate=False)
    assert plan.action is RepairAction.GIVE_UP


def test_feedback_is_owner_tagged():
    c = contract("a", provides=[sym("pkg.a.f")], owns=["pkg/a.py"])
    r = verify_node(c, {"pkg/a.py": "x = 1\n"}, owner="a")
    plan = plan_repair(r, ["a"], repair_rounds_used=0, max_repair_rounds=2,
                       decompose_retries_used=0, max_decompose_retries=2, can_escalate=False)
    assert "a" in plan.feedback
    assert plan.feedback["a"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
