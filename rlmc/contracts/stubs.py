"""``python_stub`` boundary: .pyi emission and deterministic AST verification.

Two directions:

* ``emit_stub`` renders a Contract's ``provides`` as a .pyi type stub. This is
  what a sibling is shown instead of source, and what is handed to a child as
  "the interface you must expose".
* ``check_stub_contract`` walks the AST of the files a node actually wrote and
  confirms every declared symbol exists with matching kind, arity and
  annotations. Each declared symbol is one CVR denominator element.

Everything here is deterministic and LLM-free. ``mypy --strict`` is available as
an optional second pass (``run_mypy``) but is never required -- the AST walk is
the checker of record, so results do not depend on whether mypy is installed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from rlmc.contracts.types import CheckReport, Contract, Param, SymbolSpec, Violation

# ---------------------------------------------------------------------------
# Emission: Contract -> .pyi text
# ---------------------------------------------------------------------------


def emit_stub(contract: Contract) -> str:
    """Render a contract's ``provides`` as .pyi stub text, grouped by module."""
    by_module: dict[str, list[SymbolSpec]] = {}
    for sym in sorted(contract.provides, key=lambda s: s.qualname):
        by_module.setdefault(sym.module or "<root>", []).append(sym)

    out: list[str] = []
    for module, syms in by_module.items():
        out.append("# --- " + module + " ---")
        for sym in syms:
            out.append(_emit_symbol(sym, indent=""))
        out.append("")
    return "\n".join(out).strip()


def _emit_symbol(sym: SymbolSpec, indent: str) -> str:
    if sym.kind == "class":
        lines = [indent + "class " + sym.name + ":"]
        if sym.doc:
            lines.append(indent + '    """' + _one_line(sym.doc) + '"""')
        if not sym.methods:
            lines.append(indent + "    ...")
        for m in sym.methods:
            lines.append(_emit_symbol(m, indent + "    "))
        return "\n".join(lines)
    if sym.kind == "constant":
        ann = ": " + sym.returns if sym.returns else ""
        return indent + sym.name + ann + " = ..."
    lines = [indent + "def " + sym.name + sym.signature() + ": ..."]
    if sym.doc:
        lines.insert(0, indent + "# " + _one_line(sym.doc))
    return "\n".join(lines)


def _one_line(text: str) -> str:
    return " ".join(text.split())[:200]


# ---------------------------------------------------------------------------
# Observation: source files -> what actually exists
# ---------------------------------------------------------------------------


class _ModuleIndex:
    """Index of the public symbols actually defined in a set of source files."""

    def __init__(self) -> None:
        # (module, name) -> ast node
        self.symbols: dict[tuple[str, str], ast.AST] = {}
        self.syntax_errors: dict[str, str] = {}   # path -> message
        self.modules_seen: set[str] = set()

    def add_file(self, module: str, path: str, source: str) -> None:
        self.modules_seen.add(module)
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self.syntax_errors[path] = "line " + str(e.lineno) + ": " + str(e.msg)
            return
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.symbols[(module, node.name)] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.symbols[(module, target.id)] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.symbols[(module, node.target.id)] = node

    def lookup(self, module: str, name: str) -> ast.AST | None:
        return self.symbols.get((module, name))


def build_index(files: dict[str, str]) -> _ModuleIndex:
    """Build a symbol index from ``{relative_path: source}``."""
    index = _ModuleIndex()
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        index.add_file(path_to_module(path), path, source)
    return index


def path_to_module(path: str) -> str:
    """``pkg/sub/mod.py`` -> ``pkg.sub.mod``; ``pkg/__init__.py`` -> ``pkg``."""
    p = path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.strip("/").replace("/", ".")


# ---------------------------------------------------------------------------
# Verification: declared vs observed
# ---------------------------------------------------------------------------


def check_stub_contract(
    contract: Contract,
    files: dict[str, str],
    owner: str | None = None,
) -> CheckReport:
    """Check that every symbol in ``contract.provides`` exists as declared.

    ``files`` maps workspace-relative path to source text. ``owner`` defaults to
    the contract's own node_id and is stamped onto each violation so the blame
    router can attribute it without inspecting source.
    """
    owner = owner or contract.node_id
    report = CheckReport(node_id=contract.node_id)
    index = build_index(files)

    for path, msg in index.syntax_errors.items():
        report.add(
            Violation(
                kind="syntax_error",
                boundary="python_stub",
                subject=path,
                detail=msg,
                owner=owner,
                path=path,
            )
        )

    for sym in contract.provides:
        # Each declared top-level symbol, plus each declared method, is one
        # checked element -- this is the CVR denominator.
        report.record("python_stub", 1 + len(sym.methods))
        _check_symbol(sym, index, report, owner)

    return report


def _check_symbol(
    sym: SymbolSpec,
    index: _ModuleIndex,
    report: CheckReport,
    owner: str,
) -> None:
    node = index.lookup(sym.module, sym.name)
    if node is None:
        report.add(
            Violation(
                kind="missing_symbol",
                boundary="python_stub",
                subject=sym.qualname,
                detail="declared " + sym.kind + " not defined in " + (sym.module or "<root>"),
                owner=owner,
                path=sym.path,
            )
        )
        # Methods of a missing class are unreachable; count them as missing too
        # so the denominator recorded above stays honest.
        for m in sym.methods:
            report.add(
                Violation(
                    kind="missing_method",
                    boundary="python_stub",
                    subject=sym.qualname + "." + m.name,
                    detail="owning class is missing",
                    owner=owner,
                    path=sym.path,
                )
            )
        return

    if sym.kind == "class":
        if not isinstance(node, ast.ClassDef):
            report.add(
                Violation(
                    kind="kind_mismatch",
                    boundary="python_stub",
                    subject=sym.qualname,
                    detail="declared class, found " + type(node).__name__,
                    owner=owner,
                    path=sym.path,
                )
            )
            return
        methods = {
            n.name: n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for m in sym.methods:
            found = methods.get(m.name)
            if found is None:
                report.add(
                    Violation(
                        kind="missing_method",
                        boundary="python_stub",
                        subject=sym.qualname + "." + m.name,
                        detail="declared method not defined on class",
                        owner=owner,
                        path=sym.path,
                    )
                )
                continue
            _check_callable(m, found, sym.qualname + "." + m.name, report, owner, sym.path,
                            skip_first=True)
        return

    if sym.kind == "constant":
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            report.add(
                Violation(
                    kind="kind_mismatch",
                    boundary="python_stub",
                    subject=sym.qualname,
                    detail="declared constant, found " + type(node).__name__,
                    owner=owner,
                    path=sym.path,
                )
            )
        return

    # function
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        report.add(
            Violation(
                kind="kind_mismatch",
                boundary="python_stub",
                subject=sym.qualname,
                detail="declared function, found " + type(node).__name__,
                owner=owner,
                path=sym.path,
            )
        )
        return
    _check_callable(sym, node, sym.qualname, report, owner, sym.path, skip_first=False)


def _check_callable(
    declared: SymbolSpec,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    subject: str,
    report: CheckReport,
    owner: str,
    path: str,
    skip_first: bool,
) -> None:
    """Compare declared arity/annotations against the observed def."""
    observed = observed_params(node, skip_first=skip_first)
    d_req, d_total, d_star, d_kw = declared.arity()
    o_req, o_total, o_star, o_kw = _arity_of(observed)

    # A callable that accepts *args/**kwargs can absorb any declared arity, so
    # only flag when the observed signature genuinely cannot be called as declared.
    absorbs = o_star or o_kw
    if not absorbs and (o_req > d_req or o_total < d_total):
        report.add(
            Violation(
                kind="arity_mismatch",
                boundary="python_stub",
                subject=subject,
                detail=(
                    "declared " + str(d_req) + " required / " + str(d_total) + " named, "
                    "found " + str(o_req) + " required / " + str(o_total) + " named"
                ),
                owner=owner,
                path=path,
            )
        )
        return

    # Annotation comparison is deliberately lenient: only flag when BOTH sides
    # carry an annotation and they differ after normalisation. A missing
    # annotation in the implementation is a style issue, not a contract breach.
    obs_by_name = {p.name: p for p in observed}
    for p in declared.params:
        if not p.annotation:
            continue
        got = obs_by_name.get(p.name)
        if got is None or not got.annotation:
            continue
        if _norm_annotation(got.annotation) != _norm_annotation(p.annotation):
            report.add(
                Violation(
                    kind="annotation_mismatch",
                    boundary="python_stub",
                    subject=subject + ":" + p.name,
                    detail="declared " + p.annotation + ", found " + got.annotation,
                    owner=owner,
                    path=path,
                )
            )
            return

    if declared.returns and node.returns is not None:
        got_ret = _unparse(node.returns)
        if _norm_annotation(got_ret) != _norm_annotation(declared.returns):
            report.add(
                Violation(
                    kind="annotation_mismatch",
                    boundary="python_stub",
                    subject=subject + ":return",
                    detail="declared " + declared.returns + ", found " + got_ret,
                    owner=owner,
                    path=path,
                )
            )


def observed_params(
    node: ast.FunctionDef | ast.AsyncFunctionDef, skip_first: bool = False
) -> list[Param]:
    """Extract parameters from a def, optionally dropping self/cls."""
    a = node.args
    out: list[Param] = []
    positional = list(a.posonlyargs) + list(a.args)
    if skip_first and positional and not _is_staticmethod(node):
        positional = positional[1:]
    n_defaults = len(a.defaults)
    n_pos = len(list(a.posonlyargs) + list(a.args))
    # Index of the first positional arg that carries a default, in the ORIGINAL list.
    first_default = n_pos - n_defaults
    offset = (len(list(a.posonlyargs) + list(a.args)) - len(positional))
    for i, arg in enumerate(positional):
        out.append(
            Param(
                name=arg.arg,
                annotation=_unparse(arg.annotation) if arg.annotation else None,
                has_default=(i + offset) >= first_default,
                kind="positional",
            )
        )
    if a.vararg is not None:
        out.append(Param(name=a.vararg.arg, kind="var_positional"))
    for i, arg in enumerate(a.kwonlyargs):
        default = a.kw_defaults[i] if i < len(a.kw_defaults) else None
        out.append(
            Param(
                name=arg.arg,
                annotation=_unparse(arg.annotation) if arg.annotation else None,
                has_default=default is not None,
                kind="keyword",
            )
        )
    if a.kwarg is not None:
        out.append(Param(name=a.kwarg.arg, kind="var_keyword"))
    return out


def _is_staticmethod(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "staticmethod":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "staticmethod":
            return True
    return False


def _arity_of(params: list[Param]) -> tuple[int, int, bool, bool]:
    named = [p for p in params if p.kind in ("positional", "keyword")]
    required = [p for p in named if not p.has_default]
    return (
        len(required),
        len(named),
        any(p.kind == "var_positional" for p in params),
        any(p.kind == "var_keyword" for p in params),
    )


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _norm_annotation(text: str) -> str:
    """Normalise an annotation for comparison.

    Collapses whitespace, strips quotes from forward references, and unifies the
    ``Optional[X]``/``X | None`` and ``List[x]``/``list[x]`` spellings so that a
    cosmetic difference is not reported as a contract violation.
    """
    t = " ".join(text.split()).replace('"', "").replace("'", "")
    t = t.replace(" ", "")
    lowered = {
        "List": "list",
        "Dict": "dict",
        "Set": "set",
        "Tuple": "tuple",
        "Type": "type",
        "FrozenSet": "frozenset",
    }
    for old, new in lowered.items():
        t = t.replace(old + "[", new + "[")
    if t.startswith("Optional[") and t.endswith("]"):
        t = t[len("Optional[") : -1] + "|None"
    if t.startswith("typing."):
        t = t[len("typing.") :]
    parts = sorted(t.split("|")) if "|" in t else [t]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Optional second pass
# ---------------------------------------------------------------------------


def run_mypy(workspace: Path, stub_path: Path | None = None, timeout: int = 120) -> tuple[bool, str]:
    """Run ``mypy --strict`` over a workspace. Advisory only.

    Returns ``(ok, output)``. If mypy is not installed the call reports
    ``(True, "mypy unavailable")`` so results never silently depend on the
    developer's environment -- the AST walk remains the checker of record.
    """
    cmd = [sys.executable, "-m", "mypy", "--strict", "--no-error-summary", str(workspace)]
    if stub_path is not None:
        cmd.append(str(stub_path))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(workspace)
        )
    except FileNotFoundError:
        return True, "mypy unavailable"
    except subprocess.TimeoutExpired:
        return False, "mypy timed out after " + str(timeout) + "s"
    if "No module named mypy" in (proc.stderr or ""):
        return True, "mypy unavailable"
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
