"""Parsing and validating the decomposition a model emits.

The model is asked for JSON, so this module is the airlock between "text a
language model produced" and "typed objects the deterministic core operates
on". It is deliberately forgiving about surface form (fenced blocks, prose
wrappers, single quotes) and strict about structure: anything that does not
parse into a well-formed set of child contracts is rejected with a short,
actionable message that goes back to the model on the retry.

RLM's Appendix B measured 16% of turns misusing ``FINAL()`` and 13% misusing
``FINAL_VAR()`` even with a 480B model, and needed a programmatic patching step.
The same lesson applies here: structured output from an LLM must be treated as
untrusted input, and repair must be cheap.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rlmc.contracts.types import Contract, Param, SymbolSpec

# The JSON contract shape shown to the model. Kept small on purpose -- every
# extra required field is another chance to fail parsing.
DECOMPOSITION_SCHEMA_DOC = """{
  "action": "DECOMPOSE" | "IMPLEMENT",
  "reason": "<one sentence>",
  "children": [
    {
      "node_id": "<short_snake_case_id>",
      "title": "<what this node builds>",
      "spec": "<the slice of the requirements this node is responsible for>",
      "owns_paths": ["pkg/mod.py", "pkg/other.py"],
      "provides": [
        {
          "qualname": "pkg.mod.function_name",
          "kind": "function" | "class" | "constant",
          "path": "pkg/mod.py",
          "params": [
            {"name": "x", "annotation": "int", "has_default": false}
          ],
          "returns": "list[str]",
          "doc": "<one line>",
          "methods": [
            {"qualname": "pkg.mod.Cls.method", "kind": "function",
             "path": "pkg/mod.py",
             "params": [{"name": "n", "annotation": "int"}],
             "returns": "None"}
          ]
        }
      ],
      "requires": ["pkg.other.symbol_from_a_sibling"]
    }
  ]
}"""


class SchemaError(ValueError):
    """Raised when a model's decomposition cannot be parsed into contracts."""


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Handles: bare JSON, ```json fences, ``` fences, and JSON embedded in prose.
    """
    if not text or not text.strip():
        raise SchemaError("empty response")

    candidates: list[str] = []

    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    candidates.extend(f.strip() for f in fenced)

    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)

    # Greedy brace match as a last resort.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    errors: list[str] = []
    for cand in candidates:
        for attempt in (cand, _repair_json(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError as e:
                errors.append(str(e))
    raise SchemaError(
        "no parseable JSON object found"
        + ((" (" + errors[0] + ")") if errors else "")
    )


def _repair_json(text: str) -> str:
    """Cheap fixes for the ways models commonly break JSON."""
    out = text.strip()
    out = re.sub(r",(\s*[}\]])", r"\1", out)          # trailing commas
    out = re.sub(r"\bTrue\b", "true", out)
    out = re.sub(r"\bFalse\b", "false", out)
    out = re.sub(r"\bNone\b", "null", out)
    return out


def parse_symbol(d: Any, default_path: str = "") -> SymbolSpec:
    if not isinstance(d, dict):
        raise SchemaError("symbol entry must be an object, got " + type(d).__name__)
    qualname = str(d.get("qualname") or d.get("name") or "").strip()
    if not qualname:
        raise SchemaError("symbol is missing 'qualname'")

    kind = str(d.get("kind") or "function").strip().lower()
    if kind not in ("function", "class", "constant"):
        kind = "function"

    params: list[Param] = []
    raw_params = d.get("params")
    if isinstance(raw_params, list):
        for p in raw_params:
            if isinstance(p, str):
                params.append(Param(name=p))
            elif isinstance(p, dict):
                params.append(
                    Param(
                        name=str(p.get("name", "")),
                        annotation=_opt_str(p.get("annotation") or p.get("type")),
                        has_default=bool(p.get("has_default") or p.get("optional")),
                        kind=_param_kind(str(p.get("kind") or "positional")),
                    )
                )
    elif isinstance(raw_params, str):
        params = _parse_signature_params(raw_params)

    # Some models emit "signature": "(x: int) -> str" instead of params/returns.
    returns = _opt_str(d.get("returns") or d.get("return_type"))
    sig = d.get("signature")
    if not params and isinstance(sig, str) and sig.strip():
        params = _parse_signature_params(sig)
        if returns is None and "->" in sig:
            returns = sig.split("->", 1)[1].strip().rstrip(":").strip() or None

    methods = [
        parse_symbol(m, default_path=str(d.get("path") or default_path))
        for m in (d.get("methods") or [])
        if isinstance(m, dict)
    ]

    return SymbolSpec(
        qualname=qualname,
        kind=kind,  # type: ignore[arg-type]
        path=str(d.get("path") or default_path),
        params=params,
        returns=returns,
        doc=_opt_str(d.get("doc")),
        methods=methods,
    )


def _param_kind(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("var_positional", "args", "*args", "vararg"):
        return "var_positional"
    if r in ("var_keyword", "kwargs", "**kwargs", "kwarg"):
        return "var_keyword"
    if r in ("keyword", "kwonly", "keyword_only"):
        return "keyword"
    return "positional"


def _parse_signature_params(sig: str) -> list[Param]:
    """Parse ``(a: int, b: str = 'x', *args, **kw)`` into Params."""
    m = re.search(r"\((.*)\)", sig, re.DOTALL)
    inner = m.group(1) if m else sig
    out: list[Param] = []
    for raw in _split_top_level(inner):
        tok = raw.strip()
        if not tok or tok in ("self", "cls", "/", "*"):
            continue
        kind = "positional"
        if tok.startswith("**"):
            kind, tok = "var_keyword", tok[2:]
        elif tok.startswith("*"):
            kind, tok = "var_positional", tok[1:]
        has_default = "=" in tok
        if has_default:
            tok = tok.split("=", 1)[0]
        if ":" in tok:
            name, ann = tok.split(":", 1)
            out.append(
                Param(
                    name=name.strip(),
                    annotation=ann.strip() or None,
                    has_default=has_default,
                    kind=kind,  # type: ignore[arg-type]
                )
            )
        else:
            out.append(
                Param(name=tok.strip(), has_default=has_default, kind=kind)  # type: ignore[arg-type]
            )
    return [p for p in out if p.name]


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside brackets."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth <= 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def parse_child_contract(d: Any, index: int) -> tuple[Contract, str, str]:
    """Parse one child entry. Returns ``(contract, title, spec)``."""
    if not isinstance(d, dict):
        raise SchemaError("child " + str(index) + " must be an object")

    node_id = str(d.get("node_id") or d.get("id") or "").strip()
    if not node_id:
        node_id = "child_" + str(index)
    node_id = re.sub(r"[^A-Za-z0-9_.-]", "_", node_id)[:48]

    owns = [str(p).replace("\\", "/").strip("/") for p in (d.get("owns_paths") or []) if p]
    default_path = owns[0] if owns else ""

    provides: list[SymbolSpec] = []
    for s in d.get("provides") or []:
        provides.append(parse_symbol(s, default_path=default_path))

    requires = [str(r).strip() for r in (d.get("requires") or []) if str(r).strip()]

    # A node that declares symbols but no files still needs somewhere to write.
    if not owns:
        owns = sorted({s.path for s in provides if s.path})

    contract = Contract(
        node_id=node_id,
        boundary="python_stub",
        provides=provides,
        requires=requires,
        owns_paths=owns,
        prose=_opt_str(d.get("prose") or d.get("interface_description")),
    )
    return contract, str(d.get("title") or node_id), str(d.get("spec") or "")


def parse_decomposition(text: str) -> dict[str, Any]:
    """Parse a full decomposition response.

    Returns ``{"action", "reason", "children": [(contract, title, spec), ...]}``.
    Raises ``SchemaError`` with a short, model-readable message on failure.
    """
    obj = extract_json(text)

    action = str(obj.get("action") or "").strip().upper()
    if action not in ("DECOMPOSE", "IMPLEMENT"):
        # Infer from shape rather than failing outright.
        action = "DECOMPOSE" if obj.get("children") else "IMPLEMENT"

    children: list[tuple[Contract, str, str]] = []
    if action == "DECOMPOSE":
        raw_children = obj.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise SchemaError("action is DECOMPOSE but 'children' is missing or empty")
        seen: set[str] = set()
        for i, c in enumerate(raw_children):
            contract, title, spec = parse_child_contract(c, i)
            # Node ids must be unique; the blame router keys on them.
            base = contract.node_id
            n = 2
            while contract.node_id in seen:
                contract.node_id = base + "_" + str(n)
                n += 1
            seen.add(contract.node_id)
            children.append((contract, title, spec))

    return {
        "action": action,
        "reason": str(obj.get("reason") or ""),
        "children": children,
    }
