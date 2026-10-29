"""System prompts for build nodes.

Three prompt families, matching the three H3 arms, so the arms differ in what
interface information a node receives and in nothing else:

  typed  -- siblings are described by .pyi stubs the node must code against
  prose  -- siblings are described in English
  none   -- siblings are not described at all

The decomposition prompt is shared by all three; only the contract detail it
asks for changes. Keeping one template with mode-conditioned sections (rather
than three separate prompts) is what stops incidental wording differences from
contaminating the H3 comparison.
"""

from __future__ import annotations

from rlmc.contracts.schema import DECOMPOSITION_SCHEMA_DOC
from rlmc.contracts.stubs import emit_stub
from rlmc.contracts.types import Contract

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------

_BASE = """You are one node in a recursive repository builder.

You are building a Python package into a workspace on disk. You interact with
the workspace ONLY through the tools listed below -- you cannot see files that
belong to other nodes, and other nodes cannot see yours. This is deliberate.

Available in the REPL (write ```repl blocks; the REPL persists across turns):
- `spec`: the slice of the requirements document you are responsible for.
- `write_file(path, content) -> str`: write a file you own. Refuses paths you
  do not own.
- `read_file(path) -> str`: read back a file YOU wrote. Refuses other nodes' files.
- `list_my_files() -> str`: paths you have written so far.
- `check_contract() -> str`: run the deterministic contract checker over what
  you have written and get back the violations. No LLM is involved. Call this
  BEFORE you finish -- it is the same checker that decides whether your work is
  accepted.
- `run_python(code) -> str`: execute a snippet against the workspace (imports,
  smoke tests). Use it to confirm your code actually runs.
- `llm_query(prompt)` / `llm_query_batched(prompts)`: sub-LLM calls for drafting
  or reviewing a chunk of content.
- `SHOW_VARS()`, `answer`: as usual. Submit by setting `answer["content"]` and
  `answer["ready"] = True`.

Work programmatically. Write files with `write_file`, verify with
`check_contract` and `run_python`, and only then finish. Do not paste large file
contents into your own reasoning -- build them in REPL variables and write them.
"""

_IMPLEMENT_TAIL = """
Your job at this node is to IMPLEMENT: write real, working code for every file
you own, satisfying the contract exactly.

Rules:
1. Every symbol in YOUR CONTRACT must exist with exactly the declared name,
   kind, parameters and return type. A missing or misnamed symbol is a failure
   even if the code is otherwise good.
2. Write complete implementations. No `pass`, no `TODO`, no `NotImplementedError`
   in a symbol your contract promises.
3. When you need something from a sibling, import it and code against the
   declared interface. Do not reimplement it, and do not try to read its file.
4. Call `check_contract()` and fix everything it reports before you finish.
5. Set `answer["ready"] = True` only after `check_contract()` reports no
   violations, or you have exhausted your turns.
"""


def build_node_system_prompt(
    contract: Contract,
    sibling_contracts: list[Contract],
    contract_mode: str,
    node_title: str,
    depth: int,
    is_root: bool = False,
) -> str:
    """System prompt for a node that is going to IMPLEMENT."""
    parts = [_BASE]

    parts.append("\n## This node\n")
    parts.append("id: " + contract.node_id + "\ntitle: " + (node_title or contract.node_id))
    parts.append("depth: " + str(depth) + ("  (root)" if is_root else ""))

    parts.append("\n## Your contract\n")
    parts.append(_render_own_contract(contract, contract_mode))

    sib = _render_siblings(sibling_contracts, contract_mode)
    if sib:
        parts.append("\n## Sibling interfaces (you may import these; you cannot read their code)\n")
        parts.append(sib)

    parts.append(_IMPLEMENT_TAIL)
    return "\n".join(parts)


def _render_own_contract(contract: Contract, mode: str) -> str:
    if mode == "none":
        return (
            "Files you own:\n"
            + "\n".join("- " + p for p in contract.owns_paths)
            + "\n\nNo interface has been specified for you. Decide the public API "
            "yourself from the spec."
        )
    if mode == "prose":
        body = contract.prose or "(no interface description was provided)"
        return (
            "Files you own:\n"
            + "\n".join("- " + p for p in contract.owns_paths)
            + "\n\nInterface you must expose, in words:\n"
            + body
        )
    # typed
    stub = emit_stub(contract) or "(no symbols declared)"
    out = [
        "Files you own:",
        "\n".join("- " + p for p in contract.owns_paths) or "- (none listed)",
        "",
        "You MUST expose exactly this interface (.pyi form):",
        "```python",
        stub,
        "```",
    ]
    if contract.requires:
        out += [
            "",
            "You may import these symbols from siblings:",
            "\n".join("- " + r for r in contract.requires),
        ]
    return "\n".join(out)


def _render_siblings(siblings: list[Contract], mode: str) -> str:
    if mode == "none" or not siblings:
        return ""
    blocks: list[str] = []
    for s in siblings:
        if mode == "prose":
            desc = s.prose or "(no description)"
            blocks.append("### " + s.node_id + "\n" + desc)
        else:
            stub = emit_stub(s)
            if not stub:
                continue
            blocks.append("### " + s.node_id + "\n```python\n" + stub + "\n```")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_BASE = """You are the planner for one node of a recursive repository builder.

Decide ONE of two actions for this node:

- IMPLEMENT: this node is small enough that a single focused agent should write
  all of its code directly.
- DECOMPOSE: this node is large enough that it should be split into {lo}-{hi}
  independent child nodes, each with its own contract, built in parallel by
  separate agents that CANNOT see each other's code.

Because children cannot see each other's code, the contracts you write are the
ONLY thing holding the pieces together. A child will code strictly against the
interfaces you declare. If two children need to interoperate and you did not
declare the interface between them, they will not fit together.

Rules for a good decomposition:
1. Split along interface seams (data model / core logic / IO / CLI), not by
   arbitrary line count.
2. Every symbol THIS node must provide has to be provided by exactly one child.
3. If child A calls into child B, put the exact symbol in A's `requires` and in
   B's `provides`, spelled identically.
4. No two children may own the same file.
5. The child dependency graph must be acyclic.
6. Declare full signatures: parameter names, annotations, defaults, return types.
   These are machine-checked against the code that gets written.

Prefer IMPLEMENT when the node is a single cohesive module (roughly under 300
lines). Splitting a small node costs more than it saves.

Respond with ONE JSON object and nothing else:

{schema}
"""

_DECOMPOSE_PROSE = """
NOTE: for this run, declare interfaces as ENGLISH DESCRIPTIONS instead of typed
signatures. For each child use the field "prose" with a paragraph describing
what it exposes and what it consumes. Omit "provides" and "requires".
"""

_DECOMPOSE_NONE = """
NOTE: for this run, do NOT declare interfaces between children. For each child
give only node_id, title, spec and owns_paths.
"""


def build_decompose_prompt(
    contract: Contract,
    contract_mode: str,
    min_children: int,
    max_children: int,
    depth_remaining: int,
    force_action: str | None = None,
) -> str:
    """System prompt for the plan/decompose decision at a node."""
    body = _DECOMPOSE_BASE.format(
        lo=min_children, hi=max_children, schema=DECOMPOSITION_SCHEMA_DOC
    )
    if contract_mode == "prose":
        body += _DECOMPOSE_PROSE
    elif contract_mode == "none":
        body += _DECOMPOSE_NONE

    if depth_remaining <= 0:
        body += (
            "\nThis node has NO recursion budget left, so you must answer with "
            'action "IMPLEMENT".\n'
        )
    if force_action:
        body += (
            "\nFor this run the action is fixed: you must answer with action \""
            + force_action
            + '".\n'
        )

    if contract_mode == "typed" and contract.provides:
        body += (
            "\n## This node's own contract\n"
            "The children's contracts together must cover exactly these symbols:\n"
            "```python\n" + emit_stub(contract) + "\n```\n"
        )
    if contract.owns_paths:
        body += (
            "\n## Files in this node's scope\n"
            + "\n".join("- " + p for p in contract.owns_paths)
            + "\n"
        )
    return body


def build_repair_prompt(briefs: list[str], contract_mode: str) -> str:
    """The bounded feedback a node receives when it is asked to repair.

    Only violation briefs are passed -- never source, never a traceback with
    code in it. This keeps the repair channel inside the same bound as every
    other return, which is what makes the invariant survive contact with the
    repair loop (Problem C).
    """
    head = (
        "Your previous attempt did not satisfy the contract. The deterministic "
        "checker reported:\n\n"
    )
    body = "\n".join("- " + b for b in briefs)
    tail = (
        "\n\nFix exactly these. Rewrite the affected files with `write_file`, then "
        "call `check_contract()` again and confirm it is clean before finishing."
    )
    if contract_mode == "none":
        tail = (
            "\n\nFix exactly these, then call `check_contract()` again before "
            "finishing."
        )
    return head + body + tail


def build_redecompose_prompt(summary: str) -> str:
    """Feedback when the composition check rejected a proposed decomposition."""
    return (
        "Your proposed decomposition does not compose. The checker reported:\n\n"
        + summary
        + "\n\nProduce a corrected decomposition. Remember: every symbol this node "
        "must provide has to be provided by exactly one child; every `requires` "
        "must name a symbol some sibling `provides`, spelled identically; no two "
        "children may own the same file; and the dependency graph must be acyclic."
    )
