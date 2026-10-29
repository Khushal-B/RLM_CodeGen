# The Mechanism

How contract-recursive RLM works, why each piece is necessary, and what it
claims that no existing system claims.

Read this before the code. It is written to be read end to end.

---

## 1. The problem, stated precisely

A language model has a fixed workspace — the context window. A repository is
bigger than that workspace, but that turns out **not** to be the binding
problem.

NL2Repo-Bench measures what actually happens when a coding agent builds a
Python library from a requirements document:

| agent | average turns per task |
|---|---|
| GPT-5 | 78.4 |
| Claude Sonnet 4.5 | 181.6 |
| Kimi-k2 | 275.1 |

Every turn appends to the conversation: the file read, the code written, the
traceback, the diff. By turn 150 the model is reasoning through 150 turns of
accumulated debris. The reported failure modes are *premature termination*,
*loss of global coherence*, *fragile cross-file dependencies*, and *inadequate
planning over hundreds of steps*. GPT-5 fails to finish 84.5% of the time. The
best agent in the world reaches 40.2% average test pass rate.

**They are not failing because they write wrong code. They run out of room and
stop.**

So the quantity that overflows is not the artifact — a 4,000-LOC Hard task is
only ~40–50 K tokens and fits in a 128 K window. What overflows is the
**accumulated trajectory**, which grows as O(LOC × turns).

## 2. What RLM did, and the gap it left

Recursive Language Models (Zhang, Kraska & Khattab, arXiv:2512.24601) had one
idea: **the input should not pass through the neural network.**

The long document is stored as a variable in a live Python REPL. The root model
sees only its length and type. It writes code to slice the variable and calls
`llm_query(chunk, question)` to spawn a fresh sub-model on each slice. The
coordinator never reads a page. This handles 10 M-token inputs at ordinary cost.

**But RLM only ever reads.** Every task in that paper is long-input,
short-output question answering.

Nobody has applied the idea to the **output**. In every repository-generating
system — CodeTeam, Co-Coder, ChatDev, OpenHands, Claude Code — the generated
code flows back through some model's context: as a git diff, an interface brief,
a file listing, a re-read for integration, a traceback.

**The artifact is generated into the conversation. That is the gap.**

## 3. The claim

> A repository should be treated as an external object that the model
> constructs by reference and verifies by execution, never by reading.

The root model's context holds the specification, the contract graph, and
constant-size telemetry — and no generated source code. Sub-calls write into
files. Verification is compilation, type checking, import resolution and test
execution. The root receives booleans, counts and hashes.

## 4. Why contracts are forced, not chosen

This is the argument that makes the design a mechanism rather than a pipeline.

- **Splitting an input is easy.** Chunk 5 of a document can be processed without
  knowing anything about chunk 9. The chunks are independent.
- **Splitting an output is hard.** Module 5 must fit together with module 9.
  They call each other. *Something has to carry the constraint across sibling
  branches of the recursion.*

That something is the contract. If a parent could hand a child's source down to
its siblings, contracts would be unnecessary — but the invariant forbids exactly
that. So:

```
invariant (no source travels up)  ⟹  siblings cannot see each other's code
                                  ⟹  something must carry the interface
                                  ⟹  contracts
```

Contracts here are a **necessary consequence of recursing on the output side**,
not a hand-designed heuristic. That logical chain is the spine of the thesis.

## 5. The function

One function, invoked at every node of the build tree:

```python
rlm_build(node, contract, budget) -> BuildResult

# node     : a unit of work (repo | package | module | file | function)
# contract : the machine-checkable interface this node must expose and consume
# budget   : token/cost allowance, split among children on decomposition
#
# The node picks exactly ONE action:
#   IMPLEMENT - write the code for `node` in one RLM sub-call, into the workspace
#   DECOMPOSE - emit child nodes + child contracts, split the budget, recurse
#               (children run concurrently within a level)
#   ESCALATE  - retry at a stronger tier, same decomposition
#   RETURN    - node satisfied; return (ok, contract_hash, counts, diagnostics)
#
# INVARIANT: a node returns at most C tokens to its parent, where C is
#            constant in subtree size.
```

Implemented in [`rlmc/core/driver.py`](../rlmc/core/driver.py).

### Verification is the same traversal, backwards

A node is verified only after all its children verify. Verification at each node
is a deterministic check of that node's output against its contract: an AST walk
and an import-graph resolution. **There is no LLM judge anywhere in the loop.**

That matters for a specific reason — see §9.

## 6. The invariant, and why it is bounded rather than absolute

"No generated source code is ever returned up the tree" is unimplementable as an
absolute. Consider: module B's tests fail because module A returns a list where
B expected a generator. Something must look at what A actually does. Two escape
routes, both bad:

- let diagnostics carry the traceback → tracebacks contain source lines, so the
  invariant is violated quietly and the paper becomes dishonest;
- forbid it entirely → the repair loop has no information channel, and the
  method loses on exactly the integration failures it was designed to fix.

**The implementable form:** a node returns at most **C tokens** of payload, where
C is constant and independent of subtree size — specifically
`(ok, contract_hash, counts, ≤ C-token diagnostic)`.

This still yields the O(1) root-growth property, which is what the claim
actually needs. `C = 256` is pinned in
[`PREREGISTRATION.md`](../PREREGISTRATION.md) and must not drift upward.

### Why it is enforced in Python, not in a prompt

RLM's in-REPL `rlm_query` returns the child's **entire response string** into the
parent's REPL (`rlm/environments/local_repl.py`). Nothing stops the model from
printing it. If the bound were only an instruction in a system prompt, one
disobedient turn would silently void the experiment, and no reviewer should
believe the result.

So the truncation happens in [`rlmc/core/payload.py`](../rlmc/core/payload.py),
above the REPL, and every return is recorded. `bound_respected` in each run
record is a **measurement over every return in the tree**, not an assertion.

The filesystem half of the invariant is in
[`rlmc/core/workspace.py`](../rlmc/core/workspace.py): a node may write only
files its contract claims, and may read only files it wrote. A node asking for a
sibling's file gets:

```
refused: 'pkg/parser.py' belongs to another node. You are given sibling
contracts, not sibling source -- code against the declared interface.
```

Refusals are counted, so an arm trying to reach around the invariant shows up as
a number instead of quietly succeeding.

## 7. Contracts

A contract declares what a node **provides**, what it may **consume**, and which
files it **owns**.

| Boundary | Artifact | Automatic check | Status |
|---|---|---|---|
| module → module | `.pyi` type stubs | AST walk: symbol exists, right kind, matching arity and annotations | **implemented** |
| package → package | import graph + `__all__` | imports resolve, no cycles, no undeclared cross-boundary imports | **implemented** |
| service → service | OpenAPI spec | schema validates, generated client compiles | registered, not implemented |
| code → data | SQL DDL / ORM | migration applies, ORM matches DDL | registered, not implemented |

The last two are registered in
[`rlmc/contracts/registry.py`](../rlmc/contracts/registry.py) so the
CVR-by-boundary table and the blame router already have the right shape. An
unimplemented boundary contributes **0 to both** the CVR numerator and
denominator and emits a checker error — it can never look like a pass. They are
unimplemented because NL2Repo-Bench is 104 pure-Python libraries: no frontend,
no database, so they would produce no numbers yet.

A child is shown its siblings' contracts as `.pyi` stubs — the **interface**, not
the implementation:

```python
# --- jsonlines.jsonlines ---
class Reader:
    def read(type: type | None = ..., allow_none: bool = ...) -> Any: ...
class Writer:
    def write(obj: Any) -> int: ...
```

### Contract Violation Rate (CVR)

`violations / checked elements`, broken out by boundary type. Deterministic,
cheap, and **orthogonal to test-pass rate**.

It is a genuinely new measurement. DependEval asks whether a model can *recover*
a dependency graph that already exists. CVR asks whether generated code
*honours a graph the system declared for itself*.

### Composition — checked before any child runs

A proposed decomposition is rejected unless all four hold
([`rlmc/contracts/compose.py`](../rlmc/contracts/compose.py)):

| rule | meaning |
|---|---|
| **C1 Coverage** | every symbol the parent must provide is provided by exactly one child |
| **C2 Exclusivity** | no two children claim the same symbol or the same file |
| **C3 Closure** | every `requires` names a symbol some sibling `provides` |
| **C4 Acyclicity** | the sibling dependency graph is a DAG |

This costs nothing — it is pure Python over declarations, before any code
exists — and it catches the most expensive failure class early: a split that
could not possibly satisfy the parent no matter how well each child implements
its part.

## 8. Repair routing — credit assignment without reading code

Node N's contract check fails after children C1…Ck have been built. Something
must decide what to re-run. But the parent may not read any child's source.

CodeTeam solves the analogous problem by routing QA failures back to "the
responsible developer" via file ownership — but it can only do that because its
agents share file contents. Here they deliberately do not, so that solution is
unavailable.

**The contract graph is already an index from symbol to owning node, so blame is
a lookup, not an inference:**

```
violation names a symbol some child DECLARED
        -> that child promised it and did not deliver. Rebuild that child.

violation spans ≥2 children, or names a symbol nobody declared
        -> the DECOMPOSITION is wrong, not any single child. Re-decompose.
```

The ladder, in order
([`rlmc/core/blame.py`](../rlmc/core/blame.py)):

1. `RETRY_CHILD` — one child owns every violation
2. `RETRY_CHILDREN` — several children own disjoint violations
3. `REDECOMPOSE` — violations are cross-node; the contract split is wrong
4. `ESCALATE` — retries exhausted; same decomposition, stronger model tier
5. `GIVE_UP` — budget exhausted; return `ok=False` with a bounded note

The feedback sent back to a node is itself bounded to C tokens, so the repair
channel does not become a hole in the invariant.

This is the piece the thesis review calls *"simultaneously your biggest
under-specification and your most defensible novelty"* — nobody else has this
problem, because nobody else has externalised the output.

## 9. The counter-evidence, and the answer

RLM's own Table 1 contains the strongest argument against recursion. For
Qwen3-Coder-480B on LongBench-v2 CodeQA:

| recursion depth | score |
|---|---|
| 0 | 66.0 |
| 1 | 56.0 |
| 2 | 54.0 |
| 3 | 44.0 |

A 480-billion-parameter code model gets **monotonically worse** the more it
recurses. A committee member who reads the RLM paper will find this.

**The answer — and it is the motivation for this design, not a defence against
it:**

In plain RLM, a sub-call returns **unverified natural language** straight into
the parent's reasoning context. A child's mistake immediately contaminates the
parent's next decision, and the contamination compounds with every extra level.
That is exactly why deeper recursion makes it worse.

In this design, a child's output is **verified by execution against a
machine-checkable contract before the parent ever sees it**, and what the parent
receives is one bit plus a hash. Errors are *gated* at every recursion boundary
instead of flowing through it.

The RLM paper's negative result is therefore not evidence against this design —
it is the precise motivation for the two features that distinguish it:
deterministic verification, and the bounded return.

(Worth noting: the same table's GPT-5 block reads 58.0 / 62.0 / 66.0 / 58.0 —
depth helps up to two levels, then falls off. So the degradation is
model-specific, a capability threshold, not a property of recursion.)

## 10. Where this sits in the 2026 landscape

| System | Artifact out of context? | Adaptive depth? | Verification |
|---|---|---|---|
| RLM (2512.24601) | input yes, **output no** | no | none |
| RLM App. C.3 | no | no | prompted |
| CodeTeam (2606.22082) | no — git state, interface briefs, QA tracebacks | no (flat, file-level) | QA agent |
| Co-Coder (2606.00953) | no | no | tests |
| DecompRL (2607.02390) | no (single-problem, function-level) | learned | tests |
| ROMA (2602.01848) | partial — lossy LLM summary | yes | model judgement |
| **This work** | **input + output** | yes | **execution + static analysis** |

The nearest neighbours and what each cannot do:

- **CodeTeam** is this design at repository scale, published, on the same
  benchmark, scoring 42.3%. But its artifact still moves between agents through
  git state, interface briefs, file listings and QA tracebacks — every one of
  those puts generated code back into some model's context, and what flows grows
  with the repository. *This design never returns source up the tree at all.*
- **DecompRL** owns "learn the decomposition policy", but states its own scope
  limit: *"handles only single-problem decomposition into functions, not
  repository-scale or multi-file codebases."*
- **ROMA** is a recursive node loop whose Aggregator compresses child outputs to
  control context growth — the closest published thing. But its aggregation is
  **lossy LLM summarisation** ("may omit critical evidence or over-compress"),
  verified by **model judgement**, evaluated on web research and creative
  writing — never on code, never on an interdependent artifact. Here the return
  is a boolean plus a hash, **constant** in subtree size rather than merely
  smaller, and verified by execution. ROMA's own Limitations section names both
  of this work's contributions as *their* open problems.

## 11. The hypotheses

| | claim | ablation that could falsify it |
|---|---|---|
| **H1** | Keeping generated code out of the root's context improves repo-scale correctness at matched model and cost | identical scaffold, one change: children return full source instead of `(ok, hash, diagnostics)` |
| **H2** | A learned/adaptive depth policy beats any fixed depth, and the gain grows with target size | fixed depth ∈ {0,1,2,3} vs adaptive, per difficulty bucket |
| **H3** | Machine-checkable contracts reduce integration failure more than prose or none | typed `.pyi` vs free-text vs no contract; report CVR *and* pass rate |
| **H4** | A small model trained on execution reward beats larger untrained ones in the same scaffold | *out of scope this cycle* — ~150–200 A100-hours |

**H1 is the whole thesis.** It is the only property that distinguishes this from
a multi-agent scaffold: CodeTeam already has machine-checkable contracts,
Co-Coder already has principled decomposition, DecompRL already has a learned
decomposition policy. And the RLM paper never ran this experiment, because RLM
only ever did reading tasks.

If H1 holds, there is a paper. If H1 does not hold, no amount of work on H2/H3
rescues it. So H1 is tested first, and the pre-registered pivot if it fails is
**recursion-for-verification** — using RLM recursion to *check* a large
generated artifact, where the input is huge and the output is small.

## 12. Reframing H1 so it cannot be dismissed

An earlier framing said "the artifact you are producing is larger than what the
model can hold." That is **arithmetically false** at this benchmark's scale (a
Hard task is ~40–50 K tokens, comfortably inside a 128 K window) and its truth
flips depending on which model you pick. A mechanism must not have a premise
that depends on model choice.

The framing actually used:

> **Externalisation bounds root-context growth to constant in target repository
> size, and that is what prevents non-finish.**

Model-independent, directly measurable, and it maps onto the failure mode
NL2Repo-Bench actually reports. The consequence: the **root-context-vs-target-LOC
plot stops being a secondary figure and becomes the primary evidence.**

### The right statistic for that plot

The claim is **flatness**, so the statistic is the **least-squares slope**
(root-context tokens per target LOC), not Pearson *r*.

This matters concretely. An arm whose context creeps 900 → 940 tokens while the
target grows 400 → 2,100 LOC has *r* ≈ 0.92 — which looks like strong growth and
would understate H1 — while its slope is 0.017 tok/LOC, which is flat. The
harness reports slope and a scale-free growth ratio; `correlation()` is retained
only as a secondary diagnostic and documented as not being the claim.

## 13. What is measured

Written to `runs/<arm>/<task>/seed<n>/record.json`:

| metric | why |
|---|---|
| `pass_rate` | fraction of upstream tests passing. **Continuous, not binary** — at n=20 a binary outcome has SE ≈ 7.7 points and a real 5-point effect is invisible |
| `cvr`, `cvr_by_boundary` | the new metric, per boundary type |
| `root_context_tokens` | the H1 plot: flat for the externalised arm, growing for the control |
| `max_return_tokens`, `bound_respected` | the invariant, audited over every return |
| `completed` | did it emit an importable package. NL2Repo reports 49% early-stop / 84.5% non-finish; driving this to 0 is the likeliest early win |
| `cost_usd`, `wallclock_s` | so every table row is cost-matched |
| decomposition shape | node count, depth, branching — watches for DecompRL's documented *size collapse* |

Statistics: **paired** per task (never two independent groups), **bootstrap** 95%
CIs, **three seeds** with seed-to-seed variance reported. A failed run scores
**0** and is never dropped — dropping inflates whichever arm crashes more often.

---

## Further reading in this repo

- [`PREREGISTRATION.md`](../PREREGISTRATION.md) — the H1 success criterion, the
  constant C, and the pivot, all fixed before data
- [`docs/RUNNING.md`](RUNNING.md) — installation and every command
- [`README.md`](../README.md) — code map and design decisions
