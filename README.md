# rlmc — contract-recursive RLM

**Recursive Language Models for Long-Horizon Code Synthesis: Generating
Repositories Without Reading Them**
Khushal Bhalia · MTech CSE, IIT Hyderabad · Guide: Prof. Sandipan

Implementation of `rlm_build(node, contract, budget)`: a recursive repository
builder in which **generated source code never travels up the recursion tree**,
constraints cross sibling branches as **machine-checkable contracts**, and every
recursion boundary is **gated on deterministic verification** rather than a
model's judgement.

Built on the official RLM repo (`alexzhang13/rlm`) **without modifying it**.
Each build node is a stock `rlm.RLM` instance; the contract system, the
bounded-return invariant and the recursion tree live in this package.

## Start here

| document | what it covers |
|---|---|
| **[docs/MECHANISM.md](docs/MECHANISM.md)** | **what this is and why it works** — the gap in RLM, why contracts are forced rather than chosen, the repair router, how it differs from CodeTeam / DecompRL / ROMA, and the four hypotheses |
| **[docs/RUNNING.md](docs/RUNNING.md)** | **how to run it** — install, backends, the H1 experiment, comparing against baseline RLM, NL2Repo-Bench, troubleshooting |
| [PREREGISTRATION.md](PREREGISTRATION.md) | the H1 success criterion, the constant C, and the pivot — fixed before any data |

```bash
pip install -e ../rlm_repo && pip install -e .
cp .env.example .env            # add OPENROUTER_API_KEY
python -m rlmc.cli check        # self-test, no API key needed
python -m pytest tests -q       # 99 tests
./run_h1.sh                     # the H1 ablation on the smoke suite
```

---

## Why it is a sibling package and not a fork

H1's ablation requires *"identical scaffold, one bit changed"*. Because both
arms are literally the same `RLM` class with different driver configuration,
that claim is structural rather than something a reviewer has to take on trust:

```python
>>> a = ArmConfig.preset("h1-bounded").to_dict()
>>> b = ArmConfig.preset("h1-full-source").to_dict()
>>> {k for k in a if a[k] != b[k]}
{'name', 'return_mode'}
```

That assertion is a test (`test_arms_differ_only_in_return_mode`), so the two
arms cannot silently drift apart.

---


## Before running anything: the pre-registration

[`PREREGISTRATION.md`](PREREGISTRATION.md) fixes, in advance, what counts as
H1 succeeding and what happens if it does not. It pins the bounded-return
constant **C = 256**, the paired-bootstrap criterion, the pivot to
recursion-for-verification if the criterion is not met, and the threats to
validity (reconstructed suites, platform-capped oracles, free-endpoint latency).

A pre-registered null result is a finding. A post-hoc one is a failed thesis.
Any change to C or to the criterion invalidates every run recorded before it.

---

## The mechanism

One function, invoked at every node of the build tree. The node picks exactly
one action:

| Action | Meaning |
|---|---|
| `IMPLEMENT` | Write this node's code in one RLM sub-call, into the workspace. |
| `DECOMPOSE` | Emit child nodes + child contracts, split the budget, recurse. Children run concurrently. |
| `ESCALATE` | Retry at a stronger tier, same decomposition. |
| `RETURN` | Node satisfied; return the bounded payload. |

**Invariant.** What a node returns to its parent is
`(ok, contract_hash, counts, ≤ C tokens of diagnostic)`, where `C` is constant
in subtree size. Enforced in Python (`rlmc/core/payload.py`), not requested in
a prompt — RLM's in-REPL `rlm_query` hands the child's whole response back into
the parent's REPL, so a prompt-level rule would be unenforceable and a reviewer
would be right to dismiss it. Every return is recorded, so
`bound_respected` in the run record is a measurement.

The invariant is **bounded, not absolute** (validation plan, Problem C):
"no source ever returns" leaves the repair loop with no information channel.
`C = 256` tokens is the pre-registered default and must not drift upward.

**Verification is the same traversal in reverse.** A node verifies only after
its children do, and every check is an AST walk or an import-graph resolution.
There is no LLM judge anywhere in the loop. This is the answer to the RLM
paper's depth-degradation result (Table 1: Qwen3-Coder-480B goes 66.0 → 44.0 as
depth rises): plain RLM feeds *unverified natural language* into the parent's
reasoning, so a child's error contaminates the parent immediately and compounds
with depth. Here every boundary is gated on execution and static analysis.

---

## Contracts

A contract declares what a node **provides**, what it may **consume**, and which
files it **owns**.

| Boundary | Artifact | Check | Status |
|---|---|---|---|
| module → module | `.pyi` type stubs | AST walk: symbol exists, kind, arity, annotations; `mypy --strict` optional | **implemented** |
| package → package | import graph + `__all__` | resolution, no cycles, no undeclared cross-boundary imports | **implemented** |
| service → service | OpenAPI | schema validation + client compiles | registered, not implemented |
| code → data | SQL DDL / ORM | migration applies, ORM matches DDL | registered, not implemented |

The last two are registered in `rlmc/contracts/registry.py` so the CVR-by-
boundary table and the blame router already have the right shape. An
unimplemented boundary contributes **0 to both** the CVR numerator and
denominator and emits a checker error — it can never look like a pass. They are
unimplemented because NL2Repo-Bench is 104 pure-Python libraries: no frontend,
no database, so they would produce no thesis numbers yet.

**Contract Violation Rate (CVR)** = violations / checked elements, broken out by
boundary type. Deterministic, cheap, orthogonal to test-pass rate.

### Composition — checked *before* any child runs

A decomposition is rejected unless:

- **C1 Coverage** — every symbol the parent provides is provided by exactly one child
- **C2 Exclusivity** — no two children claim the same symbol or file
- **C3 Closure** — every `requires` names a symbol some sibling `provides`
- **C4 Acyclicity** — the sibling dependency graph is a DAG

This is free (pure Python over declarations, no code exists yet) and it catches
the most expensive failure class early: a split that could not possibly satisfy
the parent however well each child implements its part.

---

## Repair routing — the answer to Problem D

Node N's check fails after children C1..Ck are built. Something must decide what
to re-run — **without reading any child's source.** CodeTeam routes QA failures
to "the responsible developer" via file ownership, but it can only do that
because its agents share file contents. We deliberately do not, so blame must
come from somewhere else.

**The contract graph is already an index from symbol to owning node, so blame is
a lookup, not an inference.**

```
violation names a symbol a child DECLARED   ->  that child failed to deliver it
violation spans ≥2 children, or names a     ->  the DECOMPOSITION is wrong,
symbol nobody declared                          not any single child
```

The ladder: `RETRY_CHILD` → `RETRY_CHILDREN` → `REDECOMPOSE` → `ESCALATE` →
`GIVE_UP`. Repair feedback passed back to a node is itself bounded to `C`
tokens, so the repair channel does not become a hole in the invariant.

---

## The ablation arms

Every hypothesis is a config field, so an experiment is a config diff.

```bash
python -m rlmc.cli arms
```

| Axis | Field | Arms |
|---|---|---|
| **H1** artifact externalisation | `return_mode` | `h1-bounded`, `h1-full-source` |
| **H2** adaptive depth | `depth_mode`, `max_depth` | `h2-adaptive`, `h2-depth0..3` |
| **H3** contracts as interface | `contract_mode` | `h3-typed`, `h3-prose`, `h3-none` |
| baselines | — | `vanilla-rlm` (paper's method, no contracts), `base-lm` (single call) |

---

## Running

```bash
# H1, both arms, all three smoke tasks, scored immediately
python -m rlmc.cli run \
  --arms h1-bounded h1-full-source \
  --model deepseek/deepseek-chat \
  --seeds 0 1 2 \
  --eval

# the paired comparison
python -m rlmc.cli report --compare h1-bounded h1-full-source
```

Useful flags: `--backend vllm --base-url http://localhost:8000/v1` for the A100,
`--max-budget-usd` to cap spend, `--skip-existing` to resume, `--dev-subset` for
the frozen 5 easy / 8 medium / 7 hard stratification.

OpenRouter is the default backend because it is the only one whose usage payload
carries real USD cost (`rlm/clients/openai.py` reads `usage.cost`), which the
cost-matched table needs.

---

## What gets measured

Written to `runs/<arm>/<task>/seed<n>/record.json`:

- **`pass_rate`** — fraction of upstream tests passing. Continuous, not binary:
  at n=20 with a binary outcome the standard error is ~7.7 points and a real
  5-point effect is invisible (validation plan, Problem B).
- **`cvr`**, **`cvr_by_boundary`** — the new metric.
- **`root_context_tokens`** — the H1 plot. Should be flat in target LOC for the
  externalised arm and grow for the control. `report` prints the correlation.
- **`max_return_tokens`** / **`bound_respected`** — the invariant, audited.
- **`completed`** — did it produce an importable package. NL2Repo-Bench reports
  49% early-stop and 84.5% non-finish for some models; driving this to 0 is the
  most likely early win.
- **`cost_usd`**, **`wallclock_s`**, decomposition shape.

`report` gives paired bootstrap 95% CIs and seed-to-seed variance.

---

## Benchmarks

**Smoke suite** (`rlmc/bench/smoke/`) — three tasks shaped exactly like
NL2Repo: one requirements document, an empty workspace, a real pytest suite the
agent never sees.

| task | difficulty | target LOC | tests |
|---|---|---|---|
| `tinycsv` | easy | 250 | 29 |
| `eventbus` | medium | 600 | 31 |
| `miniorm` | hard | 1100 | 40 |

Each is deliberately **layered with cross-module dependencies**, so contracts
have something to do. Each has a reference implementation in `reference/` that
**passes 100% of its suite** — so a score of 0 means the agent failed, not that
the task is impossible. Re-verify any time:

```bash
python -c "from rlmc.bench.evaluate import evaluate_workspace as e; \
print(e('reference/miniorm','rlmc/bench/smoke/miniorm/tests').pass_rate)"
```

**NL2Repo-Bench** — clone it, then `--benchmark nl2repo --nl2repo-root <path>`.
The loader is tolerant about field naming across revisions. Note that the
headline table should ultimately be produced with the benchmark's own Docker
harness; local evaluation is right for development and for H1 (both arms scored
identically), not for numbers compared against published ones.

---

## Layout

```
rlmc/
  config.py              ArmConfig — every hypothesis as a field
  contracts/
    types.py             Contract, SymbolSpec, Violation, CheckReport
    stubs.py             .pyi emission + AST verification
    imports.py           import graph, ownership, cycles
    compose.py           C1-C4, checked before children run
    checker.py           verify_node / verify_subtree
    schema.py            parsing model-emitted decompositions
    registry.py          boundary -> checker
  core/
    driver.py            rlm_build: the recursion
    payload.py           the bounded-return invariant  <- H1 lives here
    blame.py             credit assignment  <- Problem D
    workspace.py         per-node file ownership
    tools.py             the REPL tools a node gets
    prompts.py           per-contract-mode prompts  <- H3 lives here
  agents/
    backend.py           the one place that talks to a model
    contract_rlm.py      the contract arm
    vanilla_rlm.py       baselines
  bench/                 tasks, smoke suite, evaluation
  metrics/               telemetry, paired bootstrap, reporting
  cli.py                 the common interface
reference/               reference impls proving the smoke tasks are solvable
tests/                   81 tests, no network required
```

---

## Design decisions worth knowing

**Node IDs are parent-qualified** (`root.parser.tokens`). Workspace ownership
and blame routing key on `node_id` across the whole tree; the model only
guarantees uniqueness within a level, so unqualified IDs at different depths
would silently share a file grant and cross-attribute violations.

**Annotation checking is lenient.** A mismatch is flagged only when *both* sides
carry an annotation and they differ after normalising `Optional[X]` / `X | None`
and `List` / `list`. A missing annotation in an implementation is a style issue,
not an interface breach — counting it would inflate CVR with noise.

**Parent verification checks only the residual.** Re-checking the union of all
files at the parent would double-count every child violation and inflate CVR by
roughly the branching factor.

**Harness-created `__init__.py` files are marked as scaffolding**, excluded from
a node's product, and created under a lock — siblings run concurrently and would
otherwise race to create the same file and both be charged for it.

**A failed run scores 0; it is never dropped.** Dropping failures inflates
whichever arm crashes more often.

---

## Running against NL2Repo-Bench

```bash
git clone https://github.com/multimodal-art-projection/NL2RepoBench ../NL2RepoBench
python -m rlmc.cli tasks --benchmark nl2repo --nl2repo-root ../NL2RepoBench
```

104 tasks load. Each is a directory under `test_files/` holding `start.md` (the
requirements document), `test_case_count.txt`, `test_commands.json` and
`test_files.json`.

### The suites are not in the clone

NL2Repo scores by running each task's **original upstream pytest suite**, which
lives in a per-task Docker image tagged `<task>:1.0`. The clone ships the
requirements documents but not the images. Two ways forward:

**With Docker** — the official harness. Reproduces published numbers.

**Without Docker** — `provision` reconstructs a suite from the project's PyPI
sdist, which is the same upstream source the image was built from:

```bash
python -m rlmc.cli provision --nl2repo-root ../NL2RepoBench --tasks jsonlines
```

It reports collected-vs-expected test counts so version drift is visible:

| task | collected | benchmark expects | verdict |
|---|---|---|---|
| `jsonlines` | 27 | 27 | exact match |
| `bleach` | 452 | 448 | close |
| `records` | 8 | 31 | heavy drift |
| `pyperclip` | 92 | 2 | heavy drift |

**These scores are valid for comparing arms to each other** — both arms run the
identical suite, which is all H1 needs. They are **not** comparable to NL2Repo's
published leaderboard without reproducing under the official Docker harness.
Prefer tasks whose counts match.

Validate a suite is satisfiable before trusting a 0, exactly as with the smoke
tasks — install the real upstream package and score it as an oracle. For
`jsonlines` on Windows the oracle is **23/27**: four tests use
`NamedTemporaryFile` in a way Windows forbids. That is the ceiling on this
machine, and a score should be read against it rather than against 1.0.

### Free / slow endpoints (NVIDIA NIM)

```bash
python -m rlmc.cli run --arm h1-bounded \
  --benchmark nl2repo --nl2repo-root ../NL2RepoBench --tasks jsonlines \
  --backend nvidia --model deepseek-ai/deepseek-v4-flash-0731 \
  --thinking off --rpm-limit 18 --request-timeout-s 900 --max-retries 6 \
  --max-depth 1 --max-concurrent-children 3 \
  --node-max-iterations 6 --root-max-iterations 8 --eval
```

`run_h1.sh <arm> --tasks <task>` wraps exactly this.

`rlmc/agents/ratelimit.py` installs a **process-wide** sliding-window limiter
and a retry policy by patching `OpenAIClient.completion`. Patching the class
rather than wrapping our own client object is deliberate: `rlm.RLM` constructs
clients internally per completion and per sub-call, so a wrapper on the object
we happen to hold would leave most calls ungoverned.

Measured on the NVIDIA free tier: roughly **175 s per call**, with some requests
timing out at 300 s and succeeding on retry. Latency, not the 40 rpm cap, is the
binding constraint — so concurrency across children is what buys wall-clock, and
timeouts must be generous. Note that this endpoint reports **no USD cost**, so
`cost_usd` stays 0; use token counts for cost-matching, or OpenRouter when a real
cost column is needed.
