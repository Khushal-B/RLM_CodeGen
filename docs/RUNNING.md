# Running the code

Everything you need to install, verify, run, and compare against baseline RLM.

Read [`docs/MECHANISM.md`](MECHANISM.md) first if you want to know *what* this
does. This document is purely operational.

---

## 1. Install

```bash
# 1. the unmodified RLM package (alexzhang13/rlm) -- NOT patched by us
git clone https://github.com/alexzhang13/rlm.git ../rlm_repo
pip install -e ../rlm_repo

# 2. this package
pip install -e .

# 3. credentials
cp .env.example .env      # then edit
```

`.env`:

```ini
OPENROUTER_API_KEY=sk-or-v1-...
NL2REPO_ROOT=../NL2RepoBench      # optional
```

`.env` is gitignored. Never commit it.

### Verify with no API key and no network

```bash
python -m rlmc.cli check      # self-tests the contract core
python -m pytest tests -q     # 99 tests
```

`check` should print:

```
Contract core OK.
  stub emission, AST verification, composition, schema parsing,
  bounded payload (35 tokens vs 7503 unbounded), blame routing.
```

If that passes, the entire deterministic half of the system works. Only the
model calls need credentials.

---

## 2. Choosing a backend

The backend is one flag. Everything else is identical.

| backend | flag | reports USD cost? | notes |
|---|---|---|---|
| **OpenRouter** | `--backend openrouter` | **yes** | **recommended.** ~3 s/call |
| vLLM (local A100) | `--backend vllm --base-url http://localhost:8000/v1` | no | for the open-weights runs |
| OpenAI | `--backend openai` | no | |
| NVIDIA NIM | `--backend nvidia` | no | free tier, but **~150–440 s per call** |

**Use OpenRouter.** It is the only backend whose usage payload carries real USD
cost (`rlm/clients/openai.py` reads `usage.cost`), which every cost-matched table
row depends on.

Measured 2 Sep 2026, same prompt:

| endpoint | latency |
|---|---|
| OpenRouter / `deepseek/deepseek-chat` | **3.5 s** |
| NVIDIA NIM / `deepseek-v4-flash` | 175 s (and 441 s for a 16 K-char planner prompt) |

NVIDIA's free tier also appears to serialise per account: running two arms with
concurrent children pushed every request past a 900 s timeout. If you must use
it, run arms **sequentially** with `--max-concurrent-children 1` and
`--request-timeout-s 1500`.

Model choice is orthogonal:

```bash
--model deepseek/deepseek-chat        # cheap, good default
--model anthropic/claude-sonnet-4.5   # frontier reference row
--model qwen/qwen3-coder              # open-weights comparison
```

---

## 3. The main experiment: H1

H1 is the go/no-go gate. Two arms, **identical in every respect except what a
child returns to its parent**.

```bash
./run_h1.sh                      # smoke suite, tinycsv
./run_h1.sh jsonlines nl2repo    # NL2Repo-Bench task
```

Or explicitly:

```bash
python -m rlmc.cli run --arm h1-bounded \
  --benchmark nl2repo --nl2repo-root ../NL2RepoBench --tasks jsonlines \
  --backend openrouter --model deepseek/deepseek-chat \
  --max-depth 1 --max-children 3 --max-concurrent-children 3 \
  --node-max-iterations 6 --root-max-iterations 8 \
  --seeds 0 1 2 --eval

python -m rlmc.cli run --arm h1-full-source ...same flags...

python -m rlmc.cli report --compare h1-bounded h1-full-source
```

That the arms differ in exactly one field is asserted by a test, so they cannot
silently drift apart:

```python
>>> a = ArmConfig.preset("h1-bounded").to_dict()
>>> b = ArmConfig.preset("h1-full-source").to_dict()
>>> {k for k in a if a[k] != b[k]}
{'name', 'return_mode'}
```

### Reading the report

```
arm                  n    pass  pass@1  compl    CVR     ctx   tok/LOC   growth    cost$
----------------------------------------------------------------------------------------
h1-bounded           4   0.457   0.000   1.00  0.050     919     0.017    0.011   0.0134
h1-full-source       4   0.333   0.000   1.00  0.120    6388     3.971    0.558   0.0210

Paired (h1-bounded - h1-full-source) on pass_rate: mean diff 0.125
  95% CI [0.041, 0.204]  n=12
  W/T/L: 9/1/2   -> h1-bounded is better than h1-full-source
```

| column | meaning |
|---|---|
| `pass` | mean fraction of upstream tests passing |
| `compl` | completion rate — did it emit an importable package |
| `CVR` | contract violation rate |
| `ctx` | mean root-context high-water mark, tokens |
| **`tok/LOC`** | **the H1 statistic.** Root-context tokens gained per target LOC. Flat (≈0) supports H1 |
| `growth` | scale-free version of the same thing |

**`tok/LOC` is the headline, not the correlation.** H1 claims flatness, and
Pearson *r* measures monotonicity — an arm creeping 900 → 940 tokens while the
target grows 400 → 2,100 LOC scores *r* ≈ 0.92 yet is flat.

Below ~5 paired tasks the reporter prints an explicit warning that the bootstrap
CI is not meaningful. Small runs are smoke tests of the machinery, not evidence.

---

## 4. Comparing against baseline RLM — swapping the API

This is the point of the common interface: **every arm produces the same
artifact** (a workspace directory holding an installable package) and the same
`RunRecord`, so one evaluator scores them all without knowing which system
produced them.

```bash
python -m rlmc.cli arms      # list every arm
```

| arm | what it is |
|---|---|
| `h1-bounded` | the proposed mechanism |
| `h1-full-source` | same scaffold, children return full source (the H1 control) |
| **`vanilla-rlm`** | **unmodified RLM**: one `rlm.RLM` with file tools, no contracts, no bounded returns, no decomposition discipline |
| `base-lm` | one plain completion, no REPL, no recursion — the lower bound |
| `h3-typed` / `h3-prose` / `h3-none` | typed `.pyi` vs English vs no contracts |
| `h2-depth0..3` / `h2-adaptive` | fixed vs adaptive recursion depth |

Run several arms in one command, on identical tasks and seeds:

```bash
python -m rlmc.cli run \
  --arms h1-bounded h1-full-source vanilla-rlm base-lm \
  --benchmark nl2repo --nl2repo-root ../NL2RepoBench --tasks jsonlines \
  --backend openrouter --model deepseek/deepseek-chat \
  --seeds 0 1 2 --eval

python -m rlmc.cli report --compare h1-bounded vanilla-rlm
```

**Swapping the model or provider is one flag** and applies identically to every
arm, so "same model, same sampling, same serving stack" is structural rather
than something to verify by reading two code paths:

```bash
# same experiment on a local A100
--backend vllm --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B

# same experiment with a frontier reference row
--backend openrouter --model anthropic/claude-sonnet-4.5
```

`vanilla-rlm` is the honest reference: it is the RLM paper's method applied to
generation. The difference between it and `h1-bounded` is what the contract
machinery actually buys.

---

## 5. NL2Repo-Bench

```bash
git clone https://github.com/multimodal-art-projection/NL2RepoBench ../NL2RepoBench
python -m rlmc.cli tasks --benchmark nl2repo --nl2repo-root ../NL2RepoBench
```

104 tasks load. Each is a directory under `test_files/` with `start.md` (the
requirements document), `test_case_count.txt`, `test_commands.json`,
`test_files.json`.

### The test suites are not in the clone

NL2Repo scores by running each task's **original upstream pytest suite**, which
lives in a per-task Docker image tagged `<task>:1.0`. The clone ships the
requirements documents but not the images.

**Option A — Docker (required for publishable numbers).** Pull
`docker.all-hands.dev/all-hands-ai/openhands:0.56` and
`runtime:0.56-nikolaik`, plus the per-task images, and use the benchmark's own
harness. This is the only route to numbers comparable with the published
leaderboard.

**Option B — reconstruct from PyPI (works today, no Docker).**

```bash
python -m rlmc.cli provision --nl2repo-root ../NL2RepoBench --tasks jsonlines
```

It fetches the project's pinned sdist and extracts its `tests/`, then reports
collected-vs-expected counts so version drift is visible:

| task | reconstructed | benchmark expects | verdict |
|---|---|---|---|
| **jsonlines** | **27** | **27** | exact — use this |
| databases | 96 | 154 | drift |
| google-images-download | 13 | 30 | drift |
| wsgidav | 3 | 36 | drift |
| arxiv-mcp-server | 384 | 23 | unusable |
| arguably, icecream, retrying, pylama, cachier, decouple | 0 | — | sdist ships no tests |

> **Scores from Option B are valid for comparing arms to each other** — both arms
> run the identical suite, which is all H1 needs. They are **not** comparable to
> NL2Repo's published leaderboard without reproducing under Docker. Say so in
> any table that uses them.

### Always validate a suite before trusting a zero

Install the real upstream package and score it as an oracle:

```bash
pip download jsonlines==4.0.0 --no-deps --no-binary :all: -d /tmp/o
tar xzf /tmp/o/jsonlines-4.0.0.tar.gz -C /tmp/o
python -c "from rlmc.bench.evaluate import evaluate_workspace as e; \
r=e('/tmp/o/jsonlines-4.0.0','../NL2RepoBench/test_files/jsonlines/tests'); \
print(r.tests_passed,'/',r.tests_total)"
```

On Windows this prints **23 / 27** — four tests reopen a `NamedTemporaryFile` by
name, which Windows forbids. **That is the ceiling on this machine**, and a
score must be read against 0.85, not 1.0. On Linux it should be 27/27.

---

## 6. The smoke suite

Three local tasks with the same shape as NL2Repo (requirements document → empty
workspace → installable package → real pytest suite the agent never sees):

| task | difficulty | target LOC | tests |
|---|---|---|---|
| `tinycsv` | easy | 250 | 29 |
| `eventbus` | medium | 600 | 31 |
| `miniorm` | hard | 1,100 | 40 |

```bash
./run_h1.sh tinycsv
python -m rlmc.cli run --arms h1-bounded h1-full-source --tasks tinycsv eventbus miniorm --eval
```

Each is deliberately **layered with cross-module dependencies**, so contracts
have something to do, and each has a reference implementation in `reference/`
that passes **100%** of its suite — so a 0 means the agent failed, not that the
task was impossible:

```bash
python -c "from rlmc.bench.evaluate import evaluate_workspace as e; \
[print(t, e('reference/'+t,'rlmc/bench/smoke/'+t+'/tests').pass_rate) \
 for t in ['tinycsv','eventbus','miniorm']]"
```

They are also useful for the H1 growth-slope plot specifically: the slope needs
points at **different target sizes**, and 250 → 1,100 LOC spans a real range.

---

## 7. Command reference

```bash
python -m rlmc.cli arms                 # list ablation arms
python -m rlmc.cli tasks [--benchmark nl2repo --nl2repo-root PATH]
python -m rlmc.cli check                # self-test, no API needed
python -m rlmc.cli provision --tasks ID # fetch upstream suites
python -m rlmc.cli run --arm ARM ...    # run
python -m rlmc.cli eval --runs runs/    # re-score existing workspaces
python -m rlmc.cli report --compare A B [--json out.json]
```

### Useful `run` flags

| flag | purpose |
|---|---|
| `--arms A B C` | several arms in one command |
| `--seeds 0 1 2` | repeat runs; needed for seed-variance reporting |
| `--eval` | score immediately after each run |
| `--skip-existing` | resume an interrupted sweep |
| `--dev-subset` | frozen stratified 5 easy / 8 medium / 7 hard |
| `--max-budget-usd 5` | hard spend ceiling |
| `--max-wallclock-s 2400` | per-run time ceiling |
| `--rpm-limit 60` | process-wide requests/min cap |
| `--max-retries 4` | retries on 429 / 5xx / timeout |
| `--diagnostic-budget-tokens 256` | **the constant C — do not raise it** |
| `--quiet` | suppress per-node progress lines |

### Output layout

```
runs/<arm>/<task>/seed<n>/
    record.json      every metric, the full build tree, all events
    workspace/       the generated package
```

---

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `OPENROUTER_API_KEY is not set` | create `.env` from `.env.example` |
| `the 'rlm' package is not importable` | `pip install -e ../rlm_repo` |
| runs hang for many minutes | slow endpoint. Check with the models endpoint first — it answers in ~2 s; if that is fast but completions hang, the model is queued. Prefer OpenRouter |
| every request times out under concurrency | the endpoint serialises per account. Use `--max-concurrent-children 1` and run arms sequentially |
| `pass_rate` is 0 for every run | check the suite is satisfiable — run the oracle (§5). Also check `eval_error` in `record.json` |
| `no tests collected` | the generated package did not import. `completed: false` in the record confirms it |
| CVR looks nondeterministic | should not happen; there are regression tests for the concurrency race that caused it. Please report |

### Cost

`deepseek/deepseek-chat` on OpenRouter is roughly **$0.01–0.05 per task-run** at
these settings. The full H1 experiment (20 tasks × 2 arms × 3 seeds = 120 runs)
is on the order of **$2–6**. Use `--max-budget-usd` as a hard stop.
