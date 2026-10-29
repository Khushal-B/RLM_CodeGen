# Pre-registration — H1

**Recursive Language Models for Long-Horizon Code Synthesis**
Khushal Bhalia · MTech CSE, IIT Hyderabad · Guide: Prof. Sandipan

**Written:** 2 September 2026, before any scored run.
**Status:** infrastructure complete and self-tested; no H1 data collected yet.

Why this file exists: if the success criterion is chosen after seeing the data,
the natural human move is to report the slice that looks best — the difficulty
bucket that won, the seed that worked. Writing it down first is what converts a
null result from "the student's method did not work" into a publishable finding
that closes a question three separate 2026 papers left open.

---

## 1. The claim under test

**H1 — artifact externalisation.** Keeping generated source code out of the root
model's context improves repository-scale correctness, at matched model and
matched cost.

Stated in the form that is actually measurable and model-independent:

> Externalisation bounds root-context growth to constant in target repository
> size, and that is what prevents non-finish.

This framing replaces "the artifact is larger than the window", which is
arithmetically false at NL2Repo-Bench's scale (a Hard task is ~4,000 LOC ≈
40–50 K tokens, comfortably inside a 128 K window) and whose truth would flip
with model choice. What actually overflows is the accumulated trajectory —
artifact × re-reads × tracebacks × diffs × turns — which grows as O(LOC × turns),
and NL2Repo-Bench reports 78–275 turns per task.

## 2. The two arms

Identical in every respect but one. Enforced by construction, not by review:
both arms are the same `rlm.RLM` class driven by the same `BuildDriver`, and the
difference is a single `ArmConfig` field.

| | arm A | arm B |
|---|---|---|
| preset | `h1-bounded` | `h1-full-source` |
| a child returns | `(ok, contract_hash, counts, ≤ C tokens)` | its complete source |
| everything else | identical | identical |

Verified mechanically by `test_arms_differ_only_in_return_mode`, which asserts
the two configs differ in exactly `{name, return_mode}`.

Held constant across arms: root model, sampling parameters, seeds, task list,
decomposition prompt, contract mode (`typed`), depth mode (`adaptive`), budgets,
serving stack.

## 3. The bounded-return constant C

**C = 256 tokens.** Locked now. Chosen to fit an `ok` flag, a 16-character
contract hash, four counters, and roughly 6–8 one-line violation briefs — enough
for the repair router to act on, far too little to carry source.

C **must not drift upward** during development. Raising it weakens H1 directly,
and the failure mode this constant guards against is exactly the slow creep
where diagnostics grow a little each week until the invariant is meaningless.
Any change to C invalidates every run recorded before it.

The invariant is **bounded, not absolute**. "No source ever returns" is
unimplementable: the repair loop would have no information channel and the
method would lose on precisely the integration failures it exists to fix. The
bounded form still yields the O(1) root-growth property, which is what H1 claims.

Audited, not asserted: every return anywhere in the tree is recorded, and
`bound_respected` in each run record is a measurement over all of them.

## 4. The criterion

> **H1 is supported if and only if:**
>
> 1. On the pooled **Medium+Hard** subset of the frozen development set, the
>    externalised arm exceeds the source-returning arm in **paired mean test
>    pass rate**, with a bootstrap 95% confidence interval on the paired
>    difference that **excludes zero**; **and**
> 2. the **root-context growth slope** is ≈ 0 for the externalised arm while
>    being clearly positive for the source-returning control.
>
> Both arms use the same root model, decomposition, budget, task list and three
> seeds. The only difference is what a child returns to its parent.

**If the criterion is not met**, the pre-agreed pivot is
**recursion-for-verification**: using RLM recursion to check a large generated
artifact (whole-repository invariant checking, cross-file consistency auditing,
repo-scale security analysis), where the input is huge and the output is small.
Lower ceiling, substantially lower risk, and it plays to RLM's proven strength.

## 5. Statistics — fixed in advance

Chosen because a binary outcome at n=20, p≈0.25 has a standard error of ~7.7
percentage points, which would make a real 5-point effect invisible.

1. **Continuous** average test pass rate, never binary success.
2. **Paired** design: arms compared per task, never as two independent groups.
   Runs of the same task under different seeds are averaged before pairing, so
   each task contributes exactly one difference.
3. **Bootstrap** 95% CIs on the paired difference — no distributional
   assumption, because pass rates are bounded and lumpy at this sample size.
4. **Three seeds** per arm, with seed-to-seed variance reported alongside the
   effect. If the effect is smaller than the seed variance, there is no effect.
5. **Pooled Medium+Hard** for the headline test; Easy is used only to look for
   the format-tax crossover, never to argue the main claim.

Below ~10 paired tasks the reporter prints an explicit warning that the CI is
not meaningful. Small-n runs are smoke tests of the machinery, not evidence.

## 6. The growth statistic, and a correction

The H1 flatness claim is measured by the **least-squares slope of root-context
tokens against target LOC** (`tok/LOC`), plus a scale-free growth ratio.

This corrects an error in an earlier draft of the harness, which reported
**Pearson r**. Correlation is the wrong statistic here: it measures
monotonicity, not flatness. An arm whose root context creeps from 900 to 940
tokens while the target grows from 400 to 2,100 LOC scores r ≈ 0.92 — and is,
for every practical purpose, flat. Reporting r would have understated H1.
`correlation()` is retained only as a secondary diagnostic and is documented as
not being the claim.

## 7. Reporting protocol

Every run records, without exception:

1. exact instance IDs evaluated;
2. context-length histogram of that subset;
3. truncation policy when an instance exceeds the window;
4. whether OOM/failed instances are scored 0 or dropped — **they are scored 0**,
   never dropped, because dropping inflates whichever arm crashes more often;
5. number of seeds and seed-to-seed variance;
6. the paired difference and its bootstrap CI;
7. the serving stack (backend, model, quantisation, endpoint), so a stack
   difference can never be silently confounded with the effect.

## 8. Scope declared in advance

**H4 (trainability) is out of scope for this cycle.** The RLM paper's recipe
used 48 H100-hours ≈ 150–200 A100-hours, plus Docker test suites inside the RL
reward loop. It is not affordable on one shared A100 in 2–3 months. It becomes
an honest future-work paragraph. "Untrained small model fails as an RLM root" is
an expected, reportable intermediate finding — not a blocker.

**H2 and H3 are conditional on H1 showing signal.** If H1 fails, the remaining
hypotheses are published elsewhere (CodeTeam has machine-checkable contracts,
Co-Coder has principled decomposition, DecompRL has a learned decomposition
policy) and no amount of work on them rescues the thesis.

## 9. Known threats to validity, declared now

- **Scores obtained without the official Docker harness are not comparable to
  NL2Repo-Bench's published leaderboard.** Where suites are reconstructed from
  PyPI sdists, both arms run the identical suite — valid for the paired
  comparison, invalid for any absolute claim. Version drift between the
  reconstructed suite and the benchmark's frozen revision is recorded per task.
- **Platform-specific test failures cap the achievable score.** On Windows the
  `jsonlines` oracle — the real upstream package — scores 23/27, because four
  tests reopen a `NamedTemporaryFile` by name. A score must be read against that
  ceiling, not against 1.0.
- **A free endpoint's latency is not a neutral variable.** At ~150 s per call,
  wall-clock limits and retry policy can truncate a run, and truncation is not
  random across arms if one arm makes more calls. Any headline comparison must
  be run on a stack where both arms complete, and completion rate must be
  reported alongside pass rate.
- **`cost_usd` is unavailable on some backends** (NVIDIA NIM reports none), so
  cost-matching there must use token counts.

---

*Amendments to this document must be dated and appended, never edited in place.
Any change to Section 3 (C) or Section 4 (the criterion) invalidates prior runs.*
