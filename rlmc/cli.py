"""The common interface.

One command runs any arm on any task set:

    python -m rlmc.cli run   --arm h1-bounded --tasks tinycsv --model deepseek/deepseek-chat
    python -m rlmc.cli run   --arm h1-full-source --tasks tinycsv
    python -m rlmc.cli eval  --runs runs/
    python -m rlmc.cli report --runs runs/ --compare h1-bounded h1-full-source

Sub-commands:
    arms     list the named ablation arms
    tasks    list available tasks
    run      run one or more arms over one or more tasks
    eval     score generated workspaces against their pytest suites
    report   aggregate records into the paired comparison table
    check    verify contract machinery works without any API access
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from rlmc.agents.base import RunRecord, TaskSpec, make_agent
from rlmc.bench.evaluate import evaluate_workspace, save_eval
from rlmc.bench.task import load_nl2repo_tasks, load_smoke_tasks, stratified_subset
from rlmc.config import ArmConfig, list_presets
from rlmc.metrics.report import build_report, render_report, save_report

DEFAULT_RUNS = "runs"


# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load rlm_contracts/.env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return


def resolve_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    if args.benchmark == "nl2repo":
        if not args.nl2repo_root:
            raise SystemExit(
                "--nl2repo-root is required for --benchmark nl2repo. Clone it first:\n"
                "  git clone https://github.com/multimodal-art-projection/NL2RepoBench"
            )
        tasks = load_nl2repo_tasks(
            args.nl2repo_root,
            task_ids=args.tasks or None,
            difficulty=args.difficulty,
        )
    else:
        tasks = load_smoke_tasks(task_ids=args.tasks or None)

    if getattr(args, "dev_subset", False):
        tasks = stratified_subset(tasks)
    if getattr(args, "limit", 0):
        tasks = tasks[: args.limit]
    return tasks


def build_config(args: argparse.Namespace) -> ArmConfig:
    cfg = ArmConfig.preset(args.arm) if args.arm in list_presets() else ArmConfig(name=args.arm)
    overrides: dict = {}
    for field_name in (
        "model", "backend", "base_url", "temperature", "max_depth",
        "return_mode", "contract_mode", "depth_mode",
        "diagnostic_budget_tokens", "max_budget_usd", "max_wallclock_s",
        "node_max_iterations", "root_max_iterations", "max_concurrent_children",
        "escalate_model", "sub_model", "top_p", "max_output_tokens",
        "rpm_limit", "max_retries", "request_timeout_s",
        "max_children", "min_children",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            overrides[field_name] = value

    ctk: dict = {}
    if getattr(args, "thinking", None) is not None:
        ctk["thinking"] = args.thinking == "on"
    if getattr(args, "reasoning_effort", None) is not None:
        ctk["reasoning_effort"] = args.reasoning_effort
    if ctk:
        overrides["chat_template_kwargs"] = {**cfg.chat_template_kwargs, **ctk}

    return cfg.copy_with(**overrides) if overrides else cfg


# ---------------------------------------------------------------------------


def cmd_arms(args: argparse.Namespace) -> int:
    print("Named arms:\n")
    for name in list_presets():
        print("  " + ArmConfig.preset(name).describe())
    print(
        "\nH1 is the go/no-go gate: run h1-bounded against h1-full-source on the same\n"
        "tasks and seeds. They differ in exactly one thing -- what a child returns."
    )
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    tasks = resolve_tasks(args)
    if not tasks:
        print("no tasks found")
        return 1
    print(f"{'task_id':<24}{'difficulty':<12}{'target_loc':>10}  tests")
    print("-" * 70)
    for t in tasks:
        print(f"{t.task_id:<24}{t.difficulty:<12}{t.target_loc:>10}  {t.tests_path or '-'}")
    print("\n" + str(len(tasks)) + " task(s)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Self-test the deterministic core. No API key, no model, no network."""
    from rlmc.contracts.checker import verify_node
    from rlmc.contracts.compose import check_composition
    from rlmc.contracts.schema import parse_decomposition
    from rlmc.contracts.stubs import emit_stub
    from rlmc.contracts.types import Contract, Param, SymbolSpec
    from rlmc.core.blame import plan_repair
    from rlmc.core.payload import build_payload

    failures: list[str] = []

    c = Contract(
        node_id="n", owns_paths=["pkg/a.py"],
        provides=[SymbolSpec("pkg.a.f", "function", "pkg/a.py", [Param("x", "int")], "str")],
    )
    if "def f(x: int) -> str" not in emit_stub(c):
        failures.append("stub emission")

    good = {"pkg/a.py": "def f(x: int) -> str:\n    return ''\n"}
    if not verify_node(c, good).ok:
        failures.append("clean code should verify")

    bad = {"pkg/a.py": "def f(x: int, y: int) -> str:\n    return ''\n"}
    if verify_node(c, bad).ok:
        failures.append("arity mismatch should fail")

    plan = parse_decomposition(
        '```json\n{"action":"DECOMPOSE","children":[{"node_id":"a","owns_paths":["p/a.py"],'
        '"provides":[{"qualname":"p.a.g","kind":"function","path":"p/a.py"}]},'
        '{"node_id":"b","owns_paths":["p/b.py"],"provides":[]}]}\n```'
    )
    if plan["action"] != "DECOMPOSE" or len(plan["children"]) != 2:
        failures.append("decomposition parsing")

    comp = check_composition(c, [x for x, _, _ in plan["children"]])
    if comp.ok:
        failures.append("composition should reject uncovered parent symbol")

    rep = verify_node(c, bad)
    payload = build_payload(
        node_id="n", ok=False, contract_hash=c.hash(), report=rep,
        files={"pkg/a.py": "x = 1\n" * 5000},
        return_mode="bounded", budget_tokens=256,
    )
    if payload.tokens > 256:
        failures.append("bounded payload exceeded C")
    if payload.raw_tokens <= payload.tokens:
        failures.append("raw payload should be much larger than bounded")

    repair = plan_repair(
        rep, ["n"], repair_rounds_used=0, max_repair_rounds=2,
        decompose_retries_used=0, max_decompose_retries=2, can_escalate=False,
    )
    if repair.action.value not in ("retry_child", "redecompose"):
        failures.append("blame routing")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  - " + f)
        return 1

    print("Contract core OK.")
    print("  stub emission, AST verification, composition, schema parsing,")
    print("  bounded payload (" + str(payload.tokens) + " tokens vs "
          + str(payload.raw_tokens) + " unbounded), blame routing.")
    return 0


def cmd_provision(args: argparse.Namespace) -> int:
    """Fetch upstream pytest suites for NL2Repo tasks (no Docker needed)."""
    from rlmc.bench.nl2repo_tests import provision_many

    if not args.nl2repo_root:
        raise SystemExit("--nl2repo-root is required")
    if not args.tasks:
        raise SystemExit("pass --tasks <id> [<id> ...]")
    results = provision_many(args.nl2repo_root, args.tasks, force=args.force)
    bad = 0
    for r in results:
        status = "ok " if r.ok else "FAIL"
        print(status + " " + r.task_id.ljust(24)
              + ("files=" + str(r.files)).ljust(11)
              + ("collected=" + str(r.collected)).ljust(16)
              + ("expected=" + str(r.expected)).ljust(15) + r.note)
        if not r.ok:
            bad += 1
    if bad:
        print(
            "\n" + str(bad) + " task(s) could not be provisioned. Add a PyPI mapping "
            "to rlmc/bench/nl2repo_tests.py KNOWN_SOURCES, or run the official "
            "Docker harness."
        )
    print(
        "\nNote: suites fetched this way are valid for comparing arms to each other "
        "(both run the identical suite). Do not quote them against NL2Repo's "
        "published leaderboard without reproducing under its Docker harness."
    )
    return 1 if bad else 0


def cmd_run(args: argparse.Namespace) -> int:
    _load_env()
    from rlmc.agents.backend import check_api_key

    tasks = resolve_tasks(args)
    if not tasks:
        print("no tasks matched")
        return 1

    arms = args.arms or [args.arm]
    seeds = args.seeds or [0]
    runs_dir = Path(args.runs)

    configs = []
    for arm in arms:
        a = argparse.Namespace(**vars(args))
        a.arm = arm
        cfg = build_config(a)
        problem = check_api_key(cfg)
        if problem:
            print("ERROR: " + problem)
            return 2
        configs.append(cfg)

    total = len(configs) * len(tasks) * len(seeds)
    print("Running " + str(total) + " job(s): " + str(len(configs)) + " arm(s) x "
          + str(len(tasks)) + " task(s) x " + str(len(seeds)) + " seed(s)")
    for cfg in configs:
        print("  " + cfg.describe())
    print()

    done = 0
    failures = 0
    for cfg in configs:
        agent = make_agent(cfg, progress=not args.quiet)
        for task in tasks:
            for seed in seeds:
                done += 1
                tag = cfg.name + "/" + task.task_id + "/seed" + str(seed)
                out_dir = runs_dir / cfg.name / task.task_id / ("seed" + str(seed))
                ws_dir = out_dir / "workspace"
                print("[" + str(done) + "/" + str(total) + "] " + tag + " ... ", end="", flush=True)

                if args.skip_existing and (out_dir / "record.json").exists():
                    print("skipped (exists)")
                    continue

                t0 = time.perf_counter()
                try:
                    record = agent.run(task, str(ws_dir), seed)
                except KeyboardInterrupt:
                    print("interrupted")
                    return 130
                except Exception as exc:
                    failures += 1
                    print("FAILED: " + type(exc).__name__ + ": " + exc)
                    if args.traceback:
                        traceback.print_exc()
                    continue

                record.config["difficulty"] = task.difficulty
                record.config["target_loc"] = task.target_loc

                if args.eval and task.tests_path:
                    ev = evaluate_workspace(ws_dir, task.tests_path, package_name=task.package_name)
                    record.tests_total = ev.tests_total
                    record.tests_passed = ev.tests_passed
                    record.pass_rate = ev.pass_rate
                    record.all_tests_pass = ev.all_tests_pass
                    record.eval_error = ev.error

                record.save(out_dir / "record.json")
                elapsed = time.perf_counter() - t0
                print(
                    ("ok " if record.ok else "done ")
                    + str(record.files) + " files, " + str(record.loc) + " loc"
                    + (", pass " + format(record.pass_rate, ".2f") if args.eval else "")
                    + ", ctx " + str(record.root_context_tokens)
                    + ", $" + format(record.cost_usd, ".4f")
                    + ", " + format(elapsed, ".0f") + "s"
                )
    if failures:
        print("\n" + str(failures) + " job(s) failed")
    print("\nRecords in " + str(runs_dir.resolve()))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    tasks = {t.task_id: t for t in resolve_tasks(args)}
    runs_dir = Path(args.runs)
    n = 0
    for record_path in sorted(runs_dir.rglob("record.json")):
        record = RunRecord.load(record_path)
        task = tasks.get(record.task_id)
        if task is None or not task.tests_path:
            continue
        ws = Path(record.workspace)
        if not ws.exists():
            ws = record_path.parent / "workspace"
        ev = evaluate_workspace(ws, task.tests_path, package_name=task.package_name)
        save_eval(record_path, ev)
        n += 1
        status = format(ev.pass_rate, ".2f") if ev.collected else ("FAIL: " + str(ev.error))
        print(record.arm + "/" + record.task_id + "/seed" + str(record.seed)
              + "  " + str(ev.tests_passed) + "/" + str(ev.tests_total) + "  " + status)
    print("\nscored " + str(n) + " run(s)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    compare = tuple(args.compare) if args.compare else None
    report = build_report(args.runs, compare)  # type: ignore[arg-type]
    print(render_report(report))
    if args.json:
        p = save_report(report, args.json)
        print("\nwrote " + str(p))
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rlmc",
        description="Contract-recursive RLM: run and compare ablation arms.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--benchmark", choices=["smoke", "nl2repo"], default="smoke")
        sp.add_argument("--nl2repo-root", default=os.getenv("NL2REPO_ROOT"))
        sp.add_argument("--tasks", nargs="*", default=None, help="task ids; default all")
        sp.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
        sp.add_argument("--dev-subset", action="store_true",
                        help="frozen stratified 5 easy / 8 medium / 7 hard subset")
        sp.add_argument("--limit", type=int, default=0)
        sp.add_argument("--runs", default=DEFAULT_RUNS)

    def model_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default=None)
        sp.add_argument("--sub-model", default=None)
        sp.add_argument("--backend", default=None,
                        choices=["openrouter", "nvidia", "openai", "vllm", "anthropic",
                                 "gemini"])
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--temperature", type=float, default=None)
        sp.add_argument("--max-depth", type=int, default=None)
        sp.add_argument("--return-mode", default=None, choices=["bounded", "full_source"])
        sp.add_argument("--contract-mode", default=None, choices=["typed", "prose", "none"])
        sp.add_argument("--depth-mode", default=None, choices=["adaptive", "fixed"])
        sp.add_argument("--diagnostic-budget-tokens", type=int, default=None,
                        help="the constant C; locked before the run (default 256)")
        sp.add_argument("--max-budget-usd", type=float, default=None)
        sp.add_argument("--max-wallclock-s", type=float, default=None)
        sp.add_argument("--node-max-iterations", type=int, default=None)
        sp.add_argument("--root-max-iterations", type=int, default=None)
        sp.add_argument("--max-concurrent-children", type=int, default=None)
        sp.add_argument("--max-children", type=int, default=None)
        sp.add_argument("--min-children", type=int, default=None)
        sp.add_argument("--escalate-model", default=None)
        sp.add_argument("--top-p", type=float, default=None)
        sp.add_argument("--max-output-tokens", type=int, default=None)
        sp.add_argument("--rpm-limit", type=int, default=None,
                        help="requests/min cap, enforced process-wide across all nodes")
        sp.add_argument("--max-retries", type=int, default=None)
        sp.add_argument("--request-timeout-s", type=float, default=None)
        sp.add_argument("--thinking", default=None, choices=["on", "off"],
                        help="provider chat-template thinking switch (NVIDIA NIM)")
        sp.add_argument("--reasoning-effort", default=None,
                        choices=["low", "medium", "high"])

    sp = sub.add_parser("arms", help="list named ablation arms")
    sp.set_defaults(func=cmd_arms)

    sp = sub.add_parser("tasks", help="list tasks")
    common(sp)
    sp.set_defaults(func=cmd_tasks)

    sp = sub.add_parser("check", help="self-test the contract core (no API needed)")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run", help="run arms over tasks")
    common(sp)
    model_args(sp)
    sp.add_argument("--arm", default="h1-bounded")
    sp.add_argument("--arms", nargs="*", default=None, help="run several arms in one go")
    sp.add_argument("--seeds", nargs="*", type=int, default=None, help="default: 0")
    sp.add_argument("--eval", action="store_true", help="score each run immediately")
    sp.add_argument("--skip-existing", action="store_true")
    sp.add_argument("--traceback", action="store_true")
    sp.add_argument("--quiet", action="store_true",
                    help="suppress per-node progress lines")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("provision", help="fetch upstream test suites for NL2Repo tasks")
    sp.add_argument("--nl2repo-root", default=os.getenv("NL2REPO_ROOT"))
    sp.add_argument("--tasks", nargs="*", default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_provision)

    sp = sub.add_parser("eval", help="score generated workspaces")
    common(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("report", help="aggregate records")
    sp.add_argument("--runs", default=DEFAULT_RUNS)
    sp.add_argument("--compare", nargs=2, default=None, metavar=("ARM_A", "ARM_B"))
    sp.add_argument("--json", default=None, help="also write the report as JSON")
    sp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
