"""``rlm_build(node, contract, budget)`` -- the recursive builder.

One function, invoked at every node of the build tree. The node picks exactly
one action:

    IMPLEMENT  emit code for this node in one RLM sub-call; write to the workspace
    DECOMPOSE  emit child nodes + child contracts, split the budget, recurse
               (children run concurrently within a level)
    ESCALATE   retry at a stronger tier / larger budget, same decomposition
    RETURN     node satisfied; return the bounded payload

Verification is the same traversal run in reverse: a node is verified only
after all its children verify, and verification at each node is a deterministic
check of that node's output against its contract. There is no LLM judge
anywhere in the loop.

The invariant
-------------
Whatever a node returns to its parent passes through
``rlmc.core.payload.build_payload`` and is bounded to C tokens, independent of
subtree size. The driver -- not the prompt -- is what enforces that, and every
return is recorded so the bound can be audited after the run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rlmc.config import ArmConfig
from rlmc.contracts.checker import verify_node, verify_subtree
from rlmc.contracts.compose import check_composition
from rlmc.contracts.imports import SymbolOwnership
from rlmc.contracts.schema import SchemaError, parse_decomposition
from rlmc.contracts.types import CheckReport, Contract
from rlmc.core.blame import RepairAction, plan_repair
from rlmc.core.node import Action, Budget, BuildNode, BuildResult, NodeKind
from rlmc.core.payload import ReturnRecord, build_payload
from rlmc.core.prompts import (
    build_decompose_prompt,
    build_node_system_prompt,
    build_redecompose_prompt,
    build_repair_prompt,
)
from rlmc.core.tools import NodeToolbox
from rlmc.core.workspace import Workspace, ensure_package_files
from rlmc.metrics.telemetry import RunTelemetry, context_high_water


class BuildDriver:
    """Drives one task from a requirements document to a workspace."""

    def __init__(
        self,
        config: ArmConfig,
        workspace: Workspace,
        lm: Any,
        telemetry: RunTelemetry | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.cfg = config
        self.ws = workspace
        self.lm = lm                      # rlmc.agents.backend.LMBackend
        self.telemetry = telemetry or RunTelemetry()
        self.returns = ReturnRecord()
        self.on_event = on_event or (lambda kind, data: None)
        self._start = time.perf_counter()
        self._node_seq = 0
        # parent_id -> the contracts of that parent's children. Instance
        # scoped: a class attribute would leak between concurrent runs.
        self._sibling_registry: dict[str, list[Contract]] = {}

    # ------------------------------------------------------------------
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def out_of_time(self) -> bool:
        limit = self.cfg.max_wallclock_s
        return limit is not None and self.elapsed() > limit

    def out_of_money(self) -> bool:
        limit = self.cfg.max_budget_usd
        return limit is not None and self.telemetry.total_cost_usd >= limit

    def _emit(self, kind: str, **data: Any) -> None:
        self.on_event(kind, data)

    # ------------------------------------------------------------------
    def build(self, node: BuildNode, budget: Budget) -> BuildResult:
        """rlm_build: entry point, called once per node."""
        t0 = time.perf_counter()
        self._emit("node_start", node_id=node.node_id, depth=node.depth, title=node.title)

        if self.out_of_time() or self.out_of_money():
            return self._give_up(node, "global budget exhausted", t0)

        action = self._choose_action(node, budget)

        if action is Action.DECOMPOSE:
            result = self._decompose(node, budget, t0)
        else:
            result = self._implement(node, budget, t0)

        self.returns.record(node.node_id, node.depth, result.payload)
        self._emit(
            "node_done",
            node_id=node.node_id,
            depth=node.depth,
            ok=result.ok,
            action=result.action.value,
            payload_tokens=result.payload.tokens,
        )
        return result

    # ------------------------------------------------------------------
    def _choose_action(self, node: BuildNode, budget: Budget) -> Action:
        """H2 lives here: adaptive (model decides) vs fixed (depth decides)."""
        if budget.depth_remaining <= 0:
            return Action.IMPLEMENT
        if self.cfg.depth_mode == "fixed":
            return (
                Action.DECOMPOSE
                if node.depth < self.cfg.max_depth
                else Action.IMPLEMENT
            )
        return Action.DECOMPOSE  # adaptive: the planner call decides, in _decompose

    # ------------------------------------------------------------------
    def _decompose(self, node: BuildNode, budget: Budget, t0: float) -> BuildResult:
        """Plan children, validate composition, recurse, verify, repair."""
        plan = self._plan_children(node, budget)

        # The planner may decide this node is a leaf after all (adaptive depth).
        if plan is None or not plan["children"]:
            return self._implement(node, budget, t0)

        children_specs = plan["children"]
        node.children = []
        child_contracts = [c for c, _, _ in children_specs]

        for contract, title, spec in children_specs:
            child = BuildNode(
                node_id=contract.node_id,
                kind=NodeKind.MODULE,
                title=title,
                spec=spec or node.spec,
                contract=contract,
                depth=node.depth + 1,
                parent_id=node.node_id,
            )
            node.children.append(child)
            self.ws.grant(contract.node_id, contract.owns_paths)

        child_budgets = budget.split(len(node.children))
        self.register_siblings(node.node_id, child_contracts)

        # --- run the level -------------------------------------------
        child_results = self._run_level(node.children, child_budgets)

        # --- verify the level ----------------------------------------
        files_by_node = {c.node_id: self.ws.files_of(c.node_id) for c in node.children}
        report = verify_subtree(node.contract, child_contracts, files_by_node)

        repair_rounds = 0
        decompose_retries = 0
        escalated = False

        while not report.ok and not self.out_of_time() and not self.out_of_money():
            repair = plan_repair(
                report,
                [c.node_id for c in node.children],
                repair_rounds_used=repair_rounds,
                max_repair_rounds=self.cfg.max_repair_rounds,
                decompose_retries_used=decompose_retries,
                max_decompose_retries=self.cfg.max_decompose_retries,
                can_escalate=bool(self.cfg.escalate_model) and not escalated,
            )
            self._emit(
                "repair",
                node_id=node.node_id,
                action=repair.action.value,
                targets=repair.targets,
                reason=repair.reason,
            )
            self.telemetry.record_repair(repair.action.value)

            if repair.action is RepairAction.GIVE_UP:
                break

            if repair.action is RepairAction.REDECOMPOSE:
                decompose_retries += 1
                summary = "\n".join(repair.feedback.get("__parent__", []))
                new_plan = self._plan_children(
                    node, budget, feedback=build_redecompose_prompt(summary)
                )
                if not new_plan or not new_plan["children"]:
                    break
                children_specs = new_plan["children"]
                child_contracts = [c for c, _, _ in children_specs]
                node.children = []
                for contract, title, spec in children_specs:
                    node.children.append(
                        BuildNode(
                            node_id=contract.node_id,
                            kind=NodeKind.MODULE,
                            title=title,
                            spec=spec or node.spec,
                            contract=contract,
                            depth=node.depth + 1,
                            parent_id=node.node_id,
                        )
                    )
                    self.ws.grant(contract.node_id, contract.owns_paths)
                child_budgets = budget.split(len(node.children))
                self.register_siblings(node.node_id, child_contracts)
                child_results = self._run_level(node.children, child_budgets)

            elif repair.action in (RepairAction.RETRY_CHILD, RepairAction.RETRY_CHILDREN,
                                   RepairAction.ESCALATE):
                if repair.action is RepairAction.ESCALATE:
                    escalated = True
                else:
                    repair_rounds += 1
                targets = set(repair.targets)
                retry_nodes = [c for c in node.children if c.node_id in targets]
                retry_budgets = budget.split(max(1, len(retry_nodes)))
                redone = self._run_level(
                    retry_nodes,
                    retry_budgets,
                    feedback={
                        n.node_id: build_repair_prompt(
                            repair.feedback.get(n.node_id, []), self.cfg.contract_mode
                        )
                        for n in retry_nodes
                    },
                    escalate=repair.action is RepairAction.ESCALATE,
                )
                by_id = {r.node_id: r for r in redone}
                child_results = [by_id.get(r.node_id, r) for r in child_results]

            files_by_node = {c.node_id: self.ws.files_of(c.node_id) for c in node.children}
            report = verify_subtree(node.contract, child_contracts, files_by_node)

        # --- return ---------------------------------------------------
        all_files: dict[str, str] = {}
        for f in files_by_node.values():
            all_files.update(f)

        payload = build_payload(
            node_id=node.node_id,
            ok=report.ok,
            contract_hash=node.contract.hash(),
            report=report,
            files=all_files,
            return_mode=self.cfg.return_mode,
            budget_tokens=self.cfg.diagnostic_budget_tokens,
        )

        return BuildResult(
            node_id=node.node_id,
            ok=report.ok,
            action=Action.DECOMPOSE,
            contract_hash=node.contract.hash(),
            payload=payload,
            report=report,
            depth=node.depth,
            files_written=sorted(all_files),
            loc=sum(v.count("\n") + 1 for v in all_files.values()),
            children=child_results,
            repair_rounds=repair_rounds,
            decompose_retries=decompose_retries,
            escalated=escalated,
            wallclock_s=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    def _run_level(
        self,
        nodes: list[BuildNode],
        budgets: list[Budget],
        feedback: dict[str, str] | None = None,
        escalate: bool = False,
    ) -> list[BuildResult]:
        """Run every node in a DAG layer, concurrently.

        RLM Appendix B flags that all its sub-calls are blocking and sequential
        and leaves it unfixed; siblings here are independent by construction
        (that is what the contracts guarantee), so they can be fired together.
        """
        feedback = feedback or {}
        if not nodes:
            return []

        if len(nodes) == 1:
            return [
                self._build_child(nodes[0], budgets[0], feedback.get(nodes[0].node_id), escalate)
            ]

        results: list[BuildResult | None] = [None] * len(nodes)
        workers = min(self.cfg.max_concurrent_children, len(nodes))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._build_child,
                    n,
                    budgets[i] if i < len(budgets) else budgets[-1],
                    feedback.get(n.node_id),
                    escalate,
                ): i
                for i, n in enumerate(nodes)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:
                    results[i] = self._error_result(nodes[i], exc)
        return [r for r in results if r is not None]

    def _build_child(
        self,
        node: BuildNode,
        budget: Budget,
        feedback: str | None,
        escalate: bool,
    ) -> BuildResult:
        if feedback or escalate:
            node.spec = (node.spec or "") + ("\n\n" + feedback if feedback else "")
        return self.build(node, budget)

    # ------------------------------------------------------------------
    def _plan_children(
        self, node: BuildNode, budget: Budget, feedback: str | None = None
    ) -> dict[str, Any] | None:
        """Ask the model for a decomposition; validate composition before use."""
        force = "DECOMPOSE" if self.cfg.depth_mode == "fixed" else None
        system = build_decompose_prompt(
            node.contract,
            self.cfg.contract_mode,
            self.cfg.min_children,
            self.cfg.max_children,
            budget.depth_remaining,
            force_action=force,
        )
        user = "## Requirements for this node\n\n" + (node.spec or "")
        if feedback:
            user += "\n\n## Correction required\n\n" + feedback

        last_error = ""
        for attempt in range(self.cfg.max_decompose_retries + 1):
            if self.out_of_time() or self.out_of_money():
                return None
            try:
                raw, usage = self.lm.complete(system, user, node_id=node.node_id)
            except Exception as exc:
                last_error = type(exc).__name__ + ": " + str(exc)
                continue
            self.telemetry.record_call(usage)

            try:
                plan = parse_decomposition(raw)
            except SchemaError as exc:
                last_error = str(exc)
                user += (
                    "\n\nYour previous reply could not be parsed: " + last_error
                    + ". Reply with ONE JSON object only."
                )
                continue

            if plan["action"] == "IMPLEMENT":
                return None  # adaptive depth chose to stop here

            # Qualify child ids with the parent's. The model only guarantees
            # uniqueness within one level, but workspace ownership and the blame
            # router key on node_id across the WHOLE tree -- an unqualified
            # "parser" at depth 1 and another at depth 2 would silently share a
            # file-ownership grant and cross-attribute every violation.
            for contract, _, _ in plan["children"]:
                if not contract.node_id.startswith(node.node_id + "."):
                    contract.node_id = node.node_id + "." + contract.node_id

            children = [c for c, _, _ in plan["children"]]
            if len(children) < self.cfg.min_children:
                return None  # too small to be worth splitting

            if self.cfg.contract_mode == "typed":
                comp = check_composition(node.contract, children)
                self.telemetry.record_composition(comp.ok)
                if not comp.ok:
                    last_error = comp.summary()
                    self._emit(
                        "composition_rejected",
                        node_id=node.node_id,
                        attempt=attempt,
                        summary=last_error,
                    )
                    user += "\n\n## Correction required\n\n" + build_redecompose_prompt(
                        last_error
                    )
                    continue

            return plan

        self._emit("decompose_failed", node_id=node.node_id, error=last_error)
        return None

    # ------------------------------------------------------------------
    def _implement(self, node: BuildNode, budget: Budget, t0: float) -> BuildResult:
        """Run one node as an RLM with a REPL, writing into the workspace."""
        self.ws.grant(node.node_id, node.contract.owns_paths)

        siblings = self._sibling_contracts(node)
        ownership = SymbolOwnership.from_contracts(
            [node.contract] + siblings, scope_complete=bool(siblings)
        )

        toolbox = NodeToolbox(
            node_id=node.node_id,
            contract=node.contract,
            workspace=self.ws,
            ownership=ownership,
            contract_mode=self.cfg.contract_mode,
            strict_undeclared_imports=self.cfg.strict_undeclared_imports,
        )

        system = build_node_system_prompt(
            node.contract,
            siblings,
            self.cfg.contract_mode,
            node.title,
            node.depth,
            is_root=(node.depth == 0),
        )

        error: str | None = None
        trajectory: dict[str, Any] | None = None
        try:
            trajectory = self.lm.run_rlm_node(
                system_prompt=system,
                context=node.spec,
                tools=toolbox.as_custom_tools(),
                max_iterations=(
                    self.cfg.root_max_iterations if node.depth == 0
                    else self.cfg.node_max_iterations
                ),
                node_id=node.node_id,
                budget_usd=budget.usd,
            )
            self.telemetry.record_call(trajectory.get("usage", {}))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            self._emit("node_error", node_id=node.node_id, error=error)

        created = ensure_package_files(
            self.ws, node.node_id, self.ws.written_paths(node.node_id)
        )
        files = self.ws.files_of(node.node_id)

        if self.cfg.contract_mode == "none":
            report = CheckReport(node_id=node.node_id)
        else:
            report = verify_node(
                node.contract,
                files,
                ownership=ownership,
                owner=node.node_id,
                extra_boundaries=["import_graph"],
            )

        ok = report.ok and bool(files) and error is None

        payload = build_payload(
            node_id=node.node_id,
            ok=ok,
            contract_hash=node.contract.hash(),
            report=report,
            files=files,
            return_mode=self.cfg.return_mode,
            budget_tokens=self.cfg.diagnostic_budget_tokens,
            note=("error: " + error) if error else "",
        )

        # run_rlm_node returns a wrapper whose 'iterations' is a COUNT; the
        # RLM logger trajectory (with per-turn message histories) is nested.
        ctx_tokens = context_high_water((trajectory or {}).get("trajectory"))
        self.telemetry.record_context(node.node_id, node.depth, ctx_tokens)

        return BuildResult(
            node_id=node.node_id,
            ok=ok,
            action=Action.IMPLEMENT,
            contract_hash=node.contract.hash(),
            payload=payload,
            report=report,
            depth=node.depth,
            files_written=sorted(files),
            loc=sum(v.count("\n") + 1 for v in files.values()),
            iterations_used=int((trajectory or {}).get("iterations", 0)),
            subcalls=int((trajectory or {}).get("subcalls", 0)),
            cost_usd=float((trajectory or {}).get("usage", {}).get("cost_usd", 0.0) or 0.0),
            wallclock_s=time.perf_counter() - t0,
            root_context_tokens=ctx_tokens,
            error=error,
            notes=(["created " + str(len(created)) + " __init__.py"] if created else []),
        )

    # ------------------------------------------------------------------
    def _sibling_contracts(self, node: BuildNode) -> list[Contract]:
        """Contracts of this node's siblings -- interfaces only, never source."""
        if node.parent_id is None or self.cfg.contract_mode == "none":
            return []
        return [
            c
            for c in self._sibling_registry.get(node.parent_id, [])
            if c.node_id != node.node_id
        ]

    def register_siblings(self, parent_id: str, contracts: list[Contract]) -> None:
        """Publish a level's contracts so each child can be shown its siblings'.

        Interfaces only -- this registry never holds source, which is what a
        child would need for the invariant to be breached here.
        """
        self._sibling_registry[parent_id] = list(contracts)

    # ------------------------------------------------------------------
    def _give_up(self, node: BuildNode, reason: str, t0: float) -> BuildResult:
        payload = build_payload(
            node_id=node.node_id,
            ok=False,
            contract_hash=node.contract.hash(),
            report=None,
            files={},
            return_mode=self.cfg.return_mode,
            budget_tokens=self.cfg.diagnostic_budget_tokens,
            note=reason,
        )
        return BuildResult(
            node_id=node.node_id,
            ok=False,
            action=Action.RETURN,
            contract_hash=node.contract.hash(),
            payload=payload,
            depth=node.depth,
            wallclock_s=time.perf_counter() - t0,
            error=reason,
        )

    def _error_result(self, node: BuildNode, exc: Exception) -> BuildResult:
        msg = type(exc).__name__ + ": " + str(exc)
        payload = build_payload(
            node_id=node.node_id,
            ok=False,
            contract_hash=node.contract.hash(),
            report=None,
            files={},
            return_mode=self.cfg.return_mode,
            budget_tokens=self.cfg.diagnostic_budget_tokens,
            note=msg,
        )
        return BuildResult(
            node_id=node.node_id,
            ok=False,
            action=Action.RETURN,
            contract_hash=node.contract.hash(),
            payload=payload,
            depth=node.depth,
            error=msg,
        )
