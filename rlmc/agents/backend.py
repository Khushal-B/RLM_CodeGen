"""The one place that talks to a model.

Both arms of every ablation go through this class, so "same model, same
sampling, same serving stack" is structural rather than something to verify by
reading two code paths.

Two entry points:

``complete``      one plain chat completion. Used for planning/decomposition,
                  where the answer is a small JSON object and a REPL would only
                  add failure modes.
``run_rlm_node``  one full RLM node: its own REPL, its own tools, its own
                  iteration loop. This is the unmodified ``rlm.RLM`` class from
                  the cloned repo -- not a reimplementation -- which is what
                  makes "identical scaffold" true for the H1 comparison.

Backends: ``openrouter`` is the default because its usage payload carries real
USD cost (rlm/clients/openai.py extracts ``usage.cost``), which the cost-matched
table needs and no other backend supplies. ``vllm`` points at a local server for
the A100 runs.
"""

from __future__ import annotations

import locale
import os
import unicodedata
from typing import Any

from rlmc.agents import ratelimit
from rlmc.config import ArmConfig

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"

# Backends that speak the OpenAI wire format but are not "openai" to rlm.
# rlm's get_client only knows a fixed set of names, so anything else is
# routed through "openai" with an explicit base_url and api_key.
_OPENAI_COMPATIBLE = {"nvidia": NVIDIA_BASE}


class MissingDependency(RuntimeError):
    pass


class LMBackend:
    """Wraps model access for one run."""

    def __init__(self, config: ArmConfig) -> None:
        self.cfg = config
        self._client: Any = None
        self._rlm_cls: Any = None
        # Install the process-wide limiter before any client exists, so RLM's
        # internally-constructed clients are governed too.
        if config.rpm_limit:
            ratelimit.install(
                rpm=config.rpm_limit,
                max_attempts=config.max_retries,
                base_delay=config.retry_base_delay,
            )

    # ------------------------------------------------------------------
    def backend_name(self) -> str:
        # rlm treats "vllm" and "openrouter" as OpenAI-compatible variants;
        # anything in _OPENAI_COMPATIBLE it has never heard of, so it goes
        # through the plain "openai" client with an explicit base_url.
        if self.cfg.backend in _OPENAI_COMPATIBLE:
            return "openai"
        return self.cfg.backend

    def backend_kwargs(self, model: str | None = None) -> dict[str, Any]:
        cfg = self.cfg
        kwargs: dict[str, Any] = {"model_name": model or cfg.model}

        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        elif cfg.backend in _OPENAI_COMPATIBLE:
            kwargs["base_url"] = _OPENAI_COMPATIBLE[cfg.backend]
        elif cfg.backend == "openrouter":
            kwargs["base_url"] = OPENROUTER_BASE

        api_key = self._api_key()
        if api_key:
            kwargs["api_key"] = api_key

        # Long timeout and no SDK-level retries: retries are handled by
        # rlmc.agents.ratelimit so that they are counted and rate-limited
        # rather than firing extra unthrottled requests underneath us.
        kwargs["timeout"] = cfg.request_timeout_s
        kwargs["max_retries"] = 0

        sampling: dict[str, Any] = {"temperature": cfg.temperature}
        if cfg.top_p is not None:
            sampling["top_p"] = cfg.top_p
        if cfg.max_output_tokens is not None:
            sampling["max_tokens"] = cfg.max_output_tokens
        if cfg.seed is not None:
            sampling["seed"] = cfg.seed

        extra_body: dict[str, Any] = {}
        # OpenRouter reports per-request USD cost only when accounting is on.
        if cfg.backend == "openrouter":
            extra_body["usage"] = {"include": True}
        if cfg.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(cfg.chat_template_kwargs)
        if extra_body:
            sampling["extra_body"] = extra_body

        kwargs["sampling_args"] = sampling
        return kwargs

    def _api_key(self) -> str | None:
        cfg = self.cfg
        if cfg.backend == "nvidia":
            return os.getenv("NVIDIA_API_KEY")
        if cfg.backend == "openrouter":
            return os.getenv("OPENROUTER_API_KEY")
        if cfg.backend == "vllm":
            return os.getenv("VLLM_API_KEY", "EMPTY")
        return None

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from rlm.clients import get_client
        except ImportError as exc:
            raise MissingDependency(
                "the 'rlm' package is not importable. Install it from the cloned "
                "repo: pip install -e ../rlm_repo"
            ) from exc
        self._client = get_client(self.backend_name(), self.backend_kwargs())
        return self._client

    def _get_rlm_cls(self) -> Any:
        if self._rlm_cls is not None:
            return self._rlm_cls
        try:
            from rlm import RLM
        except ImportError as exc:
            raise MissingDependency(
                "the 'rlm' package is not importable. Install it from the cloned "
                "repo: pip install -e ../rlm_repo"
            ) from exc
        self._rlm_cls = RLM
        return self._rlm_cls

    # ------------------------------------------------------------------
    def complete(
        self, system: str, user: str, node_id: str = "", model: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """One plain completion. Returns ``(text, usage)``."""
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = client.completion(messages, model=model)
        usage = _usage_of(client)
        return text or "", usage

    # ------------------------------------------------------------------
    def run_rlm_node(
        self,
        system_prompt: str,
        context: str,
        tools: dict[str, Any],
        max_iterations: int,
        node_id: str = "",
        budget_usd: float | None = None,
    ) -> dict[str, Any]:
        """Run one build node as an RLM with its own REPL.

        Returns a dict with ``response``, ``usage``, ``iterations``, ``subcalls``
        and ``trajectory`` -- the last of which carries the per-iteration message
        histories that the context high-water mark is computed from.
        """
        RLM = self._get_rlm_cls()
        from rlm.logger import RLMLogger

        # RLM's LocalREPL persists the context with a bare open(path, "w"),
        # which uses the platform default encoding -- cp1252 on Windows. A
        # requirements document containing a smart quote or an em dash then
        # raises UnicodeEncodeError and the node produces nothing. We do not
        # patch rlm, so the context is made safe for the local codec here.
        # Setting PYTHONUTF8=1 before launching python is the cleaner fix and
        # avoids the substitution entirely.
        context = _encodable(context)

        logger = RLMLogger()

        # The node's system prompt must survive RLM's own formatting, which
        # calls .format(custom_tools_section=...) on it. Anything else in the
        # prompt that looks like a format field would raise, so braces are
        # escaped and the expected field is appended.
        safe_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
        safe_prompt += "\n{custom_tools_section}\n"

        rlm = RLM(
            backend=self.backend_name(),
            backend_kwargs=self.backend_kwargs(),
            environment=self.cfg.rlm_environment,
            max_depth=1,               # the driver owns recursion, not the REPL
            max_iterations=max_iterations,
            max_budget=budget_usd,
            custom_system_prompt=safe_prompt,
            custom_tools=tools,
            custom_sub_tools={},       # sub-LLMs get no workspace access
            logger=logger,
            verbose=False,
            orchestrator=False,
            max_concurrent_subcalls=self.cfg.max_concurrent_children,
        )

        subcalls = 0
        iterations = 0
        response = ""
        try:
            completion = rlm.completion(context, root_prompt=None)
            response = completion.response or ""
            usage = _usage_from_summary(completion.usage_summary)
        except Exception:
            usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            raise
        finally:
            try:
                rlm.close()
            except Exception:
                pass

        trajectory = logger.get_trajectory() or {}
        iters = trajectory.get("iterations", []) or []
        iterations = len(iters)
        for it in iters:
            for block in it.get("code_blocks", []) or []:
                subcalls += len((block.get("result") or {}).get("rlm_calls", []) or [])

        return {
            "response": response,
            "usage": usage,
            "iterations": iterations,
            "subcalls": subcalls,
            "trajectory": trajectory,
        }


# ---------------------------------------------------------------------------


def _encodable(text: str) -> str:
    """Replace characters the local default codec cannot encode."""
    if not isinstance(text, str):
        return text
    enc = locale.getpreferredencoding(False) or "utf-8"
    if enc.lower().replace("-", "") in ("utf8", "utf_8"):
        return text
    try:
        text.encode(enc)
        return text
    except UnicodeEncodeError:
        pass
    # Fold to the nearest ASCII equivalent, then drop what is left.
    folded = unicodedata.normalize("NFKD", text)
    return folded.encode(enc, errors="replace").decode(enc, errors="replace")


def _usage_of(client: Any) -> dict[str, Any]:
    try:
        summary = client.get_usage_summary()
        return _usage_from_summary(summary)
    except Exception:
        return {"calls": 1, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _usage_from_summary(summary: Any) -> dict[str, Any]:
    if summary is None:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    try:
        return {
            "calls": sum(s.total_calls for s in summary.model_usage_summaries.values()),
            "input_tokens": summary.total_input_tokens,
            "output_tokens": summary.total_output_tokens,
            "cost_usd": summary.total_cost or 0.0,
        }
    except Exception:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def check_api_key(config: ArmConfig) -> str | None:
    """Return a human-readable problem with credentials, or None if fine."""
    if config.backend == "nvidia":
        if not os.getenv("NVIDIA_API_KEY"):
            return (
                "NVIDIA_API_KEY is not set. Put it in rlm_contracts/.env or "
                "export it before running."
            )
    elif config.backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            return (
                "OPENROUTER_API_KEY is not set. Put it in rlm_contracts/.env or "
                "export it before running."
            )
    elif config.backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return "OPENAI_API_KEY is not set."
    elif config.backend == "vllm" and not config.base_url:
        return "backend=vllm requires --base-url (e.g. http://localhost:8000/v1)."
    return None
