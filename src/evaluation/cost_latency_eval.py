"""Cost and latency SLO tracking for the RAG pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.settings import get_config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.evaluation.system_eval import GoldenQuery
    from src.generation.generator import GenerationResult

logger = get_logger(__name__)


# USD per 1,000,000 tokens. Override via pricing_overrides in config.
DEFAULT_LLM_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "BAAI/bge-large-en-v1.5": {"input": 0.0, "output": 0.0},
}

DEFAULT_RERANK_PRICE_PER_1K_SEARCHES: dict[str, float] = {
    "rerank-english-v3.0": 2.00,
    "rerank-multilingual-v3.0": 2.00,
}

_DEFAULT_GENERATION_MODEL_FALLBACK = "gpt-4-turbo"


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (0 <= pct <= 100). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    rank = min(rank, len(ordered))
    return ordered[rank - 1]


@dataclass
class CostLatencySLO:
    """Configurable SLOs for cost and latency. None = no limit."""

    enabled: bool = True
    p50_latency_ms: float | None = 2000.0
    p95_latency_ms: float | None = 6000.0
    p99_latency_ms: float | None = 10000.0
    max_cost_per_query_usd: float | None = 0.05
    max_tokens_per_query: int | None = 6000
    pricing_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> CostLatencySLO:
        cfg = get_config()
        slo_cfg = cfg.evaluation.get("cost_latency_slo", {}) if hasattr(cfg, "evaluation") else {}
        if not slo_cfg:
            logger.info(
                "No evaluation.cost_latency_slo config section found — using built-in defaults."
            )
            return cls()
        return cls(
            enabled=slo_cfg.get("enabled", True),
            p50_latency_ms=slo_cfg.get("p50_latency_ms", 2000.0),
            p95_latency_ms=slo_cfg.get("p95_latency_ms", 6000.0),
            p99_latency_ms=slo_cfg.get("p99_latency_ms", 10000.0),
            max_cost_per_query_usd=slo_cfg.get("max_cost_per_query_usd", 0.05),
            max_tokens_per_query=slo_cfg.get("max_tokens_per_query", 6000),
            pricing_overrides=slo_cfg.get("pricing_overrides", {}) or {},
        )


@dataclass
class QueryCostLatencyRecord:
    """Cost and latency measurements for a single query."""

    query: str
    namespace: str = "default"
    timestamp: str = ""
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    generation_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    embedding_tokens: int = 0
    rerank_searches: int = 0
    cost_breakdown_usd: dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    slo_violations: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.judge_prompt_tokens
            + self.judge_completion_tokens
            + self.embedding_tokens
        )

    @property
    def passed_slo(self) -> bool:
        return not self.slo_violations

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["passed_slo"] = self.passed_slo
        return d


@dataclass
class CostLatencyReport:
    """Aggregate cost/latency report across a batch of queries."""

    run_id: str
    timestamp: str
    num_queries: int

    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float

    total_cost_usd: float
    mean_cost_per_query_usd: float
    max_cost_per_query_usd: float

    mean_tokens_per_query: float
    total_tokens: int

    slo_pass_rate: float
    slo: CostLatencySLO
    per_query: list[QueryCostLatencyRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  COST & LATENCY EVALUATION  [{self.run_id}]",
            f"  {self.timestamp}   N={self.num_queries}",
            "=" * 60,
            "  Latency:",
            f"    p50      {self.p50_latency_ms:8.1f} ms",
            f"    p95      {self.p95_latency_ms:8.1f} ms",
            f"    p99      {self.p99_latency_ms:8.1f} ms",
            f"    mean     {self.mean_latency_ms:8.1f} ms",
            "  Cost:",
            f"    total    ${self.total_cost_usd:.4f}",
            f"    mean     ${self.mean_cost_per_query_usd:.5f} / query",
            f"    max      ${self.max_cost_per_query_usd:.5f} / query",
            "  Tokens:",
            f"    mean     {self.mean_tokens_per_query:.1f} / query",
            f"    total    {self.total_tokens}",
            "  SLO:",
            f"    pass rate {self.slo_pass_rate:.3f}",
        ]
        if self.slo.enabled:
            lines += [
                f"    targets   p50<={self.slo.p50_latency_ms} ms, "
                f"p95<={self.slo.p95_latency_ms} ms, "
                f"p99<={self.slo.p99_latency_ms} ms, "
                f"cost<=${self.slo.max_cost_per_query_usd}/query, "
                f"tokens<={self.slo.max_tokens_per_query}/query",
            ]
            failures = [r for r in self.per_query if not r.passed_slo]
            if failures:
                lines.append(f"    violations ({len(failures)}):")
                for r in failures[:10]:
                    lines.append(f"      - '{r.query[:60]}': {', '.join(r.slo_violations)}")
        else:
            lines.append("    (disabled)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "cost_latency",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "num_queries": self.num_queries,
            "latency_ms": {
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
                "mean": round(self.mean_latency_ms, 2),
            },
            "cost_usd": {
                "total": round(self.total_cost_usd, 6),
                "mean_per_query": round(self.mean_cost_per_query_usd, 6),
                "max_per_query": round(self.max_cost_per_query_usd, 6),
            },
            "tokens": {
                "mean_per_query": round(self.mean_tokens_per_query, 2),
                "total": self.total_tokens,
            },
            "slo_pass_rate": round(self.slo_pass_rate, 4),
            "slo_thresholds": {
                "enabled": self.slo.enabled,
                "p50_latency_ms": self.slo.p50_latency_ms,
                "p95_latency_ms": self.slo.p95_latency_ms,
                "p99_latency_ms": self.slo.p99_latency_ms,
                "max_cost_per_query_usd": self.slo.max_cost_per_query_usd,
                "max_tokens_per_query": self.slo.max_tokens_per_query,
            },
            "per_query": [r.to_dict() for r in self.per_query],
        }


class LatencyTimer:
    """Wall-clock timer for named pipeline stages."""

    def __init__(self) -> None:
        self.stage_ms: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stage_ms[name] = self.stage_ms.get(name, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


class CostCalculator:
    """Computes USD cost from token / search counts using a pricing table."""

    def __init__(self, pricing_overrides: dict[str, dict[str, float]] | None = None) -> None:
        self._llm_pricing = {**DEFAULT_LLM_PRICING_PER_1M}
        if pricing_overrides:
            for model, prices in pricing_overrides.items():
                self._llm_pricing[model] = {**self._llm_pricing.get(model, {}), **prices}
        self._rerank_pricing = dict(DEFAULT_RERANK_PRICE_PER_1K_SEARCHES)

    def llm_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        prices = self._llm_pricing.get(model)
        if prices is None:
            logger.debug(
                "No pricing entry for model '%s' — falling back to '%s'",
                model,
                _DEFAULT_GENERATION_MODEL_FALLBACK,
            )
            prices = self._llm_pricing.get(
                _DEFAULT_GENERATION_MODEL_FALLBACK, {"input": 0.0, "output": 0.0}
            )
        return (prompt_tokens * prices.get("input", 0.0) / 1_000_000) + (
            completion_tokens * prices.get("output", 0.0) / 1_000_000
        )

    def embedding_cost(self, model: str, num_tokens: int) -> float:
        prices = self._llm_pricing.get(model, {"input": 0.0, "output": 0.0})
        return num_tokens * prices.get("input", 0.0) / 1_000_000

    def rerank_cost(self, model: str, num_searches: int) -> float:
        price_per_1k = self._rerank_pricing.get(model, 2.00)
        return num_searches * price_per_1k / 1000


class CostLatencyEvaluator:
    """Instruments pipeline calls for cost and latency, checks against SLOs."""

    def __init__(
        self,
        score_ledger_path: str | None = None,
        slo: CostLatencySLO | None = None,
    ) -> None:
        self._ledger_path = Path(score_ledger_path) if score_ledger_path else None
        self.slo = slo or CostLatencySLO.from_config()
        self._costs = CostCalculator(pricing_overrides=self.slo.pricing_overrides)

        try:
            self._generation_model = get_config().generation["model"]
        except Exception:
            self._generation_model = _DEFAULT_GENERATION_MODEL_FALLBACK

        try:
            self._judge_model = get_config().evaluation["llm_judge"]["model"]
        except Exception:
            self._judge_model = _DEFAULT_GENERATION_MODEL_FALLBACK

        try:
            self._embedding_model = get_config().evaluation.get(
                "embedding_model", "text-embedding-3-small"
            )
        except Exception:
            self._embedding_model = "text-embedding-3-small"

        try:
            self._rerank_model = get_config().retrieval["reranking"]["cohere_model"]
        except Exception:
            self._rerank_model = "rerank-english-v3.0"

    def record_request(
        self,
        query: str,
        namespace: str,
        timer: LatencyTimer,
        generation_result: GenerationResult | None = None,
    ) -> QueryCostLatencyRecord:
        """Build, persist, and SLO-check a single live request record."""
        record = self.build_record(
            query=query,
            namespace=namespace,
            timer=timer,
            generation_result=generation_result,
        )
        self._append_to_ledger(record.to_dict())
        if record.slo_violations:
            logger.warning(
                "SLO violation: %s",
                ", ".join(record.slo_violations),
                extra={"query": query[:120], "violations": record.slo_violations},
            )
        return record

    def build_record(
        self,
        query: str,
        namespace: str,
        timer: LatencyTimer,
        generation_result: GenerationResult | None = None,
        judge_prompt_tokens: int = 0,
        judge_completion_tokens: int = 0,
        embedding_tokens: int = 0,
        rerank_searches: int = 0,
    ) -> QueryCostLatencyRecord:
        """Build a QueryCostLatencyRecord from timing and generation result."""
        record = QueryCostLatencyRecord(
            query=query,
            namespace=namespace,
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage_latencies_ms={k: round(v, 2) for k, v in timer.stage_ms.items()},
            total_latency_ms=round(timer.total_ms, 2),
            judge_prompt_tokens=judge_prompt_tokens,
            judge_completion_tokens=judge_completion_tokens,
            embedding_tokens=embedding_tokens,
            rerank_searches=rerank_searches,
        )

        gen_model = self._generation_model
        prompt_tokens = 0
        completion_tokens = 0
        if generation_result is not None:
            prompt_tokens = getattr(generation_result, "prompt_tokens", 0) or 0
            completion_tokens = getattr(generation_result, "completion_tokens", 0) or 0
            model_used = getattr(generation_result, "model_used", "") or ""
            if model_used in DEFAULT_LLM_PRICING_PER_1M:
                gen_model = model_used

        record.generation_model = gen_model
        record.prompt_tokens = prompt_tokens
        record.completion_tokens = completion_tokens

        gen_cost = self._costs.llm_cost(gen_model, prompt_tokens, completion_tokens)
        judge_cost = self._costs.llm_cost(
            self._judge_model, judge_prompt_tokens, judge_completion_tokens
        )
        embed_cost = self._costs.embedding_cost(self._embedding_model, embedding_tokens)
        rerank_cost = self._costs.rerank_cost(self._rerank_model, rerank_searches)

        record.cost_breakdown_usd = {
            "generation": round(gen_cost, 6),
            "judge": round(judge_cost, 6),
            "embedding": round(embed_cost, 6),
            "rerank": round(rerank_cost, 6),
        }
        record.total_cost_usd = round(gen_cost + judge_cost + embed_cost + rerank_cost, 6)

        record.slo_violations = self._check_slo(record)
        return record

    def _check_slo(self, record: QueryCostLatencyRecord) -> list[str]:
        if not self.slo.enabled:
            return []
        violations: list[str] = []
        if (
            self.slo.p99_latency_ms is not None
            and record.total_latency_ms > self.slo.p99_latency_ms
        ):
            violations.append(
                f"latency {record.total_latency_ms:.0f}ms > p99 budget {self.slo.p99_latency_ms:.0f}ms"
            )
        if (
            self.slo.max_cost_per_query_usd is not None
            and record.total_cost_usd > self.slo.max_cost_per_query_usd
        ):
            violations.append(
                f"cost ${record.total_cost_usd:.5f} > budget ${self.slo.max_cost_per_query_usd:.5f}"
            )
        if (
            self.slo.max_tokens_per_query is not None
            and record.total_tokens > self.slo.max_tokens_per_query
        ):
            violations.append(
                f"tokens {record.total_tokens} > budget {self.slo.max_tokens_per_query}"
            )
        return violations

    async def run_suite(
        self,
        orchestrator,
        golden_queries: list[GoldenQuery],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> CostLatencyReport:
        """Run the pipeline for each golden query and record cost and latency."""
        records: list[QueryCostLatencyRecord] = []

        for gq in golden_queries:
            timer = LatencyTimer()
            try:
                with timer.stage("end_to_end"):
                    result = await orchestrator.run(
                        raw_query=gq.query,
                        conversation_history=conversation_history,
                        namespace=gq.namespace,
                    )
            except Exception as exc:
                logger.error(
                    "Cost/latency probe failed for '%s': %s", gq.query[:60], exc, exc_info=True
                )
                continue

            record = self.build_record(
                query=gq.query,
                namespace=gq.namespace,
                timer=timer,
                generation_result=result,
            )
            records.append(record)
            logger.debug(
                "[cost/latency] '%s' -> %.0fms, $%.5f, slo=%s",
                gq.query[:60],
                record.total_latency_ms,
                record.total_cost_usd,
                "PASS" if record.passed_slo else "FAIL",
            )

        return self._build_report(records)

    def _build_report(self, records: list[QueryCostLatencyRecord]) -> CostLatencyReport:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        timestamp = datetime.now(timezone.utc).isoformat()

        latencies = [r.total_latency_ms for r in records]
        costs = [r.total_cost_usd for r in records]
        tokens = [r.total_tokens for r in records]

        report = CostLatencyReport(
            run_id=run_id,
            timestamp=timestamp,
            num_queries=len(records),
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            p99_latency_ms=_percentile(latencies, 99),
            mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
            total_cost_usd=sum(costs),
            mean_cost_per_query_usd=(sum(costs) / len(costs)) if costs else 0.0,
            max_cost_per_query_usd=max(costs) if costs else 0.0,
            mean_tokens_per_query=(sum(tokens) / len(tokens)) if tokens else 0.0,
            total_tokens=sum(tokens),
            slo_pass_rate=(
                (sum(1 for r in records if r.passed_slo) / len(records)) if records else 1.0
            ),
            slo=self.slo,
            per_query=records,
        )

        logger.info(report.summary())
        self._append_to_ledger(report.to_dict())
        return report

    def _append_to_ledger(self, record: dict[str, Any]) -> None:
        if not self._ledger_path:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Cost/latency report appended to ledger: %s", self._ledger_path)

    def load_ledger(self) -> list[dict[str, Any]]:
        """Return all previously recorded cost/latency runs from the score ledger."""
        if not self._ledger_path or not self._ledger_path.exists():
            return []
        records = []
        with self._ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("report_type") == "cost_latency":
                    records.append(rec)
        return records

    def print_trend(self, metric: str = "p95", last_n: int = 10) -> None:
        """Print an ASCII sparkline of latency p50/p95/p99 or cost over recent runs."""
        records = self.load_ledger()[-last_n:]
        if not records:
            logger.info("No cost/latency ledger entries found.")
            return

        if metric in ("p50", "p95", "p99", "mean"):
            values = [r["latency_ms"][metric] for r in records]
            unit = "ms"
        elif metric == "cost":
            values = [r["cost_usd"]["mean_per_query"] for r in records]
            unit = "$"
        else:
            values = [r.get(metric, 0.0) for r in records]
            unit = ""

        blocks = " ▁▂▃▄▅▆▇█"
        rng = max(values) - min(values) if len(values) > 1 else 1.0
        sparkline = "".join(
            blocks[round((v - min(values)) / rng * 8)] if rng > 0 else "─" for v in values
        )
        logger.info(
            "Trend [%s] last %d runs: %s  (latest=%s%.3f)",
            metric,
            len(records),
            sparkline,
            unit if unit == "$" else "",
            values[-1],
        )
