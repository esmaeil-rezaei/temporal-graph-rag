"""System-level evaluation: golden regression suite, baseline comparison, and score ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import EvaluationReport, RAGEvaluator
from src.evaluation.retrieval_eval import LabeledQuery, RetrievalEvalReport, RetrievalEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GoldenQuery:
    query: str
    relevant_chunk_ids: list[str]
    expected_answer_keywords: list[str] = field(default_factory=list)
    namespace: str = "default"
    metadata: dict = field(default_factory=dict)

    def to_labeled_query(self) -> LabeledQuery:
        return LabeledQuery(
            query=self.query,
            relevant_chunk_ids=self.relevant_chunk_ids,
            namespace=self.namespace,
            metadata=self.metadata,
        )


@dataclass
class GoldenQueryResult:
    """Per-query system evaluation result."""

    query: str
    retrieval_passed: bool
    answer_keywords_hit: int
    answer_keywords_total: int
    keyword_coverage: float
    ragas_faithfulness: float
    ragas_answer_relevancy: float
    overall_passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class SystemEvalReport:
    """Aggregate report from the golden regression suite."""

    run_id: str
    timestamp: str
    num_golden_queries: int
    pass_rate: float
    mean_keyword_coverage: float
    mean_faithfulness: float
    mean_answer_relevancy: float
    retrieval_report: RetrievalEvalReport | None
    per_query: list[GoldenQueryResult] = field(default_factory=list)
    baseline_retrieval_report: RetrievalEvalReport | None = None

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  SYSTEM EVALUATION  [{self.run_id}]",
            f"  {self.timestamp}   N={self.num_golden_queries}",
            "=" * 60,
            f"  Pass Rate                 {self.pass_rate:.3f}",
            f"  Mean Keyword Coverage     {self.mean_keyword_coverage:.3f}",
            f"  Mean Faithfulness         {self.mean_faithfulness:.3f}",
            f"  Mean Answer Relevancy     {self.mean_answer_relevancy:.3f}",
        ]
        if self.retrieval_report:
            r = self.retrieval_report
            lines += [
                f"  Retrieval Recall@{r.k:<3}     {r.mean_recall:.3f}",
                f"  Retrieval MRR             {r.mean_mrr:.3f}",
            ]
        if self.baseline_retrieval_report:
            b = self.baseline_retrieval_report
            lines += [
                "",
                "  Baseline (dense-only):",
                f"    Recall@{b.k:<3}            {b.mean_recall:.3f}",
                f"    MRR                   {b.mean_mrr:.3f}",
            ]
            if self.retrieval_report:
                recall_lift = self.retrieval_report.mean_recall - b.mean_recall
                mrr_lift = self.retrieval_report.mean_mrr - b.mean_mrr
                lines += [
                    f"  Pipeline Lift (Recall)    {recall_lift:+.3f}",
                    f"  Pipeline Lift (MRR)       {mrr_lift:+.3f}",
                ]
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "num_golden_queries": self.num_golden_queries,
            "pass_rate": round(self.pass_rate, 4),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_answer_relevancy": round(self.mean_answer_relevancy, 4),
        }
        if self.retrieval_report:
            d["retrieval"] = self.retrieval_report.to_dict()
        if self.baseline_retrieval_report:
            d["baseline_retrieval"] = self.baseline_retrieval_report.to_dict()
        return d


class SystemEvaluator:

    def __init__(
        self,
        retrieval_evaluator: RetrievalEvaluator,
        rag_evaluator: RAGEvaluator,
        answer_fn=None,
        score_ledger_path: str | None = None,
        keyword_pass_threshold: float = 0.6,
        k: int = 10,
    ) -> None:
        self._ret_eval = retrieval_evaluator
        self._rag_eval = rag_evaluator
        self._answer_fn = answer_fn
        self._ledger_path = Path(score_ledger_path) if score_ledger_path else None
        self._kw_threshold = keyword_pass_threshold
        self._k = k

    def _check_keywords(self, answer: str, expected_keywords: list[str]) -> tuple:
        if not expected_keywords:
            return 0, 0, 1.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return hits, len(expected_keywords), hits / len(expected_keywords)

    def _make_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    def _append_to_ledger(self, record: dict) -> None:
        if not self._ledger_path:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Score appended to ledger: %s", self._ledger_path)

    def evaluate_golden_query(
        self,
        golden: GoldenQuery,
    ) -> GoldenQueryResult:
        """Run the full pipeline on a single golden query."""

        labeled = golden.to_labeled_query()
        ret_result = self._ret_eval.evaluate_query(labeled, self._k)
        retrieval_passed = ret_result.hit

        kw_hit, kw_total, kw_cov = 0, 0, 1.0
        faithfulness = 0.0
        answer_relevancy = 0.0
        details: dict = {}

        if self._answer_fn is not None:
            try:
                answer, context_texts = self._answer_fn(golden.query, golden.namespace)
                kw_hit, kw_total, kw_cov = self._check_keywords(
                    answer, golden.expected_answer_keywords
                )
                gen_report: EvaluationReport = self._rag_eval.evaluate_with_ragas(
                    query=golden.query,
                    answer=answer,
                    context_texts=context_texts,
                )
                faithfulness = gen_report.generation.faithfulness
                answer_relevancy = gen_report.generation.answer_relevancy
                details["answer_preview"] = answer[:200]
            except Exception as exc:
                logger.error(
                    "Pipeline call failed for golden query '%s': %s",
                    golden.query[:60],
                    exc,
                    exc_info=True,
                )

        overall_passed = retrieval_passed and kw_cov >= self._kw_threshold

        return GoldenQueryResult(
            query=golden.query,
            retrieval_passed=retrieval_passed,
            answer_keywords_hit=kw_hit,
            answer_keywords_total=kw_total,
            keyword_coverage=kw_cov,
            ragas_faithfulness=faithfulness,
            ragas_answer_relevancy=answer_relevancy,
            overall_passed=overall_passed,
            details=details,
        )

    def run_golden_suite(
        self,
        golden_queries: list[GoldenQuery],
        run_baseline: bool = True,
    ) -> SystemEvalReport:
        """Execute the golden regression suite and return an aggregated report."""
        if not golden_queries:
            raise ValueError("golden_queries must not be empty.")

        run_id = self._make_run_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("Golden regression suite: %s  N=%d", run_id, len(golden_queries))

        per_query: list[GoldenQueryResult] = []
        for i, gq in enumerate(golden_queries):
            result = self.evaluate_golden_query(gq)
            per_query.append(result)
            status = "✓" if result.overall_passed else "✗"
            logger.debug(
                "[%d/%d] %s '%s'  kw_cov=%.2f",
                i + 1,
                len(golden_queries),
                status,
                gq.query[:60],
                result.keyword_coverage,
            )

        def _mean(values):
            return sum(values) / len(values) if values else 0.0

        labeled_queries = [gq.to_labeled_query() for gq in golden_queries]
        retrieval_report = self._ret_eval.evaluate(labeled_queries, k=self._k)

        baseline_report: RetrievalEvalReport | None = None
        if run_baseline:
            try:
                baseline_report = self._ret_eval.evaluate_baseline(labeled_queries, k=self._k)
            except Exception as exc:
                logger.warning("Baseline evaluation skipped: %s", exc)

        report = SystemEvalReport(
            run_id=run_id,
            timestamp=timestamp,
            num_golden_queries=len(per_query),
            pass_rate=_mean([float(r.overall_passed) for r in per_query]),
            mean_keyword_coverage=_mean([r.keyword_coverage for r in per_query]),
            mean_faithfulness=_mean([r.ragas_faithfulness for r in per_query]),
            mean_answer_relevancy=_mean([r.ragas_answer_relevancy for r in per_query]),
            retrieval_report=retrieval_report,
            per_query=per_query,
            baseline_retrieval_report=baseline_report,
        )

        logger.info(report.summary())
        self._append_to_ledger(report.to_dict())
        return report

    def load_ledger(self) -> list[dict]:
        """Return all previously recorded runs from the score ledger."""
        if not self._ledger_path or not self._ledger_path.exists():
            return []
        records = []
        with self._ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Could not parse ledger line: %s", line[:80])
        return records

    def print_trend(self, metric: str = "pass_rate", last_n: int = 10) -> None:
        """Print the last N runs for a given metric as an ASCII sparkline."""
        records = self.load_ledger()[-last_n:]
        if not records:
            logger.info("No ledger entries found.")
            return

        values = [r.get(metric, 0.0) for r in records]
        blocks = " ▁▂▃▄▅▆▇█"
        rng = max(values) - min(values) if len(values) > 1 else 1.0
        sparkline = "".join(
            blocks[round((v - min(values)) / rng * 8)] if rng > 0 else "─" for v in values
        )
        logger.info(
            "Trend [%s] last %d runs: %s  (latest=%.3f)",
            metric,
            len(records),
            sparkline,
            values[-1],
        )
