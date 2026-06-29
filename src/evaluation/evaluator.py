from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config.settings import get_config, get_secrets
from src.utils.logger import get_logger

_RAGAS_RUN_CONFIG = RunConfig(timeout=30, max_retries=1, max_wait=10)

logger = get_logger(__name__)

_RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0


@dataclass
class EvaluationReport:
    query: str
    answer: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    ragas_scores: dict[str, float] = field(default_factory=dict)
    custom_judge_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0


def _avg(values: list[float]) -> float:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _reset_ragas_state(metrics) -> None:
    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = None
        if hasattr(m, "embeddings"):
            m.embeddings = None


def _report_from_scores(query: str, answer: str, scores: dict[str, float]) -> EvaluationReport:
    report = EvaluationReport(query=query, answer=answer)
    report.ragas_scores = scores
    report.generation.faithfulness = scores.get("faithfulness", 0.0)
    report.generation.answer_relevancy = scores.get("answer_relevancy", 0.0)
    report.retrieval.context_precision = scores.get("context_precision", 0.0)
    report.retrieval.context_recall = scores.get("context_recall", 0.0)
    report.overall_score = _avg(list(scores.values()))
    return report


def _log_batch_summary(report: EvaluationReport) -> None:
    logger.info(
        "\n%s\n  RAGAS BATCH RESULTS\n"
        "  Faithfulness:      %.3f\n"
        "  Answer Relevancy:  %.3f\n"
        "  Context Precision: %.3f\n"
        "  Context Recall:    %.3f\n"
        "  Overall Score:     %.3f\n%s",
        "=" * 50,
        report.generation.faithfulness,
        report.generation.answer_relevancy,
        report.retrieval.context_precision,
        report.retrieval.context_recall,
        report.overall_score,
        "=" * 50,
    )


class RAGEvaluator:
    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._eval_cfg = cfg.evaluation
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)

        judge_cfg = self._eval_cfg["llm_judge"]
        self._judge_enabled: bool = judge_cfg["enabled"]
        self._judge_method: str = judge_cfg.get("method", "ragas")

        self._ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(model=judge_cfg["model"], api_key=sec.openai_api_key, temperature=0)
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self._eval_cfg.get("embedding_model", "text-embedding-3-small"),
                api_key=sec.openai_api_key,
            )
        )

        self._reference_embeddings: list[np.ndarray] = []
        self._reference_window = self._eval_cfg["drift_detection"]["reference_window"]

    def evaluate_online(
        self,
        query: str,
        answer: str,
        context_texts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationReport:
        """Route to the configured evaluation method(s) at inference time."""
        if not self._judge_enabled:
            return EvaluationReport(query=query, answer=answer)

        if self._judge_method == "custom":
            return self.evaluate_with_custom_judge(query, answer, context_texts)

        if self._judge_method == "both":
            report = self.evaluate_with_ragas(query, answer, context_texts, ground_truth)
            custom = self.evaluate_with_custom_judge(query, answer, context_texts)
            report.custom_judge_scores = custom.custom_judge_scores
            return report

        return self.evaluate_with_ragas(query, answer, context_texts, ground_truth)

    def _run_ragas(
        self,
        dataset: EvaluationDataset,
        has_ground_truth: bool,
        show_progress: bool = False,
    ) -> Any:
        metrics = [faithfulness, answer_relevancy]
        if has_ground_truth:
            metrics += [context_precision, context_recall]

        _reset_ragas_state(_RAGAS_METRICS)

        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
            raise_exceptions=False,
            show_progress=show_progress,
            run_config=_RAGAS_RUN_CONFIG,
        )

    def evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        context_texts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationReport:
        if not self._judge_enabled:
            return EvaluationReport(query=query, answer=answer)

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=context_texts,
            reference=ground_truth,
        )
        result = self._run_ragas(
            EvaluationDataset(samples=[sample]),
            has_ground_truth=bool(ground_truth),
        )

        scores = dict(result.scores[0]) if result.scores else {}
        report = _report_from_scores(query, answer, scores)
        logger.info("RAGAS evaluation complete", extra={"scores": scores})
        return report

    def evaluate_batch_with_ragas(
        self,
        samples: list[dict[str, Any]],
    ) -> list[EvaluationReport]:
        ragas_samples = [
            SingleTurnSample(
                user_input=s["query"],
                response=s["answer"],
                retrieved_contexts=s["context_texts"],
                reference=s.get("ground_truth"),
            )
            for s in samples
        ]

        result = self._run_ragas(
            EvaluationDataset(samples=ragas_samples),
            has_ground_truth=any(s.get("ground_truth") for s in samples),
            show_progress=True,
        )

        reports = [
            _report_from_scores(
                s["query"],
                s["answer"],
                dict(result.scores[i]) if i < len(result.scores) else {},
            )
            for i, s in enumerate(samples)
        ]

        if reports:
            _log_batch_summary(reports[-1])

        return reports

    def evaluate_with_custom_judge(
        self,
        query: str,
        answer: str,
        context_texts: list[str],
    ) -> EvaluationReport:
        report = EvaluationReport(query=query, answer=answer)

        if not self._judge_enabled:
            return report

        judge_cfg = self._eval_cfg["llm_judge"]
        criteria = judge_cfg["criteria"]
        context_str = "\n---\n".join(context_texts[:5])
        criteria_list = "\n".join(f"- {c}" for c in criteria)

        prompt = (
            f"Evaluate the following question-answer pair on these criteria:\n{criteria_list}\n\n"
            f"Question: {query}\n\n"
            f"Context (retrieved passages):\n{context_str}\n\n"
            f"Answer: {answer}\n\n"
            "For each criterion, assign a score from 0.0 to 1.0 and a one-sentence justification. "
            "Return ONLY valid JSON: "
            '{"faithfulness": {"score": 0.9, "reason": "..."}, '
            '"relevance": {"score": 0.8, "reason": "..."}, ...}'
        )

        try:
            response = self._openai.chat.completions.create(
                model=judge_cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            parsed = json.loads(response.choices[0].message.content.strip())
            report.custom_judge_scores = {
                c: float(parsed[c]["score"]) for c in criteria if c in parsed
            }
            report.overall_score = _avg(list(report.custom_judge_scores.values()))
            logger.info(
                "Custom judge: overall=%.2f", report.overall_score, extra=report.custom_judge_scores
            )
        except Exception as exc:
            logger.error("Custom judge failed: %s", exc)

        return report

    def generate_synthetic_qa(self, chunk_text: str, n: int = 3) -> list[dict[str, str]]:
        prompt = (
            f"Generate {n} diverse, specific question-answer pairs from the passage below. "
            "Answers must be fully grounded in the passage text. "
            'Return ONLY valid JSON: [{"question": "...", "answer": "..."}, ...]\n\n'
            f"Passage: {chunk_text[:1000]}"
        )
        try:
            response = self._openai.chat.completions.create(
                model=self._eval_cfg["synthetic_qa"]["generator_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600,
            )
            return json.loads(response.choices[0].message.content.strip())[:n]
        except Exception as exc:
            logger.warning("Synthetic QA generation failed: %s", exc)
            return []

    def build_synthetic_eval_batch(
        self, chunks: list[str], n_per_chunk: int = 3
    ) -> list[dict[str, Any]]:
        samples = []
        for chunk in chunks:
            for pair in self.generate_synthetic_qa(chunk, n=n_per_chunk):
                samples.append(
                    {
                        "query": pair["question"],
                        "answer": pair["answer"],
                        "context_texts": [chunk],
                        "ground_truth": pair["answer"],
                    }
                )
        return samples

    def evaluate_retrieval(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 10,
    ) -> RetrievalMetrics:
        retrieved_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)

        hits = sum(1 for cid in retrieved_k if cid in relevant_set)
        recall_at_k = hits / len(relevant_set) if relevant_set else 0.0

        mrr = next(
            (1.0 / rank for rank, cid in enumerate(retrieved_k, 1) if cid in relevant_set),
            0.0,
        )

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, cid in enumerate(retrieved_k, 1)
            if cid in relevant_set
        )
        ideal = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics = RetrievalMetrics(recall_at_k=recall_at_k, ndcg=ndcg, mrr=mrr)
        logger.info(
            "Retrieval metrics: recall@%d=%.3f NDCG=%.3f MRR=%.3f", k, recall_at_k, ndcg, mrr
        )
        return metrics

    def update_reference_distribution(self, query_embedding: np.ndarray) -> None:
        self._reference_embeddings.append(query_embedding)
        if len(self._reference_embeddings) > self._reference_window:
            self._reference_embeddings.pop(0)

        if len(self._reference_embeddings) >= 50 and len(self._reference_embeddings) % 100 == 0:
            recent = self._reference_embeddings[-20:]
            drifted, score = self.detect_drift(recent)
            if drifted:
                logger.warning("Query drift detected: score=%.3f", score)

    def detect_drift(self, recent_embeddings: list[np.ndarray]) -> tuple[bool, float]:
        drift_cfg = self._eval_cfg["drift_detection"]

        if not drift_cfg["enabled"] or len(self._reference_embeddings) < 50:
            return False, 0.0

        ref_proj = np.stack(self._reference_embeddings)[:, :32].mean(axis=0)
        cur_proj = np.stack(recent_embeddings)[:, :32].mean(axis=0)

        ref_norm = ref_proj / (np.linalg.norm(ref_proj) + 1e-8)
        cur_norm = cur_proj / (np.linalg.norm(cur_proj) + 1e-8)
        divergence = float(np.linalg.norm(ref_norm - cur_norm))

        threshold = drift_cfg["drift_threshold"]
        drift_detected = divergence > threshold

        if drift_detected:
            logger.warning(
                "Query drift detected: divergence=%.3f > threshold=%.3f",
                divergence,
                threshold,
                extra={"divergence": divergence, "threshold": threshold},
            )
        else:
            logger.debug("No drift detected: divergence=%.3f", divergence)

        return drift_detected, divergence
