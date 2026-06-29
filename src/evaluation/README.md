# src/evaluation — Seven-Layer Offline Evaluation Suite

This package implements a layered evaluation framework that runs independently of the online RAGAS evaluation performed at inference time. It is invoked via `scripts/evaluate_offline.py`.

---

## Layers at a glance

| Layer | Flag | Module | What it checks |
|---|---|---|---|
| 1 — Indexing | `--layer1` | `indexing_eval.py` | Chunk coherence (LLM judge), embedding similarity spot-checks, entity coverage |
| 2 — Retrieval | `--layer2` | `retrieval_eval.py` | Precision@K, Recall@K, MRR, Hit Rate@K, NDCG@K — no LLM calls |
| 3 — Generation | inline | `evaluator.py` | Online RAGAS faithfulness / answer relevancy (runs at inference, not offline) |
| 4 — System | `--layer4` | `system_eval.py` | Golden regression suite + naive dense-only baseline comparison |
| 5 — Cost & Latency | `--layer5` | `cost_latency_eval.py` | Per-query USD cost, token count, p50/p95/p99 latency vs SLO thresholds |
| 6 — Multi-turn | `--layer6` | `multiturn_eval.py` | Context carryover (pronoun resolution), follow-up retrieval, session coherence |
| 7 — Fairness | `--layer7` | `fairness_eval.py` | Retrieval and answer consistency across counterfactual demographic query pairs |

Layers 1, 2, and 4 run by default. Layers 5–7 are opt-in because they call the full pipeline (Layer 5 always; Layers 6–7 only with `--answer`).

---

## Module reference

### `evaluator.py` — Online RAG evaluation (`RAGEvaluator`)

Runs at inference time inside `OpsMiddleware`. Wraps RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`) and an optional custom LLM judge. Returns an `EvaluationReport` used to:
- enforce the faithfulness guardrail (blocks responses below threshold 0.40)
- log per-query metrics for drift detection
- feed the synthetic QA generator for continuous golden set expansion

Key classes: `EvaluationReport`, `GenerationMetrics`, `RetrievalMetrics`, `RAGEvaluator`.

---

### `retrieval_eval.py` — Retrieval metrics (`RetrievalEvaluator`)

Computes ranking metrics against labeled golden queries. No LLM calls — purely set-intersection and rank-based arithmetic.

Metrics computed at cutoff K (default 10):
- **Precision@K** — fraction of top-K retrieved chunks that are relevant
- **Recall@K** — fraction of relevant chunks found in top-K
- **Hit Rate@K** — binary: ≥1 relevant chunk in top-K
- **MRR** — Mean Reciprocal Rank of the first relevant chunk
- **NDCG@K** — Normalized Discounted Cumulative Gain

Input: `LabeledQuery` objects loaded from `tests/golden_queries.json`.
Output: `RetrievalEvalReport` (per-query results + aggregate means).

Also supports `evaluate_baseline()` — runs the same metrics with dense-only retrieval (no BM25, no reranking) for pipeline lift comparison.

---

### `indexing_eval.py` — Indexing quality (`IndexingEvaluator`)

Three checks on the indexed chunks:

1. **Chunk coherence** — LLM judge scores each chunk 0–1 on completeness and self-containedness. Chunks below `coherence_threshold` (default 0.6) are flagged.
2. **Embedding quality** — Cosine similarity spot-check on known-related and known-unrelated chunk pairs. Related pairs should exceed `related_threshold` (0.65); unrelated pairs should stay below `unrelated_ceiling` (0.35).
3. **Entity coverage** — Verifies that expected domain entities (e.g. "Ab(1-42)", "phospho-tau-181") appear in at least one indexed chunk.

Output: `IndexingEvalReport`.

---

### `system_eval.py` — Golden regression suite (`SystemEvaluator`)

Runs each `GoldenQuery` through the full retrieval + generation pipeline and checks:
- Retrieval hit (via `RetrievalEvaluator`)
- Keyword coverage: fraction of `expected_answer_keywords` present in the generated answer
- RAGAS faithfulness and answer relevancy (if `answer_fn` provided)

A query passes if it retrieves at least one relevant chunk and achieves ≥ `keyword_pass_threshold` (default 0.6) keyword coverage.

Optionally runs a dense-only baseline for side-by-side comparison. All results are appended to a JSON-lines score ledger for trend tracking.

Key classes: `GoldenQuery`, `GoldenQueryResult`, `SystemEvalReport`, `SystemEvaluator`.

---

### `cost_latency_eval.py` — Cost & latency SLOs (`CostLatencyEvaluator`)

Times each golden query end-to-end and computes:
- **Token counts** — prompt, completion, and embedding tokens per query
- **USD cost** — from a built-in pricing table (GPT-4o, GPT-4-turbo, GPT-4o-mini, text-embedding-3-*, Cohere rerank)
- **Latency** — wall-clock time; p50/p95/p99 computed across the query set

SLO thresholds are set in `config/config.yaml` under `evaluation.cost_latency_slo`. A query fails if it exceeds `max_cost_per_query_usd`, `max_tokens_per_query`, or the p99 latency budget.

Results are tagged `"report_type": "cost_latency"` in the score ledger.

Key classes: `QueryCostLatency`, `CostLatencyReport`, `CostLatencyEvaluator`.

---

### `multiturn_eval.py` — Multi-turn evaluation (`MultiTurnEvaluator`)

Evaluates conversational robustness across the golden conversations in `tests/golden_conversations.json`. For each turn:

1. **Condensation check** — `QueryUnderstanding.condense_with_history()` is called on the raw follow-up query. The resulting standalone query must contain `condensation_must_contain` terms and must not contain `condensation_must_not_contain` terms (typically pronouns like "its" or dangling references).
2. **Retrieval check** — the condensed query is evaluated against `relevant_chunk_ids` via `RetrievalEvaluator`.
3. **Keyword and faithfulness checks** — with `--answer`, the full pipeline answer is keyword-checked and RAGAS-scored per turn.
4. **Session coherence** — `GenerationResult.has_conflict` is aggregated; `session_coherence_score = 1 − conflict_rate`.

Results are tagged `"report_type": "multi_turn"` in the score ledger.

Key helpers: `_word_present`, `_check_condensation`, `_check_keywords`, `load_golden_conversations`.
Key classes: `ConversationTurn`, `GoldenConversation`, `TurnResult`, `ConversationResult`, `MultiTurnEvalReport`.

---

### `fairness_eval.py` — Fairness evaluation (`FairnessEvaluator`)

Evaluates retrieval and answer consistency across counterfactual query pairs in `tests/golden_fairness_pairs.json`. Each pair holds two or more variants of the same underlying question that differ only in a demographic descriptor (age, sex, education, study cohort).

For each pair:
- **Retrieval Jaccard similarity** — mean pairwise overlap of retrieved chunk-ID sets across variants. Below `evaluation.fairness.retrieval_jaccard_threshold` → flagged for review.
- **Answer similarity** (with `--answer`) — mean pairwise cosine similarity of answer embeddings across variants. Below `evaluation.fairness.answer_similarity_threshold` → flagged for review.

Flagging is a signal for human review, not an automated bias verdict — legitimate source-grounded subgroup differences (e.g. age-stratified reference ranges) would also appear here.

Results are tagged `"report_type": "fairness"` in the score ledger.

---

## Golden datasets

| File | Used by | Contents |
|---|---|---|
| `tests/golden_queries.json` | Layers 2, 4 | 15 AD biomarker queries with real Qdrant chunk IDs, keywords, namespace |
| `tests/golden_conversations.json` | Layer 6 | 3 multi-turn AD conversations with condensation constraints per follow-up turn |
| `tests/golden_fairness_pairs.json` | Layer 7 | 10 counterfactual pairs across age, sex, education, cohort dimensions |

All chunk IDs are real paragraph-level IDs extracted from Qdrant (`namespace: "Alzheimer's disease"`, source: `Biomarkers for AD.pdf`).

---

## Score ledger

All layers append JSON-lines records to a shared ledger (e.g. `eval_results/scores.jsonl`). Each record is tagged with `run_id`, `timestamp`, and `report_type` (absent for `SystemEvaluator`, `"cost_latency"` / `"multi_turn"` / `"fairness"` for the others). Use `SystemEvaluator.print_trend()` to render an ASCII sparkline of a metric over recent runs.
