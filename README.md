# Temporal Graph RAG

> ⚠️ **Work in Progress** — This project is under active development and is not yet complete. The current state is an initial skeleton adapted from a graph RAG codebase. The temporal layer (time-aware graph edges, timeline queries, entity tracking over time) has not been built yet.

---

A Retrieval-Augmented Generation (RAG) system designed to track entities — such as patients, diseases, and medications — over time using a knowledge graph. The core use case is clinical: given a patient's records over multiple visits, the system should answer questions like *"how did this patient's biomarkers change after starting treatment X?"* or *"what happened to this character at each stage of the story?"*

The system combines dense vector search, sparse BM25 retrieval, a Neo4j knowledge graph (GraphRAG), and a multi-agent orchestration layer built on the OpenAI Agents SDK.

---

## Architecture Overview

```
User Query
    │
    ▼
QueryUnderstanding  ──►  intent classification · HyDE expansion · NER · decomposition
    │
    ├──► SemanticCache (Redis)  ──►  cache hit → return immediately
    │
    ▼
OrchestratorAgent  ──►  routes to RetrievalAgent · ConversationalAgent · FollowUpAgent
    │
    ▼
HybridSearchEngine
    ├── DenseVectorStore  (Qdrant HNSW, cosine ANN)
    ├── SparseIndex       (Elasticsearch BM25)
    └── GraphRetriever    (Neo4j — LOCAL k-hop · GLOBAL community · HYBRID)
    │
    ▼
Post-Retrieval  ──►  parent expansion · Cohere rerank · context compression
    │
    ▼
Generator  ──►  conflict detection · grounded prompting · citation extraction · faithfulness check
    │
    ▼
PIIGuard  ──►  Presidio output scan
```

---

## Project Structure

```
temporal-graph-rag/
├── src/
│   ├── agents/        # Orchestrator, retrieval agent, conversational agent, graph tools, guardrails
│   ├── config/        # settings.py — typed config loader from config/config.yaml
│   ├── core/          # container.py — dependency wiring and startup
│   ├── evaluation/    # retrieval_eval, indexing_eval, system_eval, cost_latency_eval, evaluator
│   ├── generation/    # generator.py — conflict detection, grounded prompting, citations
│   ├── graphrag/      # extractor, neo4j_store, graph_retriever, community, schema
│   ├── indexing/      # embedder.py, vector_store.py (Qdrant + Elasticsearch)
│   ├── ingestion/     # parser, chunker, consolidator, deduplicator, pipeline, graph_handler
│   ├── operations/    # ops_middleware.py — PII guard, semantic cache, access control, tracing
│   ├── query/         # understanding.py — query rewriting, HyDE, NER, decomposition
│   ├── retrieval/     # retriever.py — hybrid search, RRF, reranking, context management
│   └── utils/         # logger.py, file_utils.py
├── scripts/
│   ├── ingest.py            # CLI: ingest documents into the knowledge base
│   ├── query.py             # CLI: run a query against the system
│   └── build_communities.py # Run Louvain/Leiden community detection (post-ingest)
├── tools/
│   └── suggest_chunk_ids.py # Dev helper for curating golden query sets
├── config/
│   └── config.yaml          # All system configuration
├── knowledge_base/          # Source documents, organized by namespace subfolder
├── models/                  # Fine-tuned model outputs (empty placeholder)
├── data/
│   └── neo4j/               # Neo4j data volume (used by docker-compose)
├── Dockerfile
├── docker-compose.yml       # App + Qdrant + Elasticsearch + Redis + Neo4j
├── pyproject.toml
├── requirements.in
├── requirements.txt
└── requirements-dev.txt
```

---

## Quickstart

### 1. Prerequisites

| Service | Purpose | Default port |
|---|---|---|
| Qdrant | Dense vector store | 6333 |
| Elasticsearch | BM25 sparse index | 9200 |
| Redis | Semantic cache | 6379 |
| Neo4j | Knowledge graph (GraphRAG) | 7687 |

Bring up all services with Docker Compose:

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 3. Configure environment

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=...
COHERE_API_KEY=...
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

### 4. Add documents

Place documents inside `knowledge_base/<namespace>/`. Subfolders become namespaces (e.g. one per patient, one per project).

```
knowledge_base/
└── patient_001/
    ├── visit_2023_01.pdf
    ├── visit_2023_06.pdf
    └── visit_2024_01.pdf
```

### 5. Ingest

```bash
python scripts/ingest.py --namespace patient_001

# Optionally build GraphRAG communities after ingestion
python scripts/build_communities.py
```

### 6. Query

```bash
python scripts/query.py --namespace patient_001 \
  --query "How did the patient's biomarkers change after starting treatment?"
```

---

## GraphRAG

Enable in `config/config.yaml`:

```yaml
graphrag:
  enabled: true
  retrieval:
    mode: "hybrid"      # local | global | hybrid
    local_hop_depth: 2
    community_top_k: 5
```

The extractor uses GPT-4o to pull entities (e.g. Patient, Disease, Drug, Biomarker) and relationships (e.g. DIAGNOSED_WITH, TREATED_WITH, MEASURED_AT) from each chunk, stores them in Neo4j, and builds community summaries via Louvain/Leiden clustering.

Three retrieval modes:
- **LOCAL** — k-hop neighbourhood traversal from entities found in the query
- **GLOBAL** — ANN search over community summaries
- **HYBRID** — all sources fused via Reciprocal Rank Fusion (default)

GraphRAG falls back gracefully to vector-only retrieval if Neo4j is unavailable.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | GPT-4o for generation, extraction, and query understanding |
| `COHERE_API_KEY` | ✅ | Cohere reranker |
| `QDRANT_URL` | ✅ | Qdrant instance URL |
| `ELASTICSEARCH_URL` | ✅ | Elasticsearch instance URL |
| `REDIS_URL` | ✅ | Redis instance URL |
| `NEO4J_URI` | ✅ | Neo4j bolt URI |
| `NEO4J_USERNAME` | ✅ | Neo4j username |
| `NEO4J_PASSWORD` | ✅ | Neo4j password |
| `JWT_SECRET_KEY` | optional | For namespace-level access control |
| `LANGSMITH_API_KEY` | optional | LangSmith tracing |

---

## What's Next (Temporal Layer)

The key work not yet done is making the graph **time-aware**:

- Attaching timestamps to graph edges (e.g. `MEASURED_AT {date: "2023-06"}`)
- Time-scoped graph queries: *"what was the state of this entity at time T?"*
- Timeline retrieval: ordering and reasoning over a sequence of events for a tracked entity
- Temporal query understanding: detecting time references in user questions and translating them to graph filters
