# Local RAG Assistant — Advanced Offline RAG with Foundry Local

An offline-first document Q&A assistant that answers questions grounded in
your own documents (Markdown, PDF, Word, Excel/CSV), running entirely
on-device with [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/).

This isn't a naive "embed and cosine-similarity" RAG demo. Documents are
parsed into a **hierarchical knowledge tree** (headings, tables, figures,
warnings — not flat text), retrieval combines **hybrid search (dense +
BM25)**, **cross-encoder re-ranking**, **retrieval grading**, and **query
expansion**, and every stage exposes its scores and latency so the
mechanism is visible, not a black box.

The same codebase can also switch to a **cloud mode** (Azure AI Search +
Azure OpenAI) with a single config flag, since Foundry Local exposes an
OpenAI-compatible API. Local-to-cloud portability is a config change, not a
rewrite.

> Full roadmap and design rationale for every architectural decision:
> [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Why this exists

Most AI assistants assume a stable connection to the cloud. This one
doesn't. It's built for the scenario where a user has no internet access at
all — a field engineer, an air-gapped facility, a regulated environment —
and it optionally upgrades to a cloud-backed setup only when that tradeoff
is worth it (bigger document sets, shared team access, no local hardware).

## What makes this different from a tutorial RAG project

| Naive RAG (typical tutorial) | This project |
|---|---|
| Single dense (embedding) retrieval | Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, index built once and cached |
| Top-K by raw similarity score | Cross-encoder re-ranking on top-K candidates before generation |
| Always trusts retrieved chunks | Retrieval grader checks relevance before the LLM sees the context; falls back to an honest "insufficient context" reply instead of hallucinating or returning a canned answer |
| Flat text, fixed-size chunking | Documents parsed into a **hierarchical node tree** (heading/paragraph/table/figure/warning/note), chunked on heading boundaries with atomic tables/figures never split mid-content |
| Markdown only | Markdown, PDF, DOCX, XLSX/CSV via a pluggable parser interface, each producing the same node-tree structure |
| No visibility into *why* an answer was produced | Explainability panel: per-chunk BM25/dense/rerank scores, section/heading provenance, latency breakdown per pipeline stage |
| Re-embeds every file on every run | Hash-based incremental ingestion: unchanged files are skipped entirely, changed files are cleanly re-indexed |
| Silently degrades on small/local models | Repetition-loop detection catches degenerate output and reports failure honestly instead of returning garbage |

## Architecture

```mermaid
flowchart TB
    subgraph Client["Client"]
        UI["Chat UI — dashboard with latency trace,\nsource citations, explainability panel"]
    end

    subgraph Server["Server — FastAPI (api/app.py)"]
        API["/chat  /upload  /documents  /health"]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        PARSE["Format parsers (src/parsers/)\nMarkdown / PDF / DOCX / XLSX\nAll produce a KnowledgeNode tree:\nheading / paragraph / table / figure / warning / note / code"]
        CHUNK["chunk_nodes() (src/chunking.py)\nHeading-boundary grouping\nTables/figures/warnings indexed atomically (never split)"]
        HASH["Hash-based dedup (scripts/ingest.py)\nSkip unchanged files, wipe+reindex changed ones"]
    end

    subgraph Retrieval["Retrieval Pipeline (src/retrieval/)"]
        RW["Query expansion (query_rewriter.py)\nDeterministic synonym-based rewriting\n(grease→lubricant, spindle→axis, scara→arm)\n— not LLM/agentic yet"]
        HYB["Hybrid search (hybrid.py)\nBM25 + dense cosine, RRF fusion\nIndex built once, cached across queries"]
        RR["Cross-encoder re-ranker (reranker.py)\nbge-reranker-base"]
        GR["Retrieval grader (grader.py)\nJaccard dedup + rerank-score / keyword-hit filter"]
        COMP["Compression + parent expansion (compression.py)\nAtomic nodes pass through unpruned\nParent heading looked up from knowledge_nodes"]
    end

    subgraph Local["Local mode"]
        SQL[("SQLite — data/rag.db\nknowledge_nodes / document_chunks / documents")]
        FL_CHAT["Foundry Local chat model\nphi-4-mini-instruct-generic-gpu:5"]
        FL_EMB["Local embedding model\nSentenceTransformer all-MiniLM-L6-v2"]
    end

    subgraph Cloud["Cloud mode (optional, Azure)"]
        AIS[("Azure AI Search")]
        BLOB[("Azure Blob Storage")]
        AOAI["Azure OpenAI (gpt-4o-mini)"]
    end

    GEN["Generation (llm_client.py)\nRepetition-penalty sampling\nRepetition-loop detection on output\nNo hardcoded fallback answers — honest failure reporting"]
    TEL["Telemetry: per-stage latency + chunks_matrix\nsurfaced live in the Explainability panel"]

    UI --> API
    PARSE --> CHUNK --> HASH --> SQL
    API --> RW --> HYB --> RR --> GR --> COMP --> GEN
    HYB -->|MODE=local| SQL
    HYB -->|MODE=cloud| AIS
    FL_EMB -.embeds queries + chunks.-> HYB
    GEN -->|MODE=local| FL_CHAT
    GEN -->|MODE=cloud| AOAI
    BLOB -.sync.-> SQL
    BLOB -.sync.-> AIS
    API -.-> TEL
    GEN --> API
```

**Local mode**: everything runs on the machine, no network calls after the
one-time model download. Documents are parsed into a knowledge-node tree,
chunked on heading boundaries, embedded with a local SentenceTransformer
model, and indexed in SQLite with both dense vectors and BM25 term
statistics. Chat generation runs through Foundry Local.

**Cloud mode**: the same pipeline logic, but retrieval goes through Azure AI
Search and generation through Azure OpenAI. Useful for larger document sets,
shared/team access, or when local hardware isn't available.

### RAG query flow (`process_chat_query()`, step by step)

This is what actually runs on every chat request — no step here is
aspirational, all of it is implemented in `src/rag_pipeline.py`.

```mermaid
flowchart TB
    Q(["User query"]) --> P1

    subgraph P1["1 — Query Expansion"]
        RW["rewrite_query()\nDeterministic synonym substitution\nUp to 3 query tracks total"]
    end

    P1 --> P2["2 — Embed query\n(local SentenceTransformer or Azure OpenAI embeddings)"]

    P2 --> P3

    subgraph P3["3 — Multi-track Hybrid Retrieval"]
        direction TB
        L["For each query track:"]
        L --> BM["BM25 sparse score\n(cached index)"]
        L --> DN["Dense cosine similarity\n(vectorized matrix product)"]
        BM --> RRF["Reciprocal Rank Fusion"]
        DN --> RRF
        RRF --> DD["Dedup by chunk id across tracks"]
    end

    P3 --> P4

    subgraph P4["4 — Rerank / Grade / Compress"]
        direction TB
        RR2["Cross-encoder rerank\ntop_n = 6"]
        GR2["Grade: Jaccard dedup +\nrerank_score > 0.0 OR keyword hit_ratio ≥ 0.2\n(atomic nodes exempt from Jaccard dedup)"]
        CO2["Compress:\ntables/figures/warnings pass through unpruned\ncover-page cleanup, sentence-window pruning otherwise"]
        PE["Parent lookup\nget_node(parent_id) → expanded_heading"]
        RR2 --> GR2 --> CO2 --> PE
    end

    PE --> BUD["Adaptive chunk budget\ntop_score > 1.2 → 2 · > 0.65 → 4 · else → 5"]

    BUD --> P5["5 — Context packaging\nper-chunk source/page/type/section header\n+ chunks_matrix for UI telemetry"]

    P5 --> P6

    subgraph P6["6 — Generation"]
        direction TB
        PR["Prompt: answer only from chunks,\nsynthesize across all, no chunk-by-chunk narration"]
        GN["Foundry Local chat model\nfrequency/presence penalty applied"]
        GD{"Repetition loop\ndetected?"}
        PR --> GN --> GD
    end

    GD -->|yes / generation error| FAIL["Honest failure message\nchunks still listed under Reference"]
    GD -->|no| OK["Final answer + Reference\n+ telemetry + chunks_matrix"]

    KN[("knowledge_nodes")] -.-> PE
    DC[("document_chunks")] -.-> P3
```

**What's deliberately *not* in this flow yet** (see Feature status below):
there is no LLM-driven query planner and no multi-hop loop — query expansion
is static synonym substitution, and parent-context expansion looks up a
single parent node rather than reconstructing a full section from all its
children. Both are tracked in the roadmap, not claimed as done.

## Feature status

Legend: [x] implemented · [~] in progress / partial · [ ] planned (see roadmap for order)

**Ingestion**
- [x] Markdown, PDF, DOCX, XLSX/CSV parsers, all producing a shared
      hierarchical node tree (heading/paragraph/table/figure/warning/note/code)
- [x] Heading-boundary structural parsing (numbering + font/style based
      heading detection, parent/child linking)
- [x] Table extraction to Markdown tables (atomic nodes, never split)
- [x] Figure detection (placeholder nodes; vision captioning not yet implemented)
- [x] Heading-boundary chunking (replaces fixed-size chunking for tree-aware parsers)
- [x] Incremental ingestion (SHA-256 content hash dedup: skip unchanged,
      wipe + reindex changed files)

**Retrieval pipeline**
- [x] Hybrid search (BM25 + dense, RRF fusion), index built once and cached
- [x] Cross-encoder re-ranking
- [x] Query expansion (deterministic synonym substitution — not LLM-based)
- [x] Retrieval grader (relevance threshold + Jaccard dedup)
- [x] Context compression (sentence-window pruning; atomic nodes exempt)
- [~] Parent-context expansion (single-level parent heading lookup;
      full child-node reconstruction not yet implemented)
- [ ] LLM-driven query planning / agentic multi-hop loop
- [ ] Document graph / entity graph (GraphRAG-style)
- [ ] Vision-model figure captioning

**Generation reliability**
- [x] Repetition-loop detection (catches degenerate small-model output,
      reports failure honestly instead of returning it)
- [x] No hardcoded/canned fallback answers — failures are reported as failures
- [x] Configurable generation params (max_tokens, temperature, repetition penalties)

**UI / Observability**
- [x] Chat interface with per-query advanced-mode toggle
- [x] Real per-stage latency trace (wired to actual `telemetry` dict, not placeholders)
- [x] Explainability panel with real per-chunk source/page/section/rerank score
- [x] Live knowledge-base document list (`GET /documents`)
- [ ] Source citation viewer (click-through to original page/bbox)

**Engineering**
- [x] Local/cloud mode switch via config
- [x] Configurable Foundry Local base URL (port is not assumed stable across restarts)
- [ ] Test suite (pytest, unit + integration)
- [ ] Structured logging with request tracing
- [ ] CI pipeline (lint + tests on push)

**Evaluation**
- [ ] Labeled eval set (20-30 Q&A pairs with ground-truth sources)
- [ ] Retrieval metrics (Precision@K, Recall@K, MRR)
- [ ] Generation faithfulness scoring (local LLM-as-judge)
- [ ] Automated benchmark report (naive vs. advanced comparison)

Full detail, rationale, and build order for every item above:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Tech stack

| Layer | Local mode | Cloud mode |
|---|---|---|
| Server | FastAPI | FastAPI (same app) |
| Parsers | Markdown, PDF (`pdfplumber`), DOCX (`python-docx`), XLSX/CSV (`pandas`) — all tree-aware | same |
| Embeddings | `SentenceTransformer` (`all-MiniLM-L6-v2`), local, CPU | Azure OpenAI embeddings / Azure AI Search vectorizer |
| Chat generation | Foundry Local (`Phi-4-mini-instruct-generic-gpu:5`) | Azure OpenAI (`gpt-4o-mini`) |
| Sparse retrieval | `rank-bm25`, index cached in memory | Azure AI Search (built-in) |
| Re-ranking | Local cross-encoder (`bge-reranker-base`) | Azure AI Search semantic ranker (optional) |
| Storage | SQLite (`data/rag.db`) — knowledge node tree + chunks + BM25/dense index + document registry | Azure Blob Storage + Azure AI Search index |
| Telemetry | In-response per-stage timings, rendered in the UI | Application Insights |

> **Local chat model:** this project currently runs
> `Phi-4-mini-instruct-generic-gpu:5` (3.72 GB, MIT license) via Foundry
> Local — small enough to run comfortably on 16GB unified memory (e.g.
> Apple Silicon M-series), while being far less prone to repetition-loop
> failures than sub-1B models on multi-chunk synthesis prompts. Swap it in
> `.env` via `FOUNDRY_CHAT_MODEL` if you have the hardware for something
> larger (`foundry model list` shows what's available).
>
> **Embedding model is intentionally separate from the chat model** — it's
> always the local `all-MiniLM-L6-v2` SentenceTransformer regardless of
> which chat model you pick, since embedding and chat are different tasks
> requiring different models.

## Project layout

```
├── api/app.py                FastAPI app: routes, serves the dashboard UI
├── src/
│   ├── config.py              Central config, reads .env, MODE switch
│   ├── db.py                  SQLite schema: knowledge_nodes, document_chunks,
│   │                          documents registry; schema migration on startup
│   ├── chunking.py             chunk_nodes() (tree-aware) + chunk_document() (legacy)
│   ├── parsers/                Pluggable document parsers, all tree-aware
│   │   ├── base.py              KnowledgeNode model, NodeType enum, parser interface
│   │   ├── markdown_parser.py
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   └── xlsx_parser.py
│   ├── retrieval/
│   │   ├── hybrid.py             BM25 + dense fusion (RRF), cached index
│   │   ├── reranker.py           Cross-encoder re-ranking
│   │   ├── grader.py             Retrieval relevance grading + Jaccard dedup
│   │   ├── query_rewriter.py     Deterministic synonym-based query expansion
│   │   └── compression.py        Sentence-window pruning + parent heading lookup
│   ├── llm_client.py           Foundry Local + Azure OpenAI client wrappers
│   ├── rag_pipeline.py         Orchestrates expansion -> retrieval -> generation
│   ├── azure_search.py         Azure AI Search index + query helpers
│   ├── azure_storage.py        Blob Storage document sync
│   └── telemetry.py            (placeholder — not yet implemented)
├── scripts/
│   ├── ingest.py                Parse + chunk + embed + index, with hash-based dedup
│   ├── sync_azure.py            Push docs to Blob Storage + Azure AI Search
│   └── run_eval.py              Benchmark harness (planned)
├── static/                     Dashboard UI (chat, latency trace, explainability panel)
├── docs/
│   ├── ROADMAP.md               Full advanced-RAG roadmap and build order
│   ├── sample_docs/             Example knowledge base (multi-format)
│   └── eval_set.json            Labeled Q&A pairs for benchmarking (planned)
├── tests/                       Unit + integration tests (planned)
├── data/                        SQLite DB (gitignored)
├── .env.example                 Template for environment variables
└── requirements.txt
```

## Setup — local mode (no Azure needed)

**1. Install Foundry Local**

```bash
# Windows
winget install Microsoft.FoundryLocal

# macOS
brew install microsoft/foundrylocal/foundrylocal
```

**2. Start Foundry Local and pull the chat model**

```bash
foundry service start
foundry model run Phi-4-mini-instruct-generic-gpu:5
```

`foundry service start` prints the local port it's listening on — it is
**not guaranteed to stay the same across restarts**. Note the URL it
prints (e.g. `http://127.0.0.1:49327/`).

**3. Python environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Leave `MODE=local` in `.env`. Set `FOUNDRY_BASE_URL` to match the port
Foundry Local printed in step 2, and `FOUNDRY_CHAT_MODEL` to
`Phi-4-mini-instruct-generic-gpu:5` (or whatever model you pulled).

**4. Ingest the sample documents**

```bash
python scripts/ingest.py
```

Parses every supported file in `docs/sample_docs/` into a knowledge-node
tree (headings, paragraphs, tables, figures), chunks it on heading
boundaries, generates embeddings and BM25 statistics, and indexes
everything in `data/rag.db`. Re-running this is safe and fast — unchanged
files are skipped via content hash comparison.

**5. Run the app**

```bash
uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000`. Turn off Wi-Fi and it still works.

## Setup — cloud mode (optional, Azure)

Requires an Azure subscription (e.g. the Azure for Students $100 credit).

**1. Provision resources** — Azure AI Search (Free tier), a Storage Account,
an Azure OpenAI resource with a `gpt-4o-mini` deployment, and Application
Insights. See `scripts/sync_azure.py` header comment for exact SKUs.

**2. Fill in `.env`**

```
MODE=cloud
AZURE_SEARCH_ENDPOINT=...
AZURE_SEARCH_KEY=...
AZURE_SEARCH_INDEX=rag-index
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
APPLICATIONINSIGHTS_CONNECTION_STRING=...
```

**3. Sync documents and switch mode**

```bash
python scripts/sync_azure.py
uvicorn api.app:app --reload
```

The dashboard UI and API surface are identical — only `MODE` changes.

## Evaluation

Planned, not yet implemented (see Feature status above and
[`docs/ROADMAP.md`](docs/ROADMAP.md)):

```bash
python scripts/run_eval.py
```

Will run a labeled eval set (`docs/eval_set.json`) against both naive and
advanced retrieval configurations, and write a comparison report
(Precision@K, Recall@K, MRR, faithfulness) to `docs/eval_report.md`.

## Cost & safety notes (cloud mode)

- Azure AI Search Free tier and Application Insights' free ingestion quota
  cover this project's needs at $0.
- Azure OpenAI is billed per token — set a **budget alert** in the Azure
  portal before testing.
- Never commit `.env`. `.env.example` is the only file that should be
  tracked.
- If deploying a public demo, put a request-rate limit in front of the
  `/chat` endpoint so a public repo doesn't turn into an open tap on your
  credit.

## Testing

Not yet implemented — tracked in Feature status / roadmap.

```bash
pytest tests/
```

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full advanced-RAG build
plan, prioritized day-by-day, with the reasoning behind each architectural
choice.

## Changelog

**docs(readme): rewrite architecture to match current node-tree RAG implementation**

Replaced the aspirational architecture diagram (agentic sub-query router,
loop/scratchpad retrieval) with one matching the actual
`process_chat_query()` flow. Added a step-by-step RAG query flow diagram.
Corrected the feature status table: parent-child expansion marked partial
(parent-only lookup, no child reconstruction), agentic multi-hop downgraded
to planned (query expansion is deterministic synonym substitution, not
LLM-based). Updated tech stack and setup docs for
`Phi-4-mini-instruct-generic-gpu:5` and the configurable Foundry Local base
URL. Marked incremental ingestion as implemented.

## License

MIT — see `LICENSE`.