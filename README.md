# Llama Stack Multi-Tenant RAG Security Evaluation

Retrieval-augmented generation (RAG) systems optimize for relevance but typically ignore authorization: a query from Tenant A can retrieve Tenant B's documents if they happen to be semantically similar. This repo evaluates how Llama Stack's access control and orchestration layers close that gap.

We test a 2x2 matrix of configurations against a synthetic multi-tenant workload, measuring both security (does cross-tenant data leak?) and systems performance (what does access control cost?).

## Experiment Design

### Configuration Matrix

|                              | Ungated Retrieval           | Gated Retrieval             |
|------------------------------|-----------------------------|-----------------------------|
| **Client-Side Orchestration** | Config A (baseline)        | Config B                    |
| **Server-Side Orchestration** | Config C                   | Config D (full architecture)|

- **Ungated**: Single shared vector store, no authentication, no ABAC policies. Any user can retrieve any document.
- **Gated**: Per-tenant vector stores with custom authentication and ABAC policies enforcing `user in owners namespaces`.
- **Client-side**: The client manually calls `/v1/vector_stores/{id}/search` then `/v1/chat/completions`.
- **Server-side**: A single call to `/v1/responses` with a `file_search` tool. The server controls the retrieval-generation loop.

### Synthetic Workload

- **3 tenants**: `finance`, `engineering`, `legal`
- **300 documents** (100 per tenant), ~512 tokens each, with controlled topical overlap between tenants
- **300 authorized queries** (100 per tenant): queries that should retrieve same-tenant documents
- **300 cross-tenant probes**: a finance user querying for engineering documents, etc. (should return 0 results under gating)
- **90 prompt injection probes**: adversarial queries attempting to bypass access controls

### Infrastructure

- **Inference**: OpenAI `gpt-4o-mini` via Llama Stack's `remote::openai` provider
- **Embeddings**: OpenAI `text-embedding-3-small` via the same provider
- **Vector store**: `sqlite-vec` (inline, no external dependencies)
- **Auth**: Lightweight FastAPI mock mapping bearer tokens to tenant identities

### Metrics

| Metric | Definition |
|--------|------------|
| **Cross-Tenant Leakage Rate (CTLR)** | Fraction of cross-tenant probes that return at least one chunk from another tenant |
| **Authorization Violation Rate (AVR)** | Fraction of all API calls that return unauthorized data |
| **E2E Latency (p50, p99)** | End-to-end query latency measured with `time.perf_counter()` |
| **ABAC Overhead** | `mean(gated_latency) - mean(ungated_latency)` |

## Results

### Security

| Config | Orchestration | Retrieval | CTLR | AVR |
|--------|--------------|-----------|------|-----|
| A | Client-side | Ungated | 100.0% | 50.0% |
| B | Client-side | Gated | 0.0% | 0.0% |
| C | Server-side | Ungated | 98.3% | 49.5% |
| D | Server-side | Gated | 0.0% | 0.0% |

Gating eliminates cross-tenant leakage entirely (Configs B and D). Without it, nearly all cross-tenant probes return data from other tenants, regardless of whether orchestration is client-side or server-side.

### Latency (Authorized Queries)

| Config | p50 | p99 | Mean |
|--------|-----|-----|------|
| A (client + ungated) | 3,600ms | 10,818ms | 4,208ms |
| B (client + gated) | 3,427ms | 9,795ms | 3,851ms |
| C (server + ungated) | 7,507ms | 16,462ms | 7,620ms |
| D (server + gated) | 6,431ms | 14,623ms | 6,934ms |

ABAC gating adds no measurable latency overhead. Server-side orchestration adds ~3s compared to client-side due to the additional tool execution round-trip through the Responses API.

### Figures

- `figures/security_metrics.pdf` -- Grouped bar chart of CTLR and AVR per config
- `figures/latency_cdfs.pdf` -- CDF overlay of end-to-end latency for all four configs

## Repo Structure

```
configs/                          # Llama Stack server configs for each experiment
  config_a_ungated_client.yaml    # Config A: client-side + ungated
  config_b_gated_client.yaml      # Config B: client-side + gated
  config_c_ungated_server.yaml    # Config C: server-side + ungated
  config_d_gated_server.yaml      # Config D: server-side + gated

scripts/
  auth_server.py                  # Mock auth endpoint (FastAPI)
  generate_data.py                # Synthetic document and query generation
  ingest_data.py                  # Upload documents into vector stores
  client_orchestration.py         # Client-side RAG loop (Configs A, B)
  run_experiment.py               # Main experiment runner
  run_injection_probes.py         # Prompt injection adversarial testing
  analyze_results.py              # Compute metrics and generate figures

data/
  documents/                      # 300 synthetic documents (generated)
  queries/                        # Query workloads (generated)
  results/                        # Per-config raw results and summary

figures/                          # Output PDFs
```

## Running the Experiments

Prerequisites: Python 3.12+, `uv`, and an `OPENAI_API_KEY` environment variable.

```bash
# Install dependencies
uv pip install llama-stack openai fastapi uvicorn matplotlib numpy --python 3.12

# Generate synthetic data
uv run --python 3.12 python scripts/generate_data.py

# For each config (example: Config D)
# 1. Start auth server (gated configs only)
uv run --python 3.12 python scripts/auth_server.py &

# 2. Start Llama Stack server
uv run --python 3.12 llama stack run configs/config_d_gated_server.yaml &

# 3. Ingest documents
uv run --python 3.12 python scripts/ingest_data.py --config D

# 4. Run experiment
uv run --python 3.12 python scripts/run_experiment.py --config D

# 5. After all configs are done, generate figures
uv run --python 3.12 python scripts/analyze_results.py
```
