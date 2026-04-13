# Experiment 1: Cross-Tenant Data Leakage Under Orchestration and Access Control Variants

## Motivation

RAG systems retrieve documents based on semantic similarity, not authorization. In a multi-tenant deployment, a query from one tenant can surface documents belonging to another tenant simply because the content is topically related. This experiment quantifies how much cross-tenant data leakage occurs across four configurations that vary the orchestration mode (client-side vs. server-side) and retrieval gating (ungated vs. ABAC-gated).

## Setup

### Configuration Matrix

|                              | Ungated Retrieval           | Gated Retrieval             |
|------------------------------|-----------------------------|-----------------------------|
| **Client-Side Orchestration** | Config A (baseline)        | Config B                    |
| **Server-Side Orchestration** | Config C                   | Config D (full architecture)|

- **Ungated** configs use a single shared vector store with no authentication. All 300 documents from all tenants are co-mingled.
- **Gated** configs use per-tenant vector stores with custom authentication and ABAC policies. Each vector store is owned by its tenant, and the policy `user in owners namespaces` restricts access.
- **Client-side** orchestration calls `/v1/vector_stores/{id}/search` and `/v1/chat/completions` separately.
- **Server-side** orchestration makes a single call to `/v1/responses` with a `file_search` tool, letting the server control the retrieval-generation loop.

### Workload

- **3 tenants**: `finance`, `engineering`, `legal`
- **300 documents** (100 per tenant), ~512 tokens each, generated with controlled topical overlap (e.g., "budget" appears in both finance and engineering documents)
- **300 authorized queries** (100 per tenant): queries that should retrieve same-tenant documents
- **300 cross-tenant probes**: queries where a user from one tenant attempts to retrieve another tenant's documents (e.g., a finance user querying for engineering content)

Each query was run once per config. Inference used OpenAI `gpt-4o-mini` via Llama Stack's `remote::openai` provider; embeddings used `text-embedding-3-small`; the vector store was `sqlite-vec`.

### Metrics

- **Cross-Tenant Leakage Rate (CTLR)**: fraction of cross-tenant probes that returned at least one chunk from the target tenant
- **Authorization Violation Rate (AVR)**: fraction of all queries (authorized + probes) that returned data from a tenant other than the querying user's tenant

## Results

### Security

| Config | Orchestration | Retrieval | CTLR | AVR |
|--------|--------------|-----------|------|-----|
| A | Client-side | Ungated | **100.0%** | 50.0% |
| B | Client-side | Gated | **0.0%** | 0.0% |
| C | Server-side | Ungated | **98.3%** | 49.5% |
| D | Server-side | Gated | **0.0%** | 0.0% |

### Latency (Authorized Queries Only)

| Config | p50 | p99 | Mean | Std | N |
|--------|-----|-----|------|-----|---|
| A (client + ungated) | 3,600ms | 10,818ms | 4,208ms | 2,529ms | 300 |
| B (client + gated) | 3,427ms | 9,795ms | 3,851ms | 1,924ms | 300 |
| C (server + ungated) | 7,507ms | 16,462ms | 7,620ms | 2,629ms | 300 |
| D (server + gated) | 6,431ms | 14,623ms | 6,934ms | 5,435ms | 295 |

### Latency by Query Outcome

Breaking latency down by whether the query was authorized (passed ABAC) vs. denied (cross-tenant probe blocked by ABAC) reveals the clearer picture:

| Config | Authorized (pass ABAC) | Cross-Tenant Probes |
|--------|----------------------|---------------------|
| A (client, ungated) | 4,208ms | 5,516ms (data returned) |
| B (client, gated) | 3,851ms | **821ms** (denied, fast fail) |
| C (server, ungated) | 7,620ms | 11,378ms (data returned) |
| D (server, gated) | 6,934ms | **2,951ms** (0 chunks, no inference) |

Under gated configs, denied queries short-circuit before reaching the inference provider: Config B rejects at the vector store search layer (821ms), and Config D returns zero chunks from file search and skips the expensive LLM call (2,951ms). These are **4-5x faster** than queries that go through the full retrieval + inference path.

For authorized queries that pass the ABAC check, the total latency difference between gated and ungated is within the variance of OpenAI response times (std ~2-5s). Isolating the retrieval component from inference confirms this:

### Search vs. Inference Breakdown (Client-Side, Authorized Queries)

| Component | Config A (ungated) | Config B (gated) | Delta |
|-----------|-------------------|------------------|-------|
| Search latency | 572ms | 591ms | +19ms |
| Inference latency | 3,636ms | 3,260ms | -376ms |
| Total | 4,208ms | 3,851ms | -357ms |

The gated search path adds ~19ms compared to ungated. This includes the round-trip to the custom auth server (`localhost:9999`), ABAC policy evaluation (sub-millisecond per Experiment 4's unit benchmarks), and per-tenant vector store lookup. The ABAC policy check itself is negligible; the 19ms reflects the full marginal cost of the gated path. The total latency difference is dominated by inference variance from the external OpenAI API (configs ran sequentially, not interleaved), not by the access control layer. Server-side configs (C, D) do not expose separate search and inference timings.

## Interpretation

**Gating is the security mechanism.** Configs B and D both achieve 0% CTLR and 0% AVR. The orchestration mode (client vs. server) has no effect on leakage when gating is absent: Config A leaks 100% and Config C leaks 98.3% of cross-tenant probes.

**Server-side orchestration alone does not prevent leakage.** Config C demonstrates that moving retrieval to the server side without access control still results in near-complete leakage. The server faithfully retrieves the most semantically relevant documents regardless of tenant boundaries.

**The gated search path adds ~19ms of overhead for authorized queries and saves time on denied ones.** Isolating the search component shows the full gated path (auth round-trip + ABAC policy evaluation + per-tenant store lookup) adds ~19ms -- trivial relative to the ~3.5s inference round-trip. For unauthorized queries, gating provides a substantial latency benefit by failing fast before the expensive retrieval + inference steps.

**Server-side orchestration adds ~3s of latency** compared to client-side. This is expected: the Responses API executes the `file_search` tool internally, adding a round-trip through the tool execution layer. The latency cost is constant and does not interact with gating.

## Figures

- `figures/security_metrics.pdf` -- Grouped bar chart of CTLR and AVR
- `figures/latency_cdfs.pdf` -- CDF overlay of end-to-end latency

## Data

- `data/results/config_{A,B,C,D}_results.json` -- Raw per-query results (600 records each)
- `data/results/summary.json` -- Aggregated metrics

## How to Reproduce

```bash
# For each config X in {A, B, C, D}:
# Start auth server (gated configs B, D only)
uv run --python 3.12 python scripts/auth_server.py &

# Start Llama Stack
uv run --python 3.12 llama stack run configs/config_x_*.yaml &

# Ingest documents
uv run --python 3.12 python scripts/ingest_data.py --config X

# Run experiment
uv run --python 3.12 python scripts/run_experiment.py --config X --num-runs 1

# After all configs, analyze
uv run --python 3.12 python scripts/analyze_results.py
```
