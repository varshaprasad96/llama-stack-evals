# Experiment 5: End-to-End Latency Overhead on GPU Infrastructure

## Motivation

Experiments 1-4 measure security properties and use external API-based inference (OpenAI), where API response variance dominates latency. This experiment measures the latency overhead of Llama Stack's security layers on self-hosted GPU infrastructure, isolating the cost of ABAC, routing, and metadata filtering from external API latency.

The key question: **how much latency does server-side orchestration with tenant isolation add when you control the inference backend?**

## Setup

### Infrastructure

- **Platform**: Red Hat OpenShift 4.21 on AWS (`g4dn.2xlarge` instance)
- **GPU**: NVIDIA T4 (16GB VRAM, Turing architecture)
- **Inference**: vLLM serving `meta-llama/Llama-3.2-1B-Instruct` on GPU
- **Embeddings**: `nomic-ai/nomic-embed-text-v1.5` via Llama Stack's `inline::sentence-transformers` provider (CPU)
- **Vector store**: `inline::sqlite-vec` with tenant-scoped documents
- **Llama Stack**: v0.7.1 (`distribution-starter` image), inference-only + vector_io providers

### Deployment Topology

```
┌──────────────────────────────────────────────────────┐
│  OpenShift Cluster (g4dn.2xlarge, T4 GPU)            │
│                                                      │
│  ┌──────────────┐     ┌───────────────────────────┐  │
│  │  vLLM Pod    │     │  Llama Stack Pod          │  │
│  │  (GPU)       │◄────│  (CPU)                    │  │
│  │              │     │                           │  │
│  │  Llama-3.2   │     │  ABAC + Routing           │  │
│  │  -1B-Instruct│     │  sqlite-vec               │  │
│  │              │     │  sentence-transformers    │  │
│  └──────────────┘     └───────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Benchmark Design

Two comparisons, each with N=50 requests:

1. **Inference overhead**: Direct vLLM vs. Llama Stack (measures ABAC + routing + provider dispatch)
2. **Retrieval filtering overhead**: Ungated search vs. tenant-gated search (measures metadata filter cost)

All requests use the same prompts/queries across configurations to ensure fair comparison.

### Configurations

| Path | Description |
|------|-------------|
| vLLM Direct | `POST /v1/chat/completions` directly to vLLM (no Llama Stack) |
| LS Inference | `POST /v1/chat/completions` through Llama Stack → vLLM |
| Search (ungated) | `POST /v1/vector_stores/{id}/search` without tenant filter |
| Search (gated) | Same endpoint with `{"type": "eq", "key": "tenant_id", "value": "tenant-a"}` filter |

## Results

### Inference Latency

| Configuration | Median | Mean | P95 | P99 | StdDev | N |
|--------------|--------|------|-----|-----|--------|---|
| vLLM Direct (baseline) | 447.9ms | 421.4ms | 531.5ms | 535.9ms | 65.7ms | 50 |
| Llama Stack Inference | 452.6ms | 427.4ms | 537.9ms | 554.4ms | 71.5ms | 50 |

**Inference security overhead: 4.7ms (1.0% of baseline)**

The 4.7ms overhead includes ABAC policy evaluation, model routing table lookup, and the internal HTTP hop from Llama Stack to vLLM. ABAC policy evaluation itself is sub-millisecond (validated in Experiment 4's unit benchmarks); the remainder is network and serialization overhead inherent to any proxy architecture.

### Retrieval Latency

| Configuration | Median | Mean | P95 | P99 | StdDev | N |
|--------------|--------|------|-----|-----|--------|---|
| Vector Search (ungated) | 283.9ms | 285.5ms | 294.3ms | 305.0ms | 6.4ms | 50 |
| Vector Search (gated) | 289.4ms | 289.6ms | 306.0ms | 329.7ms | 9.5ms | 50 |

**Retrieval filter overhead: 5.5ms (1.9% of search time)**

Metadata filtering adds ~5.5ms to the search path. Both configurations ran after a 3-request warmup to ensure the embedding model (nomic-embed-text-v1.5, running on CPU) was loaded into memory, eliminating cold-start variance.

### Component Breakdown

| Component | Latency | % of Inference Total |
|-----------|---------|---------------------|
| LLM Inference (vLLM, T4 GPU) | 447.9ms | 99.0% |
| Llama Stack overhead (ABAC + routing) | 4.7ms | 1.0% |
| Tenant metadata filter | 5.5ms | 1.2% |

## Interpretation

**Security overhead is negligible on self-hosted infrastructure.** At 4.7ms (1.0%), the combined cost of ABAC policy evaluation, model routing, and provider dispatch is dwarfed by LLM inference time.

**Metadata filtering has minimal practical cost.** The 5.5ms filter overhead includes the metadata comparison and slightly larger result set processing. This confirms Experiment 4's unit-level finding that the per-chunk filter cost is sub-millisecond; the remaining overhead is from the search API's filter parsing and result marshaling.

**GPU inference changes the overhead ratio.** On a T4 GPU, inference takes ~448ms (vs. ~3-7s with OpenAI API). The security overhead is proportionally larger (1.0% vs. <0.5%) but still well within acceptable bounds. For larger models or higher token counts, inference time grows while security overhead remains constant, so the ratio improves further.

**Comparison with Experiments 1-3:**

| Metric | Exp 1-3 (OpenAI API) | Exp 5 (vLLM on T4) |
|--------|---------------------|---------------------|
| Baseline inference | 3,600-7,500ms | 448ms |
| ABAC overhead | ~19ms | ~5ms |
| Overhead % | <0.5% | 1.0% |
| Filter overhead | — | 5.5ms |

The security overhead is consistent in being a small, fixed cost independent of the inference backend.

## Configs

- [`configs/config_e2e_vllm_gpu.yaml`](../configs/config_e2e_vllm_gpu.yaml) -- Llama Stack config pointing to vLLM with sentence-transformers embeddings

## Data

- [`data/results/e2e_latency_gpu.csv`](../data/results/e2e_latency_gpu.csv) -- Raw latency measurements
- Benchmark script: [`scripts/bench_e2e_latency.py`](../scripts/bench_e2e_latency.py)

## How to Reproduce

```bash
# Prerequisites: OpenShift cluster with GPU node, vLLM serving Llama-3.2-1B-Instruct

# 1. Deploy vLLM (see configs/config_e2e_vllm_gpu.yaml for Llama Stack config)
# 2. Deploy Llama Stack with the provided config
# 3. Port-forward both services:
oc port-forward -n vllm svc/vllm 8000:8000 &
oc port-forward -n llama-stack-eval svc/llama-stack 8321:8321 &

# 4. Run benchmark
python scripts/bench_e2e_latency.py \
    --llama-stack-url http://localhost:8321 \
    --vllm-url http://localhost:8000 \
    --num-requests 50 \
    --output-csv data/results/e2e_latency_gpu.csv
```
