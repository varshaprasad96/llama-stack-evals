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
| vLLM Direct (baseline) | 447.8ms | 421.1ms | 502.5ms | 580.3ms | 67.6ms | 50 |
| Llama Stack Inference | 462.4ms | 447.7ms | 556.8ms | 579.8ms | 78.9ms | 50 |

**Inference security overhead: 14.7ms (3.3% of baseline)**

The 14.7ms overhead includes ABAC policy evaluation, model routing table lookup, and the internal HTTP hop from Llama Stack to vLLM. ABAC policy evaluation itself is sub-millisecond (validated in Experiment 4's unit benchmarks); the remainder is network and serialization overhead inherent to any proxy architecture.

### Retrieval Latency

| Configuration | Median | Mean | P95 | P99 | StdDev | N |
|--------------|--------|------|-----|-----|--------|---|
| Vector Search (ungated) | 299.0ms | 297.6ms | 339.1ms | 330.9ms | 16.0ms | 50 |
| Vector Search (gated) | 299.3ms | 297.6ms | 327.9ms | 330.9ms | 16.0ms | 50 |

**Retrieval filter overhead: 0.3ms (0.1% of search time)**

Metadata filtering adds effectively zero overhead. The gated P95 (327.9ms) is actually lower than ungated P95 (339.1ms), suggesting that filtering reduces variance by shrinking the result set.

### Component Breakdown

| Component | Latency | % of Inference Total |
|-----------|---------|---------------------|
| LLM Inference (vLLM, T4 GPU) | 447.8ms | 96.8% |
| Llama Stack overhead (ABAC + routing) | 14.7ms | 3.2% |
| Tenant metadata filter | 0.3ms | <0.1% |

## Interpretation

**Security overhead is negligible on self-hosted infrastructure.** At 14.7ms (3.3%), the combined cost of ABAC policy evaluation, model routing, and provider dispatch is dwarfed by LLM inference time. This is a stronger result than Experiments 1-3, which showed ~19ms ABAC overhead but measured against ~3-7s OpenAI API latency where network variance obscured the signal.

**Metadata filtering has zero practical cost.** The 0.3ms filter overhead confirms Experiment 4's unit-level finding (<0.01ms per chunk) at the API level. Tenant isolation through chunk-level gating is free in terms of latency.

**GPU inference changes the overhead ratio.** On a T4 GPU, inference takes ~450ms (vs. ~3-7s with OpenAI API). The security overhead is proportionally larger (3.3% vs. <1%) but still well within acceptable bounds. For larger models or higher token counts, inference time grows while security overhead remains constant, so the ratio improves further.

**Comparison with Experiments 1-3:**

| Metric | Exp 1-3 (OpenAI API) | Exp 5 (vLLM on T4) |
|--------|---------------------|---------------------|
| Baseline inference | 3,600-7,500ms | 448ms |
| ABAC overhead | ~19ms | ~15ms |
| Overhead % | <0.5% | 3.3% |
| Filter overhead | — | 0.3ms |

The ABAC overhead is consistent (~15-19ms) across both setups, confirming it's a fixed cost of Llama Stack's security layer, independent of the inference backend.

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
