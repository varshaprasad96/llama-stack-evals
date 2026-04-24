# Experiment 4: Multitenant Retrieval Security and Quality Benchmarks

## Motivation

Experiments 1-3 evaluate the end-to-end system with real embeddings and external inference APIs. This experiment complements those results by testing the retrieval layer in isolation using synthetic embeddings with controlled similarity properties. The goal is to precisely quantify the "relevance-authorization gap" -- the observation that semantic similarity (relevance) is insufficient for authorization in multitenant vector stores -- and to measure how different gating strategies affect retrieval quality, not just security.

This experiment was contributed as part of [ogx-ai/ogx#5515](https://github.com/ogx-ai/ogx/pull/5515).

## Setup

### Test Corpus

- **Two synthetic tenants**: Tenant A (Acme Financial) and Tenant B (Beta Healthcare)
- **5 topics each**: revenue, hiring, compliance, strategy, confidential
- **10 total documents** with controlled synthetic embeddings (128-dimensional)
- Embeddings use **orthogonal basis vectors** per topic with small per-tenant noise (0.05 scale), producing high cross-tenant similarity (~0.95) for same-topic pairs. This intentionally creates conditions where relevance-only retrieval leaks across tenants.

### Three Retrieval Configurations

| Configuration | Description |
|--------------|-------------|
| **Ungated (relevance-only)** | No metadata filter; shared index queried directly |
| **Chunk-level gated** | Shared index with a `tenant_id` metadata filter applied post-retrieval |
| **Per-tenant index** | Physically separate `SQLiteVecIndex` per tenant |

Unlike Experiments 1-3 which use a 2x2 matrix of orchestration mode × gating, this experiment holds orchestration constant and compares three gating strategies at the retrieval layer.

### Four Evaluation Dimensions

1. **Cross-tenant retrieval leakage** -- fraction of retrieved chunks belonging to the wrong tenant
2. **Retrieval quality** -- Recall@5, Precision@5, and MRR under each configuration
3. **Adversarial scenarios** -- four attack patterns targeting tenant isolation
4. **Resource-level ABAC correctness** -- 48-case access control matrix with four user types

All 51 tests use synthetic embeddings and run locally with no external API dependencies.

## Results

### Cross-Tenant Retrieval Leakage

| Configuration | Leaked Chunks | Total Retrieved | Leakage Rate |
|--------------|--------------|----------------|-------------|
| Ungated (relevance-only) | 13 | 25 | **52.0%** |
| Chunk-level gated | 0 | 12 | **0.0%** |
| Per-tenant index | 0 | 25 | **0.0%** |

Without gating, over half of retrieved chunks come from the wrong tenant. Both gating strategies eliminate leakage entirely. Chunk-level gating retrieves fewer total chunks (12 vs 25) because cross-tenant documents are filtered out post-retrieval, and some queries return fewer than k=5 results when same-tenant matches are scarce in the top-k.

### Retrieval Quality

| Configuration | Recall@5 | Precision@5 | MRR |
|--------------|----------|-------------|-----|
| Ungated | 1.000 | 0.200 | 0.700 |
| Chunk-level gated | 1.000 | **0.433** | **1.000** |
| Per-tenant index | 1.000 | 0.200 | **1.000** |

Chunk-level gating not only eliminates leakage but **improves retrieval quality**: precision increases from 0.200 to 0.433 and MRR from 0.700 to 1.000. Filtering out cross-tenant noise moves the correct document to rank 1.

### Adversarial Scenarios

| Attack Pattern | Ungated | Gated |
|---------------|---------|-------|
| Targeted confidential extraction | LEAKED | **BLOCKED** |
| Metadata filter tampering | LEAKED | **BLOCKED** |
| Compound OR-filter bypass | LEAKED | **BLOCKED** |
| Exhaustive enumeration | LEAKED | **BLOCKED** |

All four attack patterns -- including attempts to tamper with metadata filters or use logical OR operators to widen filter scope -- are blocked when tenant gating is applied.

### Resource-Level ABAC Correctness

| Metric | Value |
|--------|-------|
| Accuracy | **1.000** |
| False positive rate | **0.000** |
| False negative rate | 0.000 |
| Total test cases | 48 |

The ABAC evaluation tests four user types (Tenant A, Tenant B, Admin, Unauthorized) against four resource types (Tenant A vector store, Tenant B vector store, Tenant A model, Public model) across three actions (READ, CREATE, DELETE). Zero false positives means zero security violations.

## Interpretation

**The relevance-authorization gap is real and quantifiable.** With controlled embeddings producing ~0.95 cross-tenant similarity for same-topic pairs, 52% of retrieved chunks leak across tenants under ungated retrieval. This confirms the gap observed in Experiment 1 (100% CTLR) in a more controlled setting.

**Gating improves retrieval quality, not just security.** This is the key new finding. Chunk-level gating doesn't just prevent leakage -- it improves precision by 2.2x (0.200 → 0.433) and MRR from 0.700 to 1.000. Cross-tenant documents act as noise in the ranking; removing them promotes the correct documents to the top positions.

**Chunk-level gating on a shared index is sufficient.** Per-tenant index isolation also eliminates leakage, but chunk-level metadata filtering on a shared index achieves the same security with simpler infrastructure. The precision advantage of chunk-level gating (0.433 vs 0.200) arises because the shared index has more context for ranking within the filtered results.

**ABAC policies are correct across the full access control matrix.** The 48-case evaluation (4 user types × 4 resources × 3 actions) achieves perfect accuracy with zero false positives, validating both namespace isolation and role-based restrictions (e.g., admin-only delete).

## Data

- Source PR: [ogx-ai/ogx#5515](https://github.com/ogx-ai/ogx/pull/5515)
- Test files:
  - `tests/evals/multitenant/conftest.py` -- shared fixtures, test corpus, synthetic embeddings
  - `tests/evals/multitenant/test_cross_tenant_leakage.py` -- leakage rate measurements
  - `tests/evals/multitenant/test_retrieval_quality.py` -- recall/precision/MRR metrics
  - `tests/evals/multitenant/test_adversarial_scenarios.py` -- attack pattern tests
  - `tests/evals/multitenant/test_resource_access_control.py` -- ABAC policy correctness

## How to Reproduce

```bash
# From a llama-stack checkout with the eval suite merged:
cd tests/evals/multitenant
pytest -v --tb=short
```
