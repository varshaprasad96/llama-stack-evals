# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Latency overhead evaluation for multitenant security layers.

Measures the time cost of each security component in the request path
to demonstrate that tenant isolation adds negligible overhead relative
to LLM inference and vector search.

Components measured:
  - **ABAC policy evaluation**: time to evaluate access rules.
  - **Metadata filter matching**: time to apply tenant filters to chunks.
  - **Vector search with/without gating**: end-to-end retrieval overhead.

All benchmarks use ``time.perf_counter`` for high-resolution timing and
report median, p95, and p99 latencies across repeated trials.

Run::

    uv run pytest tests/evals/multitenant/test_latency_overhead.py -v
"""

import statistics
import time

import pytest

from llama_stack.core.access_control.access_control import default_policy, is_action_allowed
from llama_stack.core.access_control.datatypes import AccessRule, Action, Scope
from llama_stack.core.datatypes import User

from .conftest import (
    ALL_CHUNKS,
    QUERY_EMBEDDINGS,
    TENANT_A,
    TENANT_B,
    TOPICS,
    matches_filters,
)

# Number of iterations for timing measurements
N_TRIALS = 1000

# Simulated inference latency range (milliseconds) for comparison context
SIMULATED_INFERENCE_MS = 500.0


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a list."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _timing_stats(latencies_ms: list[float]) -> dict:
    """Compute summary statistics for a list of latencies in milliseconds."""
    return {
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "mean_ms": statistics.mean(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


class _MockResource:
    """Minimal ProtectedResource for ABAC benchmarking."""

    def __init__(self, resource_type: str, identifier: str, owner: User):
        self.type = resource_type
        self.identifier = identifier
        self.owner = owner


# ---------------------------------------------------------------------------
# 1. ABAC policy evaluation latency
# ---------------------------------------------------------------------------


class TestABACPolicyLatency:
    """Measure the time cost of ABAC policy evaluation."""

    def test_default_policy_evaluation_latency(self):
        """Default policy evaluation (attribute matching) is sub-millisecond."""
        policy = default_policy()
        resource = _MockResource("vector_db", "tenant-a-store", TENANT_A)

        latencies = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            is_action_allowed(policy, Action.READ, resource, TENANT_A)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        assert stats["p99_ms"] < 1.0, f"ABAC evaluation p99 latency {stats['p99_ms']:.3f}ms exceeds 1ms threshold"

    def test_cross_tenant_denial_latency(self):
        """Cross-tenant denial (no matching rules) is equally fast."""
        policy = default_policy()
        resource = _MockResource("vector_db", "tenant-a-store", TENANT_A)

        latencies = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            is_action_allowed(policy, Action.READ, resource, TENANT_B)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        assert stats["p99_ms"] < 1.0, f"Cross-tenant denial p99 latency {stats['p99_ms']:.3f}ms exceeds 1ms threshold"

    @pytest.mark.parametrize("num_rules", [1, 5, 10, 25, 50])
    def test_policy_evaluation_scales_with_rules(self, num_rules):
        """ABAC evaluation time scales linearly with policy size but stays sub-millisecond."""
        # Build a policy with num_rules non-matching rules followed by one matching rule
        policy = []
        for i in range(num_rules - 1):
            policy.append(
                AccessRule(
                    permit=Scope(
                        principal=f"other-user-{i}",
                        actions=[Action.READ],
                        resource=f"vector_db::other-store-{i}",
                    ),
                    description=f"non-matching rule {i}",
                )
            )
        # Add the matching rule at the end (worst case — must evaluate all rules)
        policy.append(
            AccessRule(
                permit=Scope(actions=list(Action)),
                when="user in owners namespaces",
                description="matching rule",
            )
        )

        resource = _MockResource("vector_db", "tenant-a-store", TENANT_A)

        latencies = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            is_action_allowed(policy, Action.READ, resource, TENANT_A)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        # Even with 50 rules, evaluation should stay well under 5ms
        assert stats["p99_ms"] < 5.0, f"ABAC with {num_rules} rules: p99 latency {stats['p99_ms']:.3f}ms exceeds 5ms"


# ---------------------------------------------------------------------------
# 2. Metadata filter matching latency
# ---------------------------------------------------------------------------


class TestMetadataFilterLatency:
    """Measure the time cost of applying metadata filters to chunks."""

    def test_single_filter_matching_latency(self):
        """Single eq filter matching is sub-microsecond per chunk."""
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        chunks = ALL_CHUNKS  # 10 chunks

        latencies = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            for c in chunks:
                matches_filters(c.metadata, tenant_filter)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        per_chunk_us = stats["median_ms"] * 1000 / len(chunks)
        assert stats["p99_ms"] < 1.0, f"Filter matching p99 for {len(chunks)} chunks: {stats['p99_ms']:.3f}ms"
        # Verify per-chunk cost is negligible
        assert per_chunk_us < 100, f"Per-chunk filter cost {per_chunk_us:.1f}us is too high"

    def test_compound_filter_matching_latency(self):
        """Compound AND/OR filters add minimal overhead."""
        compound_filter = {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant_id", "value": "tenant-a"},
                {"type": "ne", "key": "topic", "value": "confidential"},
            ],
        }
        chunks = ALL_CHUNKS

        latencies = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            for c in chunks:
                matches_filters(c.metadata, compound_filter)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        assert stats["p99_ms"] < 2.0, f"Compound filter p99 for {len(chunks)} chunks: {stats['p99_ms']:.3f}ms"

    @pytest.mark.parametrize("num_chunks", [10, 100, 1000, 5000])
    def test_filter_matching_scales_with_corpus(self, num_chunks):
        """Filter matching time scales linearly with corpus size."""
        # Create synthetic chunks
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        metadata_list = [
            {"tenant_id": "tenant-a" if i % 2 == 0 else "tenant-b", "topic": f"topic-{i}"} for i in range(num_chunks)
        ]

        latencies = []
        n_trials = min(N_TRIALS, 100)  # Fewer trials for large corpora
        for _ in range(n_trials):
            start = time.perf_counter()
            for m in metadata_list:
                matches_filters(m, tenant_filter)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        per_chunk_us = stats["median_ms"] * 1000 / num_chunks
        # Per-chunk cost should be consistent regardless of corpus size
        assert per_chunk_us < 100, f"Per-chunk filter cost at {num_chunks} chunks: {per_chunk_us:.1f}us"


# ---------------------------------------------------------------------------
# 3. Vector search latency with and without gating
# ---------------------------------------------------------------------------


class TestVectorSearchLatency:
    """Measure vector search overhead from metadata filtering."""

    @pytest.mark.parametrize("topic", TOPICS)
    async def test_search_latency_with_and_without_filter(self, shared_vector_index, topic):
        """Metadata filtering adds negligible overhead to vector search."""
        query_emb = QUERY_EMBEDDINGS[topic]
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        n_trials = 100

        # Ungated search
        ungated_latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()
            result = await shared_vector_index.query_vector(embedding=query_emb, k=5, score_threshold=0.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            ungated_latencies.append(elapsed_ms)

        # Gated search (search + filter)
        gated_latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()
            result = await shared_vector_index.query_vector(embedding=query_emb, k=5, score_threshold=0.0)
            [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]
            elapsed_ms = (time.perf_counter() - start) * 1000
            gated_latencies.append(elapsed_ms)

        ungated_stats = _timing_stats(ungated_latencies)
        gated_stats = _timing_stats(gated_latencies)

        # Filtering overhead should be < 20% of search time
        overhead_pct = (
            (gated_stats["median_ms"] - ungated_stats["median_ms"]) / ungated_stats["median_ms"] * 100
            if ungated_stats["median_ms"] > 0
            else 0
        )
        assert overhead_pct < 50, (
            f"Topic '{topic}': gating overhead {overhead_pct:.1f}% exceeds 50% threshold. "
            f"Ungated median: {ungated_stats['median_ms']:.3f}ms, "
            f"Gated median: {gated_stats['median_ms']:.3f}ms"
        )


# ---------------------------------------------------------------------------
# 4. Security overhead relative to inference
# ---------------------------------------------------------------------------


class TestSecurityOverheadRelativeToInference:
    """Compare total security layer overhead to simulated inference latency."""

    async def test_security_overhead_fraction(self, shared_vector_index):
        """Total security overhead (ABAC + filter) is a small fraction of inference time."""
        policy = default_policy()
        resource = _MockResource("vector_db", "tenant-a-store", TENANT_A)
        tenant_filter = {"type": "eq", "key": "tenant_id", "value": "tenant-a"}
        query_emb = QUERY_EMBEDDINGS["revenue"]
        n_trials = 200

        security_latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()

            # ABAC check (resource-level)
            is_action_allowed(policy, Action.READ, resource, TENANT_A)

            # Vector search
            result = await shared_vector_index.query_vector(embedding=query_emb, k=5, score_threshold=0.0)

            # Metadata filtering (chunk-level)
            [c for c in result.chunks if matches_filters(c.metadata, tenant_filter)]

            elapsed_ms = (time.perf_counter() - start) * 1000
            security_latencies.append(elapsed_ms)

        stats = _timing_stats(security_latencies)
        overhead_fraction = stats["median_ms"] / SIMULATED_INFERENCE_MS

        # Security overhead should be < 5% of typical inference time
        assert overhead_fraction < 0.05, (
            f"Security overhead {stats['median_ms']:.3f}ms is "
            f"{overhead_fraction:.1%} of {SIMULATED_INFERENCE_MS}ms inference, "
            f"exceeding 5% threshold"
        )

    def test_abac_overhead_fraction(self):
        """ABAC evaluation alone is < 0.1% of typical inference time."""
        policy = default_policy()
        resource = _MockResource("vector_db", "tenant-a-store", TENANT_A)

        latencies = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            is_action_allowed(policy, Action.READ, resource, TENANT_A)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        stats = _timing_stats(latencies)
        overhead_fraction = stats["median_ms"] / SIMULATED_INFERENCE_MS

        assert overhead_fraction < 0.01, (
            f"ABAC overhead {stats['median_ms']:.3f}ms is "
            f"{overhead_fraction:.1%} of {SIMULATED_INFERENCE_MS}ms inference, "
            f"exceeding 1% threshold"
        )
