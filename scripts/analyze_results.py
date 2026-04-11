"""
Analyze experiment results and generate figures for the paper.

Computes:
  - Cross-Tenant Leakage Rate (CTLR)
  - Authorization Violation Rate (AVR)
  - Latency statistics (p50, p99, mean)
  - ABAC overhead
  - Throughput (QPS) at various concurrency levels
  - Prompt injection results summary

Generates:
  - figures/security_metrics.pdf   -- Grouped bar chart of CTLR/AVR
  - figures/latency_cdfs.pdf       -- CDF curves per config
  - figures/throughput_scaling.pdf  -- QPS vs concurrency
  - Prints summary tables to stdout for easy paper integration

Usage:
    python scripts/analyze_results.py [--data-dir data/] [--figures-dir figures/]
"""

import argparse
import json
import os
import sys

import numpy as np

# Try to import matplotlib; provide fallback if not available
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for PDF generation
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Figures will not be generated.")
    print("Install with: pip install matplotlib")


CONFIGS = ["A", "B", "C", "D"]
CONFIG_LABELS = {
    "A": "Client + Ungated",
    "B": "Client + Gated",
    "C": "Server + Ungated",
    "D": "Server + Gated",
}
CONFIG_COLORS = {
    "A": "#e74c3c",  # red
    "B": "#3498db",  # blue
    "C": "#e67e22",  # orange
    "D": "#2ecc71",  # green
}

SENSITIVITY_WEIGHTS = {
    "public": 0.1,
    "internal": 0.5,
    "confidential": 1.0,
}


def load_results(data_dir: str, config: str) -> list[dict] | None:
    path = os.path.join(data_dir, "results", f"config_{config}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_throughput(data_dir: str, config: str) -> list[dict] | None:
    path = os.path.join(data_dir, "results", f"config_{config}_throughput.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_injection_results(data_dir: str, config: str) -> list[dict] | None:
    path = os.path.join(data_dir, "results", f"config_{config}_injection_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---- Security Metrics ----

def compute_ctlr(results: list[dict]) -> float:
    """Cross-Tenant Leakage Rate: fraction of probes that returned cross-tenant data."""
    probes = [r for r in results if r["query_type"] == "cross_tenant_probe"]
    if not probes:
        return 0.0

    leaked = 0
    for r in probes:
        target = r.get("target_tenant")
        retrieved = r.get("retrieved_tenants", [])
        if target and target in retrieved:
            leaked += 1

    return leaked / len(probes)


def compute_avr(results: list[dict]) -> float:
    """Authorization Violation Rate: fraction of all queries returning unauthorized data."""
    if not results:
        return 0.0

    violations = 0
    total = 0

    for r in results:
        if r.get("error"):
            continue
        total += 1
        tenant = r["tenant"]
        retrieved = r.get("retrieved_tenants", [])
        # Any retrieved tenant that doesn't match the querying tenant is a violation
        other_tenants = [t for t in retrieved if t != tenant]
        if other_tenants:
            violations += 1

    return violations / total if total > 0 else 0.0


# ---- Latency Metrics ----

def compute_latency_stats(results: list[dict], query_type: str = "authorized") -> dict:
    """Compute latency percentiles for a given query type."""
    latencies = [
        r["total_latency_ms"]
        for r in results
        if r["query_type"] == query_type and r.get("error") is None and r.get("total_latency_ms") is not None
    ]

    if not latencies:
        return {"p50": 0, "p99": 0, "mean": 0, "std": 0, "count": 0}

    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": len(arr),
    }


def compute_abac_overhead(ungated_results: list[dict], gated_results: list[dict]) -> float:
    """Mean latency difference between gated and ungated for authorized queries."""
    ungated_lats = [
        r["total_latency_ms"]
        for r in ungated_results
        if r["query_type"] == "authorized" and r.get("error") is None and r.get("total_latency_ms") is not None
    ]
    gated_lats = [
        r["total_latency_ms"]
        for r in gated_results
        if r["query_type"] == "authorized" and r.get("error") is None and r.get("total_latency_ms") is not None
    ]

    if not ungated_lats or not gated_lats:
        return 0.0

    return float(np.mean(gated_lats) - np.mean(ungated_lats))


# ---- Figures ----

def plot_security_metrics(metrics: dict, figures_dir: str):
    """Grouped bar chart of CTLR and AVR per config."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [c for c in CONFIGS if c in metrics]
    x = np.arange(len(configs))
    width = 0.35

    ctlr_vals = [metrics[c]["ctlr"] * 100 for c in configs]
    avr_vals = [metrics[c]["avr"] * 100 for c in configs]

    bars1 = ax.bar(x - width/2, ctlr_vals, width, label="CTLR (%)", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, avr_vals, width, label="AVR (%)", color="#3498db", alpha=0.85)

    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_title("Security Metrics: Cross-Tenant Leakage and Authorization Violations", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(ctlr_vals), max(avr_vals)) * 1.2 + 5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(figures_dir, "security_metrics.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_latency_cdfs(all_results: dict, figures_dir: str):
    """CDF curves for E2E latency per config."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for config in CONFIGS:
        results = all_results.get(config)
        if not results:
            continue

        latencies = sorted([
            r["total_latency_ms"]
            for r in results
            if r["query_type"] == "authorized" and r.get("error") is None and r.get("total_latency_ms") is not None
        ])

        if not latencies:
            continue

        cdf = np.arange(1, len(latencies) + 1) / len(latencies)
        ax.plot(latencies, cdf, label=CONFIG_LABELS[config], color=CONFIG_COLORS[config], linewidth=2)

    ax.set_xlabel("End-to-End Latency (ms)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Latency Distribution: Authorized Queries", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(figures_dir, "latency_cdfs.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_throughput(all_throughput: dict, figures_dir: str):
    """Line chart of QPS vs concurrency."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for config in CONFIGS:
        throughput = all_throughput.get(config)
        if not throughput:
            continue

        concurrencies = [t["concurrency"] for t in throughput]
        qps_values = [t["qps"] for t in throughput]
        ax.plot(concurrencies, qps_values, marker="o", label=CONFIG_LABELS[config],
                color=CONFIG_COLORS[config], linewidth=2, markersize=6)

    ax.set_xlabel("Concurrency Level", fontsize=12)
    ax.set_ylabel("Queries Per Second (QPS)", fontsize=12)
    ax.set_title("Throughput Under Concurrent Load", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(figures_dir, "throughput_scaling.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--figures-dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Load all results
    all_results = {}
    all_throughput = {}
    all_injection = {}

    for config in CONFIGS:
        results = load_results(args.data_dir, config)
        if results:
            all_results[config] = results
        throughput = load_throughput(args.data_dir, config)
        if throughput:
            all_throughput[config] = throughput
        injection = load_injection_results(args.data_dir, config)
        if injection:
            all_injection[config] = injection

    if not all_results:
        print("No results found. Run experiments first.")
        sys.exit(1)

    available = list(all_results.keys())
    print(f"Found results for configs: {', '.join(available)}")
    print()

    # ---- Compute Security Metrics ----
    print("=" * 60)
    print("SECURITY METRICS")
    print("=" * 60)

    security_metrics = {}
    for config in available:
        ctlr = compute_ctlr(all_results[config])
        avr = compute_avr(all_results[config])
        security_metrics[config] = {"ctlr": ctlr, "avr": avr}

    print(f"\n{'Config':<25} {'CTLR (%)':<12} {'AVR (%)':<12}")
    print("-" * 49)
    for config in available:
        m = security_metrics[config]
        print(f"{CONFIG_LABELS[config]:<25} {m['ctlr']*100:>8.1f}%    {m['avr']*100:>8.1f}%")

    # ---- Compute Latency Metrics ----
    print(f"\n{'=' * 60}")
    print("LATENCY METRICS (Authorized Queries)")
    print("=" * 60)

    print(f"\n{'Config':<25} {'p50 (ms)':<12} {'p99 (ms)':<12} {'Mean (ms)':<12} {'N':<8}")
    print("-" * 69)
    for config in available:
        stats = compute_latency_stats(all_results[config])
        print(f"{CONFIG_LABELS[config]:<25} {stats['p50']:>8.0f}    {stats['p99']:>8.0f}    {stats['mean']:>8.0f}    {stats['count']:>6}")

    # ---- ABAC Overhead ----
    print(f"\n{'=' * 60}")
    print("ABAC OVERHEAD")
    print("=" * 60)

    if "A" in all_results and "B" in all_results:
        overhead_client = compute_abac_overhead(all_results["A"], all_results["B"])
        print(f"\n  Client-side (A vs B): {overhead_client:+.1f}ms")
    if "C" in all_results and "D" in all_results:
        overhead_server = compute_abac_overhead(all_results["C"], all_results["D"])
        print(f"  Server-side (C vs D): {overhead_server:+.1f}ms")

    # ---- Throughput ----
    if all_throughput:
        print(f"\n{'=' * 60}")
        print("THROUGHPUT (QPS)")
        print("=" * 60)

        # Get all concurrency levels
        all_concurrencies = set()
        for t_list in all_throughput.values():
            for t in t_list:
                all_concurrencies.add(t["concurrency"])
        concurrencies = sorted(all_concurrencies)

        header = f"\n{'Config':<25}" + "".join(f"{'c=' + str(c):<12}" for c in concurrencies)
        print(header)
        print("-" * (25 + 12 * len(concurrencies)))
        for config in available:
            if config not in all_throughput:
                continue
            t_map = {t["concurrency"]: t["qps"] for t in all_throughput[config]}
            row = f"{CONFIG_LABELS[config]:<25}"
            for c in concurrencies:
                qps = t_map.get(c, 0)
                row += f"{qps:>8.1f}    "
            print(row)

    # ---- Injection Probes ----
    if all_injection:
        print(f"\n{'=' * 60}")
        print("PROMPT INJECTION RESULTS (Reported Separately)")
        print("=" * 60)

        print(f"\n{'Config':<25} {'Total':<8} {'Leaked':<10} {'Leak Rate':<12} {'Denied':<10}")
        print("-" * 65)
        for config in available:
            if config not in all_injection:
                continue
            results = all_injection[config]
            total = len(results)
            leaked = sum(1 for r in results if r.get("other_tenant_data_leaked"))
            denied = sum(1 for r in results if r.get("error") and "not found" in r["error"].lower())
            leak_rate = leaked / total if total > 0 else 0
            print(f"{CONFIG_LABELS[config]:<25} {total:<8} {leaked:<10} {leak_rate*100:>8.1f}%    {denied:<10}")

    # ---- Generate Figures ----
    if HAS_MATPLOTLIB:
        print(f"\n{'=' * 60}")
        print("GENERATING FIGURES")
        print("=" * 60)
        plot_security_metrics(security_metrics, args.figures_dir)
        plot_latency_cdfs(all_results, args.figures_dir)
        if all_throughput:
            plot_throughput(all_throughput, args.figures_dir)
    else:
        print("\nSkipping figure generation (matplotlib not installed).")

    # ---- Save summary JSON ----
    summary = {
        "security": {config: security_metrics[config] for config in available if config in security_metrics},
        "latency": {config: compute_latency_stats(all_results[config]) for config in available},
    }
    if all_throughput:
        summary["throughput"] = {config: all_throughput[config] for config in available if config in all_throughput}
    if all_injection:
        summary["injection"] = {}
        for config in available:
            if config in all_injection:
                results = all_injection[config]
                total = len(results)
                leaked = sum(1 for r in results if r.get("other_tenant_data_leaked"))
                summary["injection"][config] = {
                    "total": total,
                    "leaked": leaked,
                    "leak_rate": leaked / total if total > 0 else 0,
                }

    summary_path = os.path.join(args.data_dir, "results", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
