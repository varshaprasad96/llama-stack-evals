# Reproducing the Evaluation Results

This document maps each claim in the paper to the exact command needed to verify or reproduce it.

## Quick verification (5 minutes, no API key, no GPU)

```bash
git clone https://github.com/varshaprasad96/llama-stack-evals.git
cd llama-stack-evals
uv sync --frozen
./run_all.sh --analysis-only
```

This runs:
- **Experiments 1-3**: Regenerates figures and summary from pre-computed results (4 PDFs)
- **Experiment 4**: Runs 80 pytest tests validating cross-tenant leakage, retrieval quality, adversarial scenarios, and ABAC correctness (~6 seconds)
- **Experiment 6**: Runs predicate pushdown scaling benchmark across 100-50K chunks (~2 minutes)

Or with Docker (no Python needed):
```bash
docker build -t llama-stack-evals .
docker run --rm -v $(pwd)/figures:/eval/figures llama-stack-evals
```

---

## Paper claims → reproduction commands

### Security (Section 5.2)

| Claim | How to verify |
|-------|---------------|
| CTLR=100% ungated, 0% gated | `uv run python scripts/analyze_results.py` — prints security metrics table |
| CTLR=0% under gating for both client and server orchestration | Same command, check Configs B and D |
| 52% leakage with synthetic embeddings | `uv run pytest tests/multitenant/test_cross_tenant_leakage.py -v` |
| 0% false positive rate (48-case ABAC matrix) | `uv run pytest tests/multitenant/test_resource_access_control.py -v` |
| Prompt injection: 0% leak under gating | `uv run python scripts/analyze_results.py` — prints injection table |

### Retrieval quality (Section 5.4)

| Claim | How to verify |
|-------|---------------|
| Gating improves Precision@5 by 2.2x | `uv run pytest tests/multitenant/test_retrieval_quality.py -v` |
| MRR improves from 0.700 to 1.000 | Same command |
| All 4 adversarial attack patterns blocked | `uv run pytest tests/multitenant/test_adversarial_scenarios.py -v` |

### Performance (Section 5.3)

| Claim | How to verify |
|-------|---------------|
| ~19ms ABAC overhead (API-based) | `uv run python scripts/analyze_results.py` — prints ABAC overhead |
| Throughput scales linearly, no gating bottleneck | Same command — prints throughput table |
| ABAC evaluation is sub-millisecond | `uv run pytest tests/multitenant/test_latency_overhead.py -v` |
| Filter overhead < 5ms at all corpus sizes | `uv run pytest tests/multitenant/test_latency_overhead.py -v` |

### Predicate pushdown (Section 5.4)

| Claim | How to verify |
|-------|---------------|
| Recall@5 = 1.000 at 100 chunks, 0.002 at 50K | `uv run python scripts/bench_predicate_pushdown.py` |
| Filter overhead 0.7-3ms at 5x multiplier | Same command |
| Latency overhead small regardless of corpus size | Same command |

### GPU infrastructure overhead (Section 5.3)

| Claim | How to verify |
|-------|---------------|
| 4.7ms routing overhead (1.0%) | Pre-computed: `cat data/results/e2e_latency_gpu.csv` |
| 5.5ms filter overhead (1.9%) | Same file |

See [Reproducing Experiment 5](#reproducing-experiment-5-gpu-infrastructure) below for full reproduction.

---

## Full reproduction of Experiments 1-3 (~2 hours, ~$5-10)

Requires an OpenAI API key. Estimated cost: ~$5-10 for gpt-4o-mini inference + text-embedding-3-small embeddings across 4 configs × (300 authorized + 300 cross-tenant + 90 injection) queries.

```bash
export OPENAI_API_KEY=sk-...
./run_all.sh
```

This runs all 4 configurations (A, B, C, D) sequentially, managing server lifecycle automatically. Results are written to `data/results/` and figures to `figures/`.

To run a single config:
```bash
./run_all.sh --config D
```

---

## Reproducing Experiment 5 (GPU infrastructure)

Experiment 5 measures Llama Stack's routing and filtering overhead on self-hosted GPU infrastructure. The pre-computed results are in `data/results/e2e_latency_gpu.csv`.

### Verify pre-computed results

```bash
python3 -c "
import csv
with open('data/results/e2e_latency_gpu.csv') as f:
    for row in csv.DictReader(f):
        p95, p99 = float(row['p95']), float(row['p99'])
        assert p99 >= p95, f'P99 < P95 for {row[\"label\"]}'
        print(f'{row[\"label\"]:30} median={float(row[\"median\"]):.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms')
print('All percentile invariants hold.')
"
```

### Reproduce on any machine with a GPU

The benchmark script works with any vLLM instance and Llama Stack deployment, not just OpenShift. You need:

1. A GPU that can serve a small LLM (any NVIDIA GPU with >= 4GB VRAM)
2. A HuggingFace account with access to `meta-llama/Llama-3.2-1B-Instruct`

```bash
# Terminal 1: Start vLLM
pip install vllm
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --max-model-len 2048 --port 8000

# Terminal 2: Start Llama Stack
pip install llama-stack
llama stack run configs/config_e2e_vllm_gpu.yaml --port 8321

# Terminal 3: Run benchmark
python scripts/bench_e2e_latency.py \
    --vllm-url http://localhost:8000 \
    --llama-stack-url http://localhost:8321 \
    --num-requests 50 \
    --output-csv data/results/e2e_latency_gpu_reproduced.csv
```

### Original infrastructure

The results in the paper were collected on:

| Component | Specification |
|-----------|--------------|
| Platform | Red Hat OpenShift 4.21 on AWS |
| Instance | g4dn.2xlarge |
| GPU | NVIDIA T4 (16GB VRAM, Turing) |
| Inference | vLLM serving meta-llama/Llama-3.2-1B-Instruct |
| Embeddings | nomic-ai/nomic-embed-text-v1.5 via inline::sentence-transformers |
| Vector store | inline::sqlite-vec |
| Llama Stack | v0.7.1 (distribution-starter image) |

The Llama Stack server config used is in `configs/config_e2e_vllm_gpu.yaml`.

---

## Experiment-to-file mapping

| Experiment | Paper section | Script | Results file | Reproducible locally? |
|-----------|--------------|--------|-------------|----------------------|
| 1. Cross-tenant leakage | 5.2 | `scripts/run_experiment.py` | `data/results/config_*_results.json` | Yes (needs OPENAI_API_KEY) |
| 2. Throughput scaling | 5.3 | `scripts/run_experiment.py` | `data/results/config_*_throughput.json` | Yes (needs OPENAI_API_KEY) |
| 3. Prompt injection | 5.2 | `scripts/run_injection_probes.py` | `data/results/config_*_injection_results.json` | Yes (needs OPENAI_API_KEY) |
| 4. Synthetic retrieval | 5.4 | `tests/multitenant/test_*.py` | pytest output (80 tests) | Yes (free, ~6s) |
| 5. GPU latency | 5.3 | `scripts/bench_e2e_latency.py` | `data/results/e2e_latency_gpu.csv` | Needs GPU + vLLM |
| 6. Predicate pushdown | 5.4 | `scripts/bench_predicate_pushdown.py` | `data/results/predicate_pushdown_scaling.csv` | Yes (free, ~2min) |

---

## Estimated costs

| Path | Time | Cost |
|------|------|------|
| Verify pre-computed results | ~5 minutes | Free |
| Experiments 4 + 6 (local) | ~3 minutes | Free |
| Experiments 1-3 (full) | ~2 hours | ~$5-10 (OpenAI API) |
| Experiment 5 (GPU) | ~30 minutes setup + 5 min run | GPU instance cost |
