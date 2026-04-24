FROM python:3.12-slim

LABEL maintainer="farceo@redhat.com, vnarsing@redhat.com"
LABEL description="OGX multitenant RAG security evaluation - artifact for paper review"

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /eval

# Copy dependency spec first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies from lock file
RUN uv sync --frozen --no-dev

# Copy everything else
COPY . .

# Default: regenerate figures from pre-computed results
# No API key needed for this path
CMD ["uv", "run", "python", "scripts/analyze_results.py"]
