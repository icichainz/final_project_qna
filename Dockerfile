# syntax=docker/dockerfile:1.7
# GCF Q&A — production image (chainlit app + RAG stack).
#
# debian-slim, not alpine: torch/faiss ship manylinux (glibc) wheels only.
# CPU-only torch: query-time embedding is one query at a time; keeps the
# image GPU-agnostic and ~2 GB smaller than the CUDA wheels.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch first from the CPU index so the package install below sees it satisfied
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir ".[app]"


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/hf_cache \
    GCF_QNA_ROOT=/app \
    PORT=18100

RUN apt-get update && apt-get install -y --no-install-recommends tini curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r gcf && useradd -r -g gcf -d /app gcf

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=gcf:gcf pyproject.toml chainlit.md ./
COPY --chown=gcf:gcf src/ src/
COPY --chown=gcf:gcf scripts/ scripts/
COPY --chown=gcf:gcf .chainlit/ .chainlit/
COPY --chown=gcf:gcf public/ public/

# data/ (index, page cache, app.db), public/app_files/ and hf_cache/ are
# bind-mounted at runtime; create the mount points with the right owner.
RUN mkdir -p /app/data /app/public/app_files /app/hf_cache && \
    chown -R gcf:gcf /app

USER gcf
EXPOSE 18100

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["chainlit", "run", "src/gcf_qna/app/chainlit_app.py", \
     "--headless", "--host", "0.0.0.0", "--port", "18100"]
