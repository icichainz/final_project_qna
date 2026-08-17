# syntax=docker/dockerfile:1.7

FROM python:3.12-alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apk add --no-cache \
        build-base \
        jpeg-dev \
        libffi-dev \
        linux-headers \
        openssl-dev \
        postgresql-dev \
        zlib-dev

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.12-alpine AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=18000

RUN apk add --no-cache \
        libffi \
        libjpeg-turbo \
        libpq \
        su-exec \
        tini \
        zlib && \
    addgroup -S pimento && \
    adduser -S -D -H -G pimento pimento

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=pimento:pimento . .

RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /app/db /app/static/uploads && \
    chown -R pimento:pimento /app/db /app/static/uploads

EXPOSE 18000

ENTRYPOINT ["/sbin/tini", "--", "/app/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "18000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*", "--timeout-keep-alive", "5"]
