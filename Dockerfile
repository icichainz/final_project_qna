FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[app]"

COPY chainlit.md ./
COPY .chainlit/ .chainlit/
COPY scripts/ scripts/

CMD ["chainlit", "run", "src/gcf_qna/app/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
