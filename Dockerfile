FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests
COPY examples /app/examples

RUN pip install --no-cache-dir -e ".[langchain,openai,community,dev]"

CMD ["python", "scripts/smoke_check.py"]
