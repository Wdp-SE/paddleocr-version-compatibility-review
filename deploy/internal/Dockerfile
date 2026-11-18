# Supply an approved python:3.11-slim-bookworm@sha256:... build argument.
ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/internal/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY versioned-rag-service/src/*.py /app/versioned-rag-service/src/
COPY versioned-rag-service/config/*.json /app/versioned-rag-service/config/
COPY demo-ui/*.py /app/demo-ui/
COPY demo-ui/components/*.py /app/demo-ui/components/
COPY demo-ui/services/*.py /app/demo-ui/services/
COPY demo-ui/assets/*.svg /app/demo-ui/assets/
COPY change-review-agent/app /app/change-review-agent/app
COPY change-review-agent/config/*.json /app/change-review-agent/config/
COPY config/internal_applications.json /app/config/internal_applications.json
COPY deploy/internal/entrypoint.py /app/deploy/internal/entrypoint.py
RUN useradd --uid 10001 --create-home reviewer
USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/versioned-rag-service
WORKDIR /app/versioned-rag-service
ENTRYPOINT ["python","/app/deploy/internal/entrypoint.py"]
CMD ["uvicorn","src.public_server:app","--host","0.0.0.0","--port","8765"]
