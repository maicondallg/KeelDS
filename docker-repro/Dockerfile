FROM python:3.11.9-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PROJECT_ENVIRONMENT=/tmp/keelds-docker-venv
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV PYTHONHASHSEED=0
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv>=0.7,<0.9"

WORKDIR /work

COPY docker-repro/compare_in_docker.py /opt/keelds-docker-repro/compare_in_docker.py

ENTRYPOINT ["python", "/opt/keelds-docker-repro/compare_in_docker.py"]
