# Docker reproduction for `process.py`

This folder contains a Docker-based comparison that avoids using the host Python, host GCC, host Conda environment, or host `uv` environment.

The container:

1. Uses `python:3.11.9-bookworm`.
2. Installs `build-essential`, `git`, `ca-certificates`, and `uv`.
3. Copies the repository mounted at `/src` into `/work/KeelDS`.
4. Runs `uv sync --locked`.
5. Runs `uv run python process.py`.
6. Compares `keel_ds/data/balanced/processed/australian.npz` against `keel_ds/data/balanced/processed/australian (Cópia).npz`.

## Requirements

- Docker installed.
- The reference file must exist in the repository before running:

```text
keel_ds/data/balanced/processed/australian (Cópia).npz
```

## Build

From the repository root:

```bash
docker build -f docker-repro/Dockerfile -t keelds-process-repro .
```

## Run

From the repository root:

```bash
docker run --rm -v "$PWD:/src:ro" keelds-process-repro
```

Expected final result:

```text
RESULT MATCH
```

## If it is different

Check these values in the Docker output first:

- `raw_sha256`
- `reference_sha256`
- Python version
- NumPy version
- pandas version
- scikit-learn version
- `mdlp` path
- First reported `DIFF_ARRAY`, if any

The expected validated values are:

```text
raw_sha256 ccc64bf31674bc1c282e11f9ba2bb3c5777ca15f03e3d96142ed0817bf7fedce
reference_sha256 88ed71029877c6ced3a9243c5475a1895358353a5bec16ad645552a628b58977
numpy 1.26.4
pandas 2.3.3
sklearn 1.5.2
```
