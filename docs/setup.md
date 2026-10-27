# Setup Guide

## 1. Clone and enter the project
```bash
git clone https://github.com/omarahameds2010-lab/sanad-rag.git
cd sanad-rag
```

## 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```
For tests, lint and type checks use `pip install -r requirements-dev.txt`
instead (it includes the runtime pins plus pytest/ruff/mypy/pip-audit).

## 4. Start the vector store (Qdrant)
```bash
docker compose up -d
```

## 5. Set environment variables
Create a `.env` file (never commit this — it's already in `.gitignore`):
```
OPENAI_API_KEY=your_key_here_only_if_testing_the_openai_comparison_arm
QDRANT_URL=http://localhost:6333
SANAD_API_KEYS=replace-with-a-long-random-key
```
`SANAD_API_KEYS` is a comma-separated allow-list. If it is unset the API
runs **without authentication** (dev fail-open), so any shared or public
deployment must set it.

## 6. Run the tests to confirm everything works
```bash
EMBEDDING_MODEL=hash-dev python -m pytest tests/ -v
```
Tests read the API key allow-list from `.env`; without `SANAD_API_KEYS`
the five unauthenticated-request tests fail by design (the app is in
dev fail-open mode and returns 200 instead of 401).

## 7. Run the pipeline (once ingestion/retrieval scripts exist in later phases)
```bash
python scripts/run_pipeline.py --input data/raw/
```

## 8. Run the evaluation harness (Phase 6)
```bash
python scripts/run_evaluation.py --golden data/golden_dataset/sample_golden.json --config baseline
```
This populates `docs/benchmark_results.md` with real, traceable numbers.
Set `EMBEDDING_MODEL=multilingual-e5` and `ANTHROPIC_API_KEY` in `.env`
first — without them it falls back to dev-only stubs and prints a warning.
