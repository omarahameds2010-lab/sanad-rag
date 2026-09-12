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

## 4. Start the vector store (Qdrant)
```bash
docker compose up -d
```

## 5. Set environment variables
Create a `.env` file (never commit this — it's already in `.gitignore`):
```
OPENAI_API_KEY=your_key_here_only_if_testing_the_openai_comparison_arm
QDRANT_URL=http://localhost:6333
```

## 6. Run the tests to confirm everything works
```bash
python -m pytest tests/ -v
```

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
