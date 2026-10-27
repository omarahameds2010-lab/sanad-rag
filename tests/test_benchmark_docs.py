"""Benchmark documentation must stay machine-consistent with the harness.

The first real run exposed a stale 11-column table header while the append
code wrote 13 columns, which would silently shift every value. These tests
pin the schema so the Markdown cannot drift from the metrics again.
"""

from pathlib import Path

DOC = Path(__file__).parent.parent / "docs" / "benchmark_results.md"


def _table_lines() -> list[str]:
    lines = DOC.read_text(encoding="utf-8").splitlines()
    return [
        line
        for line in lines
        if line.startswith("|") and set(line) - {"|", "-"} != set()
    ]


def test_benchmark_table_has_13_columns():
    table = _table_lines()
    assert table, "benchmark_results.md must contain the results table"
    widths = {len(line.strip().strip("|").split("|")) for line in table}
    assert widths == {13}, f"inconsistent column count: {widths}"


def test_benchmark_header_matches_script_schema():
    header = next(line for line in _table_lines() if line.startswith("| Timestamp"))
    cells = [cell.strip() for cell in header.strip().strip("|").split("|")]
    assert cells == [
        "Timestamp",
        "Config",
        "Chunker",
        "Reranker",
        "Embedding",
        "LLM",
        "N",
        "Precision@k",
        "Recall@k",
        "p50 latency (s)",
        "p95 latency (s)",
        "Dev-only?",
        "run_id",
    ]
