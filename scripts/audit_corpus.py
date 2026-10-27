"""Audit the committed knowledge corpus for filler before it can be indexed.

SANAD's #1 rule is that the library must contain genuine Arabic content — no
line inflation, no repeated lines, no English padding, no mojibake. This
script makes that rule executable instead of aspirational.

It reads every committed ``data/knowledge/*.txt`` (tracked by git), computes
objective repetition/script statistics, and prints a PASS/REVIEW/FAIL row.
``data/knowledge/gen_supreme_core_*.txt`` is local-only by policy (gitignored)
and is listed separately so it can never be confused with the shipped corpus.

Usage:
    python scripts/audit_corpus.py            # human table
    python scripts/audit_corpus.py --json     # machine-readable
    python scripts/audit_corpus.py --strict   # exit 1 if any FAIL
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / "data" / "knowledge"

# Filler often differs only by a leading index ("1. same line"). Normalise it
# away before judging diversity, otherwise numbering hides line inflation.
_INDEX_PREFIX = re.compile(r"^\s*[\(\[{]*\s*[\d٠-٩]+[\.\)\]}:：-]*\s*")


def content_line(line: str) -> str:
    return _INDEX_PREFIX.sub("", line).strip()


def tracked_files() -> list[Path]:
    """Only files git actually ships. Untracked local shards are not corpus."""
    out = subprocess.run(
        ["git", "ls-files", "data/knowledge"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        ROOT / line.strip()
        for line in out.splitlines()
        if line.strip().endswith(".txt")
    )


def stats(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    total = len(lines)
    unique = len(set(lines))
    normalized = [content_line(line) for line in lines]
    normalized = [line for line in normalized if line]
    normalized_unique = len(set(normalized))
    top_line, top_count = Counter(normalized).most_common(1)[0] if normalized else ("", 0)
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(1 for ch in letters if ch.isascii())
    arabic = sum(1 for ch in letters if "\u0600" <= ch <= "\u06ff")
    n_letters = max(len(letters), 1)
    latin_ratio = latin / n_letters
    arabic_ratio = arabic / n_letters
    unique_ratio = unique / total if total else 0.0
    normalized_unique_ratio = normalized_unique / len(normalized) if normalized else 0.0
    top_ratio = top_count / len(normalized) if normalized else 0.0

    reasons = []
    if total < 5:
        reasons.append("too-short")
    if normalized_unique_ratio < 0.50 and total >= 20:
        reasons.append("template-repetition")
    if top_ratio > 0.10 and total >= 20:
        reasons.append("single-line-inflation")
    if latin_ratio > 0.05:
        reasons.append("english-padding")
    if arabic_ratio < 0.70:
        reasons.append("not-mostly-arabic")

    if reasons:
        status = "FAIL"
    elif normalized_unique_ratio < 0.80 or unique_ratio < 0.80:
        status = "REVIEW"
        reasons.append("low-diversity")
    else:
        status = "PASS"

    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "chars": len(text),
        "lines": total,
        "unique_lines": unique,
        "unique_ratio": round(unique_ratio, 3),
        "normalized_unique_ratio": round(normalized_unique_ratio, 3),
        "top_line_repeats": top_count,
        "top_ratio": round(top_ratio, 3),
        "arabic_ratio": round(arabic_ratio, 3),
        "latin_ratio": round(latin_ratio, 3),
        "top_line": top_line[:60],
        "status": status,
        "reasons": reasons,
    }


def local_only_shards() -> list[str]:
    return sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in KNOWLEDGE.glob("gen_supreme_core_*.txt")
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = [stats(path) for path in tracked_files()]
    shards = local_only_shards()
    failed = [row for row in rows if row["status"] == "FAIL"]

    if args.json:
        print(json.dumps({"tracked": rows, "local_only_count": len(shards)}, ensure_ascii=False, indent=2))
        return 1 if args.strict and failed else 0

    print(f"{'status':7} {'lines':>6} {'uniq':>5} {'top':>5} {'ar':>5} {'en':>5}  file")
    for row in rows:
        print(
            f"{row['status']:7} {row['lines']:6d} {row['unique_ratio']:5.2f} "
            f"{row['top_ratio']:5.2f} {row['arabic_ratio']:5.2f} {row['latin_ratio']:5.2f}  "
            f"{row['file']}  {','.join(row['reasons'])}"
        )

    print(
        f"\n{len(rows)} tracked files | {len(failed)} FAIL | "
        f"{sum(row['status'] == 'REVIEW' for row in rows)} REVIEW | "
        f"{sum(row['status'] == 'PASS' for row in rows)} PASS"
    )
    print(f"local-only ignored shards (never indexed): {len(shards)}")
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    sys.exit(main())
