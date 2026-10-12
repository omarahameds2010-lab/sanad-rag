"""SANAD safe knowledge injector (honest, throttled).

What it does:
- Scans data/knowledge/*.txt (skips seeds_*)
- Splits any text over --limit chars on SENTENCE boundaries only
  (same rule as src/chunking/arabic_chunker.py, plus newlines),
  falls back to word boundaries for a single over-long sentence.
- Sends each part to POST {base}/ingest with X-API-Key,
  max 3 concurrent requests, retry on 429/5xx only.

What it does NOT do:
- No background queue, no 202, no Kafka. The API is synchronous
  (see src/api/main.py:196) and returns 200 after ingest.
- No byte-slicing: split is on str chars, never bytes, so Arabic
  multibyte characters are never corrupted.

Usage:
  set SANAD_API_KEY=demo-key-123
  .venv/Scripts/python scripts/sanad_safe_injector.py --dry-run
  .venv/Scripts/python scripts/sanad_safe_injector.py
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Same sentence-final rule as src/chunking/arabic_chunker.py:29,
# extended with explicit newline splitting for pre-sharded banks.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!؟?؛۔])\s+|\n+")

DEFAULT_LIMIT = 180000  # chars, safely under API max 200000
DEFAULT_CONCURRENCY = 3

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"


def split_on_sentences(text: str, limit: int) -> list[str]:
    """Split text into parts <= limit chars, never mid-sentence/word."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    sentences = [s for s in SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        if current:
            parts.append(" ".join(current).strip())

    for sent in sentences:
        # Single sentence longer than limit: fall back to word boundaries.
        while len(sent) > limit:
            # Take whole words up to limit.
            cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit  # one very long token, hard cut unavoidable
            flush_chunk = sent[:cut].strip()
            if current:
                flush()
                current.clear()
            parts.append(flush_chunk)
            sent = sent[cut:].strip()
            current_len = 0
        sep = 1 if current else 0
        if current_len + sep + len(sent) > limit:
            flush()
            current.clear()
            current_len = 0
        current.append(sent)
        current_len += (sep + len(sent))

    flush()
    return [p for p in parts if p]


def plan_files(limit: int) -> list[tuple[Path, list[str]]]:
    plan = []
    for fp in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        if fp.name.startswith("seeds_"):
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except OSError as e:
            print(f"SKIP {fp.name}: unreadable ({e})")
            continue
        if not text.strip():
            print(f"SKIP {fp.name}: empty")
            continue
        parts = split_on_sentences(text, limit)
        plan.append((fp, parts))
    return plan


async def send_one(session, base: str, api_key: str, source_id: str,
                   text: str, metadata: dict, chunker_type: str,
                   sem: asyncio.Semaphore, retries: int = 3) -> tuple[str, bool, str]:
    payload = {
        "source_id": source_id,
        "text": text,
        "metadata": metadata,
        "chunker_type": chunker_type,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    url = base.rstrip("/") + "/ingest"

    async with sem:
        last_status = ""
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=60) as r:
                    body = await r.text()
                    last_status = f"HTTP {r.status}"
                    if 200 <= r.status < 300:
                        return (source_id, True, last_status)
                    if r.status in (401, 422):
                        # Client error: retrying will not help.
                        return (source_id, False, f"{last_status} {body[:200]}")
                    # 429 / 5xx: retry with backoff.
            except asyncio.TimeoutError:
                last_status = "timeout"
            except Exception as e:  # network errors are retryable
                last_status = f"conn-error: {type(e).__name__}"
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
        return (source_id, False, last_status)


async def run(base: str, api_key: str, limit: int, concurrency: int,
              chunker_type: str, only: str | None) -> int:
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp not installed. Run: pip install -r requirements.txt")
        return 2

    plan = plan_files(limit)
    if only:
        plan = [(fp, parts) for fp, parts in plan if fp.name == only]
        if not plan:
            print(f"No file matched --only={only}")
            return 2

    sem = asyncio.Semaphore(concurrency)
    failures = 0
    async with aiohttp.ClientSession() as session:
        for fp, parts in plan:
            print(f"{fp.name}: {len(parts)} part(s), "
                  f"{sum(len(p) for p in parts)} chars total")
            tasks = []
            for i, part in enumerate(parts, 1):
                if len(parts) == 1:
                    source_id = f"{fp.stem}_v1"
                else:
                    source_id = f"{fp.stem}_part{i}_v1"
                metadata = {"dataset": "sanad-library", "file": fp.name}
                if len(parts) > 1:
                    metadata["part"] = f"{i}/{len(parts)}"
                tasks.append(send_one(session, base, api_key, source_id,
                                      part, metadata, chunker_type, sem))
            for coro in asyncio.as_completed(tasks):
                source_id, ok, status = await coro
                print(f"  {source_id}: {'OK' if ok else 'FAIL'} ({status})")
                if not ok:
                    failures += 1
    print(f"DONE. failures={failures}")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Safe throttled SANAD injector")
    ap.add_argument("--base", default=os.environ.get("SANAD_BASE_URL",
                                                     "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("SANAD_API_KEY", ""),
                    help="X-API-Key (or set SANAD_API_KEY env var)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--chunker-type", default="hierarchical",
                    choices=["arabic", "baseline", "hierarchical"])
    ap.add_argument("--only", default=None, help="inject a single file name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        plan = plan_files(args.limit)
        if args.only:
            plan = [(fp, p) for fp, p in plan if fp.name == args.only]
        total_parts = sum(len(p) for _, p in plan)
        total_chars = sum(sum(len(x) for x in p) for _, p in plan)
        over = [(fp.name, len(x)) for fp, p in plan for x in p
                if len(x) > 200000]
        print(f"files={len(plan)} parts={total_parts} chars={total_chars}")
        for fp, parts in plan[:10]:
            print(f"  {fp.name}: {len(parts)} part(s)")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more files")
        if over:
            print(f"OVER-LIMIT (would 422): {over}")
            sys.exit(2)
        print("dry-run OK: all parts under API limit.")
        return

    if not KNOWLEDGE_DIR.is_dir():
        print(f"ERROR: knowledge dir not found: {KNOWLEDGE_DIR}")
        sys.exit(2)
    code = asyncio.run(run(args.base, args.api_key, args.limit,
                           args.concurrency, args.chunker_type, args.only))
    sys.exit(code)


if __name__ == "__main__":
    main()
