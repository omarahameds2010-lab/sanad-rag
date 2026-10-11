"""Ingest wave-3 files (idempotent upserts)."""
import json
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "X-API-Key": "demo-key-123"}
ROOT = Path(__file__).parent.parent

NEW_FILES = [
    "amthal_hikam.txt",
    "insha_models.txt",
    "quran_vocab.txt",
    "seera_nabaweya.txt",
    "tarikh_masr.txt",
]


def call(method, path, payload=None, timeout=600):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> None:
    for name in NEW_FILES:
        text = (ROOT / "data" / "knowledge" / name).read_text(encoding="utf-8")
        status, body = call("POST", "/ingest", {
            "source_id": Path(name).stem + "_v1",
            "text": text,
            "metadata": {"dataset": "sanad-library"},
            "chunker_type": "hierarchical",
        })
        n = body.get("n_chunks") if isinstance(body, dict) else body
        print(f"{name}: {status} chunks={n}", flush=True)


if __name__ == "__main__":
    main()
