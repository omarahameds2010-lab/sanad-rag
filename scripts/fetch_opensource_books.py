"""Fetches public-domain Arabic grammar verse fromar.wikisource and saves
a cleaned corpus file with attribution headers.

Sources (Wikisource, freely licensed texts):
    - نظم الآجرومية (pageid 175768) — versified Ajurrumiyya primer

Usage:
    python scripts/fetch_opensource_books.py
"""

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
KNOWLEDGE = ROOT / "data" / "knowledge"
API = "https://ar.wikisource.org/w/api.php"


def fetch_wikitext(pageid: int) -> str:
    url = (
        f"{API}?action=parse&pageid={pageid}"
        f"&prop=wikitext&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SANAD-edu/0.4"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["parse"]["wikitext"]["*"]


def clean_nazm(wikitext: str) -> list[str]:
    # Verses live inside many {{أبيات| ... }} blocks with "\\" as the
    # hemistich separator — collect every block.
    blocks = re.findall(r"\{\{أبيات\|(.*?)\}\}", wikitext, re.DOTALL)
    body = "\n".join(blocks) if blocks else wikitext
    # Drop any remaining simple templates
    body = re.sub(r"\{\{[^{}]*\}\}", "", body)
    # Verse separator in this edition is "\\" -> newline per hemistich
    body = body.replace("\\\\", "\n")
    lines = []
    for raw in body.split("\n"):
        line = raw.strip().strip("|").strip()
        line = re.sub(r"\[\[|\]\]", "", line)
        line = re.sub(r"\s+", " ", line)
        if len(line) < 8:
            continue
        if line.startswith("تصنيف:") or line.startswith("ملاحظات:"):
            continue
        lines.append(line)
    return lines


def main() -> None:
    lines = clean_nazm(fetch_wikitext(175768))
    header = [
        "المصدر: نظم الآجرومية للإمام محمد بن آب القلاوي الشنقيطي.",
        "مأخوذ من ويكي مصدر العربية (نص حر).",
        "استخدام تعليمي ضمن مكتبة سند.",
        "",
    ]
    path = KNOWLEDGE / "opensource_nazm_ajurrumiyya.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")
    print(f"{path.name}: {len(header) + len(lines)} lines")


if __name__ == "__main__":
    main()
