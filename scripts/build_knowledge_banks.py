"""
Builds generated knowledge banks from compact seed files.

Seeds (one entry per line, pipe-separated, hand-written and reviewable):
    seeds_verbs.txt  -> ماض|مضارع|أمر|مصدر|فاعل|مفعول|مثال|نوع
    seeds_vocab_a/b  -> كلمة|معنى|مضاد|جمع|مثال
    seeds_fixed.txt  -> كلمة|إعراب ثابت|مثال|القاعدة

Outputs (committed, ingested by SANAD):
    data/knowledge/gen_sarf_tables_p1.txt
    data/knowledge/gen_sarf_tables_p2.txt
    data/knowledge/gen_vocab_bank.txt
    data/knowledge/gen_fixed_irab.txt

Usage:
    python scripts/build_knowledge_banks.py
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
KNOWLEDGE = ROOT / "data" / "knowledge"


def read_seeds(name: str) -> list[list[str]]:
    rows = []
    with open(KNOWLEDGE / name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([c.strip() for c in line.split("|")])
    return rows


def _gen_sarf_rows(rows: list[list[str]]) -> list[str]:
    """Generate sarf output lines from a list of seed rows."""
    out = []
    for row in rows:
        if len(row) != 8:
            continue
        p, pr, imp, mas, af, mf, ex, typ = row
        out.append(f"تصريف الفعل {p}: الماضي {p} والمضارع {pr} والأمر {imp}.")
        out.append(f"مصدر الفعل {p} هو {mas}.")
        out.append(f"اسم الفاعل من الفعل {p} هو {af} واسم المفعول هو {mf}.")
        out.append(f"الفعل {p} فعل {typ}.")
        out.append(f"مثال تطبيقي على الفعل {p}: {ex}.")
        out.append(f"مثال الرفع: {pr} الطالب بجد واجتهاد.")
        out.append(f"مثال النصب: لن {pr} المهمل أبدا.")
        out.append(f"مثال الجزم: لم {pr} المهمل أمس.")
    return out


def build_sarf() -> dict[str, list[str]]:
    """Returns sharded sarf tables: no single part may exceed the /ingest limit (200k chars).

    Output files:
        data/knowledge/gen_sarf_tables_p1.txt  (part 1)
        data/knowledge/gen_sarf_tables_p2.txt  (part 2)
    """
    # Collect all valid rows from all verb seed files
    all_rows: list[list[str]] = []
    for name in sorted(KNOWLEDGE.glob("seeds_verbs*.txt")):
        for row in read_seeds(name.name):
            all_rows.append(row)

    # Balance rows across two parts keeping each part < 200k chars
    mid = len(all_rows) // 2
    part1_rows = all_rows[:mid]
    part2_rows = all_rows[mid:]

    # Generate output lines for each part
    part1_output = _gen_sarf_rows(part1_rows)
    part2_output = _gen_sarf_rows(part2_rows)

    # Prepend section headers
    part1_lines = ["## بنك جداول التصريف المولد (الجزء الأول)"] + part1_output
    part2_lines = ["## بنك جداول التصريف المولد (الجزء الثاني)"] + part2_output

    # Generate output lines for each part
    part1_output = _gen_sarf_rows(part1_rows)
    part2_output = _gen_sarf_rows(part2_rows)

    # Prepend section headers
    part1_lines = ["## بنك جداول التصريف المولد (الجزء الأول)"] + part1_output
    part2_lines = ["## بنك جداول التصريف المولد (الجزء الثاني)"] + part2_output

    return {
        "gen_sarf_tables_p1.txt": part1_lines,
        "gen_sarf_tables_p2.txt": part2_lines,
    }


def build_vocab() -> list[str]:
    out = ["## بنك المفردات المولد"]
    for path in sorted(KNOWLEDGE.glob("seeds_vocab*.txt")):
        for row in read_seeds(path.name):
            if len(row) != 5:
                continue
            w, meaning, ant, pl, ex = row
            out.append(f"معنى كلمة {w}: {meaning}.")
            out.append(f"مضاد كلمة {w}: {ant}.")
            out.append(f"جمع كلمة {w}: {pl}.")
            out.append(f"مثال على كلمة {w}: {ex}.")
    return out


def build_fixed() -> list[str]:
    out = ["## بنك الإعراب الثابت المولد"]
    for row in read_seeds("seeds_fixed.txt"):
        if len(row) != 4:
            continue
        w, irab, ex, rule = row
        out.append(f"الكلمة {w} إعرابها ثابت لا يتغير: {irab}.")
        out.append(f"مثال على {w}: {ex}.")
        out.append(f"القاعدة: {rule}.")
    return out


def main() -> None:
    sarf_banks = build_sarf()
    vocab_bank = build_vocab()
    fixed_irab = build_fixed()

    banks = {
        **sarf_banks,  # gen_sarf_tables_p1.txt, gen_sarf_tables_p2.txt
        "gen_vocab_bank.txt": vocab_bank,
        "gen_fixed_irab.txt": fixed_irab,
    }
    total = 0
    for name, lines in banks.items():
        path = KNOWLEDGE / name
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        total += len(lines)
        print(f"{name}: {len(lines)} lines")
    print(f"TOTAL generated: {total} lines")


if __name__ == "__main__":
    main()