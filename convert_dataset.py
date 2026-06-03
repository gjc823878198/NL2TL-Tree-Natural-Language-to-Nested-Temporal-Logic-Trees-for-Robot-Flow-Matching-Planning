"""
Convert an NL2TL-style dataset of (natural language, STL formula) pairs into
JSONL with parse trees attached. Auto-detects CSV / JSON / JSONL and tries
common column names.

Output line schema:
  {"natural": <str>, "formula": <str>, "tree": <ast-dict-or-null>,
   "ok": true | false, "error": <str-if-failed>}
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from stl_parser import parse_stl

NL_KEYS = ["natural", "nl", "english", "text", "input",
           "instruction", "sentence", "description", "logic_sentence"]
FORMULA_KEYS = ["formula", "stl", "ltl", "tl", "output", "target", "logic",
                "stl_formula", "tl_formula", "logic_ltl"]


def _pick(row: Dict[str, Any], keys) -> Optional[str]:
    """Pick the first matching field, joining token-list values into a string."""
    if not isinstance(row, dict):
        return None
    lc = {k.lower(): k for k in row.keys()}
    for want in keys:
        if want in lc:
            v = row[lc[want]]
            if v is None:
                continue
            # NL2TL stores tokens as lists; join with spaces
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            return str(v).strip()
    return None


def load_rows(path: Path) -> Iterable[Dict[str, Optional[str]]]:
    """Yield {'natural': ..., 'formula': ...} rows from a CSV/JSON/JSONL file."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {"natural": _pick(row, NL_KEYS), "formula": _pick(row, FORMULA_KEYS)}

    elif suffix == ".json":
        data = json.loads(text)
        items = data if isinstance(data, list) else [data]
        for row in items:
            yield {"natural": _pick(row, NL_KEYS), "formula": _pick(row, FORMULA_KEYS)}

    elif suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        for row in reader:
            yield {"natural": _pick(row, NL_KEYS), "formula": _pick(row, FORMULA_KEYS)}

    else:
        raise SystemExit(f"Unsupported file type: {path.suffix} (expected csv/tsv/json/jsonl)")


def main():
    ap = argparse.ArgumentParser(description="Convert NL2TL-style data into JSONL with parse trees.")
    ap.add_argument("input", help="Input CSV / TSV / JSON / JSONL file.")
    ap.add_argument("output", help="Output JSONL path (will be overwritten).")
    ap.add_argument("--limit", type=int, default=0, help="Convert only first N rows (0 = all).")
    ap.add_argument("--keep-failed", action="store_true",
                    help="Also write rows where parsing failed (useful to feed back into grammar work).")
    args = ap.parse_args()

    in_p = Path(args.input)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_ok = n_skip = n_fail = 0
    fail_examples = []

    with out_p.open("w", encoding="utf-8") as out:
        for i, row in enumerate(load_rows(in_p)):
            if args.limit and i >= args.limit:
                break
            n_total += 1
            nl, formula = row.get("natural"), row.get("formula")
            if not formula:
                n_skip += 1
                continue
            try:
                tree = parse_stl(formula)
                out.write(json.dumps(
                    {"natural": nl, "formula": formula, "tree": tree, "ok": True},
                    ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if len(fail_examples) < 5:
                    fail_examples.append((formula, type(e).__name__ + ": " + str(e)))
                if args.keep_failed:
                    out.write(json.dumps(
                        {"natural": nl, "formula": formula, "tree": None,
                         "ok": False, "error": str(e)},
                        ensure_ascii=False) + "\n")

    print(f"Read   : {n_total}")
    print(f"OK     : {n_ok}")
    print(f"Skipped: {n_skip}  (no formula field)")
    print(f"Failed : {n_fail}  (parse error)")
    if fail_examples:
        print("\nFirst few parse failures (extend the grammar to handle these):")
        for f, e in fail_examples:
            print("  formula:", repr(f))
            print("  error  :", e)
            print()
    print(f"\nWrote: {out_p}")


if __name__ == "__main__":
    main()
