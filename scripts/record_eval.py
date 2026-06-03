"""
Record one daily NL2TL-Tree evaluation run and maintain the 3-day-average
success rate that goes into the paper.

Each day you run the eval:

    python3 nl_to_tree_groq.py --input data/nl2tl_converted.jsonl \
        --output data/predictions.jsonl --limit 50 --eval

then run THIS once to bank the result:

    python3 scripts/record_eval.py

It reads data/predictions.jsonl (whose `--eval` rows carry per-row `metrics`),
aggregates them, appends ONE dated record to
outputs/nl2tl_tree_eval/results.jsonl (one record per calendar day; re-running
the same day overwrites that day), archives that day's predictions as
predictions_<date>.jsonl, and rewrites SUMMARY.md with the per-day table plus
the running average and the rolling 3-day average -- the number to report.

Flags:
    --predictions  path to the eval output (default data/predictions.jsonl)
    --date         YYYY-MM-DD (default: today)
    --window       rolling-average window in days (default 3)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tree_metrics import aggregate

STORE_DIR = ROOT / "outputs" / "nl2tl_tree_eval"
STORE = STORE_DIR / "results.jsonl"
SUMMARY = STORE_DIR / "SUMMARY.md"


def _load_metrics(predictions_path: Path):
    rows = [json.loads(l) for l in predictions_path.read_text().splitlines()
            if l.strip()]
    metrics = [r["metrics"] for r in rows if "metrics" in r]
    n_err = sum(1 for r in rows if "error" in r)
    return metrics, len(rows), n_err


def _mean(key, recs):
    return (sum(r[key] for r in recs) / len(recs)) if recs else 0.0


def _pool_archives(store_dir: Path):
    """Pool per-row metrics across ALL archived days, deduped by the NL
    sentence -- so overlapping windows don't double-count.  Returns the
    aggregate over the union of distinct test sentences seen so far."""
    seen = {}
    for f in sorted(store_dir.glob("predictions_*.jsonl")):
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            if "metrics" in o and o.get("natural"):
                seen[o["natural"]] = o["metrics"]      # dedup by sentence
    return list(seen.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(ROOT / "data" / "predictions.jsonl"))
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--avg-days", type=int, default=3,
                    help="rolling-average window in DAYS (default 3)")
    ap.add_argument("--offset", type=int, default=0,
                    help="row offset used for this run (stored, so the next day "
                    "can pick a disjoint window)")
    ap.add_argument("--window", type=int, default=0,
                    help="row window (--limit) used for this run (stored)")
    args = ap.parse_args(argv)

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found -- run nl_to_tree_groq.py --eval first.",
              file=sys.stderr)
        return 2
    metrics, n_total, n_err = _load_metrics(pred_path)
    if not metrics:
        print(f"ERROR: no rows with 'metrics' in {pred_path}. "
              f"Re-run nl_to_tree_groq.py WITH --eval (and gold trees present).",
              file=sys.stderr)
        return 2

    agg = aggregate(metrics)
    today = args.date or date.today().isoformat()
    rec = {
        "date": today, "n": agg["n"], "n_total": n_total, "n_err": n_err,
        "offset": args.offset, "window": args.window or n_total,
        "exact_match": round(agg["exact_match"], 4),
        "op_f1": round(agg.get("op_f1", 0.0), 4),
        "path_f1": round(agg.get("path_f1", 0.0), 4),
        "ted_norm": round(agg.get("ted_norm", 0.0), 4),
        "mean_ted": round(agg.get("mean_ted") or 0.0, 3),
        "model": "llama-3.3-70b-versatile", "source": pred_path.name,
    }

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    store = Path(args.store)
    records = ([json.loads(l) for l in store.read_text().splitlines() if l.strip()]
               if store.exists() else [])
    records = [r for r in records if r["date"] != today]   # one record per day
    records.append(rec)
    records.sort(key=lambda r: r["date"])
    store.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    # archive this day's raw predictions so they are not overwritten tomorrow
    archive = STORE_DIR / f"predictions_{today}.jsonl"
    shutil.copyfile(pred_path, archive)

    all_em = _mean("exact_match", records)
    win = records[-args.avg_days:]
    win_em = _mean("exact_match", win)
    pooled = aggregate(_pool_archives(STORE_DIR))      # union of distinct rows

    print(f"recorded {today}: n={rec['n']} (of {n_total}, {n_err} err)  "
          f"EM={rec['exact_match']:.1%}  op_f1={rec['op_f1']:.3f}  "
          f"path_f1={rec['path_f1']:.3f}  ted_norm={rec['ted_norm']:.3f}")
    print(f"days recorded: {len(records)} ; all-day avg EM = {all_em:.1%}")
    print(f"POOLED over {pooled['n']} distinct sentences: EM={pooled['exact_match']:.1%}  "
          f"op_f1={pooled.get('op_f1', 0):.3f}  path_f1={pooled.get('path_f1', 0):.3f}  "
          f"ted_norm={pooled.get('ted_norm', 0):.3f}")
    if len(records) >= args.avg_days:
        print(f"*** {args.avg_days}-day mean-of-days EM = {win_em:.1%} ; "
              f"POOLED EM = {pooled['exact_match']:.1%} over {pooled['n']} rows  "
              f"<- PAPER NUMBER (prefer pooled) ***")
    else:
        need = args.avg_days - len(records)
        print(f"(need {need} more day(s) for the {args.avg_days}-day average; "
              f"use a different --offset each day so rows stay disjoint)")

    # human-readable summary
    L = ["# NL2TL-Tree daily evaluation",
         "",
         f"Zero-shot NL→STL-tree parsing (`nl_to_tree_groq.py --eval`), "
         f"llama-3.3-70b. The rolling **{args.avg_days}-day average** exact-match "
         f"is the success rate reported in the paper.",
         "",
         "| date | n | exact_match | op_f1 | path_f1 | ted_norm |",
         "|---|---|---|---|---|---|"]
    for r in records:
        L.append(f"| {r['date']} | {r['n']} | {r['exact_match']:.1%} | "
                 f"{r['op_f1']:.3f} | {r['path_f1']:.3f} | {r['ted_norm']:.3f} |")
    L += ["",
          f"- days recorded: **{len(records)}**",
          f"- mean-of-days EM: **{all_em:.1%}**",
          f"- **pooled EM over {pooled['n']} distinct sentences: "
          f"{pooled['exact_match']:.1%}** "
          f"(op_f1 {pooled.get('op_f1', 0):.3f}, "
          f"path_f1 {pooled.get('path_f1', 0):.3f}, "
          f"ted_norm {pooled.get('ted_norm', 0):.3f}) — the number to report"]
    if len(records) < args.avg_days:
        L.append(f"- _{args.avg_days}-day target pending: "
                 f"need {args.avg_days - len(records)} more day(s); run each day "
                 f"with a different `--offset` (0, 50, 100, …) so rows are "
                 f"disjoint._")
    L += ["", "Per-day raw predictions archived as `predictions_<date>.jsonl`. "
          "Pooled = union of distinct sentences across archives (deduped by "
          "sentence, so overlapping windows are not double-counted)."]
    SUMMARY.write_text("\n".join(L) + "\n")

    print(f"wrote {store}\n      {SUMMARY}\n      {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
