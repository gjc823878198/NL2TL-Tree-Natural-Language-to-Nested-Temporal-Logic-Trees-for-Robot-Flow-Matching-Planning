#!/usr/bin/env bash
# One-command daily NL2TL-Tree evaluation.
#
# Runs 50 NEW (disjoint) test rows on the 70b model, then banks the result and
# updates the pooled 3-day success rate.  The offset auto-advances by 50 each
# day (day 1 = rows 0-49, day 2 = 50-99, day 3 = 100-149), so the 3-day pooled
# estimate covers ~150 DISTINCT sentences -- NOT 3x the same deterministic 50
# (temperature=0 + fixed few-shot seed would otherwise reproduce day 1 exactly,
# making a "3-day average" meaningless).
#
# The 70b free tier is ~100k tokens/day == ~50 of these calls, so run this
# ONCE per day, after the quota resets at UTC midnight:
#
#   export GROQ_API_KEY=gsk_...
#   bash scripts/daily_eval.sh
#
# After 3 days, outputs/nl2tl_tree_eval/SUMMARY.md shows the pooled EM to report.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
cd "$CODE"

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "ERROR: export GROQ_API_KEY=gsk_... first (free key: https://console.groq.com/keys)" >&2
  exit 1
fi

STORE="outputs/nl2tl_tree_eval/results.jsonl"
PER_DAY="${PER_DAY:-40}"          # 40 < ~48/day 70b token-quota ceiling (buffer)
# Next offset = furthest row reached so far, so each day's window is DISJOINT
# from every prior day (robust even if PER_DAY changes between days).
DAYS=0
[ -f "$STORE" ] && DAYS=$(grep -c . "$STORE" 2>/dev/null || echo 0)
OFFSET=$(python3 - "$STORE" <<'PY'
import sys, json, pathlib
p = pathlib.Path(sys.argv[1]); nxt = 0
if p.exists():
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            nxt = max(nxt, r.get("offset", 0) + r.get("window", r.get("n", 0)))
print(nxt)
PY
)

# Optional accuracy boosters (cost extra tokens -> fewer rows/day on the 70b
# 100k-token quota): SC=K enables #1 self-consistency (K samples, majority
# vote); ROUNDTRIP=1 enables #4 round-trip scope self-refine.
#   SC=5 ROUNDTRIP=1 bash scripts/daily_eval.sh
SC="${SC:-1}"; ROUNDTRIP="${ROUNDTRIP:-0}"
EXTRA=""
if [ "$SC" -gt 1 ] 2>/dev/null; then EXTRA="$EXTRA --self-consistency $SC"; fi
if [ "$ROUNDTRIP" = "1" ]; then EXTRA="$EXTRA --roundtrip"; fi

echo "[daily_eval] day #$((DAYS + 1)): testing rows ${OFFSET}..$((OFFSET + PER_DAY - 1)) (disjoint from prior days)${EXTRA:+ ; boosters:$EXTRA}"
python3 nl_to_tree_groq.py \
  --input  data/nl2tl_converted.jsonl \
  --output data/predictions.jsonl \
  --limit "$PER_DAY" --offset "$OFFSET" --eval $EXTRA

echo "[daily_eval] banking result + updating pooled average ..."
python3 scripts/record_eval.py --offset "$OFFSET" --window "$PER_DAY"
