# NL2TL-Tree daily evaluation

Zero-shot NL→STL-tree parsing (`nl_to_tree_groq.py --eval`), llama-3.3-70b. The rolling **3-day average** exact-match is the success rate reported in the paper.

| date | n | exact_match | op_f1 | path_f1 | ted_norm |
|---|---|---|---|---|---|
| 2026-05-29 | 48 | 47.9% | 0.925 | 0.557 | 0.823 |

- days recorded: **1**
- mean-of-days EM: **47.9%**
- **pooled EM over 48 distinct sentences: 47.9%** (op_f1 0.925, path_f1 0.557, ted_norm 0.823) — the number to report
- _3-day target pending: need 2 more day(s); run each day with a different `--offset` (0, 50, 100, …) so rows are disjoint._

Per-day raw predictions archived as `predictions_<date>.jsonl`. Pooled = union of distinct sentences across archives (deduped by sentence, so overlapping windows are not double-counted).
