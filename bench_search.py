#!/usr/bin/env python3
"""Empirical comparison of recoll vs sqlite search backends.

Measures wall-clock time and result overlap (Jaccard on step-level hits)
for a set of representative queries.

Run after `trajectoriz-cli refresh` has populated both indexes.

## Key findings (measured on ~5400 records / 114 000 step rows)

Speed (median, 3 runs):
  recoll  65 ms – 1.8 s  (Xapian lookup → parse top-200 candidate files → grep)
  sqlite   1 ms – 2.6 s  (FTS5 MATCH on 114 000 step rows)
  recoll wins on rare/specific queries; sqlite wins on dense common-word queries.

Result overlap (Jaccard of step-level (traj_id, step_id) pairs):
  0.07 – 0.34 — substantially different result sets.

Two root causes:
  1. Scope: recoll indexes JSONL topdirs only (~2500 files); sqlite covers all
     5851 records including DB-backed sessions (opencode, hermes, codex_db).
  2. Cap: RecollBackend fetches at most _RECOLL_MAX_CANDIDATES=200 Xapian hits
     then greps within those files. Raising to 2000 improves Jaccard to 0.28–0.56
     but blows latency to 6–26 s — not practical.
  3. Tokenisation: recoll does substring match inside step blobs; sqlite FTS5
     (unicode61) does word-boundary match. Recoll catches subword hits like
     "cpython" or "python3"; sqlite misses those but finds more step rows overall
     because it searches the full corpus.

Practical guidance:
  --backend recoll  → fast, relevance-ranked, top-200-files recall
  --backend sqlite  → comprehensive (full corpus), word-boundary matching
  --backend grep    → full corpus + substring matching, linear scan (slowest)
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import trajectoriz as tz
from trajectoriz._search import RecollBackend, SqliteBackend, _parse_terms

QUERIES = [
    "python",           # very common — many hits expected
    "refactor",         # moderately common
    "sonnet",           # specific model name (also appears mid-token: claude-sonnet-4)
    "authentication",   # multi-syllable word
    r"error\|exception",  # OR query
    "xyzzy_nonexistent_qqq",  # no hits
]

REPEATS = 3          # timing runs per query
ALL_RECORDS = list(tz.iter_all_records())

backends: dict[str, Callable] = {
    "recoll": RecollBackend().search,
    "sqlite": SqliteBackend().search,
}


def run_query(backend_fn, terms: list[str]) -> tuple[float, list[tuple[str, int]]]:
    """Return (median_seconds, sorted list of (traj_id, step_id))."""
    times: list[float] = []
    result_set: list[tuple[str, int]] = []
    for i in range(REPEATS):
        t0 = time.perf_counter()
        matches = backend_fn(ALL_RECORDS, terms)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            result_set = sorted((r.id, sid) for r, sid, _ in matches)
    return statistics.median(times), result_set


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def fmt(t: float) -> str:
    return f"{t*1000:7.1f}ms"


print(f"{'Query':<30}  {'recoll':>9}  {'sqlite':>9}  {'|recoll|':>9}  {'|sqlite|':>9}  {'Jaccard':>8}  {'only-recoll':>11}  {'only-sqlite':>11}")
print("-" * 120)

for query in QUERIES:
    terms = _parse_terms(query)
    results: dict[str, tuple[float, list]] = {}
    errors: dict[str, str] = {}

    for name, fn in backends.items():
        try:
            t, hits = run_query(fn, terms)
            results[name] = (t, hits)
        except Exception as exc:
            errors[name] = str(exc)[:60]

    if errors:
        for name, msg in errors.items():
            print(f"  {name} ERROR: {msg}")
        continue

    t_r, hits_r = results["recoll"]
    t_s, hits_s = results["sqlite"]

    only_r = len(set(hits_r) - set(hits_s))
    only_s = len(set(hits_s) - set(hits_r))
    j = jaccard(hits_r, hits_s)

    print(
        f"{query:<30}  {fmt(t_r):>9}  {fmt(t_s):>9}  "
        f"{len(hits_r):>9}  {len(hits_s):>9}  "
        f"{j:>8.3f}  {only_r:>11}  {only_s:>11}"
    )

print()
print("Notes:")
print("  Jaccard: overlap of (traj_id, step_id) pairs between backends (1.0 = identical)")
print("  only-recoll: step hits in recoll not in sqlite (substring matches sqlite misses)")
print("  only-sqlite: step hits in sqlite not in recoll (stemming / recoll indexing scope)")
print(f"  Timing: median of {REPEATS} runs. ALL_RECORDS={len(ALL_RECORDS)} records.")
