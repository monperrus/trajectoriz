#!/usr/bin/env python3
"""Empirical comparison of recoll vs sqlite search backends.

Measures wall-clock time and result overlap (Jaccard on step-level hits)
for a set of representative queries.

Run after `trajectoriz-cli refresh` has populated both indexes.

## Key findings (measured on ~5400 records / 114 000 step rows)

Speed (median, 3 runs):
  recoll  420 ms – 2.0 s  (Xapian → grep top-200 JSONL candidates + DB-backed grep)
  sqlite    1 ms – 2.4 s  (FTS5 MATCH on 114 000 step rows)
  sqlite wins on rare queries; both are comparable on common ones.

Result overlap (Jaccard of step-level (traj_id, step_id) pairs):
  0.11 – 0.41 — both backends now cover the same corpus. Remaining divergence
  is entirely the 200-candidate cap: recoll only parses the top-200 Xapian hits
  for JSONL files, missing lower-ranked but still matching files. Raising the
  cap to 2000 improves Jaccard to 0.28–0.56 but blows latency to 6–26 s.

Tokenisation difference (explains only-recoll column, always small):
  recoll does substring grep inside step blobs; sqlite FTS5 (unicode61) does
  word-boundary matching. Recoll catches subword hits like "cpython" or
  "python3"; sqlite misses those (hence a few hundred only-recoll hits).

Practical guidance:
  --backend recoll  → fast, relevance-ranked, top-200-JSONL recall + all DB sessions
  --backend sqlite  → comprehensive full-corpus FTS, word-boundary matching
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
