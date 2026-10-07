"""Print misses (leaks) and unmatched predictions for one config.

    .venv/bin/python bench/inspect_errors.py presidio+sm [hard|sanctum] [threshold]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import CORPORA  # noqa: E402
from run import load_run  # noqa: E402
from score import _ov, group_of  # noqa: E402


def main(key: str, corpus: str = "hard", thr: str | None = None) -> None:
    docs = CORPORA[corpus]()
    run = load_run(key)
    by_thr = run["corpora"][corpus]["by_thr"]
    if thr is None:
        thr = "none" if run.get("default_thr") is None else str(run["default_thr"])
    preds = by_thr[thr]
    for name, d in docs.items():
        ps = [p for p in preds.get(name, []) if group_of(p["entity_type"])]
        miss = [g for g in d["entities"] if not any(_ov(p, g) for p in ps)]
        wrong = [g for g in d["entities"]
                 if any(_ov(p, g) for p in ps)
                 and not any(_ov(p, g) and group_of(p["entity_type"]) == group_of(g["entity_type"]) for p in ps)]
        fps = [p for p in ps if not any(_ov(p, x) for x in d["entities"] + d["optional"])]
        if not (miss or fps or wrong):
            continue
        print(f"\n## {name}")
        for g in miss:
            print(f"  MISS  {g['entity_type']:<16} {g['text']!r}")
        for g in wrong:
            got = {p['entity_type'] for p in ps if _ov(p, g)}
            print(f"  TYPE  {g['entity_type']:<16} {g['text']!r} -> {sorted(got)}")
        for p in fps:
            print(f"  FP    {p['entity_type']:<16} {p['text']!r} ({p['score']:.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
