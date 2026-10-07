"""Score cached predictions (results/preds/*.json) and write results/*.csv + summary.

Metrics (all overlap-based, like Sanctum's EntityScorer OVERLAP mode):

* leak_recall   — share of ground-truth entities touched by *any* prediction,
                  whatever its label. For a redaction tool this is the number
                  that matters: an untouched entity leaks.
* char_recall   — share of ground-truth characters covered by any prediction.
                  Catches partial redaction ("Charlotte" redacted, "Rose Hill" not).
* precision     — share of predictions that overlap some ground-truth or
                  optional span (any label). 1 - precision = over-redaction.
* typed_f1      — micro F1 where a hit also needs the right coarse group
                  (PERSON / ORGANIZATION / LOCATION / DATE_TIME / EMAIL / PHONE /
                  URL / IP / FINANCIAL / GOV_ID). Coarse groups avoid penalising
                  e.g. SSN-vs-national-ID label choices that don't change the
                  anonymised output.
* macro_f1      — mean of per-group typed F1 (comparable in spirit to the
                  macro number `pytest -m evaluation` prints).

Predictions overlapping only an *optional* span are ignored entirely.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import CORPORA  # noqa: E402
from run import load_runs  # noqa: E402

GROUP = {
    "PERSON": "PERSON", "ORGANIZATION": "ORGANIZATION", "LOCATION": "LOCATION",
    "DATE_TIME": "DATE_TIME", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE",
    "URL": "URL", "IP_ADDRESS": "IP",
    "IBAN_CODE": "FINANCIAL", "CREDIT_CARD": "FINANCIAL", "US_BANK_NUMBER": "FINANCIAL",
    "CRYPTO": "FINANCIAL",
    "US_SSN": "GOV_ID", "US_DRIVER_LICENSE": "GOV_ID", "ID_NUMBER": "GOV_ID",
}
DROP = {"NRP"}  # nationality / religion / political group: not annotated


def group_of(etype: str) -> str | None:
    if etype in DROP:
        return None
    # Presidio's country-specific recognizers (UK_NHS, IT_FISCAL_CODE, US_PASSPORT,
    # SG_NRIC_FIN, ...) are all personal identifiers.
    return GROUP.get(etype, "GOV_ID")


def _ov(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def score_corpus(preds: dict[str, list[dict]], docs: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    gt_total = gt_leak_hit = 0
    gt_chars = covered_chars = 0
    p_total = p_hit_any = 0
    tp_g: dict[str, int] = defaultdict(int)   # GT entities found with right group
    n_g: dict[str, int] = defaultdict(int)    # GT entities per group
    pp_g: dict[str, int] = defaultdict(int)   # predictions per group
    ptp_g: dict[str, int] = defaultdict(int)  # predictions matching a same-group GT
    for name, d in docs.items():
        gts = [{**g, "g": group_of(g["entity_type"])} for g in d["entities"]]
        opts = d["optional"]
        ps = []
        for p in preds.get(name, []):
            g = group_of(p["entity_type"])
            if g is None:
                continue
            if not any(_ov(p, x) for x in gts) and any(_ov(p, o) for o in opts):
                continue  # only touches an optional span: neutral
            ps.append({**p, "g": g})
        mask = bytearray(len(d["text"]))
        for p in ps:
            mask[p["start"]:p["end"]] = b"\x01" * (p["end"] - p["start"])
        for g in gts:
            gt_total += 1
            n_g[g["g"]] += 1
            gt_chars += g["end"] - g["start"]
            covered_chars += sum(mask[g["start"]:g["end"]])
            if any(_ov(p, g) for p in ps):
                gt_leak_hit += 1
            if any(_ov(p, g) and p["g"] == g["g"] for p in ps):
                tp_g[g["g"]] += 1
        for p in ps:
            p_total += 1
            pp_g[p["g"]] += 1
            if any(_ov(p, x) for x in gts) or any(_ov(p, o) for o in opts):
                p_hit_any += 1
            if any(_ov(p, x) and p["g"] == x["g"] for x in gts):
                ptp_g[p["g"]] += 1

    rows = []
    for g in sorted(set(n_g) | set(pp_g)):
        r = tp_g[g] / n_g[g] if n_g[g] else 0.0
        pr = ptp_g[g] / pp_g[g] if pp_g[g] else 0.0
        f = 2 * pr * r / (pr + r) if pr + r else 0.0
        rows.append({"group": g, "n_gt": n_g[g], "n_pred": pp_g[g],
                     "precision": pr, "recall": r, "f1": f})
    per_group = pd.DataFrame(rows)
    rec = sum(tp_g.values()) / gt_total if gt_total else 0.0
    prec = sum(ptp_g.values()) / p_total if p_total else 0.0
    gt_groups = per_group[per_group.n_gt > 0]
    summary = {
        "leak_recall": gt_leak_hit / gt_total if gt_total else 0.0,
        "char_recall": covered_chars / gt_chars if gt_chars else 0.0,
        "precision": p_hit_any / p_total if p_total else 0.0,
        "typed_precision": prec,
        "typed_recall": rec,
        "typed_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        # F2 weights recall 2x: a miss is a leak, a false alarm is a nuisance
        "typed_f2": 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0,
        "macro_f1": float(gt_groups.f1.mean()) if len(gt_groups) else 0.0,
        "n_pred": p_total,
        "missed": gt_total - gt_leak_hit,
    }
    return summary, per_group


def main() -> None:
    corpora = {k: f() for k, f in CORPORA.items()}
    rows, group_rows = [], []
    for run in load_runs():
        default = "none" if run.get("default_thr") is None else str(run["default_thr"])
        for cname, c in run["corpora"].items():
            for thr, preds in c["by_thr"].items():
                s, pg = score_corpus(preds, corpora[cname])
                rows.append({"config": run["key"], "corpus": cname, "thr": thr,
                             "is_default": thr == default, "family": run["family"],
                             "licence": run["licence"], "params_m": run["params_m"],
                             "disk_mb": round(run["disk_mb"]), "peak_rss_mb": round(run["peak_rss_mb"]),
                             "load_s": round(run["load_s"], 1),
                             "ms_per_1k_chars": round(1000 * c["seconds"] / c["chars"] * 1000, 1),
                             **{k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}})
                for _, r in pg.iterrows():
                    group_rows.append({"config": run["key"], "corpus": cname, "thr": thr, **r.to_dict()})
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    df.to_csv(out / "summary_all_thresholds.csv", index=False)
    pd.DataFrame(group_rows).round(3).to_csv(out / "per_group.csv", index=False)
    cols = ["config", "thr", "leak_recall", "char_recall", "precision", "typed_f1", "macro_f1",
            "missed", "n_pred", "ms_per_1k_chars", "disk_mb", "peak_rss_mb", "licence"]
    for cname in corpora:
        sub = df[df.corpus == cname]
        print(f"\n== {cname}: default threshold ==")
        print(sub[sub.is_default].sort_values("typed_f1", ascending=False)[cols].to_string(index=False))
        best = sub.loc[sub.groupby("config").typed_f1.idxmax()]
        print(f"\n== {cname}: best threshold by typed F1 (tuned on this corpus) ==")
        print(best.sort_values("typed_f1", ascending=False)[cols].to_string(index=False))
    df[df.is_default].to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
