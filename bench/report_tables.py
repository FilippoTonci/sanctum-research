"""Build the markdown tables + chart used in REPORT.md.

Threshold choice is cross-validated so the "tuned" numbers are not fitted to the
documents they are scored on: the hard corpus is split into two folds
(odd / even documents); for each fold the threshold with the best typed F1 on
the *other* fold is applied, and the two held-out results are pooled.

Writes results/tables.md and results/tradeoff.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import CORPORA  # noqa: E402
from postprocess import propagate  # noqa: E402
from run import load_runs as load_cached_runs  # noqa: E402
from score import score_corpus  # noqa: E402

KEY_GROUPS = ["PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME", "GOV_ID", "FINANCIAL"]


def load_runs() -> list[dict]:
    """Cached runs, plus a "+propagate" variant of every Presidio/hybrid run."""
    runs = load_cached_runs()
    texts = {c: {n: d["text"] for n, d in CORPORA[c]().items()} for c in CORPORA}
    derived = []
    for run in runs:
        if run["family"] not in {"presidio", "hybrid"}:
            continue
        new = {**run, "key": run["key"] + " +propagate", "family": run["family"] + "+prop",
               "corpora": {}}
        for cname, c in run["corpora"].items():
            new["corpora"][cname] = {**c, "by_thr": {
                thr: {n: propagate(ps, texts[cname][n]) for n, ps in preds.items()}
                for thr, preds in c["by_thr"].items()}}
        derived.append(new)
    return runs + derived


def cv_pick(run: dict, docs: dict) -> tuple[dict, list[str]]:
    """2-fold CV over documents. Returns pooled held-out metrics + chosen thresholds."""
    by_thr = run["corpora"]["hard"]["by_thr"]
    names = sorted(docs)
    folds = [names[0::2], names[1::2]]
    chosen, held_preds = [], {}
    for i, test in enumerate(folds):
        train = folds[1 - i]
        sub = {n: docs[n] for n in train}
        best = max(by_thr, key=lambda t: score_corpus(by_thr[t], sub)[0]["typed_f2"])
        chosen.append(best)
        for n in test:
            held_preds[n] = by_thr[best][n]
    summary, per_group = score_corpus(held_preds, docs)
    return {**summary, "groups": per_group.set_index("group")}, chosen


def fmt(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    hard = CORPORA["hard"]()
    sanctum = CORPORA["sanctum"]()
    rows = []
    for run in load_runs():
        default = "none" if run.get("default_thr") is None else str(run["default_thr"])
        h = run["corpora"]["hard"]
        d_sum, d_groups = score_corpus(h["by_thr"][default], hard)
        s_sum, _ = score_corpus(run["corpora"]["sanctum"]["by_thr"][default], sanctum)
        if len(h["by_thr"]) > 1:
            cv, chosen = cv_pick(run, hard)
        else:
            cv, chosen = {**d_sum, "groups": d_groups.set_index("group")}, ["n/a"]
        g = cv["groups"]
        rows.append({
            "config": run["key"], "family": run["family"], "licence": run["licence"],
            "disk_mb": run["disk_mb"], "rss_mb": run["peak_rss_mb"],
            "ms_1k": 1000 * h["seconds"] / h["chars"] * 1000,
            "def_thr": default, "def_leak": d_sum["leak_recall"], "def_prec": d_sum["precision"],
            "def_f1": d_sum["typed_f1"], "def_missed": d_sum["missed"],
            "cv_thr": "/".join(chosen), "cv_leak": cv["leak_recall"], "cv_char": cv["char_recall"],
            "cv_prec": cv["precision"], "cv_f1": cv["typed_f1"], "cv_f2": cv["typed_f2"],
            "cv_macro": cv["macro_f1"],
            "cv_missed": cv["missed"], "cv_npred": cv["n_pred"],
            "sanctum_leak": s_sum["leak_recall"], "sanctum_f1": s_sum["typed_f1"],
            **{f"r_{k}": float(g.loc[k, "recall"]) if k in g.index else 0.0 for k in KEY_GROUPS},
            **{f"p_{k}": float(g.loc[k, "precision"]) if k in g.index else 0.0 for k in KEY_GROUPS},
        })
    df = pd.DataFrame(rows).sort_values("cv_f2", ascending=False)
    df.to_csv(ROOT / "results" / "leaderboard.csv", index=False)

    lines = ["## Leaderboard (hard corpus, 487 entities; thresholds 2-fold cross-validated on F2)\n",
             "| Config | Leak recall | Char recall | Precision | Typed F2 | Typed F1 | Missed | Thr (fold A/B) "
             "| Sanctum-fixtures F1 | ms / 1k chars | Weights MB | Peak RAM MB | Licence |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.config} | {fmt(r.cv_leak)} | {fmt(r.cv_char)} | {fmt(r.cv_prec)} | "
                     f"**{fmt(r.cv_f2)}** | {fmt(r.cv_f1)} | {r.cv_missed} | {r.cv_thr} | {fmt(r.sanctum_f1)} | "
                     f"{r.ms_1k:.0f} | {r.disk_mb:.0f} | {r.rss_mb:.0f} | {r.licence} |")
    lines += ["\n## At each model's default threshold (no tuning)\n",
              "| Config | Thr | Leak recall | Precision | Typed F1 | Missed |", "|---|---|---|---|---|---|"]
    for r in df.sort_values("def_f1", ascending=False).itertuples():
        lines.append(f"| {r.config} | {r.def_thr} | {fmt(r.def_leak)} | {fmt(r.def_prec)} | "
                     f"{fmt(r.def_f1)} | {r.def_missed} |")
    lines += ["\n## Recall by entity group (cross-validated threshold)\n",
              "| Config | " + " | ".join(KEY_GROUPS) + " |", "|---|" + "---|" * len(KEY_GROUPS)]
    for r in df.itertuples():
        lines.append(f"| {r.config} | " + " | ".join(fmt(getattr(r, f"r_{k}")) for k in KEY_GROUPS) + " |")
    lines += ["\n## Precision by entity group (cross-validated threshold)\n",
              "| Config | " + " | ".join(KEY_GROUPS) + " |", "|---|" + "---|" * len(KEY_GROUPS)]
    for r in df.itertuples():
        lines.append(f"| {r.config} | " + " | ".join(fmt(getattr(r, f"p_{k}")) for k in KEY_GROUPS) + " |")
    (ROOT / "results" / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:len(df) + 3]))
    chart(df)


# One point per model: the best shippable shape (hybrid + propagate when it exists).
POINTS = {  # config -> (short label, label position: offset in points, or ("data", x, y) with leader)
    "presidio+sm": ("en_core_web_sm (shipped)", (8, -4)),
    "presidio+lg +propagate": ("en_core_web_lg", (8, -4)),
    "presidio+trf +propagate": ("en_core_web_trf", (8, -4)),
    "presidio+gliner_medium (shipped Pro) +propagate": ("gliner_medium (shipped Pro)", ("data", 2600, 0.845)),
    "presidio+gliner_multi_pii +propagate": ("gliner_multi_pii", ("data", 2600, 0.86)),
    "presidio+kg_pii_base +propagate": ("kg gliner-pii-base (recommended)", ("data", 120, 0.893)),
    "presidio+kg_pii_base_onnx_q8 +propagate": ("kg gliner-pii-base, ONNX uint8", (-8, -4)),
    "presidio+kg_pii_small +propagate": ("kg gliner-pii-small", (8, -4)),
    "presidio+kg_pii_edge +propagate": ("kg gliner-pii-edge", (8, -4)),
    "presidio+fastino_gliner2_pii +propagate": ("fastino GLiNER2-PII", ("data", 2600, 0.875)),
    "presidio+nvidia_gliner_pii +propagate": ("nvidia gliner-PII", ("data", 2600, 0.892)),
    "presidio+openai_privacy_filter +propagate": ("openai privacy-filter", (-8, 8)),
    "presidio+gretel_bi_small +propagate": ("gretel bi-small", (-8, 8)),
    "bardsai_eu_pii": ("bardsai eu-pii", (8, -4)),
    "soelmgd_bert_pii": ("SoelMgd bert-pii*", (8, -4)),
    "dslim_bert_ner": ("dslim bert-NER", (8, -4)),
}


def chart(df: pd.DataFrame) -> None:
    """Static PNG for the markdown report. The leaderboard table is the accessible
    view; series use the reference palette in fixed order plus distinct markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = {  # legend group -> (colour, marker)
        "spaCy (Presidio today)": ("#2a78d6", "s"),
        "GLiNER in Presidio + propagate": ("#eb6834", "o"),
        "token classifier (standalone)": ("#1baf7a", "^"),
    }

    def kind(cfg: str) -> str:
        if cfg.startswith("presidio+") and any(m in cfg for m in ("+sm", "+lg", "+trf")):
            return "spaCy (Presidio today)"
        return "GLiNER in Presidio + propagate" if cfg.startswith("presidio+") else "token classifier (standalone)"

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    sub = df[df.config.isin(POINTS)]
    for k, (color, marker) in kinds.items():
        pts = sub[sub.config.map(kind) == k]
        ax.scatter(pts.disk_mb.clip(lower=10), pts.cv_f2, s=70, c=color, marker=marker, label=k,
                   edgecolor="#fcfcfb", linewidth=2, zorder=3)
    for r in sub.itertuples():
        label, pos = POINTS[r.config]
        weight = "bold" if "recommended" in label else "normal"
        xy = (max(r.disk_mb, 10), r.cv_f2)
        if pos[0] == "data":
            ax.annotate(label, xy, xytext=pos[1:], textcoords="data", fontsize=8, color="#0b0b0b",
                        fontweight=weight, va="center", ha="left" if pos[1] > xy[0] else "right",
                        arrowprops={"arrowstyle": "-", "color": "#a8a7a2", "linewidth": 0.8,
                                    "shrinkA": 2, "shrinkB": 5})
        else:
            ax.annotate(label, xy, fontsize=8, xytext=pos, textcoords="offset points",
                        color="#0b0b0b", fontweight=weight, ha="right" if pos[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(8, 9000)
    ax.set_ylim(0.49, 0.905)
    ax.set_xlabel("Model weights on disk (MB, log scale)")
    ax.set_ylabel("Typed F2 on hard corpus (cross-validated threshold)")
    ax.set_title("Accuracy vs. size: one point per model, best setup", loc="left", color="#0b0b0b",
                 fontsize=11)
    ax.grid(True, color="#e4e3df", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.0, -0.12, "* trained on ai4privacy data (commercial licence required) - reference only",
            transform=ax.transAxes, fontsize=7.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "tradeoff.png", dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
