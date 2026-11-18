"""Score the real Sanctum engine on real, human-annotated text.

The other benchmarks here reimplement the production setup; this one runs the
engine itself (``sanctum.cli.commands._create_engine``) from a sanctum
checkout, so it measures exactly what the app would ship. It was written to
choose the Precise sensitivity preset and to vet detection filters before
they land (sanctum-desktop#66, sanctum#61).

Corpora:
  hard   data/corpus/ (22 hand-annotated documents; synthetic text)
  conll  CoNLL-2003 test: real news, people / organisations / places (MISC neutral)
  cuad   CUAD v1: real SEC contracts; party names only, first 5000 chars of each

Steps:
  fetch           download CoNLL (tner/conll2003) and CUAD into data/real/ (git-ignored)
  run <corpus>    one engine pass at a low threshold; every finding keeps its score
                  -> results/real/<corpus>-preds.json (git-ignored)
  score           per-threshold table: missed, not fully covered, false alarms

``run`` needs the sanctum checkout's interpreter (``make real`` does this):
    ../sanctum/.venv/bin/python bench/real_text.py run conll
``fetch`` and ``score`` run in this repo's venv.

Scoring each threshold by filtering one low-threshold run is the same cut the
analyzer applies; on the hard corpus it slightly overstates misses versus
separate runs per threshold (0.5: 136 vs 118), so compare settings, not
absolute counts. CUAD only labels parties: its false-alarm column is not
meaningful, and generic role labels ("the Company") are excluded because
Sanctum drops them on purpose.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "real"
OUT = ROOT / "results" / "real"
LOW = 0.3
THRESHOLDS = [0.35, 0.4, 0.45, 0.5]
NAME_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "NRP"}
CUAD_LIMIT = 200
# tner/conll2003 label ids -> (kind, starts a span)
CONLL_TAGS = {1: ("ORG", True), 2: ("MISC", True), 3: ("PER", True), 4: ("PER", False),
              5: ("LOC", True), 6: ("ORG", False), 7: ("MISC", False), 8: ("LOC", False)}
SOURCES = [("tner/conll2003", "dataset/test.json"), ("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json")]


# ----- corpora -------------------------------------------------------------


def load_conll() -> list[dict]:
    rows = [json.loads(line) for line in (DATA / "dataset/test.json").read_text().splitlines()]
    out = []
    for i in range(0, len(rows), 15):  # this copy has no document boundaries
        text, ents, opt = "", [], []
        for r in rows[i : i + 15]:
            toks, tags = r["tokens"], r["tags"]
            starts = []
            for tok in toks:
                starts.append(len(text))
                text += tok + " "
            span = None
            for j in range(len(toks) + 1):
                kind, begin = CONLL_TAGS.get(tags[j], (None, False)) if j < len(toks) else (None, False)
                if span and (kind is None or begin or kind != span[1]):
                    end = starts[j - 1] + len(toks[j - 1])
                    (opt if span[1] == "MISC" else ents).append({"start": span[0], "end": end, "type": span[1]})
                    span = None
                if kind and span is None:
                    span = (starts[j], kind)
            text = text.rstrip() + "\n"
        out.append({"id": f"conll{i}", "text": text, "entities": ents, "optional": opt})
    return out


def load_cuad() -> list[dict]:
    from sanctum.core.role_terms import is_role_term

    data = json.loads((DATA / "CUAD_v1/CUAD_v1.json").read_text())["data"]
    out = []
    for doc in data[:CUAD_LIMIT]:
        para = doc["paragraphs"][0]
        ctx = para["context"][:5000]
        ents = []
        for qa in para["qas"]:
            if "Parties" not in qa["id"]:
                continue
            for a in qa["answers"]:
                s, txt = a["answer_start"], a["text"]
                if s + len(txt) > len(ctx) or not any(c.isupper() for c in txt):
                    continue
                if is_role_term("ORGANIZATION", txt):
                    continue
                if len(txt.split()) == 1 and txt.istitle() and len(txt) < 12:
                    continue  # a single capitalised word: usually a defined short label
                ents.append({"start": s, "end": s + len(txt), "type": "PARTY"})
        if ents:
            out.append({"id": doc["title"][:60], "text": ctx, "entities": ents, "optional": []})
    return out


def load_hard() -> list[dict]:
    from bench.corpus import load_hard as lh

    def conv(es: list[dict]) -> list[dict]:
        return [{"start": e["start"], "end": e["end"], "type": e["entity_type"]} for e in es]

    return [{"id": k, "text": v["text"], "entities": conv(v["entities"]), "optional": conv(v["optional"])}
            for k, v in lh().items()]


LOADERS = {"hard": load_hard, "conll": load_conll, "cuad": load_cuad}


# ----- steps ---------------------------------------------------------------


def fetch() -> None:
    from huggingface_hub import hf_hub_download

    for repo, name in SOURCES:
        print(hf_hub_download(repo, name, repo_type="dataset", local_dir=DATA))


def run(corpus: str) -> None:
    from sanctum.cli.commands import _create_engine

    docs = LOADERS[corpus]()
    engine = _create_engine()
    t0 = time.time()
    for i, d in enumerate(docs, 1):
        d["preds"] = [{"start": p.start, "end": p.end, "type": p.entity_type, "score": p.score}
                      for p in engine.analyze(d["text"], score_threshold=LOW)]
        if i % 50 == 0 or i == len(docs):
            print(f"[{corpus}] {i}/{len(docs)} docs, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{corpus}-preds.json").write_text(json.dumps(docs))


def _ov(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _covered(g: dict, preds: list[dict], text: str) -> bool:
    chars = [i for i in range(g["start"], g["end"]) if text[i].isalnum()]
    hit = sum(1 for i in chars if any(p["start"] <= i < p["end"] for p in preds))
    return not chars or hit / len(chars) >= 0.9


def score() -> None:
    for corpus in LOADERS:
        path = OUT / f"{corpus}-preds.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text())
        gold = sum(len(d["entities"]) for d in docs)
        print(f"\n{corpus} ({len(docs)} docs, {gold} annotated entities)")
        print(f"{'threshold':>9} {'missed':>7} {'recall':>7} {'not fully':>10} {'false al.':>10} {'precision':>10}")
        for t in THRESHOLDS:
            missed = partial = flags = ok = 0
            for d in docs:
                preds = [p for p in d["preds"] if p["score"] >= t - 1e-9]
                if corpus == "conll":
                    preds = [p for p in preds if p["type"] in NAME_TYPES]
                for g in d["entities"]:
                    near = [p for p in preds if _ov(p, g)]
                    missed += not near
                    partial += not _covered(g, near, d["text"])
                flags += len(preds)
                ok += sum(1 for p in preds if any(_ov(p, g) for g in d["entities"] + d["optional"]))
            print(f"{t:>9} {missed:>7} {1 - missed / gold:>7.3f} {partial:>10} {flags - ok:>10} "
                  f"{ok / flags if flags else 0:>10.3f}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    step, *args = sys.argv[1:] or ["score"]
    {"fetch": fetch, "run": lambda: run(args[0]), "score": score}[step]()
