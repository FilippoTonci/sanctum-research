"""Load the two evaluation corpora into a common shape.

Each document is ``{"text": str, "entities": [...], "optional": [...]}`` where
entities/optional are dicts with ``entity_type``, ``start``, ``end``, ``text``.

* ``hard``     — hand-annotated documents in ``data/corpus/`` (inline markup)
* ``sanctum``  — copy of Sanctum's ``tests/fixtures`` (Faker-generated, JSON offsets)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
_MARK = re.compile(r"\[\[(\??)([A-Z_]+)\|(.+?)\]\]")


def parse_markup(raw: str) -> tuple[str, list[dict], list[dict]]:
    """Strip ``[[TYPE|text]]`` / ``[[?TYPE|text]]`` markup, returning offsets."""
    out: list[str] = []
    entities: list[dict] = []
    optional: list[dict] = []
    pos = 0  # position in the clean text
    last = 0  # position in the raw text
    for m in _MARK.finditer(raw):
        chunk = raw[last : m.start()]
        out.append(chunk)
        pos += len(chunk)
        is_opt, etype, span = m.groups()
        ent = {"entity_type": etype, "start": pos, "end": pos + len(span), "text": span}
        (optional if is_opt else entities).append(ent)
        out.append(span)
        pos += len(span)
        last = m.end()
    out.append(raw[last:])
    text = "".join(out)
    assert "[[" not in text and "]]" not in text, "unbalanced markup"
    for e in entities + optional:
        assert text[e["start"] : e["end"]] == e["text"]
    return text, entities, optional


def load_hard() -> dict[str, dict]:
    docs = {}
    for p in sorted((DATA / "corpus").glob("*.txt")):
        text, ents, opt = parse_markup(p.read_text(encoding="utf-8"))
        docs[p.stem] = {"text": text, "entities": ents, "optional": opt}
    return docs


def load_sanctum() -> dict[str, dict]:
    docs = {}
    for jp in sorted((DATA / "sanctum_fixtures").glob("*.json")):
        tp = jp.with_suffix(".txt")
        ann = json.loads(jp.read_text(encoding="utf-8"))
        docs[jp.stem] = {
            "text": tp.read_text(encoding="utf-8"),
            "entities": ann["entities"],
            "optional": [],
        }
    return docs


CORPORA = {"hard": load_hard, "sanctum": load_sanctum}


if __name__ == "__main__":
    from collections import Counter

    for name, loader in CORPORA.items():
        docs = loader()
        c = Counter(e["entity_type"] for d in docs.values() for e in d["entities"])
        n_opt = sum(len(d["optional"]) for d in docs.values())
        print(f"{name}: {len(docs)} docs, {sum(c.values())} entities, {n_opt} optional")
        for k, v in c.most_common():
            print(f"   {k:18s} {v}")
