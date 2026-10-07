"""Model-agnostic post-processing experiments.

``propagate`` — document-level name propagation. NER models judge each mention
in isolation, so once "Dwayne Kowalczyk" is found, a later bare "Dwayne" or
"Mr Price" is often missed. After detection we collect the strings of every
PERSON / ORGANIZATION span (plus the individual tokens of PERSON names) and
mark every other occurrence in the same document with the same type.

This is cheap (regex over the text), deterministic, and would live in Sanctum's
analyzer adapter after `_normalize_overlaps` — it does not depend on which model
produced the seed spans.
"""
from __future__ import annotations

import re

# tokens that must never be propagated on their own
_STOP = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame", "mme", "mlle", "herr", "frau",
    "d", "dña", "jr", "sr", "esq", "the", "and", "of", "de", "del", "la", "le", "van", "von",
    "der", "den", "da", "di", "du", "bin", "al",
}
# generic organisation words that are not identifying on their own
_ORG_GENERIC = {
    "bank", "group", "capital", "partners", "holdings", "limited", "ltd", "llp", "llc", "inc",
    "corp", "company", "co", "services", "consulting", "national", "first", "international",
    "health", "the", "and", "of", "ag", "sa", "srl", "gmbh", "bv", "plc", "pllc", "kg",
}


def _candidates(span: dict) -> list[str]:
    text = span["text"].strip()
    out = [text]
    if span["entity_type"] == "PERSON":
        for tok in re.findall(r"[^\W\d_][\w'’-]*", text):
            if len(tok) >= 3 and tok.lower().strip(".") not in _STOP:
                out.append(tok)
    elif span["entity_type"] == "ORGANIZATION":
        toks = re.findall(r"[^\W\d_][\w&'’-]*", text)
        if len(toks) > 1 and len(toks[0]) >= 4 and toks[0].lower() not in _ORG_GENERIC:
            out.append(toks[0])  # "Verdana Pharmaceuticals" -> "Verdana"
    return out


def propagate(spans: list[dict], text: str) -> list[dict]:
    covered = bytearray(len(text))
    for s in spans:
        covered[s["start"]:s["end"]] = b"\x01" * (s["end"] - s["start"])
    seeds: dict[str, tuple[str, float]] = {}
    for s in spans:
        if s["entity_type"] in {"PERSON", "ORGANIZATION"}:
            for c in _candidates(s):
                key = c.lower()
                if key not in seeds or seeds[key][1] < s["score"]:
                    seeds[key] = (s["entity_type"], s["score"])
    added: list[dict] = []
    # longest strings first so "Mark Price" wins over "Price"
    for key in sorted(seeds, key=len, reverse=True):
        etype, score = seeds[key]
        for m in re.finditer(r"(?<![\w])" + re.escape(key) + r"(?![\w])", text, flags=re.IGNORECASE):
            s, e = m.span()
            # case guard: single tokens must look like names (Capitalised or ALL CAPS)
            word = text[s:e]
            if " " not in key and not (word[:1].isupper() or word.isupper()):
                if not text[max(0, s - 40):e + 40].islower():  # allow all-lowercase chat text
                    continue
            if any(covered[s:e]):
                continue
            covered[s:e] = b"\x01" * (e - s)
            added.append({"entity_type": etype, "start": s, "end": e, "score": score,
                          "text": word})
    return spans + added
