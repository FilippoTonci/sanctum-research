"""Model wrappers. Every predictor maps ``text -> list[Span]`` in Sanctum's taxonomy.

Three families:

* ``presidio``  — the production setup: Presidio + a spaCy NLP engine, built the
  same way ``sanctum/cli/commands.py::_create_engine`` builds it (ORG kept,
  AnyDomainEmailRecognizer added, overlaps normalised). Optionally a model
  from the other families replaces ``SpacyRecognizer`` as the NER source —
  the "hybrid" setup Sanctum's GLiNER tier uses.
* ``gliner`` / ``gliner2`` — span models prompted with natural-language labels.
* ``hf_token`` — plain token-classification checkpoints (BIO / BIOES heads).

Nothing here imports Sanctum; the bits we need are reproduced so the research
repo stays standalone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

Span = dict  # {"entity_type", "start", "end", "score", "text"}

# ----------------------------------------------------------------------------
# Chunking — keep every model inside its context window. We split on line
# breaks first, then on sentence ends, packing pieces into <= MAX_CHARS.
# ----------------------------------------------------------------------------
MAX_CHARS = 1000
FLOOR = 0.15  # models run once at this cut-off; thresholds are swept afterwards
THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def chunks(text: str, max_chars: int = MAX_CHARS) -> list[tuple[int, str]]:
    pieces: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\n]+", text):
        s, e = m.span()
        while e - s > max_chars:  # very long line: break at a sentence end
            cut = text.rfind(". ", s, s + max_chars)
            cut = cut + 2 if cut > s else s + max_chars
            pieces.append((s, cut))
            s = cut
        pieces.append((s, e))
    out: list[tuple[int, str]] = []
    cur_s = cur_e = None
    for s, e in pieces:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= max_chars:
            cur_e = e
        else:
            out.append((cur_s, text[cur_s:cur_e]))
            cur_s, cur_e = s, e
    if cur_s is not None:
        out.append((cur_s, text[cur_s:cur_e]))
    return out


def _span(etype: str, start: int, end: int, score: float, text: str) -> Span:
    return {"entity_type": etype, "start": int(start), "end": int(end),
            "score": float(score), "text": text[start:end]}


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s] in " \t\n,;:()[]\"'":
        s += 1
    while e > s and text[e - 1] in " \t\n,;:()[]\"'.":
        e -= 1
    return s, e


# ----------------------------------------------------------------------------
# GLiNER (urchade / knowledgator / gretel / nvidia / E3-JSI)
# ----------------------------------------------------------------------------
# prompt -> Sanctum type. Two prompt sets: the generic one (what Sanctum ships
# today in nlp_config._GLINER_ENTITY_MAPPING, plus address / ID prompts) and
# the label names Knowledgator's PII models were trained on.
GLINER_GENERIC = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "address": "LOCATION",
    "date": "DATE_TIME",
    "email address": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "url": "URL",
    "ip address": "IP_ADDRESS",
    "iban": "IBAN_CODE",
    "social security number": "US_SSN",
    "driver license number": "US_DRIVER_LICENSE",
    "bank account number": "US_BANK_NUMBER",
    "credit card number": "CREDIT_CARD",
    "passport number": "ID_NUMBER",
    "national id number": "ID_NUMBER",
    "tax identification number": "ID_NUMBER",
}
GLINER_KNOWLEDGATOR = {
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "location address": "LOCATION",
    "location city": "LOCATION",
    "location country": "LOCATION",
    "date": "DATE_TIME",
    "dob": "DATE_TIME",
    "email address": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "url": "URL",
    "ip address": "IP_ADDRESS",
    "iban": "IBAN_CODE",
    "ssn": "US_SSN",
    "driver license": "US_DRIVER_LICENSE",
    "bank account": "US_BANK_NUMBER",
    "credit card": "CREDIT_CARD",
    "passport number": "ID_NUMBER",
    "tax id": "ID_NUMBER",
}
# Sanctum's current prompt set (13 labels, no address / ID prompts) — used to
# reproduce the shipped "Professional" tier exactly.
GLINER_SANCTUM = {k: v for k, v in GLINER_GENERIC.items()
                  if k not in {"address", "passport number", "national id number",
                               "tax identification number"}}


def gliner_predictor(repo_id: str, prompts: dict[str, str], threshold: float = FLOOR,
                     onnx_file: str | None = None) -> Callable[[str], list[Span]]:
    from gliner import GLiNER

    path = str(MODELS_DIR / repo_id.replace("/", "__"))
    if onnx_file:
        model = GLiNER.from_pretrained(path, load_onnx_model=True, load_tokenizer=True,
                                       onnx_model_file=onnx_file)
    else:
        model = GLiNER.from_pretrained(path, map_location="cpu")
    model.eval()
    labels = list(prompts)

    def predict(text: str) -> list[Span]:
        out: list[Span] = []
        for off, chunk in chunks(text):
            for r in model.predict_entities(chunk, labels, threshold=threshold, flat_ner=True):
                etype = prompts.get(r["label"].lower())
                if etype:
                    s, e = _trim(text, off + r["start"], off + r["end"])
                    if e > s:
                        out.append(_span(etype, s, e, r["score"], text))
        return out

    return predict


def gliner_onnx_predictor(repo_id: str, prompts: dict[str, str], threshold: float = FLOOR,
                          onnx_file: str = "onnx/model_quint8.onnx") -> Callable[[str], list[Span]]:
    """Same as gliner_predictor on an ONNX file, but through bench/gliner_onnx.py:
    onnxruntime + tokenizers + numpy, no PyTorch (what the Sanctum sidecar ships)."""
    from gliner_onnx import GlinerOnnx

    model = GlinerOnnx(MODELS_DIR / repo_id.replace("/", "__"), onnx_file, intra_op_threads=4)
    labels = list(prompts)

    def predict(text: str) -> list[Span]:
        out: list[Span] = []
        for off, chunk in chunks(text):
            for r in model.predict_entities(chunk, labels, threshold=threshold, flat_ner=True):
                etype = prompts.get(r.label.lower())
                if etype:
                    s, e = _trim(text, off + r.start, off + r.end)
                    if e > s:
                        out.append(_span(etype, s, e, r.score, text))
        return out

    return predict


# ----------------------------------------------------------------------------
# GLiNER2 (fastino)
# ----------------------------------------------------------------------------
GLINER2_PII = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "address": "LOCATION",
    "city": "LOCATION",
    "country": "LOCATION",
    "date": "DATE_TIME",
    "date_of_birth": "DATE_TIME",
    "email": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "url": "URL",
    "ip_address": "IP_ADDRESS",
    "iban": "IBAN_CODE",
    "card_number": "CREDIT_CARD",
    "bank_account": "US_BANK_NUMBER",
    "drivers_license_number": "US_DRIVER_LICENSE",
    "national_id_number": "US_SSN",
    "passport_number": "ID_NUMBER",
    "tax_id": "ID_NUMBER",
}


def gliner2_predictor(repo_id: str, prompts: dict[str, str] = GLINER2_PII,
                      threshold: float = FLOOR) -> Callable[[str], list[Span]]:
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(str(MODELS_DIR / repo_id.replace("/", "__")))
    labels = list(prompts)

    def predict(text: str) -> list[Span]:
        out: list[Span] = []
        for off, chunk in chunks(text):
            res = model.extract_entities(chunk, labels, threshold=threshold,
                                         include_confidence=True, include_spans=True)
            for label, items in (res.get("entities") or {}).items():
                etype = prompts.get(label)
                if not etype:
                    continue
                for it in items:
                    if isinstance(it, str):  # defensive: older API returns bare strings
                        continue
                    s, e = _trim(text, off + it["start"], off + it["end"])
                    if e > s:
                        out.append(_span(etype, s, e, it.get("confidence", 1.0), text))
        return out

    return predict


# ----------------------------------------------------------------------------
# HF token classification (BIO / BIOES). We decode ourselves: transformers'
# "simple" aggregation does not understand E-/S- tags (OpenAI Privacy Filter).
# ----------------------------------------------------------------------------
HF_LABEL_MAPS: dict[str, dict[str, str]] = {
    "dslim/bert-base-NER": {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"},
    "openai/privacy-filter": {
        "private_person": "PERSON", "private_address": "LOCATION",
        "private_email": "EMAIL_ADDRESS", "private_phone": "PHONE_NUMBER",
        "private_url": "URL", "private_date": "DATE_TIME",
        "account_number": "US_BANK_NUMBER",
    },
    # filled in from each checkpoint's id2label (see bench/inspect_labels.py)
    # ai4privacy pii-masking-200k taxonomy
    "SoelMgd/bert-pii-detection": {
        "FIRSTNAME": "PERSON", "MIDDLENAME": "PERSON", "LASTNAME": "PERSON",
        "COMPANYNAME": "ORGANIZATION",
        "CITY": "LOCATION", "STATE": "LOCATION", "COUNTY": "LOCATION", "STREET": "LOCATION",
        "BUILDINGNUMBER": "LOCATION", "SECONDARYADDRESS": "LOCATION", "ZIPCODE": "LOCATION",
        "DATE": "DATE_TIME", "DOB": "DATE_TIME", "TIME": "DATE_TIME",
        "EMAIL": "EMAIL_ADDRESS", "PHONENUMBER": "PHONE_NUMBER", "URL": "URL",
        "IP": "IP_ADDRESS", "IPV4": "IP_ADDRESS", "IPV6": "IP_ADDRESS",
        "IBAN": "IBAN_CODE", "CREDITCARDNUMBER": "CREDIT_CARD", "ACCOUNTNUMBER": "US_BANK_NUMBER",
        "SSN": "US_SSN", "VEHICLEVIN": "ID_NUMBER", "VEHICLEVRM": "ID_NUMBER",
    },
    "bardsai/eu-pii-anonimization-multilang": {
        "PERSON_NAME": "PERSON", "PERSON_ALIAS": "PERSON",
        "ORGANIZATION_NAME": "ORGANIZATION",
        "LOCATION": "LOCATION", "POSTAL_ADDRESS": "LOCATION", "GEO_LOCATION": "LOCATION",
        "DATE_OF_BIRTH": "DATE_TIME",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS", "PHONE_NUMBER": "PHONE_NUMBER",
        "IDENTIFYING_LINK": "URL", "IP_ADDRESS": "IP_ADDRESS",
        "PAYMENT_CARD": "CREDIT_CARD", "BANK_ACCOUNT_IDENTIFIER": "US_BANK_NUMBER",
        "ACCOUNT_IDENTIFIER": "ID_NUMBER", "PERSON_IDENTIFIER": "ID_NUMBER",
        "DOCUMENT_IDENTIFIER": "ID_NUMBER", "VEHICLE_IDENTIFIER": "ID_NUMBER",
    },
}
_PREFIX = re.compile(r"^([BIESLU])[-_]")


def hf_token_predictor(repo_id: str, label_map: dict[str, str] | None = None,
                       threshold: float = FLOOR) -> Callable[[str], list[Span]]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    path = str(MODELS_DIR / repo_id.replace("/", "__"))
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path, dtype=torch.float32)
    model.eval()
    id2label = model.config.id2label
    lmap = label_map if label_map is not None else HF_LABEL_MAPS[repo_id]

    def decode(chunk: str) -> list[tuple[str, int, int, float]]:
        enc = tok(chunk, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=512)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            probs = model(**enc).logits[0].softmax(-1)
        conf, ids = probs.max(-1)
        spans: list[list] = []  # [etype, start, end, [scores]]
        for (s, e), i, c in zip(offsets, ids.tolist(), conf.tolist()):
            if e <= s:
                continue  # special tokens
            raw = id2label[i]
            m = _PREFIX.match(raw)
            tag, base = (m.group(1), raw[2:]) if m else ("I", raw)
            etype = lmap.get(base) or lmap.get(base.upper())
            if raw == "O" or not etype:
                spans.append(None)  # break marker
                continue
            last = spans[-1] if spans else None
            same_word = last is not None and last[0] == etype and s == last[2]
            joinable = last is not None and last[0] == etype and s - last[2] <= 1
            if same_word or (joinable and tag not in "BSU"):
                last[2] = e
                last[3].append(c)
            else:
                spans.append([etype, s, e, [c]])
        return [(t, s, e, sum(sc) / len(sc)) for t, s, e, sc in filter(None, spans)]

    def predict(text: str) -> list[Span]:
        out: list[Span] = []
        for off, chunk in chunks(text):
            for etype, s, e, sc in decode(chunk):
                while s > 0 and chunk[s - 1].isalnum():  # extend to word boundary
                    s -= 1
                while e < len(chunk) and chunk[e].isalnum():
                    e += 1
                gs, ge = _trim(text, off + s, off + e)
                if ge > gs and sc >= threshold:
                    out.append(_span(etype, gs, ge, sc, text))
        return out

    return predict


# ----------------------------------------------------------------------------
# Presidio (production) and Presidio + model hybrid
# ----------------------------------------------------------------------------
# Presidio's default noisy-label list minus ORGANIZATION (copied from
# sanctum/analyzer/nlp_config.py).
LABELS_TO_IGNORE = ["CARDINAL", "EVENT", "LANGUAGE", "LAW", "MONEY", "ORDINAL",
                    "PERCENT", "PRODUCT", "QUANTITY", "WORK_OF_ART"]
_EMAIL = (r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
          r"[A-Za-z]{2,63}(?![\w-])")
NER_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME", "NRP"}


def presidio_predictor(spacy_model: str = "en_core_web_sm",
                       ner: Callable[[str], list[Span]] | None = None,
                       threshold: float = 0.35) -> Callable[[str], list[Span]]:
    from presidio_analyzer import (AnalyzerEngine, EntityRecognizer, Pattern,
                                   PatternRecognizer, RecognizerResult)
    from presidio_analyzer.nlp_engine import NerModelConfiguration, NlpEngineProvider

    ner_cfg = NerModelConfiguration(labels_to_ignore=LABELS_TO_IGNORE)
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
        "ner_model_configuration": ner_cfg.to_dict(),
    }).create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp)
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="EMAIL_ADDRESS", name="AnyDomainEmailRecognizer",
        patterns=[Pattern("email, any domain", _EMAIL, 0.9)]))

    cache: dict[str, list[Span]] = {}
    state = {"thr": 0.0}

    def model_spans(text: str) -> list[Span]:
        if text not in cache:
            cache[text] = ner(text)
        t = state["thr"]
        # Presidio drops anything below its own 0.35 cut, so map [t, 1] -> [0.35, 1]
        return [{**s, "score": threshold + (1 - threshold) * (s["score"] - t) / max(1e-9, 1 - t)}
                for s in cache[text] if s["score"] >= t]

    if ner is not None:
        class ModelRecognizer(EntityRecognizer):
            def load(self) -> None:  # model already loaded
                pass

            def analyze(self, text, entities, nlp_artifacts=None):  # noqa: ANN001
                return [RecognizerResult(s["entity_type"], s["start"], s["end"], s["score"],
                                         recognition_metadata={"recognizer_name": "ModelRecognizer"})
                        for s in model_spans(text) if s["entity_type"] in entities]

        supported = sorted({*GLINER_GENERIC.values(), "ID_NUMBER"})
        engine.registry.add_recognizer(ModelRecognizer(supported_entities=supported,
                                                       name="ModelRecognizer"))
        engine.registry.remove_recognizer("SpacyRecognizer")

    def predict(text: str, ner_thr: float = 0.0) -> list[Span]:
        state["thr"] = ner_thr
        res = engine.analyze(text=text, language="en", score_threshold=threshold)
        spans = [_span(r.entity_type, r.start, r.end, r.score, text) for r in res]
        return normalize_overlaps(spans)

    return predict


def normalize_overlaps(spans: list[Span]) -> list[Span]:
    """Port of PresidioAnalyzer._normalize_overlaps: containment keeps the outer
    span; partial overlaps go to the longer span and the loser is trimmed."""
    items = sorted(spans, key=lambda d: (d["start"], d["end"], -d["score"]))
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["end"] <= b["start"]:
                    break
                if not (a["start"] < b["end"] and b["start"] < a["end"]):
                    continue
                if a["start"] <= b["start"] and b["end"] <= a["end"]:
                    rep = [a]
                elif b["start"] <= a["start"] and a["end"] <= b["end"]:
                    rep = [b]
                else:
                    la, lb = a["end"] - a["start"], b["end"] - b["start"]
                    if la != lb:
                        w, l = (a, b) if la > lb else (b, a)
                    else:
                        w, l = (a, b) if a["score"] >= b["score"] else (b, a)
                    ns, ne = (l["start"], w["start"]) if l["start"] < w["start"] else (w["end"], l["end"])
                    rep = [w] if ns >= ne else sorted(
                        [w, {**l, "start": ns, "end": ne}], key=lambda d: d["start"])
                items[i:j + 1] = rep
                items.sort(key=lambda d: (d["start"], d["end"], -d["score"]))
                changed = True
                break
            if changed:
                break
    return items


# ----------------------------------------------------------------------------
# Registry of benchmark configurations
# ----------------------------------------------------------------------------
@dataclass
class Config:
    key: str
    build: Callable[[], Callable[[str], list[Span]]]
    repo: str            # HF repo id or spaCy package
    params_m: float      # parameters, millions (model card / config)
    licence: str
    family: str
    note: str = ""
    default_thr: float | None = 0.4  # model-card / Sanctum default; None = no model threshold
    weights: tuple[str, ...] = ()    # files counted as "weights on disk"; () = whole dir minus onnx/
    torch_free: bool = False         # run without importing torch (peak RSS then excludes it)


def _gl(repo, prompts=GLINER_GENERIC, onnx_file=None):
    return lambda: gliner_predictor(repo, prompts, onnx_file=onnx_file)


def _hyb(spacy_model, inner):
    return lambda: presidio_predictor(spacy_model, ner=inner())


KG = GLINER_KNOWLEDGATOR
FASTINO = "fastino/gliner2-privacy-filter-PII-multi"

CONFIGS: list[Config] = [
    # --- production today (Presidio + spaCy; no model threshold) -------------
    Config("presidio+sm", lambda: presidio_predictor("en_core_web_sm"), "spacy/en_core_web_sm",
           12, "MIT", "presidio", "What the desktop app ships today", None),
    Config("presidio+lg", lambda: presidio_predictor("en_core_web_lg"), "spacy/en_core_web_lg",
           0, "MIT", "presidio", "spaCy large (CNN + vectors)", None),
    Config("presidio+trf", lambda: presidio_predictor("en_core_web_trf"), "spacy/en_core_web_trf",
           125, "MIT", "presidio", "spaCy RoBERTa-base pipeline", None),
    # --- GLiNER standalone ---------------------------------------------------
    Config("gliner_small_v21", _gl("urchade/gliner_small-v2.1"), "urchade/gliner_small-v2.1",
           166, "Apache-2.0", "gliner"),
    Config("gliner_medium_v21", _gl("urchade/gliner_medium-v2.1"), "urchade/gliner_medium-v2.1",
           209, "Apache-2.0", "gliner", "Sanctum's current Professional default"),
    Config("gliner_multi_pii_v1", _gl("urchade/gliner_multi_pii-v1"), "urchade/gliner_multi_pii-v1",
           279, "Apache-2.0", "gliner"),
    Config("e3jsi_multi_pii_domains", _gl("E3-JSI/gliner-multi-pii-domains-v1"),
           "E3-JSI/gliner-multi-pii-domains-v1", 279, "Apache-2.0", "gliner"),
    Config("gretel_bi_small", _gl("gretelai/gretel-gliner-bi-small-v1.0"),
           "gretelai/gretel-gliner-bi-small-v1.0", 194, "Apache-2.0", "gliner"),
    Config("kg_pii_edge", _gl("knowledgator/gliner-pii-edge-v1.0", KG), "knowledgator/gliner-pii-edge-v1.0",
           45, "Apache-2.0", "gliner", "ONNX quint8 export is 46 MB", 0.3),
    Config("kg_pii_small", _gl("knowledgator/gliner-pii-small-v1.0", KG), "knowledgator/gliner-pii-small-v1.0",
           82, "Apache-2.0", "gliner", "", 0.3),
    Config("kg_pii_base", _gl("knowledgator/gliner-pii-base-v1.0", KG), "knowledgator/gliner-pii-base-v1.0",
           166, "Apache-2.0", "gliner", "", 0.3),
    Config("nvidia_gliner_pii", _gl("nvidia/gliner-PII"), "nvidia/gliner-PII",
           570, "NVIDIA Open Model License", "gliner", "Reference: large"),
    # --- GLiNER2 ---------------------------------------------------------------
    Config("fastino_gliner2_pii", lambda: gliner2_predictor(FASTINO), FASTINO,
           205, "Apache-2.0", "gliner2", "", 0.5),
    # --- token classifiers -----------------------------------------------------
    Config("dslim_bert_ner", lambda: hf_token_predictor("dslim/bert-base-NER"), "dslim/bert-base-NER",
           108, "MIT", "hf_token", "CoNLL PER/ORG/LOC only", 0.4),
    Config("soelmgd_bert_pii", lambda: hf_token_predictor("SoelMgd/bert-pii-detection"),
           "SoelMgd/bert-pii-detection", 66, "MIT", "hf_token", "", 0.4),
    Config("bardsai_eu_pii", lambda: hf_token_predictor("bardsai/eu-pii-anonimization-multilang"),
           "bardsai/eu-pii-anonimization-multilang", 278, "Apache-2.0", "hf_token", "", 0.4),
    Config("openai_privacy_filter", lambda: hf_token_predictor("openai/privacy-filter"), "openai/privacy-filter",
           1500, "Apache-2.0", "hf_token", "Reference: 1.5B total / 50M active MoE; argmax decode", 0.4),
    # --- hybrids: Presidio regex/context recognizers + model replacing spaCy NER
    Config("presidio+gliner_medium (shipped Pro)",
           _hyb("en_core_web_sm", _gl("urchade/gliner_medium-v2.1", GLINER_SANCTUM)),
           "urchade/gliner_medium-v2.1", 209, "Apache-2.0", "hybrid", "Exactly Sanctum's ner_backend=gliner"),
    Config("presidio+gliner_multi_pii", _hyb("en_core_web_sm", _gl("urchade/gliner_multi_pii-v1")),
           "urchade/gliner_multi_pii-v1", 279, "Apache-2.0", "hybrid"),
    Config("presidio+kg_pii_edge", _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-edge-v1.0", KG)),
           "knowledgator/gliner-pii-edge-v1.0", 45, "Apache-2.0", "hybrid", "", 0.3),
    Config("presidio+kg_pii_small", _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-small-v1.0", KG)),
           "knowledgator/gliner-pii-small-v1.0", 82, "Apache-2.0", "hybrid", "", 0.3),
    Config("presidio+kg_pii_base", _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-base-v1.0", KG)),
           "knowledgator/gliner-pii-base-v1.0", 166, "Apache-2.0", "hybrid", "", 0.3),
    Config("presidio+kg_pii_base (sanctum prompts)",
           _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-base-v1.0", GLINER_SANCTUM)),
           "knowledgator/gliner-pii-base-v1.0", 166, "Apache-2.0", "hybrid",
           "Config-only swap: Sanctum's existing 13 prompts", 0.3),
    Config("presidio+fastino_gliner2_pii", _hyb("en_core_web_sm", lambda: gliner2_predictor(FASTINO)),
           FASTINO, 205, "Apache-2.0", "hybrid", "", 0.5),
    # --- quantised ONNX exports (what would actually ship without torch) -------
    Config("kg_pii_edge_onnx_q8", _gl("knowledgator/gliner-pii-edge-v1.0", KG, "onnx/model_quint8.onnx"),
           "knowledgator/gliner-pii-edge-v1.0", 45, "Apache-2.0", "gliner", "uint8 ONNX, 46 MB", 0.3, weights=("onnx/model_quint8.onnx", "tokenizer.json", "*config.json")),
    Config("presidio+kg_pii_edge_onnx_q8",
           _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-edge-v1.0", KG, "onnx/model_quint8.onnx")),
           "knowledgator/gliner-pii-edge-v1.0", 45, "Apache-2.0", "hybrid", "uint8 ONNX, 46 MB", 0.3, weights=("onnx/model_quint8.onnx", "tokenizer.json", "*config.json")),
    Config("presidio+kg_pii_small_onnx_q8",
           _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-small-v1.0", KG, "onnx/model_quint8.onnx")),
           "knowledgator/gliner-pii-small-v1.0", 82, "Apache-2.0", "hybrid", "uint8 ONNX, 83 MB", 0.3, weights=("onnx/model_quint8.onnx", "tokenizer.json", "*config.json")),
    Config("presidio+kg_pii_base_onnx_q8",
           _hyb("en_core_web_sm", _gl("knowledgator/gliner-pii-base-v1.0", KG, "onnx/model_quint8.onnx")),
           "knowledgator/gliner-pii-base-v1.0", 166, "Apache-2.0", "hybrid", "uint8 ONNX, 197 MB", 0.3, weights=("onnx/model_quint8.onnx", "tokenizer.json", "*config.json")),
    # --- the torch-free loader (bench/gliner_onnx.py) on the same ONNX file --------
    Config("kg_pii_base_onnx_loader",
           lambda: gliner_onnx_predictor("knowledgator/gliner-pii-base-v1.0", KG),
           "knowledgator/gliner-pii-base-v1.0", 166, "Apache-2.0", "gliner",
           "uint8 ONNX via onnxruntime + tokenizers, no torch", 0.3,
           weights=("onnx/model_quint8.onnx", "tokenizer.json", "gliner_config.json"), torch_free=True),
    Config("presidio+kg_pii_base_onnx_loader",
           _hyb("en_core_web_sm", lambda: gliner_onnx_predictor("knowledgator/gliner-pii-base-v1.0", KG)),
           "knowledgator/gliner-pii-base-v1.0", 166, "Apache-2.0", "hybrid",
           "uint8 ONNX via onnxruntime + tokenizers, no torch", 0.3,
           weights=("onnx/model_quint8.onnx", "tokenizer.json", "gliner_config.json"), torch_free=True),
    Config("presidio+gretel_bi_small", _hyb("en_core_web_sm", _gl("gretelai/gretel-gliner-bi-small-v1.0")),
           "gretelai/gretel-gliner-bi-small-v1.0", 194, "Apache-2.0", "hybrid"),
    Config("presidio+nvidia_gliner_pii", _hyb("en_core_web_sm", _gl("nvidia/gliner-PII")),
           "nvidia/gliner-PII", 570, "NVIDIA Open Model License", "hybrid", "Reference: large"),
    Config("presidio+openai_privacy_filter", _hyb("en_core_web_sm", lambda: hf_token_predictor("openai/privacy-filter")),
           "openai/privacy-filter", 1500, "Apache-2.0", "hybrid", "Reference", 0.4),
]
CONFIG_BY_KEY = {c.key: c for c in CONFIGS}
