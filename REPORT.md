# Which NER model should Sanctum ship by default?

*Benchmark run 7 Oct 2026 on an Apple M5 (32 GB, CPU only, 4 torch threads).
Everything is reproducible from this repo. See [Reproduce](#reproduce).*

## TL;DR

**Ship [`knowledgator/gliner-pii-base-v1.0`](https://huggingface.co/knowledgator/gliner-pii-base-v1.0)
as the NER model inside Sanctum's existing Presidio pipeline:**

- Use it the same way as the current `ner_backend = "gliner"` path: it replaces
  `SpacyRecognizer`, and all regex recognizers stay.
- Set the threshold to **0.2**.
- Use Knowledgator's label prompts.
- Add a small **name-propagation** pass after detection.

| | Shipped today: Presidio + `en_core_web_sm` | **Recommended** |
|---|---|---|
| PII entities missed (hard corpus, 487) | **101** (21%) | **18** (4%) |
| Leak recall / precision | 0.79 / 0.74 | 0.96 / 0.83 |
| PERSON / ORG / LOCATION recall | 0.59 / 0.51 / 0.53 | 0.97 / 0.88 / 0.86 |
| Generic ID numbers recall (passport, NHS, tax ID, ...) | 0.28 | 0.72 |
| Sanctum's own fixtures (154 entities): missed / typed F1 | 10 / 0.66 | **0** / 0.89 |
| Weights | 15 MB | 664 MB PyTorch, or **197 MB** uint8 ONNX |
| Speed (M5, 4 threads) | 24 ms / 1k chars | 111 ms / 1k chars (≈ 3 s for a 10-page contract) |
| Peak RAM | 0.5 GB | 2.3 GB PyTorch, 1.3 GB ONNX |
| Licence | MIT | Apache-2.0 weights, MIT backbone |

The quantised ONNX build loses little accuracy: 22 misses instead of 18.

**Runner-up: [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)**
(Apache-2.0, 2026). It has the best balance: F1 0.88, precision
0.89, and it is very stable across thresholds. But it misses 41 entities
instead of 18, is twice the size (1.2 GB), and needs `gliner2` + `peft`.
Choose it if over-redaction complaints turn out to matter more than leaks.

**Quick win, no new model:** the current Professional model
(`gliner_medium-v2.1`) misses 63 entities at Sanctum's threshold of 0.4. At
0.2, with propagation, it misses only 31.

![Accuracy vs size](results/tradeoff.png)

---

## 1. Why the current default misses so much

The desktop app ships Presidio + `en_core_web_sm` (`spacy_model` defaults to
`en_core_web_sm` in `sanctum/config/settings.py`, and
`sanctum-desktop/scripts/build-sidecar.sh` bundles only that model). Its NER
layer is the weak part:

| Group (hard corpus) | Recall with `en_core_web_sm` | Typical misses |
|---|---|---|
| PERSON | 0.59 | lowercase chat names, non-Anglo names, ALL-CAPS statements, "Mr Price" |
| ORGANIZATION | 0.51 (precision 0.25) | firm names; and it flags "Board", "Supplier", "HR" |
| LOCATION | 0.53 | full street addresses (it only catches the city part) |
| GOV_ID | 0.28 | passports, NHS numbers, tax codes, VINs: no recognizer exists |

The regex recognizers (email, phone, IBAN, card, IP, URL) are already near
their ceiling, and every model inherits them in the hybrid setup. So the
choice of NER model is really about PERSON, ORGANIZATION, LOCATION and IDs.

**Why switching to GLiNER crashed the packaged app.** The sidecar build
installs `sanctum[security,api,documents]`, not the `gliner` extra. So the
frozen sidecar has neither `gliner` nor `torch`; I checked
`sidecar-build/mac-arm64/_internal`. The Pro-tier downloader
(`src/main/models.ts`) is a stub that nothing calls, and it points at a
placeholder CDN. Picking "GLiNER" in Settings restarts the sidecar with
`SANCTUM_NLP__NER_BACKEND=gliner`. `create_gliner_recognizer()` then raises
`ImportError` while building the engine. I did not change any Sanctum code;
see §6 for what shipping the recommendation would take.

## 2. Finalists

Hard corpus, fixed thresholds as stated. "+ propagate" = the name-propagation
pass from §4. Full tables: [`results/finalists.md`](results/finalists.md),
[`results/tables.md`](results/tables.md),
[`results/leaderboard.csv`](results/leaderboard.csv).

| Setup | Leak recall | Precision | Typed F1 | Typed F2 | Missed (of 487) |
|---|---|---|---|---|---|
| Shipped default: Presidio + en_core_web_sm | 0.79 | 0.74 | 0.61 | 0.63 | 101 |
| Presidio + en_core_web_lg + propagate | 0.89 | 0.76 | 0.72 | 0.77 | 55 |
| Shipped Pro: Presidio + gliner_medium @0.4 | 0.87 | 0.89 | 0.85 | 0.85 | 63 |
| gliner_medium @0.2 + propagate | 0.94 | 0.82 | 0.84 | 0.87 | 31 |
| **kg gliner-pii-base @0.2 + propagate** | **0.96** | 0.83 | 0.84 | **0.88** | **18** |
| kg gliner-pii-base, ONNX uint8 @0.2 + propagate | 0.95 | 0.81 | 0.81 | 0.86 | 22 |
| fastino GLiNER2-PII @0.5 + propagate | 0.92 | 0.89 | 0.88 | 0.89 | 41 |
| nvidia gliner-PII @0.4 + propagate | 0.87 | 0.95 | 0.89 | 0.87 | 61 |

Recall by group:

| Recall | PERSON | ORG | LOCATION | DATE_TIME | EMAIL | PHONE | FINANCIAL | GOV_ID |
|---|---|---|---|---|---|---|---|---|
| Shipped default (sm) | 0.59 | 0.51 | 0.53 | 0.94 | 0.96 | 0.96 | 0.53 | 0.28 |
| Shipped Pro (gliner_medium @0.4) | 0.86 | 0.86 | 0.98 | 0.86 | 1.00 | 1.00 | 0.63 | 0.35 |
| **kg gliner-pii-base @0.2 + propagate** | 0.97 | 0.88 | 0.86 | 0.89 | 1.00 | 0.87 | **1.00** | **0.72** |
| fastino GLiNER2-PII @0.5 + propagate | 0.96 | 0.93 | 0.83 | 0.91 | 1.00 | 0.96 | 0.74 | 0.55 |
| nvidia gliner-PII @0.4 + propagate | 0.95 | 0.75 | 0.88 | 0.89 | 1.00 | 0.96 | 0.68 | 0.55 |

Knowledgator's edge is in **IDs and account numbers**: it was trained on 60+
PII/PHI/PCI labels. Its weak spot is organisation precision (0.52). It tags
role nouns and defined terms ("the Target", "Supplier", "HR") as
organisations. That over-redacts but never leaks.

### Threshold sensitivity (important)

Knowledgator-base is **steep**: tuned for 0.2–0.3, it falls apart at
Sanctum's current default of 0.4.

| Threshold | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| kg-base + propagate, missed | 11 | **18** | 27 | 37 | 69 | 122 |
| kg-base + propagate, precision | 0.78 | 0.83 | 0.86 | 0.88 | 0.93 | 0.95 |
| fastino + propagate, missed | 30 | 33 | 33 | 36 | 40 | 41 |
| gliner_medium + propagate, missed | 27 | 31 | 34 | 39 | 46 | 57 |

So the threshold must ship with the model, not be inherited from
`gliner_threshold = 0.4`. The desktop's score-threshold slider should also be
re-labelled or re-scaled. In the cross-validated selection (each fold tunes
on the other half of the documents), both folds independently picked 0.2–0.25
for this model.

## 3. Everything tested

31 configurations: 16 checkpoints, standalone and/or inside Presidio. All
weights are in `models/`.

| Model | Params | Licence | Shippable? | Notes |
|---|---|---|---|---|
| spaCy `en_core_web_sm` / `_lg` / `_trf` | 12M / – / 125M | MIT | yes | baselines; `trf` = F2 0.83 but needs torch and 3 GB RAM |
| `urchade/gliner_small-v2.1` | 166M | Apache-2.0 | yes | weak on contact fields |
| `urchade/gliner_medium-v2.1` | 209M | Apache-2.0 | yes | current Pro default; good if threshold lowered |
| `urchade/gliner_multi_pii-v1` | 279M | Apache-2.0 | yes | good char coverage, 1.2 GB |
| `E3-JSI/gliner-multi-pii-domains-v1` | 279M | Apache-2.0 | yes | ≈ multi_pii |
| `gretelai/gretel-gliner-bi-small-v1.0` | 194M | Apache-2.0 | yes | very precise, recall 0.48: does not suit our prompts |
| **`knowledgator/gliner-pii-base-v1.0`** | ~166M | Apache-2.0 | **yes** | **recommended** |
| `knowledgator/gliner-pii-small-v1.0` | ~80M | Apache-2.0 | yes | 20 misses but precision 0.78; ONNX 83 MB |
| `knowledgator/gliner-pii-edge-v1.0` | ~45M | Apache-2.0 | yes | ONNX 46 MB, but F2 0.73–0.80: too lossy |
| `fastino/gliner2-privacy-filter-PII-multi` | 205M | Apache-2.0 | yes | runner-up |
| `nvidia/gliner-PII` | 570M | NVIDIA Open Model License | yes (custom terms) | most precise; 1.8 GB, 3× slower, 4 GB RAM |
| `openai/privacy-filter` | 1.5B (50M active) | Apache-2.0 | technically | 2.8 GB, 9 GB RAM, no ORG label; see caveats |
| `bardsai/eu-pii-anonimization-multilang` | 278M | Apache-2.0 | yes | precise (0.95), misses 81; Sanctum-fixtures F1 only 0.71 |
| `dslim/bert-base-NER` | 108M | MIT | yes | CoNLL PER/ORG/LOC only; poor |
| `SoelMgd/bert-pii-detection` | 66M | MIT on the card | **no** | trained on ai4privacy data. Their terms require a paid licence for orgs with more than 3 staff and forbid redistributing derivatives |

Earlier-round reference models stay excluded: Piiranha (CC-BY-NC-ND) and
Isotonic (CC-BY-NC). `ab-ai/pii_model` is gated.

Licence check on the recommendation:

- The weights are Apache-2.0.
- The backbone `microsoft/deberta-v3-small` is MIT.
- The model card does not name its training data. Knowledgator's public
  synthetic GLiNER datasets are Apache-2.0, and nothing points to ai4privacy
  (unlike SoelMgd). If you want certainty before shipping commercially, ask
  Knowledgator / Wordcab to confirm the training-data terms.
- Apache-2.0 requires keeping the licence and NOTICE text in the bundle. Put
  it next to the existing third-party notices.

## 4. Name propagation: a model-independent fix

The most common miss for every model was a **repeat mention**. A model finds
"Dwayne Kowalczyk" but then misses "Dwayne" two paragraphs later, or "Mr
Price" after "Mark Price". NER models judge each mention in isolation.
`bench/postprocess.py::propagate` works per document:

1. Collect the text of every detected PERSON / ORGANIZATION span.
2. Add the individual name tokens of each person, skipping titles and particles.
3. Add the distinctive first word of each organisation, e.g. "Verdana" from
   "Verdana Pharmaceuticals".
4. Mark every other occurrence of those strings with the same type.

It costs a regex pass and helps every hybrid:

| Setup | Missed without | Missed with |
|---|---|---|
| kg gliner-pii-base @0.2 | 31 | 18 |
| gliner_medium @0.2 | 42 | 31 |
| fastino @0.5 | 57 | 41 |
| en_core_web_sm | 101 | 79 |

In Sanctum this belongs in `PresidioAnalyzer.analyze()`, right after
`_normalize_overlaps`. It is independent of the model choice and worth doing
regardless.

## 5. What the recommendation still misses

The 18 remaining leaks:

| Kind | Count | Examples |
|---|---|---|
| Times and bare years | 8 | "10:00 a.m. Pacific Time", "7:45", "2019", "late 2024" |
| Lone first names never written in full | 2 | "Seun" (nickname of "Oluwaseun"), "Maria" |
| IDs with no surrounding cue | 4 | VIN `1HGCV1F34MA012345`, plate "NH 4417KX", hospital no. `RX8841207`, case ref `HRI-2025-031` |
| Other | 4 | "Huntington National Bank", "Route 3", and "PROVIDENCE HEALTH" / "DIVISION ST" in an ALL-CAPS statement table |

Most of these could be fixed with pattern recognizers (VIN, UK plate, a
generic "ref/no.: <alnum>" context recognizer). Times and years are
low-sensitivity, depending on policy.

**Over-redaction** (precision 0.83) is mostly role nouns and defined terms
("the Board", "the Employee", "Supplier"), department names ("HR", "Sales"),
and a few product names. A small stop-list of defined-term patterns
("the X" where X is capitalised and defined in quotes) would remove most of
them.

## 6. What shipping this would take (not done: no Sanctum code was changed)

1. **Prompts.** Sanctum's `_GLINER_ENTITY_MAPPING` uses generic prompts
   ("person", "location"). With Knowledgator, its own label names work better:
   `name`, `location address`, `location city`, `ssn`, `passport number`, ...
   (see `GLINER_KNOWLEDGATOR` in `bench/predictors.py`). With Sanctum's existing
   prompts (threshold 0.2, with propagation) it misses 26 instead of 18 on the
   hard corpus, and 12 instead of 0 on the fixtures. A config-only swap works
   but leaves accuracy on the table.
2. **Threshold.** `gliner_threshold = 0.2` for this model. Add an `ID_NUMBER`
   (or similar) entity, so passport, NHS and tax IDs get a tag instead of being
   squeezed into `US_SSN`.
3. **Packaging.** `gliner` imports PyTorch even on its ONNX path (verified).
   PyTorch is 582 MB unpacked, while the whole current sidecar is 161 MB. Options:
   - **(a)** Bundle `torch` + `gliner` + the 664 MB checkpoint. My estimate
     (not built) is a 1.2–1.5 GB sidecar. Simple.
   - **(b)** Ship the 197 MB uint8 ONNX file with `onnxruntime` (80 MB) and a
     small hand-written GLiNER pre/post-processor. A Rust port
     ([gline-rs](https://github.com/fbilhaut/gline-rs)) and a JS port exist to
     copy from. My estimate (not built) is a ~450 MB sidecar with 1.3 GB peak
     RAM (measured with the ONNX model in this benchmark). More work, best
     result.
   - Either way, the model must ship **inside** the app or a verified download.
     Today's runtime `from_pretrained()` hub fetch breaks the airgap rule.
4. **Airgap gotcha.** GLiNER checkpoints store only the GLiNER head config.
   At load time they also resolve the backbone config and tokenizer by Hub id
   (`microsoft/deberta-v3-small` for this model). The bundle must include those
   files, and the loader must be pointed at local paths with
   `HF_HUB_OFFLINE=1`. Otherwise first launch makes a network call. `run.py`
   caches them under `models/_hf_cache` to show what is needed (≈ 7 MB).
5. **Standard tier without torch.** If a torch-free tier must remain, use
   `en_core_web_lg` + propagation (55 misses vs 101, +430 MB, MIT). It is a
   strict improvement over `sm`.

## 7. Method

**Corpora.**

- **hard** (`data/corpus/`): 22 documents I wrote for this benchmark, with
  487 entities plus 79 "optional" borderline spans. It covers UK, US, Italian,
  German, French, Spanish, Dutch, Nordic and Singapore legal and consulting
  material:
  - formats: engagement letters, witness statements, board minutes, Slack and
    WhatsApp exports, a payroll CSV, an ALL-CAPS scanned bank statement, KYC
    forms, a medical referral, an insurance claim
  - hard negatives: a zero-PII contract and a tech memo full of product names
    and statute references
  - annotation rules: `data/corpus/README.md`
- **sanctum** (`data/sanctum_fixtures/`): a copy of Sanctum's 12 Faker
  fixtures, 154 entities.

**Metrics** (`bench/score.py`). All are overlap-based, like Sanctum's
`EntityScorer`:

- **Leak recall:** share of entities touched by any prediction. This is the
  privacy metric.
- **Char recall:** share of entity characters covered. It catches partial
  redaction.
- **Precision:** share of predictions that hit a real or optional span.
- **Typed F1 / F2:** a hit also needs the right coarse group (PERSON / ORG /
  LOCATION / DATE / EMAIL / PHONE / URL / IP / FINANCIAL / GOV_ID). F2 weights
  recall double, because a miss is a leak.

**Setup** (`bench/predictors.py`):

- "presidio+X" reproduces `sanctum/cli/commands.py::_create_engine`: ORG
  kept, `AnyDomainEmailRecognizer`, overlap normalisation, threshold 0.35. X
  replaces `SpacyRecognizer`, exactly as `ner_backend="gliner"` does.
- Text is chunked to ≤ 1000 chars on line boundaries.
- Thresholds are swept from 0.15 to 0.8. The leaderboard picks each model's
  threshold by 2-fold cross-validation over documents, so no model is tuned on
  the documents it is scored on.

**Caveats.**

- **Small corpus, same author.** I wrote both the documents and the labels.
  The corpus has 487 entities, so differences under about 0.02 F2 are noise.
  The top four setups are statistically close; the recommendation rests on
  recall plus size, speed and licence, not on a decimal.
- **Synthetic text.** The documents are realistic in shape, but the people
  and companies are invented. Real client documents (longer, messier, more
  OCR noise) should be spot-checked before release.
- **OpenAI Privacy Filter** was decoded with plain argmax, not the
  constrained Viterbi decoder its authors ship. That probably understates its
  recall. It would still be excluded on size (2.8 GB, 9 GB RAM) and for
  having no organisation or location labels.
- **English only.** Multilingual tests were not run, though Fastino,
  NVIDIA and bardsai claim EU-language support.
- Latency is CPU-only (4 threads) on an M5. Expect roughly 2× slower on an
  older Intel laptop.

## Reproduce

See [README.md](README.md#reproduce). In short:

```bash
.venv/bin/python bench/download_models.py   # ~13 GB into ./models (git-ignored)
.venv/bin/python bench/run.py               # cached in results/preds/
.venv/bin/python bench/report_tables.py     # tables + chart
.venv/bin/python bench/inspect_errors.py "presidio+kg_pii_base" hard 0.2
```

Weights are in `models/<org>__<name>/`. `du -sh models/*` shows what each one
costs; delete any directory to reclaim the space.
