# sanctum-research

Evaluation playground for [Sanctum](https://github.com/FilippoTonci/sanctum):
model benchmarks, exploratory notebooks and cached predictions that don't
belong in the production repo.

It is kept separate so the main codebase stays airgap-clean. The research
workflows pull transformer weights from Hugging Face, which would break
Sanctum's "no runtime network calls" rule.

**Latest result:** [`REPORT.md`](REPORT.md) recommends a default NER model
to ship with Sanctum.

![Accuracy vs size](results/tradeoff.png)

## Layout

```
REPORT.md                 recommendation + findings (start here)
bench/
  download_models.py      fetch every benchmarked checkpoint into ./models/
  corpus.py               load the two evaluation corpora
  predictors.py           model wrappers + the registry of benchmark configs
  run.py                  run configs (one subprocess each), cache predictions
  score.py                overlap-based metrics (leak recall, typed F1/F2, ...)
  report_tables.py        cross-validated leaderboard + chart -> results/
  postprocess.py          document-level name propagation experiment
  inspect_errors.py       print misses / false positives for one config
data/
  corpus/                 22 hand-annotated hard documents (inline markup)
  sanctum_fixtures/       copy of sanctum/tests/fixtures (12 Faker docs)
results/
  preds/                  cached predictions per config (all thresholds)
  leaderboard.csv         one row per config, CV-tuned + default thresholds
  tables.md               markdown tables used in REPORT.md
  tradeoff.png            accuracy vs size chart
models/                   downloaded weights (git-ignored, ~13 GB)
```

## Reproduce

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install torch transformers 'gliner>=0.2.16' gliner2 peft \
    huggingface_hub pandas matplotlib presidio-analyzer 'spacy>=3.8,<3.9' \
    'thinc>=8.3.12,<8.4' 'spacy-curated-transformers<1.0' onnxruntime
.venv/bin/python -m spacy download en_core_web_sm   # also _lg, _trf
.venv/bin/python bench/download_models.py           # ~13 GB into ./models
.venv/bin/python bench/run.py                       # ~10 min on an M-series Mac
.venv/bin/python bench/report_tables.py
```

Nothing here imports Sanctum. The production Presidio setup
(`sanctum/cli/commands.py::_create_engine`) is reproduced in
`bench/predictors.py::presidio_predictor`, so the numbers match what the CLI
and desktop sidecar would produce.

GLiNER v1 checkpoints still fetch their backbone config and tokenizer by Hub
id (for example `microsoft/deberta-v3-base`). `run.py` points `HF_HOME` at
`models/_hf_cache` so those files also stay inside this repo.

The earlier notebook round (7 models on the 12 Sanctum fixtures) was removed;
`bench/` replaces it and is still in git history.
