# sanctum-research

Evaluation playground for [Sanctum](https://github.com/FilippoTonci/sanctum) —
exploratory notebooks, model benchmarks, and cached prediction artifacts
that don't belong in the production repo.

Kept separate so the main codebase stays airgap-clean: the research
workflows pull transformer weights from HuggingFace on first run,
which would otherwise break Sanctum's "no runtime network calls"
invariant.

![Example Results](/assets/heatmap.png)


## Layout

```
notebooks/
  pii_model_benchmark.ipynb    small-transformer PII/NER models vs Sanctum's Presidio baseline
  bench_results/               cached per-model predictions (JSON, one file per model)
```

## Prerequisites

This repo expects a sibling Sanctum checkout so notebooks can import
the production pipeline and read the fixture corpus:

```
projects/
├── sanctum/              # main repo — https://github.com/FilippoTonci/sanctum
└── sanctum-research/     # this repo
```

Override the default location with the `SANCTUM_ROOT` environment
variable if your layout differs.

## Running the notebooks

1. Create a virtualenv and install the notebook dependencies. The
   install cell at the top of `pii_model_benchmark.ipynb` lists them
   (torch CPU wheel, transformers, gliner, pandas, matplotlib,
   ipywidgets, tqdm).
2. Make sure the sibling `sanctum/` checkout has its dev env set up
   (`pip install -e .` + the spaCy model) — the notebook imports
   `sanctum.core`, `sanctum.analyzer`, and `tests.evaluation.scorer`
   from there.
3. Open `notebooks/pii_model_benchmark.ipynb` and run top-to-bottom.
   Cached results under `bench_results/` let you re-score without
   reloading any model; delete a file to force a re-run for that
   model.

Disk footprint for the default model set is ≈ 3.5 GB in
`~/.cache/huggingface/`. Expect 2–8 minutes per model on CPU.

## What's in the benchmark

The notebook compares Sanctum's Presidio baseline against a set of
small transformer models (GLiNER variants, Stanford/obi clinical
de-id, Piiranha, DeBERTa-based PII fine-tunes) on the 12-document
fixture corpus shipped in `sanctum/tests/fixtures/`. Scoring uses
Sanctum's overlap-matched, type-aware `EntityScorer`, so numbers are
directly comparable to `pytest -m evaluation` output in the main
repo.

Licence-restricted models (Piiranha, Isotonic) are included as
reference numbers only — they cannot ship in a commercial Sanctum
build.
