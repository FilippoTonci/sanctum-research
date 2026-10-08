# Reproduce the NER model benchmark behind REPORT.md.
#
#   make            venv + spaCy models + HF weights + run + report (~13 GB, ~15 min)
#   make report     rebuild tables/chart from cached results/preds (seconds)
#   make inspect CONFIG="presidio+kg_pii_base" THR=0.2
#   make parity     torch-free ONNX loader vs the gliner library
#
# Weights land in ./models (git-ignored); delete a model's folder to reclaim space.

PYTHON  ?= /opt/homebrew/bin/python3.12
VENV    := .venv
PY      := $(VENV)/bin/python -I
SPACY_MODELS := en_core_web_sm en_core_web_lg en_core_web_trf
SPACY_MODEL_VERSION := 3.8.0
SPACY_MODEL_URL := https://github.com/explosion/spacy-models/releases/download

CONFIG  ?= presidio+kg_pii_base
CORPUS  ?= hard
THR     ?=

.PHONY: all venv spacy-models models run rerun report score inspect parity run-notorch clean-results clean-models clean

all: models run report

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip wheel
	$(VENV)/bin/pip install -q -r requirements.txt
	touch $@

venv: $(VENV)/.installed

# pip sometimes reads 0 bytes from GitHub's release redirect and calls the wheel
# "invalid"; curl follows it reliably, so fetch first and install the local file.
WHEELS := models/_wheels
$(WHEELS)/%.whl:
	mkdir -p $(WHEELS)
	m=$$(echo $* | cut -d- -f1); \
	  curl -fsSL --retry 5 -o $@.part "$(SPACY_MODEL_URL)/$$m-$(SPACY_MODEL_VERSION)/$*.whl" && mv $@.part $@

$(VENV)/.spacy-models: $(VENV)/.installed
	for m in $(SPACY_MODELS); do \
	  $(PY) -c "import $$m" 2>/dev/null && continue; \
	  $(MAKE) -s $(WHEELS)/$$m-$(SPACY_MODEL_VERSION)-py3-none-any.whl || exit 1; \
	  $(VENV)/bin/pip install -q $(WHEELS)/$$m-$(SPACY_MODEL_VERSION)-py3-none-any.whl || exit 1; \
	done
	touch $@

spacy-models: $(VENV)/.spacy-models

# Hugging Face checkpoints into ./models/<org>__<name>/ (skips files already present)
models: spacy-models
	$(PY) bench/download_models.py

# Run every config without cached predictions in results/preds/
run: spacy-models
	$(PY) bench/run.py

# Force a re-run of every config
rerun: spacy-models
	$(PY) bench/run.py -f

report: venv
	$(PY) bench/report_tables.py > /dev/null
	@echo "wrote results/tables.md, results/leaderboard.csv, results/tradeoff.png"

score: venv
	$(PY) bench/score.py

inspect: venv
	$(PY) bench/inspect_errors.py "$(CONFIG)" $(CORPUS) $(THR)

# Torch-free loader (bench/gliner_onnx.py) vs the gliner library on the same ONNX file
parity: venv
	$(PY) bench/check_onnx_parity.py

# The torch-free configs in a venv with no torch, so peak RSS is what the sidecar sees
NOTORCH := .venv-notorch
$(NOTORCH)/.installed: requirements-notorch.txt
	$(PYTHON) -m venv $(NOTORCH)
	$(NOTORCH)/bin/pip install -q --upgrade pip
	$(NOTORCH)/bin/pip install -q -r requirements-notorch.txt
	$(MAKE) -s $(WHEELS)/en_core_web_sm-$(SPACY_MODEL_VERSION)-py3-none-any.whl
	$(NOTORCH)/bin/pip install -q $(WHEELS)/en_core_web_sm-$(SPACY_MODEL_VERSION)-py3-none-any.whl
	touch $@

run-notorch: $(NOTORCH)/.installed
	$(NOTORCH)/bin/python -I bench/run.py kg_pii_base_onnx_loader presidio+kg_pii_base_onnx_loader -f

clean-results:
	rm -f results/preds/*.json.gz

clean-models:
	rm -rf models

clean:
	rm -rf $(VENV) $(NOTORCH)
