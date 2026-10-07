"""Download every benchmarked checkpoint into ./models/<org>__<name>/.

Weights live inside this repo (git-ignored) rather than ~/.cache/huggingface
so they are easy to inspect, copy into a sidecar build, or delete.
Only one weight format per model is fetched (see ALLOW).
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

META = ["*.json", "*.md", "*.model", "*.txt", "LICENSE*"]
ALLOW: dict[str, list[str]] = {
    "knowledgator/gliner-pii-edge-v1.0": META + ["pytorch_model.bin", "onnx/model_quint8.onnx"],
    "knowledgator/gliner-pii-small-v1.0": META + ["pytorch_model.bin", "onnx/model_quint8.onnx"],
    "fastino/gliner2-privacy-filter-PII-multi": META + ["model.safetensors"],
    "knowledgator/gliner-pii-base-v1.0": META + ["pytorch_model.bin", "onnx/model_quint8.onnx"],
    "SoelMgd/bert-pii-detection": META + ["*.safetensors"],
    "dslim/bert-base-NER": META + ["model.safetensors"],
    "bardsai/eu-pii-anonimization-multilang": META + ["model.safetensors"],
    "urchade/gliner_small-v2.1": META + ["pytorch_model.bin"],
    "urchade/gliner_medium-v2.1": META + ["pytorch_model.bin"],
    "urchade/gliner_multi_pii-v1": META + ["pytorch_model.bin"],
    "gretelai/gretel-gliner-bi-small-v1.0": META + ["pytorch_model.bin"],
    "nvidia/gliner-PII": META + ["pytorch_model.bin"],
    "openai/privacy-filter": META + ["model.safetensors"],
    "E3-JSI/gliner-multi-pii-domains-v1": META + ["model.safetensors"],
}


def local_dir(repo_id: str) -> Path:
    return MODELS_DIR / repo_id.replace("/", "__")


def main(only: list[str]) -> None:
    for repo_id, patterns in ALLOW.items():
        if only and repo_id not in only:
            continue
        print(f"[dl] {repo_id}", flush=True)
        try:
            snapshot_download(repo_id, local_dir=local_dir(repo_id), allow_patterns=patterns)
        except Exception as exc:  # noqa: BLE001 — keep going on one failure
            print(f"     FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
