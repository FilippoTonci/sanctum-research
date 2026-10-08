"""Run benchmark configurations and cache predictions under results/preds/.

    .venv/bin/python bench/run.py                 # every config without cached preds
    .venv/bin/python bench/run.py kg_pii_edge -f  # one config, force re-run

Each config runs in its own subprocess so peak RSS is attributable to that
model alone and memory is fully released between models.
"""
from __future__ import annotations

import gzip
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "results" / "preds"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def slug(key: str) -> str:
    return "".join(c if c.isalnum() or c in "_-+" else "_" for c in key).strip("_")


def pred_path(key: str) -> Path:
    return PREDS / f"{slug(key)}.json.gz"


def load_runs() -> list[dict]:
    """Every cached run (gzipped JSON, one file per config)."""
    return [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(PREDS.glob("*.json.gz"))]


def load_run(key: str) -> dict:
    return json.loads(gzip.decompress(pred_path(key).read_bytes()))


def _dir_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def disk_mb(repo: str, weights: tuple[str, ...] = ()) -> float:
    """Size of what would ship: the given files, else the model dir minus
    onnx/ exports and download caches."""
    if repo.startswith("spacy/"):
        import importlib.util
        spec = importlib.util.find_spec(repo.split("/", 1)[1])
        return _dir_mb(Path(spec.origin).parent) if spec else 0.0
    d = ROOT / "models" / repo.replace("/", "__")
    if not d.exists():
        return 0.0
    if weights:
        files = {p for pat in weights for p in d.glob(pat) if p.is_file()}
    else:
        files = {p for p in d.rglob("*") if p.is_file()
                 and not {"onnx", ".cache"} & set(p.relative_to(d).parts)}
    return sum(p.stat().st_size for p in files) / 1e6


def run_one(key: str) -> None:
    """Child process entry point."""
    # Weights come from ./models. GLiNER v1 checkpoints still resolve their
    # backbone's config/tokenizer by hub id (e.g. microsoft/deberta-v3-base);
    # keep that cache inside the repo too so every artifact is reviewable here.
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "_hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from corpus import CORPORA
    from predictors import CONFIG_BY_KEY, THRESHOLDS

    cfg = CONFIG_BY_KEY[key]
    if not cfg.torch_free:
        import torch
        torch.set_num_threads(4)  # a fixed, laptop-like budget for latency numbers
    t0 = time.perf_counter()
    predict = cfg.build()
    load_s = time.perf_counter() - t0
    predict("Warm-up sentence for John Smith in London.")

    out: dict = {"key": key, "repo": cfg.repo, "family": cfg.family, "licence": cfg.licence,
                 "params_m": cfg.params_m, "note": cfg.note, "load_s": load_s,
                 "default_thr": cfg.default_thr, "disk_mb": disk_mb(cfg.repo, cfg.weights), "corpora": {}}
    for cname, loader in CORPORA.items():
        docs = loader()
        chars = sum(len(d["text"]) for d in docs.values())
        by_thr: dict[str, dict] = {}
        t = time.perf_counter()
        if cfg.family == "hybrid":
            # first pass (lowest threshold) includes model inference; later passes hit the cache
            for i, thr in enumerate(THRESHOLDS):
                by_thr[str(thr)] = {name: predict(d["text"], thr) for name, d in docs.items()}
                if i == 0:
                    elapsed = time.perf_counter() - t
        elif cfg.default_thr is None:
            by_thr["none"] = {name: predict(d["text"]) for name, d in docs.items()}
            elapsed = time.perf_counter() - t
        else:
            raw = {name: predict(d["text"]) for name, d in docs.items()}
            elapsed = time.perf_counter() - t
            for thr in THRESHOLDS:
                by_thr[str(thr)] = {n: [s for s in ps if s["score"] >= thr] for n, ps in raw.items()}
        out["corpora"][cname] = {"seconds": elapsed, "chars": chars, "by_thr": by_thr}
    if cfg.torch_free:
        # spaCy imports torch whenever it is installed: run these via `make run-notorch`
        assert "torch" not in sys.modules, f"{key} is torch-free but torch got imported"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["peak_rss_mb"] = rss / (1e6 if sys.platform == "darwin" else 1e3)
    PREDS.mkdir(parents=True, exist_ok=True)
    data = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode()
    pred_path(key).write_bytes(gzip.compress(data, mtime=0))


def main(argv: list[str]) -> None:
    from predictors import CONFIGS

    force = "-f" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    for cfg in CONFIGS:
        if wanted and cfg.key not in wanted:
            continue
        if pred_path(cfg.key).exists() and not force:
            print(f"[cache] {cfg.key}", flush=True)
            continue
        print(f"[run  ] {cfg.key}", flush=True)
        t = time.perf_counter()
        r = subprocess.run([sys.executable, "-I", __file__, "--child", cfg.key],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"        FAILED ({r.returncode}):\n" + "\n".join(r.stderr.splitlines()[-12:]), flush=True)
        else:
            print(f"        ok in {time.perf_counter() - t:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        run_one(sys.argv[2])
    else:
        main(sys.argv[1:])
