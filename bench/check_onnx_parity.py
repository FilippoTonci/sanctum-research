"""Check bench/gliner_onnx.py against the gliner library on the same ONNX file.

Runs both on every chunk of both corpora with Knowledgator's prompts at the
benchmark floor threshold and compares spans (char offsets + label) and scores.

    python bench/check_onnx_parity.py [repo_id] [onnx_file]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import CORPORA
from gliner_onnx import GlinerOnnx
from predictors import FLOOR, GLINER_KNOWLEDGATOR, MODELS_DIR, chunks

REPO = sys.argv[1] if len(sys.argv) > 1 else "knowledgator/gliner-pii-base-v1.0"
ONNX = sys.argv[2] if len(sys.argv) > 2 else "onnx/model_quint8.onnx"


def main() -> int:
    from gliner import GLiNER

    path = MODELS_DIR / REPO.replace("/", "__")
    ref = GLiNER.from_pretrained(str(path), load_onnx_model=True, load_tokenizer=True,
                                 onnx_model_file=ONNX)
    mine = GlinerOnnx(path, ONNX)
    labels = list(GLINER_KNOWLEDGATOR)

    n_chunks = n_ref = n_mine = n_match = 0
    max_dscore = 0.0
    mismatches: list[str] = []
    t_ref = t_mine = 0.0
    tok = ref.data_processor.transformer_tokenizer
    prompt = [t for label in labels for t in (mine.ent_token, label)] + [mine.sep_token]
    n_tok_diff = 0
    for corpus in ("hard", "sanctum"):
        for doc_id, doc in CORPORA[corpus]().items():
            for _, chunk in chunks(doc["text"]):
                n_chunks += 1
                t0 = time.perf_counter()
                a = ref.predict_entities(chunk, labels, threshold=FLOOR, flat_ner=True)
                t1 = time.perf_counter()
                b = mine.predict_entities(chunk, labels, threshold=FLOOR, flat_ner=True)
                t2 = time.perf_counter()
                t_ref += t1 - t0
                t_mine += t2 - t1
                words = [w for w, _, _ in mine.split_words(chunk)]
                if mine._encode(words, labels)["input_ids"][0].tolist() != \
                        tok(prompt + words, is_split_into_words=True)["input_ids"]:
                    # the library uses the slow sentencepiece tokenizer; ``tokenizers``
                    # NFKC-normalises a few compatibility characters (e.g. "º" -> "o")
                    # that the slow one maps to [UNK]. Report, don't compare.
                    n_tok_diff += 1
                    print(f"  tokenization differs in a chunk of {doc_id}; skipped")
                    continue
                ka = {(r["start"], r["end"], r["label"]): r["score"] for r in a}
                kb = {(e.start, e.end, e.label): e.score for e in b}
                n_ref += len(ka)
                n_mine += len(kb)
                for key in ka.keys() & kb.keys():
                    n_match += 1
                    max_dscore = max(max_dscore, abs(ka[key] - kb[key]))
                for key in ka.keys() ^ kb.keys():
                    side = "gliner only" if key in ka else "onnx-loader only"
                    score = ka.get(key, kb.get(key))
                    mismatches.append(f"{doc_id}: {side} {key} {chunk[key[0]:key[1]]!r} "
                                      f"score={score:.3f}")

    print(f"chunks: {n_chunks} ({n_tok_diff} skipped: tokenizer normalisation differs)")
    print(f"spans: gliner={n_ref} loader={n_mine} identical={n_match}")
    print(f"max |score diff| on identical spans: {max_dscore:.2e}")
    print(f"time: gliner {t_ref:.1f}s, loader {t_mine:.1f}s")
    for line in mismatches[:40]:
        print("  ", line)
    return 0 if not mismatches and max_dscore < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
