"""Torch-free GLiNER span-model inference on ONNX Runtime.

The ``gliner`` library imports PyTorch even when it runs an ONNX export, so
shipping it would put ~520 MB of PyTorch into the Sanctum sidecar. This module
re-implements the small amount of pre/post-processing a *span-level,
uni-encoder* GLiNER checkpoint needs (e.g. knowledgator/gliner-pii-*), using
only ``onnxruntime``, ``tokenizers`` and ``numpy``:

1. Split text into words with GLiNER's whitespace splitter.
2. Prepend the label prompt: ``<<ENT>> label1 <<ENT>> label2 ... <<SEP>>``.
3. Sub-word tokenize the words; mark the first sub-token of each text word.
4. Enumerate every span of up to ``max_width`` words.
5. Run the graph; sigmoid the logits; keep spans above the threshold.
6. Greedy flat decoding: highest score first, drop anything overlapping.

It mirrors gliner 0.2.29 (``UniEncoderSpanProcessor`` + ``SpanDecoder``).
``bench/check_onnx_parity.py`` checks it gives the same spans as the library.
Everything is loaded from local files; nothing touches the network.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")


@dataclass(frozen=True)
class Entity:
    start: int  # character offsets into the input text, end exclusive
    end: int
    label: str
    score: float
    text: str


class GlinerOnnx:
    """A span-level GLiNER model loaded from a local directory.

    ``model_dir`` must hold ``gliner_config.json``, ``tokenizer.json`` and the
    ONNX file (``onnx_file`` is relative to ``model_dir``).
    """

    def __init__(
        self,
        model_dir: str | Path,
        onnx_file: str = "onnx/model_quint8.onnx",
        intra_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        config = json.loads((model_dir / "gliner_config.json").read_text(encoding="utf-8"))
        if config.get("span_mode") == "token_level":
            raise ValueError("token-level GLiNER checkpoints are not supported")
        self.max_width: int = int(config["max_width"])
        self.max_len: int = int(config["max_len"])
        self.ent_token: str = config.get("ent_token", "<<ENT>>")
        self.sep_token: str = config.get("sep_token", "<<SEP>>")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(model_dir / onnx_file), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    # -- public -------------------------------------------------------------
    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[Entity]:
        labels = list(dict.fromkeys(labels))
        words = self.split_words(text)
        if not words or not labels:
            return []
        words = words[: self.max_len]
        inputs = self._encode([w for w, _, _ in words], labels)
        (logits,) = self.session.run(["logits"], inputs)
        probs = 1.0 / (1.0 + np.exp(-logits[0]))  # (L, K, C)
        return self._decode(probs, words, labels, text, threshold, flat_ner)

    @staticmethod
    def split_words(text: str) -> list[tuple[str, int, int]]:
        """GLiNER's whitespace splitter: words (with inner - or _) and single symbols."""
        return [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]

    # -- internals ----------------------------------------------------------
    def _encode(self, words: list[str], labels: list[str]) -> dict[str, Any]:
        prompt: list[str] = []
        for label in labels:
            prompt.append(self.ent_token)
            if label:
                prompt.append(label)
        prompt.append(self.sep_token)
        n_prompt = len(prompt)

        enc = self.tokenizer.encode(prompt + words, is_pretokenized=True)
        input_ids = np.asarray(enc.ids, dtype=np.int64)[None, :]
        attention_mask = np.asarray(enc.attention_mask, dtype=np.int64)[None, :]

        # words_mask: 1-based index of the text word on its *first* sub-token,
        # 0 for special tokens, prompt words and continuation sub-tokens.
        words_mask = np.zeros_like(input_ids)
        prev: int | None = None
        for i, wid in enumerate(enc.word_ids):
            if wid is not None and wid != prev and wid >= n_prompt:
                words_mask[0, i] = wid - n_prompt + 1
            prev = wid

        n = len(words)
        k = self.max_width
        starts = np.repeat(np.arange(n, dtype=np.int64), k)
        ends = starts + np.tile(np.arange(k, dtype=np.int64), n)
        span_idx = np.stack([starts, ends], axis=1)[None, :, :]
        span_mask = (ends <= n - 1)[None, :]

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": np.asarray([[n]], dtype=np.int64),
            "span_idx": span_idx,
            "span_mask": span_mask,
        }
        return {name: value for name, value in feeds.items() if name in self._input_names}

    def _decode(
        self,
        probs: np.ndarray,
        words: list[tuple[str, int, int]],
        labels: list[str],
        text: str,
        threshold: float,
        flat_ner: bool,
    ) -> list[Entity]:
        n = len(words)
        s_idx, k_idx, c_idx = np.nonzero(probs > threshold)
        candidates: list[tuple[int, int, int, float]] = []
        for s, k, c in zip(s_idx.tolist(), k_idx.tolist(), c_idx.tolist()):
            if s + k + 1 <= n and c < len(labels):
                candidates.append((s, s + k, c, float(probs[s, k, c])))

        # greedy: best score first, skip anything overlapping a kept span
        candidates.sort(key=lambda x: -x[3])
        kept: list[tuple[int, int, int, float]] = []
        for cand in candidates:
            if not any(_overlaps(cand, other, flat_ner) for other in kept):
                kept.append(cand)
        kept.sort(key=lambda x: x[0])

        out = []
        for s, e, c, score in kept:
            cs, ce = words[s][1], words[e][2]
            out.append(Entity(cs, ce, labels[c], score, text[cs:ce]))
        return out


def _overlaps(a: tuple[int, int, int, float], b: tuple[int, int, int, float], flat: bool) -> bool:
    if (a[0], a[1]) == (b[0], b[1]):
        return True  # multi_label=False: one label per span
    disjoint = a[0] > b[1] or b[0] > a[1]
    if flat:
        return not disjoint
    nested = (a[0] <= b[0] and a[1] >= b[1]) or (b[0] <= a[0] and b[1] >= a[1])
    return not disjoint and not nested
