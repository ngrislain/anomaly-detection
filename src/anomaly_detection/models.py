"""Character-level log-probability models.

A model exposes a single method, `char_log_probs(text) -> list[float]`, giving
one log-probability per character of `text`, each conditioned on the preceding
characters. Two implementations are provided:

- `TrigramModel`: a character trigram model with interpolated smoothing.
- `QwenModel`: a HuggingFace causal LM whose per-token log-probability is
  divided by the token's character length and attributed to each character.
"""

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Protocol, runtime_checkable

# Sentinel prepended to every text so the first characters have a context.
BOT = "\x02"

DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-0.8B"


@runtime_checkable
class CharLogProbModel(Protocol):
    """Anything that can score each character of a text."""

    def char_log_probs(self, text: str) -> list[float]:
        """Return one log-probability per character of `text`."""
        ...


class TrigramModel:
    """Character trigram model with interpolated (Jelinek-Mercer) smoothing.

    P(c | c1, c2) mixes the trigram, bigram, unigram and uniform estimates, so
    no character ever receives zero probability -- including characters never
    seen during training.
    """

    def __init__(self, weights: tuple[float, float, float, float] = (0.7, 0.2, 0.09, 0.01)):
        if not math.isclose(sum(weights), 1.0):
            raise ValueError(f"weights must sum to 1, got {sum(weights)}")
        self.weights = weights
        self.trigrams: Counter[tuple[str, str, str]] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        self.unigrams: Counter[str] = Counter()
        self.trigram_contexts: Counter[tuple[str, str]] = Counter()
        self.bigram_contexts: Counter[str] = Counter()
        self.total = 0
        self.vocab: set[str] = set()

    def fit(self, texts: str | list[str]) -> "TrigramModel":
        """Accumulate counts from one or more training texts."""
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            padded = BOT + BOT + text
            self.vocab.update(text)
            for i in range(2, len(padded)):
                c2, c1, c = padded[i - 2], padded[i - 1], padded[i]
                self.trigrams[(c2, c1, c)] += 1
                self.trigram_contexts[(c2, c1)] += 1
                self.bigrams[(c1, c)] += 1
                self.bigram_contexts[c1] += 1
                self.unigrams[c] += 1
                self.total += 1
        return self

    @property
    def vocab_size(self) -> int:
        """Training vocabulary plus one slot for unseen characters."""
        return len(self.vocab) + 1

    def char_log_probs(self, text: str) -> list[float]:
        if self.total == 0:
            raise RuntimeError("TrigramModel must be fitted before scoring")
        w3, w2, w1, w0 = self.weights
        uniform = 1.0 / self.vocab_size
        padded = BOT + BOT + text
        out = []
        for i in range(2, len(padded)):
            c2, c1, c = padded[i - 2], padded[i - 1], padded[i]
            tri_ctx = self.trigram_contexts[(c2, c1)]
            p3 = self.trigrams[(c2, c1, c)] / tri_ctx if tri_ctx else 0.0
            bi_ctx = self.bigram_contexts[c1]
            p2 = self.bigrams[(c1, c)] / bi_ctx if bi_ctx else 0.0
            p1 = self.unigrams[c] / self.total
            out.append(math.log(w3 * p3 + w2 * p2 + w1 * p1 + w0 * uniform))
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "TrigramModel":
        return pickle.loads(Path(path).read_bytes())


def _resolve_device(device: str | None) -> str:
    """Pick the best available torch device, preferring Apple Silicon's GPU."""
    import torch

    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenModel:
    """HuggingFace causal LM scored at character granularity.

    Each token's log-probability is divided by the number of characters the
    token spans in the original text, and that per-character share is assigned
    to every one of those characters.

    Token-to-character alignment uses the fast tokenizer's offset mapping, so
    attribution is exact even for multi-byte characters and leading spaces.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise ValueError(f"{model_name} needs a fast tokenizer for offset mapping")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype)
        )
        self.model.to(self.device)
        self.model.eval()

        # Qwen has no BOS token, so <|endoftext|> stands in as the priming
        # context that lets the first real token receive a log-probability.
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if bos is None or bos < 0:
            bos = self.tokenizer.eos_token_id
        self.bos_token_id = bos

    def char_log_probs(self, text: str) -> list[float]:
        import torch

        if not text:
            return []

        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        if not token_ids:
            return [float("nan")] * len(text)

        input_ids = torch.tensor([[self.bos_token_id] + list(token_ids)], device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids).logits

        # Row i of logits predicts token i+1, so rows 0..n-1 line up with the n
        # real tokens that follow the prepended BOS. Log-softmax in float32:
        # bfloat16 loses too much precision for meaningful log-probabilities.
        logits = logits[0, :-1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(token_ids, device=self.device)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).tolist()

        # Characters no token covers (rare, e.g. dropped control characters)
        # keep NaN rather than being silently credited to a neighbour.
        out = [float("nan")] * len(text)
        for (start, end), token_log_prob in zip(offsets, token_log_probs, strict=True):
            span = end - start
            if span <= 0:
                continue
            per_char = token_log_prob / span
            for j in range(start, end):
                out[j] = per_char
        return out
