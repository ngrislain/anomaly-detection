"""Character-level log-probability models.

A model exposes a single method, `char_log_probs(text) -> list[float]`, giving
one log-probability per character of `text`, each conditioned on the preceding
characters. Two implementations are provided:

- `NgramModel`: a character n-gram model with interpolated smoothing, for any
  order `n`.
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

# Mass reserved for the uniform term, and the factor by which each successive
# (lower) order's weight shrinks, when default interpolation weights are built.
_UNIFORM_WEIGHT = 0.01
_ORDER_DECAY = 0.3

# The base (non-post-trained) variant: its distribution is not skewed by chat
# post-training, which makes it the better likelihood estimator for scoring.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-0.8B-Base"


@runtime_checkable
class CharLogProbModel(Protocol):
    """Anything that can score each character of a text."""

    def char_log_probs(self, text: str) -> list[float]:
        """Return one log-probability per character of `text`."""
        ...


def default_weights(n: int) -> tuple[float, ...]:
    """Interpolation weights for an order-`n` model, highest order first.

    The top order takes most of the mass and each lower order takes a shrinking
    share, with a small amount held back for the uniform term. Returns `n + 1`
    weights summing to 1: one per order from `n` down to 1, then uniform.
    """
    raw = [_ORDER_DECAY**i for i in range(n)]
    scale = (1.0 - _UNIFORM_WEIGHT) / sum(raw)
    return tuple(r * scale for r in raw) + (_UNIFORM_WEIGHT,)


class NgramModel:
    """Character n-gram model with interpolated (Jelinek-Mercer) smoothing.

    P(c | previous n-1 characters) mixes the order-n, order-(n-1), ... , unigram
    and uniform estimates, so no character ever receives zero probability --
    including characters never seen during training.

    Args:
        n: order of the model. `n=3` is a trigram, `n=1` a plain unigram.
        weights: `n + 1` interpolation weights summing to 1, highest order
            first and uniform last. Defaults to `default_weights(n)`.
    """

    def __init__(self, n: int = 5, weights: tuple[float, ...] | None = None):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if weights is None:
            weights = default_weights(n)
        if len(weights) != n + 1:
            raise ValueError(f"expected {n + 1} weights for n={n}, got {len(weights)}")
        if not math.isclose(sum(weights), 1.0):
            raise ValueError(f"weights must sum to 1, got {sum(weights)}")
        self.n = n
        self.weights = tuple(weights)
        # counts[k] and contexts[k] hold the order-(k+1) statistics, i.e. those
        # conditioned on a context of k preceding characters.
        self.counts: list[Counter[tuple[str, str]]] = [Counter() for _ in range(n)]
        self.contexts: list[Counter[str]] = [Counter() for _ in range(n)]
        self.total = 0
        self.vocab: set[str] = set()

    def fit(self, texts: str | list[str]) -> "NgramModel":
        """Accumulate counts from one or more training texts."""
        if isinstance(texts, str):
            texts = [texts]
        pad = BOT * (self.n - 1)
        for text in texts:
            padded = pad + text
            self.vocab.update(text)
            for i in range(self.n - 1, len(padded)):
                char = padded[i]
                for k in range(self.n):
                    context = padded[i - k : i]
                    self.counts[k][(context, char)] += 1
                    self.contexts[k][context] += 1
                self.total += 1
        return self

    @property
    def vocab_size(self) -> int:
        """Training vocabulary plus one slot for unseen characters."""
        return len(self.vocab) + 1

    def char_log_probs(self, text: str) -> list[float]:
        if self.total == 0:
            raise RuntimeError(f"{type(self).__name__} must be fitted before scoring")
        n, weights = self.n, self.weights
        uniform = weights[n] / self.vocab_size
        padded = BOT * (n - 1) + text
        out = []
        for i in range(n - 1, len(padded)):
            char = padded[i]
            probability = uniform
            for k in range(n):
                context = padded[i - k : i]
                seen = self.contexts[k][context]
                if seen:
                    # weights are highest-order first, contexts shortest first
                    probability += weights[n - 1 - k] * self.counts[k][(context, char)] / seen
            out.append(math.log(probability))
        return out

    def top_ngrams(self, k: int = 20) -> list[tuple[str, int, float]]:
        """The `k` most frequent order-n grams, as (gram, count, frequency).

        Frequency is the share of all scored positions. Grams touching the
        start-of-text sentinel are left out: they are an artifact of padding
        rather than anything the corpus actually contains.
        """
        if self.total == 0:
            raise RuntimeError(f"{type(self).__name__} must be fitted before inspection")
        grams = (
            (context + char, count)
            for (context, char), count in self.counts[self.n - 1].items()
            if BOT not in context and char != BOT
        )
        # Sort by descending count, then by the gram itself so ties are stable.
        top = sorted(grams, key=lambda item: (-item[1], item[0]))[:k]
        return [(gram, count, count / self.total) for gram, count in top]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "NgramModel":
        return pickle.loads(Path(path).read_bytes())


def ewma(values: list[float], half_life: float) -> list[float]:
    """Causal exponentially weighted moving average over `values`.

    The weight of a sample `k` steps back is halved every `half_life` steps.
    Because it only looks backwards, a run of surprising characters ramps up
    over roughly `half_life` positions and decays again once it ends.
    Non-finite entries pass through untouched and do not update the average.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    decay = 0.5 ** (1.0 / half_life)
    out: list[float] = []
    state: float | None = None
    for value in values:
        if not math.isfinite(value):
            out.append(value)
            continue
        state = value if state is None else decay * state + (1.0 - decay) * value
        out.append(state)
    return out


class EwmaModel:
    """Wraps a model, smoothing its per-character log-probabilities.

    Character-level scores are spiky: a single unusual letter inside an
    otherwise ordinary word can outscore a genuinely foreign phrase. Smoothing
    trades that per-character precision for sensitivity to sustained regions.

    Note the smoothed values are no longer log-probabilities of anything -- they
    are a running average of them -- so they should be read as scores.
    """

    def __init__(self, model: CharLogProbModel, half_life: float = 5.0):
        self.model = model
        self.half_life = half_life

    def char_log_probs(self, text: str) -> list[float]:
        return ewma(self.model.char_log_probs(text), self.half_life)


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
