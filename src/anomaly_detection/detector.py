"""Anomaly detection over text, at character granularity."""

from anomaly_detection.models import CharLogProbModel


def detect_anomalies(model: CharLogProbModel, text: str) -> list[tuple[str, float]]:
    """Score every character of `text` under `model`.

    Args:
        model: anything implementing `char_log_probs`, e.g. `NgramModel`
            or `QwenModel`.
        text: the text to score.

    Returns:
        One `(char, log_prob)` pair per character of `text`, where `log_prob`
        is the log-probability of that character given all preceding ones.
        Characters the model cannot attribute a score to carry NaN.
    """
    return list(zip(text, model.char_log_probs(text), strict=True))
