"""Pure evaluation helpers used by CRASP v2 and tests."""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Optional


def extract_answer_letter(text: str, valid_letters: set[str]) -> Optional[str]:
    """Extract the first valid answer letter from a model completion."""
    cleaned = text.strip().upper()

    if cleaned in valid_letters:
        return cleaned

    explicit = re.search(r"(?:ANSWER|SELECT|CHOICE|OPTION)[:\s]+([A-Z])", cleaned)
    if explicit and explicit.group(1) in valid_letters:
        return explicit.group(1)

    for match in re.finditer(r"\b([A-Z])\b", cleaned):
        if match.group(1) in valid_letters:
            return match.group(1)

    for char in cleaned:
        if char in valid_letters:
            return char

    return None


def _safe_retention(pruned_score: float, baseline_score: float) -> float:
    """Return ratio retention, clamped to [0, 1]."""
    if baseline_score <= 0:
        return 1.0 if pruned_score >= baseline_score else 0.0
    return max(0.0, min(1.0, pruned_score / baseline_score))


def compute_retention(
    raw_clinical: float,
    raw_safety: float,
    new_clinical: float,
    new_safety: float,
) -> dict[str, float]:
    """Compute the same retention fields CRASP uses for pruning gates."""
    clinical_retention = _safe_retention(new_clinical, raw_clinical)
    safety_retention = _safe_retention(new_safety, raw_safety)
    return {
        "clinical_retention": clinical_retention,
        "safety_retention": safety_retention,
        "mean_retention": (clinical_retention + safety_retention) / 2,
    }


def prediction_diagnostics(predictions: list[str], valid_letters: set[str]) -> dict[str, object]:
    """Summarize prediction quality for multiple-choice evaluation."""
    normalized = [str(prediction).strip().upper() for prediction in predictions]
    invalid = [prediction for prediction in normalized if prediction not in valid_letters]
    return {
        "prediction_distribution": dict(sorted(Counter(normalized).items())),
        "invalid_prediction_rate": len(invalid) / len(normalized) if normalized else 0.0,
        "invalid_predictions": len(invalid),
    }


def bootstrap_mean_ci(
    values: list[float | int | bool],
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Return a deterministic bootstrap confidence interval for a sample mean."""
    numeric = [float(value) for value in values]
    if not numeric:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "confidence": confidence}

    rng = random.Random(seed)
    n = len(numeric)
    means: list[float] = []
    for _ in range(num_resamples):
        sample = [numeric[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lower_idx = int((alpha / 2.0) * (num_resamples - 1))
    upper_idx = int((1.0 - alpha / 2.0) * (num_resamples - 1))
    return {
        "mean": sum(numeric) / n,
        "lower": means[lower_idx],
        "upper": means[upper_idx],
        "confidence": confidence,
    }
