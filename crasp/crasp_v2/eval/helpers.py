"""Pure evaluation helpers used by CRASP v2 and tests."""

from __future__ import annotations

import re
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
