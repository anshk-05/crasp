"""src/metrics.py
─────────────────────────────────────────────────────────────────────────────
Core metric calculations for CRASP (Clinical Reasoning-Aware Structured
Pruning).

All functions are *pure* — no I/O, no model loading, no side effects — so
they can be imported and unit-tested in isolation from the evaluation harness.

Metrics
-------
  clinical_accuracy        : Exact-match accuracy on MedQA USMLE-4 options.
  safety_score             : Macro-averaged accuracy across Med-HALT task
                             categories (reasoning vs. memory hallucination).
  retention_score          : pruned_score / raw_baseline_score, clamped to
                             [0, 1].  This is the *raw-reference fix* — a
                             pruned model that exactly matches the raw model
                             earns retention = 1.0 regardless of the raw
                             model's absolute score.
  compute_retention_report : Full retention breakdown for a pruned vs. raw
                             comparison.

Dataclasses
-----------
  RetentionReport  : Per-metric retention + mean.
  CRASPMetrics     : All scores for one model evaluation run.

Serialisation
-------------
  metrics_to_json  : CRASPMetrics → JSON string (round-trippable for Phase 3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ── Task-type grouping constants ──────────────────────────────────────────────

# Fine-grained Med-HALT task names that belong to the reasoning-hallucination
# high-level category.
REASONING_HALLUCINATION_TASKS: frozenset[str] = frozenset(
    {
        "reasoning_FCT",
        "reasoning_nota",
        "reasoning_fake",
        "reasoning_hallucination",
    }
)

# Fine-grained Med-HALT task names that belong to the memory-hallucination
# high-level category.
MEMORY_HALLUCINATION_TASKS: frozenset[str] = frozenset(
    {
        "memory_hallucination",
        "memory_FCT",
        "memory_fake",
        "memory_nota",
    }
)

# Canonical category name strings used in breakdown dicts throughout CRASP.
SAFETY_CATEGORY_REASONING: str = "reasoning_hallucination"
SAFETY_CATEGORY_MEMORY: str = "memory_hallucination"


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class RetentionReport:
    """Retention scores comparing a pruned model against the raw baseline.

    Attributes:
        clinical_retention: Fraction of raw clinical accuracy retained.
            A pruned model matching the raw model exactly yields 1.0.
        safety_retention: Fraction of raw safety score retained.
        mean_retention: Simple mean of clinical_retention and safety_retention.
    """

    clinical_retention: float
    safety_retention: float
    mean_retention: float


@dataclass
class CRASPMetrics:
    """Bundles all evaluation scores for a single CRASP model run.

    Attributes:
        clinical_accuracy: Exact-match accuracy on MedQA USMLE-4 (0.0–1.0).
        safety_score: Macro-averaged accuracy across Med-HALT categories
            (0.0–1.0).
        safety_breakdown: Per-category accuracy dict.  Keys are the canonical
            category names ``"reasoning_hallucination"``,
            ``"memory_hallucination"``, and ``"macro_avg"``.
        retention: Retention against the raw baseline.  ``None`` when *this*
            evaluation IS the raw baseline (nothing to compare against yet).
        model_name: HuggingFace model ID or local path used for this run.
        sparsity: Fraction of weights pruned (0.0 for the unpruned baseline).
        timestamp: ISO-8601 UTC timestamp recorded at evaluation time.
    """

    clinical_accuracy: float
    safety_score: float
    safety_breakdown: dict[str, float]
    retention: Optional[RetentionReport]
    model_name: str
    sparsity: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Metric functions ──────────────────────────────────────────────────────────


def clinical_accuracy(
    predictions: list[str],
    ground_truth: list[str],
) -> float:
    """Compute exact-match accuracy for USMLE-style 4-option questions.

    Comparison is case-insensitive after stripping surrounding whitespace.
    Both inputs must be non-empty and of equal length.

    Args:
        predictions: Model-selected answer strings — either single letters
            (``"A"``, ``"D"``) or full option text.  Format must be consistent
            with ``ground_truth``.
        ground_truth: Correct answer strings in the same format as
            ``predictions``.

    Returns:
        Exact-match accuracy as a float in ``[0.0, 1.0]``.

    Raises:
        ValueError: If either list is empty, or if the lengths differ.

    Example:
        >>> clinical_accuracy(["A", "B", "A"], ["A", "A", "A"])
        0.6666666666666666
    """
    if not predictions or not ground_truth:
        raise ValueError(
            "predictions and ground_truth must be non-empty lists."
        )
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground-truth labels."
        )

    correct = sum(
        p.strip().upper() == g.strip().upper()
        for p, g in zip(predictions, ground_truth)
    )
    return correct / len(ground_truth)


def safety_score(
    predictions: list[str],
    ground_truth: list[str],
    task_types: list[str],
) -> dict[str, float]:
    """Compute macro-averaged safety accuracy across Med-HALT task categories.

    Fine-grained Med-HALT task types (``reasoning_FCT``, ``reasoning_nota``,
    ``reasoning_fake``) are grouped into the ``"reasoning_hallucination"``
    high-level category.  Any ``memory_*`` tasks are grouped into
    ``"memory_hallucination"``.  The macro average is computed over only the
    *categories that are actually present* in the supplied data.

    Args:
        predictions: Model output strings (answer letters or full text).
        ground_truth: Correct answer strings matching the format of
            ``predictions``.
        task_types: Per-sample Med-HALT task-type label strings.  Each element
            must be one of the known fine-grained task names or one of the
            two high-level category names.

    Returns:
        A dict with four keys:

        ``"reasoning_hallucination"``
            Accuracy over reasoning-type samples.  ``0.0`` if no such samples
            are present in the batch.
        ``"memory_hallucination"``
            Accuracy over memory-type samples.  ``0.0`` if absent.
        ``"macro_avg"``
            Mean accuracy computed only over *present* high-level categories
            (avoids penalising for absent categories).

    Raises:
        ValueError: If any list is empty or the three lists have different
            lengths.

    Example:
        >>> preds = ["A", "B", "C"]
        >>> gt    = ["A", "A", "C"]
        >>> types = ["reasoning_FCT", "reasoning_nota", "memory_hallucination"]
        >>> safety_score(preds, gt, types)
        {'reasoning_hallucination': 0.5, 'memory_hallucination': 1.0,
         'macro_avg': 0.75}
    """
    if not predictions or not ground_truth or not task_types:
        raise ValueError(
            "predictions, ground_truth, and task_types must all be non-empty."
        )
    if len(predictions) != len(ground_truth) or len(predictions) != len(task_types):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"ground_truth={len(ground_truth)}, "
            f"task_types={len(task_types)}."
        )

    # Bucket each sample into the appropriate high-level category.
    buckets: dict[str, list[tuple[str, str]]] = {
        SAFETY_CATEGORY_REASONING: [],
        SAFETY_CATEGORY_MEMORY: [],
    }
    for pred, gt, tt in zip(predictions, ground_truth, task_types):
        tt_lower = tt.lower()
        if tt in REASONING_HALLUCINATION_TASKS or (
            "reasoning" in tt_lower and "memory" not in tt_lower
        ):
            buckets[SAFETY_CATEGORY_REASONING].append((pred, gt))
        elif tt in MEMORY_HALLUCINATION_TASKS or "memory" in tt_lower:
            buckets[SAFETY_CATEGORY_MEMORY].append((pred, gt))
        else:
            # Unrecognised type — treat as reasoning hallucination by default.
            buckets[SAFETY_CATEGORY_REASONING].append((pred, gt))

    # Compute per-category accuracy, tracking which categories are present.
    category_scores: dict[str, float] = {}
    for category, pairs in buckets.items():
        if pairs:
            preds_cat, gts_cat = zip(*pairs)
            category_scores[category] = clinical_accuracy(
                list(preds_cat), list(gts_cat)
            )

    reasoning_acc = category_scores.get(SAFETY_CATEGORY_REASONING, 0.0)
    memory_acc = category_scores.get(SAFETY_CATEGORY_MEMORY, 0.0)

    present_scores = list(category_scores.values())
    macro = sum(present_scores) / len(present_scores) if present_scores else 0.0

    return {
        SAFETY_CATEGORY_REASONING: reasoning_acc,
        SAFETY_CATEGORY_MEMORY: memory_acc,
        "macro_avg": macro,
    }


def retention_score(pruned_score: float, raw_baseline_score: float) -> float:
    """Compute how much of the raw model's performance a pruned model retains.

    Uses the *raw-reference fix*: the denominator is the raw (unpruned)
    model's actual score, **not** a perfect score of 1.0.  Concretely:

    * A pruned model that matches the raw model exactly → ``retention = 1.0``,
      even if the raw model only scored 0.72.
    * A pruned model that exceeds the raw model → clamped to ``1.0``.
    * Division by zero (``raw_baseline_score == 0``) → returns ``0.0``.

    Args:
        pruned_score: Metric value (accuracy) for the pruned model.
        raw_baseline_score: Metric value for the raw, unpruned model.

    Returns:
        Retention as a float clamped to ``[0.0, 1.0]``.

    Example:
        >>> retention_score(0.65, 0.72)  # pruned recovered 90.3 % of raw
        0.9027777777777778
        >>> retention_score(0.72, 0.72)  # perfect match
        1.0
        >>> retention_score(0.80, 0.72)  # pruned exceeds raw → clamped
        1.0
    """
    if raw_baseline_score == 0.0:
        return 0.0
    return max(0.0, min(1.0, pruned_score / raw_baseline_score))


def compute_retention_report(
    raw_scores: dict[str, float],
    pruned_scores: dict[str, float],
) -> RetentionReport:
    """Build a full RetentionReport comparing a pruned model to the raw baseline.

    Both dicts must contain at minimum the keys ``"clinical_accuracy"`` and
    ``"safety_score"``.  These are exactly the keys produced by
    ``CRASPMetrics`` and the output of ``evaluate_all()`` in the harness.

    Args:
        raw_scores: Metric dict from evaluating the raw (unpruned) model.
            Required keys: ``"clinical_accuracy"``, ``"safety_score"``.
        pruned_scores: Metric dict from evaluating the pruned model.
            Same required keys.

    Returns:
        A :class:`RetentionReport` with per-metric retention and their mean.

    Raises:
        KeyError: If either dict is missing ``"clinical_accuracy"`` or
            ``"safety_score"``.

    Example:
        >>> raw    = {"clinical_accuracy": 0.72, "safety_score": 0.68}
        >>> pruned = {"clinical_accuracy": 0.65, "safety_score": 0.66}
        >>> r = compute_retention_report(raw, pruned)
        >>> round(r.clinical_retention, 4)
        0.9028
    """
    for key in ("clinical_accuracy", "safety_score"):
        if key not in raw_scores:
            raise KeyError(f"raw_scores is missing required key: '{key}'")
        if key not in pruned_scores:
            raise KeyError(f"pruned_scores is missing required key: '{key}'")

    clinical_ret = retention_score(
        pruned_scores["clinical_accuracy"], raw_scores["clinical_accuracy"]
    )
    safety_ret = retention_score(
        pruned_scores["safety_score"], raw_scores["safety_score"]
    )
    return RetentionReport(
        clinical_retention=clinical_ret,
        safety_retention=safety_ret,
        mean_retention=(clinical_ret + safety_ret) / 2.0,
    )


def metrics_to_json(metrics: CRASPMetrics) -> str:
    """Serialise a :class:`CRASPMetrics` object to a JSON string.

    The :class:`RetentionReport` (if present) is inlined as a nested object.
    The output is round-trippable — a Phase 3 pruning script can ``json.loads``
    the result and access all fields directly.

    Args:
        metrics: The :class:`CRASPMetrics` instance to serialise.

    Returns:
        A pretty-printed JSON string with 2-space indentation.

    Example:
        >>> m = CRASPMetrics(
        ...     clinical_accuracy=0.42,
        ...     safety_score=0.68,
        ...     safety_breakdown={"reasoning_hallucination": 0.68,
        ...                       "memory_hallucination": 0.0,
        ...                       "macro_avg": 0.68},
        ...     retention=None,
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     sparsity=0.0,
        ... )
        >>> "clinical_accuracy" in metrics_to_json(m)
        True
    """
    data = asdict(metrics)
    return json.dumps(data, indent=2)
