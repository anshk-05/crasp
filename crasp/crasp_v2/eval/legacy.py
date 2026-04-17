"""Thin wrappers around the existing evaluator and metric stack."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .helpers import compute_retention


def load_baseline_metrics(path: Path) -> dict[str, Any]:
    """Load a baseline metrics JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    num_samples: int | None = None,
    scoring: str = "loglikelihood",
) -> dict[str, Any]:
    """Evaluate an already loaded model using the legacy evaluator."""
    from src.eval_harness import CRASPEvaluator  # type: ignore[import]
    from src.metrics import CRASPMetrics  # type: ignore[import]

    evaluator = CRASPEvaluator.from_model(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    medqa_result = evaluator.evaluate_medqa(num_samples=num_samples, scoring=scoring)
    medhalt_result = evaluator.evaluate_medhalt(num_samples=num_samples, scoring=scoring)
    metrics = CRASPMetrics(
        clinical_accuracy=medqa_result["clinical_accuracy"],
        safety_score=medhalt_result["safety_score"],
        safety_breakdown=medhalt_result["safety_breakdown"],
        retention=None,
        model_name=model_name,
        sparsity=0.0,
    )
    payload = asdict(metrics)
    payload["eval_details"] = {
        "scoring": scoring,
        "medqa": {
            "num_samples": medqa_result.get("num_samples"),
            "diagnostics": medqa_result.get("diagnostics", {}),
            "confidence_interval": medqa_result.get("confidence_interval", {}),
        },
        "medhalt": {
            "scope": medhalt_result.get("safety_scope"),
            "num_samples": medhalt_result.get("num_samples"),
            "per_task": medhalt_result.get("per_task", {}),
            "diagnostics": medhalt_result.get("diagnostics", {}),
            "confidence_interval": medhalt_result.get("confidence_interval", {}),
        },
    }
    payload["confidence_intervals"] = {
        "clinical_accuracy": medqa_result.get("confidence_interval", {}),
        "safety_score": medhalt_result.get("confidence_interval", {}),
    }
    return payload


def attach_retention(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Attach CRASP retention fields relative to the raw baseline."""
    metrics = dict(metrics)
    metrics["retention"] = compute_retention(
        raw_clinical=float(baseline["clinical_accuracy"]),
        raw_safety=float(baseline["safety_score"]),
        new_clinical=float(metrics["clinical_accuracy"]),
        new_safety=float(metrics["safety_score"]),
    )
    return metrics
