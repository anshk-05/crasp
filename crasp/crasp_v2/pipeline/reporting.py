"""Reporting helpers for CRASP v2 pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _latest_summary(directory: Path) -> dict[str, Any] | None:
    summaries = sorted(directory.glob("summary_*.json"))
    if not summaries:
        return None
    return json.loads(summaries[-1].read_text(encoding="utf-8"))


def _best_run(summary: dict[str, Any]) -> dict[str, Any] | None:
    runs = summary.get("runs", [])
    if not runs:
        return None
    return max(
        runs,
        key=lambda run: (
            float(run.get("retention", {}).get("mean_retention", 0.0)),
            float(run.get("clinical_accuracy", 0.0)),
        ),
    )


def build_comparison_rows(results_root: Path, crasp_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Create comparison rows for CRASP and available baselines."""
    rows = [
        {
            "method": "CRASP v2",
            "clinical_accuracy": crasp_metrics.get("clinical_accuracy"),
            "safety_score": crasp_metrics.get("safety_score"),
            "clinical_retention": crasp_metrics.get("retention", {}).get("clinical_retention"),
            "safety_retention": crasp_metrics.get("retention", {}).get("safety_retention"),
            "mean_retention": crasp_metrics.get("retention", {}).get("mean_retention"),
        }
    ]
    for method in ["wanda", "sparsegpt", "llmpruner"]:
        summary = _latest_summary(results_root / method)
        if not summary:
            continue
        best = _best_run(summary)
        if not best:
            continue
        rows.append(
            {
                "method": method,
                "clinical_accuracy": best.get("clinical_accuracy"),
                "safety_score": best.get("safety_score"),
                "clinical_retention": best.get("retention", {}).get("clinical_retention"),
                "safety_retention": best.get("retention", {}).get("safety_retention"),
                "mean_retention": best.get("retention", {}).get("mean_retention"),
            }
        )
    return rows


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render comparison rows as a compact Markdown table."""
    headers = [
        "Method",
        "Clinical Acc",
        "Safety",
        "Clinical Ret",
        "Safety Ret",
        "Mean Ret",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {clinical_accuracy:.4f} | {safety_score:.4f} | {clinical_retention:.4f} | {safety_retention:.4f} | {mean_retention:.4f} |".format(
                method=row["method"],
                clinical_accuracy=float(row.get("clinical_accuracy") or 0.0),
                safety_score=float(row.get("safety_score") or 0.0),
                clinical_retention=float(row.get("clinical_retention") or 0.0),
                safety_retention=float(row.get("safety_retention") or 0.0),
                mean_retention=float(row.get("mean_retention") or 0.0),
            )
        )
    return "\n".join(lines)
