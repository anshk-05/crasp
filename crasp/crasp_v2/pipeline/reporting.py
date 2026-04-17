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


def build_comparison_rows(
    results_root: Path,
    crasp_metrics: dict[str, Any],
    crasp_effective_head_sparsity: float | None = None,
) -> list[dict[str, Any]]:
    """Create comparison rows for CRASP and available baselines."""
    rows = [
        {
            "method": "CRASP v2",
            "sparsity": crasp_effective_head_sparsity,
            "sparsity_type": "effective_head_sparsity",
            "clinical_accuracy": crasp_metrics.get("clinical_accuracy"),
            "safety_score": crasp_metrics.get("safety_score"),
            "clinical_retention": crasp_metrics.get("retention", {}).get("clinical_retention"),
            "safety_retention": crasp_metrics.get("retention", {}).get("safety_retention"),
            "mean_retention": crasp_metrics.get("retention", {}).get("mean_retention"),
            "confidence_intervals": crasp_metrics.get("confidence_intervals", {}),
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
                "sparsity": best.get("sparsity"),
                "sparsity_type": "weight_sparsity",
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
    if not rows:
        return "No accepted CRASP pruning candidate; comparison table skipped."
    headers = [
        "Method",
        "Sparsity",
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
        sparsity = row.get("sparsity")
        sparsity_text = "n/a" if sparsity is None else f"{float(sparsity):.2%}"
        lines.append(
            "| {method} | {sparsity} | {clinical_accuracy:.4f} | {safety_score:.4f} | {clinical_retention:.4f} | {safety_retention:.4f} | {mean_retention:.4f} |".format(
                method=row["method"],
                sparsity=sparsity_text,
                clinical_accuracy=float(row.get("clinical_accuracy") or 0.0),
                safety_score=float(row.get("safety_score") or 0.0),
                clinical_retention=float(row.get("clinical_retention") or 0.0),
                safety_retention=float(row.get("safety_retention") or 0.0),
                mean_retention=float(row.get("mean_retention") or 0.0),
            )
        )
    return "\n".join(lines)


def render_run_report(summary: dict[str, Any]) -> str:
    """Render a compact human-readable report for a completed CRASP run."""
    status = summary.get("status", "unknown")
    lines = [
        "# CRASP v2 First-Result Report",
        "",
        f"Status: `{status}`",
        "",
        "CRASP sparsity is reported as effective attention-head sparsity from runtime masks. "
        "It is not a measured edge-latency or compact-export result.",
    ]
    if summary.get("post_prune_status") == "no_accepted_candidate":
        lines.extend(
            [
                "",
                "No pruning candidate passed both clinical and safety retention gates. "
                "The run produced diagnostics only and should not be reported as a successful pruned checkpoint.",
            ]
        )
    markdown = summary.get("comparison_markdown")
    if markdown:
        lines.extend(["", "## Comparison", "", markdown])
    rows = summary.get("comparison_rows", [])
    crasp = rows[0] if rows else {}
    ci = crasp.get("confidence_intervals", {}) if isinstance(crasp, dict) else {}
    if ci:
        lines.extend(["", "## Confidence Intervals", ""])
        for metric_name, interval in ci.items():
            if not isinstance(interval, dict):
                continue
            lines.append(
                "- {metric}: mean={mean:.4f}, 95% CI [{lower:.4f}, {upper:.4f}]".format(
                    metric=metric_name,
                    mean=float(interval.get("mean", 0.0)),
                    lower=float(interval.get("lower", 0.0)),
                    upper=float(interval.get("upper", 0.0)),
                )
            )
    return "\n".join(lines) + "\n"
