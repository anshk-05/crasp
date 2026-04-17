"""CLI for CRASP v2 Phase B iterative structured head pruning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from crasp_v2.eval.legacy import attach_retention, evaluate_loaded_model, load_baseline_metrics
from crasp_v2.pipeline.artifacts import PhaseArtifact, save_phase_artifact
from crasp_v2.pipeline.common import (
    ensure_dir,
    load_config_bundle,
    load_model_for_stage,
    serialize_json,
)
from crasp_v2.pruning.masks import build_layer_mask_map, rank_heads_for_pruning
from crasp_v2.pruning.runtime import AttentionHeadMasker


def _passes_thresholds(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    retention = metrics.get("retention", {})
    return (
        float(retention.get("clinical_retention", 0.0)) >= float(config["clinical_retention_threshold"])
        and float(retention.get("safety_retention", 0.0)) >= float(config["safety_retention_threshold"])
    )


def _mean_retention(metrics: dict[str, Any]) -> float:
    retention = metrics.get("retention") or {}
    return float(retention.get("mean_retention", 0.0))


def _effective_sparsity(pruned_heads: int, total_heads: int) -> float:
    return pruned_heads / total_heads if total_heads else 0.0


def build_candidate_schedule(
    total_heads: int,
    target_sparsities: list[float],
    step_fraction: float,
) -> list[dict[str, int | float]]:
    """Build monotonically advancing pruning candidates for all target sparsities."""
    if total_heads <= 0:
        return []
    step_heads = max(1, int(total_heads * step_fraction))
    candidate_count = 0
    candidates: list[dict[str, int | float]] = []
    for target_sparsity in target_sparsities:
        target_pruned = int(total_heads * float(target_sparsity))
        while candidate_count < target_pruned:
            candidate_count = min(candidate_count + step_heads, target_pruned)
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "target_sparsity": float(target_sparsity),
                    "pruned_heads": candidate_count,
                    "effective_head_sparsity": _effective_sparsity(candidate_count, total_heads),
                }
            )
    return candidates


def select_final_candidate(iteration_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the highest-sparsity candidate that passed both retention gates."""
    accepted = [row for row in iteration_rows if row.get("accepted")]
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda row: (
            int(row.get("pruned_heads", 0)),
            _mean_retention(row.get("metrics", {})),
        ),
    )


def _threshold_summary(pruning_cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "clinical_retention_threshold": float(pruning_cfg["clinical_retention_threshold"]),
        "safety_retention_threshold": float(pruning_cfg["safety_retention_threshold"]),
    }


def run(args: argparse.Namespace) -> Path:
    import json

    config, paths = load_config_bundle(args.config)
    pruning_cfg = config["pruning"]
    baseline_metrics = load_baseline_metrics(Path(args.baseline_metrics or config["artifacts"]["baseline_metrics"]))
    saliency_report = json.loads(args.saliency_report.read_text(encoding="utf-8"))

    model, tokenizer, maskers, resolved_model_name = load_model_for_stage(
        args.stage,
        model_cfg=config["model"],
        cache_dir=paths["model_cache"],
    )
    for masker in maskers:
        masker.remove()
    maskers.clear()

    ranked_heads = rank_heads_for_pruning(saliency_report["combined"])
    total_heads_per_layer = {
        layer_name: int(num_heads)
        for layer_name, num_heads in saliency_report["total_heads_per_layer"].items()
    }
    total_heads = sum(total_heads_per_layer.values())
    candidate_schedule = build_candidate_schedule(
        total_heads=total_heads,
        target_sparsities=[float(value) for value in pruning_cfg["target_sparsities"]],
        step_fraction=float(pruning_cfg["step_fraction"]),
    )

    output_dir = ensure_dir(args.output_dir)
    iteration_rows: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] | None = None

    try:
        for candidate in candidate_schedule:
            current_pruned = ranked_heads[: int(candidate["pruned_heads"])]
            layer_masks = build_layer_mask_map(total_heads_per_layer, current_pruned)
            current_masker = AttentionHeadMasker(model, layer_masks)
            current_masker.apply()
            try:
                metrics = evaluate_loaded_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=resolved_model_name,
                    device=str(config["model"].get("device", "cuda")),
                    batch_size=int(config["eval"].get("batch_size", 8)),
                    max_length=int(config["model"].get("max_seq_len", 2048)),
                    num_samples=config["eval"].get("num_samples"),
                    scoring=str(config["eval"].get("scoring", "loglikelihood")),
                )
                metrics = attach_retention(metrics, baseline_metrics)
            finally:
                current_masker.remove()

            passes_thresholds = _passes_thresholds(metrics, pruning_cfg)
            row = {
                **candidate,
                "passes_thresholds": passes_thresholds,
                "accepted": passes_thresholds,
                "metrics": metrics,
            }
            iteration_rows.append(row)
            last_metrics = metrics

        selected = select_final_candidate(iteration_rows)
        selected_heads = ranked_heads[: int(selected["pruned_heads"])] if selected else []
        selected_metrics = selected["metrics"] if selected else None
        final_masks = build_layer_mask_map(total_heads_per_layer, selected_heads)
        mask_path = serialize_json(final_masks, output_dir / "mask.json")
        metrics_payload: dict[str, Any] = selected_metrics or {
            "status": "no_accepted_candidate",
            "baseline_metrics": baseline_metrics,
            "last_evaluated_metrics": last_metrics,
        }
        metrics_path = serialize_json(metrics_payload, output_dir / "metrics.json")
        serialize_json(
            {
                "selection_policy": "highest_sparsity_passing_dual_retention_gate",
                "thresholds": _threshold_summary(pruning_cfg),
                "accepted_candidate_id": selected.get("candidate_id") if selected else None,
                "accepted_pruned_heads": len(selected_heads),
                "accepted_effective_head_sparsity": _effective_sparsity(len(selected_heads), total_heads),
                "status": "accepted" if selected else "no_accepted_candidate",
                "iterations": iteration_rows,
            },
            output_dir / "pruning_iterations.json",
        )
        artifact = PhaseArtifact.create(
            phase="post_prune" if selected else "post_prune_failed",
            model_name=resolved_model_name,
            base_model=config["model"]["name"],
            parent=args.stage,
            mask_path=str(mask_path),
            metrics_path=str(metrics_path),
            extra={
                "saliency_report": str(args.saliency_report),
                "selection_policy": "highest_sparsity_passing_dual_retention_gate",
                "thresholds": _threshold_summary(pruning_cfg),
                "accepted": selected is not None,
                "pruned_heads": len(selected_heads),
                "total_heads": total_heads,
                "effective_head_sparsity": _effective_sparsity(len(selected_heads), total_heads),
                "accepted_candidate_id": selected.get("candidate_id") if selected else None,
            },
        )
        save_phase_artifact(artifact, output_dir)
        return output_dir
    finally:
        for masker in maskers:
            masker.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRASP v2 iterative structured head pruning.")
    parser.add_argument("--config", type=Path, default=Path("crasp_v2/config/default_runpod.yaml"))
    parser.add_argument("--stage", default="/workspace/crasp-vol/checkpoints/post_sft")
    parser.add_argument("--saliency-report", type=Path, default=Path("/workspace/crasp-vol/results/crasp_v2/saliency/saliency_report.json"))
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/crasp-vol/checkpoints/post_prune"))
    return parser.parse_args()


def main() -> None:
    output_dir = run(parse_args())
    print(f"Saved post-prune stage to {output_dir}")


if __name__ == "__main__":
    main()
