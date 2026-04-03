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
    step_heads = max(1, int(total_heads * float(pruning_cfg["step_fraction"])))

    output_dir = ensure_dir(args.output_dir)
    iteration_rows: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_pruned_heads: list[dict[str, Any]] = []
    violations = 0

    try:
        for target_sparsity in pruning_cfg["target_sparsities"]:
            target_pruned = int(total_heads * float(target_sparsity))
            while len(best_pruned_heads) < target_pruned:
                current_pruned = ranked_heads[: min(len(best_pruned_heads) + step_heads, target_pruned)]
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
                    )
                    metrics = attach_retention(metrics, baseline_metrics)
                finally:
                    current_masker.remove()

                row = {
                    "target_sparsity": target_sparsity,
                    "pruned_heads": len(current_pruned),
                    "metrics": metrics,
                }
                iteration_rows.append(row)
                if _passes_thresholds(metrics, pruning_cfg):
                    best_metrics = metrics
                    best_pruned_heads = list(current_pruned)
                    violations = 0
                else:
                    violations += 1
                    if violations >= int(pruning_cfg["early_stop_patience"]):
                        break
            if violations >= int(pruning_cfg["early_stop_patience"]):
                break

        final_masks = build_layer_mask_map(total_heads_per_layer, best_pruned_heads)
        mask_path = serialize_json(final_masks, output_dir / "mask.json")
        metrics_path = serialize_json(best_metrics or baseline_metrics, output_dir / "metrics.json")
        serialize_json({"iterations": iteration_rows}, output_dir / "pruning_iterations.json")
        artifact = PhaseArtifact.create(
            phase="post_prune",
            model_name=resolved_model_name,
            base_model=config["model"]["name"],
            parent=args.stage,
            mask_path=str(mask_path),
            metrics_path=str(metrics_path),
            extra={
                "saliency_report": str(args.saliency_report),
                "pruned_heads": len(best_pruned_heads),
                "total_heads": total_heads,
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
