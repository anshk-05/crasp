"""CLI for CRASP v2 Phase 0 mixed LoRA SFT."""

from __future__ import annotations

import argparse
from pathlib import Path

from crasp_v2.eval.legacy import attach_retention, evaluate_loaded_model, load_baseline_metrics
from crasp_v2.pipeline.artifacts import PhaseArtifact, save_phase_artifact
from crasp_v2.pipeline.common import (
    ensure_dir,
    load_config_bundle,
    load_model_for_stage,
    load_training_texts,
    serialize_json,
)
from crasp_v2.pipeline.training import attach_lora, run_lora_training


def _resolve_metrics_path(candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    return path if path.is_absolute() else Path.cwd() / path


def run(args: argparse.Namespace) -> Path:
    config, paths = load_config_bundle(args.config)
    training_cfg = dict(config.get("training", {}))
    training_cfg["max_seq_len"] = int(config.get("model", {}).get("max_seq_len", 2048))

    model, tokenizer, maskers, resolved_model_name = load_model_for_stage(
        args.base_stage or config["model"]["name"],
        model_cfg=config["model"],
        cache_dir=paths["model_cache"],
    )
    try:
        train_texts = load_training_texts([args.mixed_sft_path])
        model = attach_lora(model, training_cfg)
        output_dir = ensure_dir(args.output_dir)
        adapter_dir = output_dir / "adapter"
        run_lora_training(model, tokenizer, train_texts, training_cfg, adapter_dir)

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

        baseline_path = _resolve_metrics_path(args.baseline_metrics or config.get("artifacts", {}).get("baseline_metrics"))
        if baseline_path and baseline_path.exists():
            metrics = attach_retention(metrics, load_baseline_metrics(baseline_path))
        metrics_path = serialize_json(metrics, output_dir / "metrics.json")
        artifact = PhaseArtifact.create(
            phase="post_sft",
            model_name=resolved_model_name,
            base_model=config["model"]["name"],
            parent=args.base_stage or config["model"]["name"],
            adapter_path=str(adapter_dir),
            metrics_path=str(metrics_path),
            extra={"mixed_sft_path": str(args.mixed_sft_path)},
        )
        save_phase_artifact(artifact, output_dir)
        return output_dir
    finally:
        for masker in maskers:
            masker.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRASP v2 Phase 0 mixed LoRA SFT.")
    parser.add_argument("--config", type=Path, default=Path("crasp_v2/config/default_runpod.yaml"))
    parser.add_argument("--mixed-sft-path", type=Path, default=Path("data/calibration/mixed_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/crasp-vol/checkpoints/post_sft"))
    parser.add_argument("--base-stage", default=None)
    parser.add_argument("--baseline-metrics", default=None)
    return parser.parse_args()


def main() -> None:
    output_dir = run(parse_args())
    print(f"Saved post-SFT stage to {output_dir}")


if __name__ == "__main__":
    main()
