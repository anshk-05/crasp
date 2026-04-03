"""CLI for evaluating any CRASP v2 stage or raw model."""

from __future__ import annotations

import argparse
from pathlib import Path

from crasp_v2.eval.legacy import attach_retention, evaluate_loaded_model, load_baseline_metrics
from crasp_v2.pipeline.common import ensure_dir, load_config_bundle, load_model_for_stage, serialize_json


def run(args: argparse.Namespace) -> Path:
    config, paths = load_config_bundle(args.config)
    model, tokenizer, maskers, resolved_model_name = load_model_for_stage(
        args.stage,
        model_cfg=config["model"],
        cache_dir=paths["model_cache"],
    )
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
        if args.baseline_metrics:
            metrics = attach_retention(metrics, load_baseline_metrics(Path(args.baseline_metrics)))
        output_dir = ensure_dir(args.output_dir)
        return serialize_json(metrics, output_dir / "eval_metrics.json")
    finally:
        for masker in maskers:
            masker.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CRASP v2 stage.")
    parser.add_argument("--config", type=Path, default=Path("crasp_v2/config/default_runpod.yaml"))
    parser.add_argument("--stage", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/crasp-vol/results/crasp_v2/eval"))
    return parser.parse_args()


def main() -> None:
    metrics_path = run(parse_args())
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
