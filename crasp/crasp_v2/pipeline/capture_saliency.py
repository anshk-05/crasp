"""CLI for CRASP v2 Phase A head saliency capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from crasp_v2.pipeline.common import (
    ensure_dir,
    load_config_bundle,
    load_model_for_stage,
    load_tokenizer,
    serialize_json,
    build_token_batches,
)
from crasp_v2.saliency.modeling import AttentionHeadCollector, discover_attention_modules
from crasp_v2.saliency.stats import combine_serialized_layer_reports


def run(args: argparse.Namespace) -> Path:
    config, paths = load_config_bundle(args.config)
    model, tokenizer, maskers, _ = load_model_for_stage(
        args.stage,
        model_cfg=config["model"],
        cache_dir=paths["model_cache"],
    )
    try:
        if tokenizer is None:
            tokenizer = load_tokenizer(config["model"]["name"], cache_dir=paths["model_cache"])

        reasoning_batches = build_token_batches(
            records_path=args.reasoning_path,
            tokenizer=tokenizer,
            seq_len=int(config["model"]["max_seq_len"]),
            batch_size=int(config["calibration"]["batch_size"]),
        )
        safety_batches = build_token_batches(
            records_path=args.safety_path,
            tokenizer=tokenizer,
            seq_len=int(config["model"]["max_seq_len"]),
            batch_size=int(config["calibration"]["batch_size"]),
        )

        reasoning_report = AttentionHeadCollector(
            model=model,
            activation_epsilon=float(config["calibration"]["activation_epsilon"]),
        ).collect(reasoning_batches)
        safety_report = AttentionHeadCollector(
            model=model,
            activation_epsilon=float(config["calibration"]["activation_epsilon"]),
        ).collect(safety_batches)

        combined = {
            layer_name: combine_serialized_layer_reports(
                reasoning_report.get(layer_name, {}),
                safety_report.get(layer_name, {}),
                reasoning_weight=float(config["pruning"]["reasoning_weight"]),
                safety_weight=float(config["pruning"]["safety_weight"]),
            )
            for layer_name in sorted(set(reasoning_report) | set(safety_report))
        }

        specs = discover_attention_modules(model)
        total_heads = {spec.layer_name: spec.num_heads for spec in specs}
        output_dir = ensure_dir(args.output_dir)
        report_path = serialize_json(
            {
                "stage": args.stage,
                "reasoning": reasoning_report,
                "safety": safety_report,
                "combined": combined,
                "total_heads_per_layer": total_heads,
                "reasoning_path": str(args.reasoning_path),
                "safety_path": str(args.safety_path),
            },
            output_dir / "saliency_report.json",
        )
        return report_path
    finally:
        for masker in maskers:
            masker.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture CRASP v2 head saliency.")
    parser.add_argument("--config", type=Path, default=Path("crasp_v2/config/default_runpod.yaml"))
    parser.add_argument("--stage", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--reasoning-path", type=Path, default=Path("data/calibration/cot_calibration.jsonl"))
    parser.add_argument("--safety-path", type=Path, default=Path("data/calibration/safety_calibration.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/crasp-vol/results/crasp_v2/saliency"))
    return parser.parse_args()


def main() -> None:
    report_path = run(parse_args())
    print(f"Saved saliency report to {report_path}")


if __name__ == "__main__":
    main()
