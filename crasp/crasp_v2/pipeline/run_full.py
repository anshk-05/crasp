"""Run the full CRASP v2 pipeline end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crasp_v2.pipeline import build_calibration, capture_saliency, run_prune, run_recovery, run_sft
from crasp_v2.pipeline.common import load_config_bundle, serialize_json
from crasp_v2.pipeline.reporting import build_comparison_rows, render_markdown_table


def run(args: argparse.Namespace) -> Path:
    config, _paths = load_config_bundle(args.config)
    calibration_manifest = build_calibration.run(
        argparse.Namespace(
            raw_medqa_dir=args.raw_medqa_dir,
            raw_medhalt_dir=args.raw_medhalt_dir,
            output_dir=args.calibration_dir,
            cot_samples=int(config["calibration"]["cot_samples"]),
            safety_samples=int(config["calibration"]["safety_samples"]),
            seed=int(config["calibration"]["seed"]),
        )
    )
    sft_dir = run_sft.run(
        argparse.Namespace(
            config=args.config,
            mixed_sft_path=args.calibration_dir / "mixed_sft.jsonl",
            output_dir=args.checkpoint_root / "post_sft",
            base_stage=None,
            baseline_metrics=args.baseline_metrics,
        )
    )
    saliency_report = capture_saliency.run(
        argparse.Namespace(
            config=args.config,
            stage=str(sft_dir),
            reasoning_path=args.calibration_dir / "cot_calibration.jsonl",
            safety_path=args.calibration_dir / "safety_calibration.jsonl",
            output_dir=args.results_root / "saliency",
        )
    )
    prune_dir = run_prune.run(
        argparse.Namespace(
            config=args.config,
            stage=str(sft_dir),
            saliency_report=saliency_report,
            baseline_metrics=args.baseline_metrics,
            output_dir=args.checkpoint_root / "post_prune",
        )
    )
    recovery_dir = run_recovery.run(
        argparse.Namespace(
            config=args.config,
            stage=str(prune_dir),
            mixed_sft_path=args.calibration_dir / "mixed_sft.jsonl",
            baseline_metrics=args.baseline_metrics,
            output_dir=args.checkpoint_root / "post_recovery",
        )
    )

    final_metrics = Path(recovery_dir) / "metrics.json"
    rows = build_comparison_rows(args.repo_results_root, json.loads(final_metrics.read_text(encoding="utf-8")))
    markdown = render_markdown_table(rows)
    comparison_path = serialize_json(
        {
            "calibration_manifest": str(calibration_manifest),
            "post_sft": str(sft_dir),
            "saliency_report": str(saliency_report),
            "post_prune": str(prune_dir),
            "post_recovery": str(recovery_dir),
            "comparison_rows": rows,
            "comparison_markdown": markdown,
        },
        args.results_root / "full_run_summary.json",
    )
    print(markdown)
    return comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full CRASP v2 pipeline.")
    parser.add_argument("--config", type=Path, default=Path("crasp_v2/config/default_runpod.yaml"))
    parser.add_argument("--raw-medqa-dir", type=Path, default=Path("data/raw/medqa"))
    parser.add_argument("--raw-medhalt-dir", type=Path, default=Path("data/raw/medhalt"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("/workspace/crasp-vol/data/calibration"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/workspace/crasp-vol/checkpoints"))
    parser.add_argument("--results-root", type=Path, default=Path("/workspace/crasp-vol/results/crasp_v2"))
    parser.add_argument("--repo-results-root", type=Path, default=Path("results"))
    parser.add_argument("--baseline-metrics", default=None)
    return parser.parse_args()


def main() -> None:
    summary_path = run(parse_args())
    print(f"Saved full pipeline summary to {summary_path}")


if __name__ == "__main__":
    main()
