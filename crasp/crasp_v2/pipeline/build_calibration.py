"""CLI for building CRASP v2 calibration datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from crasp_v2.data.calibration import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_RAW_MEDHALT_DIR,
    DEFAULT_RAW_MEDQA_DIR,
    build_cot_records,
    build_mixed_sft_records,
    build_safety_records,
)
from crasp_v2.pipeline.common import serialize_json


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    cot_manifest = build_cot_records(
        input_dir=args.raw_medqa_dir,
        output_path=output_dir / "cot_calibration.jsonl",
        num_samples=args.cot_samples,
        seed=args.seed,
        include_plain=True,
    )
    safety_manifest = build_safety_records(
        input_dir=args.raw_medhalt_dir,
        output_path=output_dir / "safety_calibration.jsonl",
        num_samples=args.safety_samples,
        seed=args.seed,
    )
    mixed_manifest = build_mixed_sft_records(
        cot_path=Path(cot_manifest["cot_path"]),
        safety_path=Path(safety_manifest["safety_path"]),
        output_path=output_dir / "mixed_sft.jsonl",
    )
    manifest_path = serialize_json(
        {
            "cot": cot_manifest,
            "safety": safety_manifest,
            "mixed": mixed_manifest,
        },
        output_dir / "calibration_manifest.json",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CRASP v2 calibration datasets.")
    parser.add_argument("--raw-medqa-dir", type=Path, default=DEFAULT_RAW_MEDQA_DIR)
    parser.add_argument("--raw-medhalt-dir", type=Path, default=DEFAULT_RAW_MEDHALT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--cot-samples", type=int, default=128)
    parser.add_argument("--safety-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    manifest_path = run(parse_args())
    print(f"Calibration manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
