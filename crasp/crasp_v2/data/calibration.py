"""Calibration dataset builders for CRASP v2."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from .io import load_jsonl, save_jsonl
from .medhalt import build_safety_prompt, validate_safety_records
from .medqa import build_cot_prompt, build_plain_prompt

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_RAW_MEDQA_DIR = DEFAULT_DATA_ROOT / "raw" / "medqa"
DEFAULT_RAW_MEDHALT_DIR = DEFAULT_DATA_ROOT / "raw" / "medhalt"
DEFAULT_CALIBRATION_DIR = DEFAULT_DATA_ROOT / "calibration"

MEDHALT_CONFIGS = ["reasoning_fake", "reasoning_nota", "reasoning_FCT"]


def _load_dataset() -> Any:
    from datasets import load_from_disk  # type: ignore[import]

    return load_from_disk


def _concatenate_datasets() -> Any:
    from datasets import concatenate_datasets  # type: ignore[import]

    return concatenate_datasets


def _sample_records(examples: list[dict[str, Any]], num_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample_size = min(num_samples, len(examples))
    if sample_size == len(examples):
        sampled = list(examples)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(examples, sample_size)


def load_medqa_split(input_dir: Path, split: str = "train") -> list[dict[str, Any]]:
    """Load a MedQA split from disk and materialize it into Python dicts."""
    load_from_disk = _load_dataset()
    ds = load_from_disk(str(input_dir / split))
    return [ds[idx] for idx in range(len(ds))]


def load_medhalt_rows(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the three reasoning Med-HALT configs into Python dicts."""
    load_from_disk = _load_dataset()
    concatenate_datasets = _concatenate_datasets()

    loaded: dict[str, list[dict[str, Any]]] = {}
    for config in MEDHALT_CONFIGS:
        config_path = input_dir / config
        if not config_path.exists():
            logger.warning("Skipping missing Med-HALT config: %s", config_path)
            continue

        split_dirs = [entry for entry in sorted(config_path.iterdir()) if entry.is_dir()]
        datasets = []
        for split_dir in split_dirs:
            datasets.append(load_from_disk(str(split_dir)))
        if not datasets:
            continue
        combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        loaded[config] = [combined[idx] for idx in range(len(combined))]
    return loaded


def build_cot_records(
    input_dir: Path = DEFAULT_RAW_MEDQA_DIR,
    output_path: Path = DEFAULT_CALIBRATION_DIR / "cot_calibration.jsonl",
    num_samples: int = 128,
    seed: int = 42,
    include_plain: bool = True,
) -> dict[str, Any]:
    """Generate CoT and plain MedQA calibration records."""
    rows = _sample_records(load_medqa_split(input_dir, split="train"), num_samples, seed)
    cot_records = [
        {
            "id": idx,
            "variant": "cot",
            "source": "medqa",
            "prompt": build_cot_prompt(row),
            "original_sample": row,
        }
        for idx, row in enumerate(rows)
    ]
    save_jsonl(cot_records, output_path)

    manifest: dict[str, Any] = {
        "cot_path": str(output_path),
        "cot_records": len(cot_records),
    }

    if include_plain:
        plain_path = output_path.with_name("plain_calibration.jsonl")
        plain_records = [
            {
                "id": idx,
                "variant": "plain",
                "source": "medqa",
                "prompt": build_plain_prompt(row),
                "original_sample": row,
            }
            for idx, row in enumerate(rows)
        ]
        save_jsonl(plain_records, plain_path)
        manifest["plain_path"] = str(plain_path)
        manifest["plain_records"] = len(plain_records)

    return manifest


def build_safety_records(
    input_dir: Path = DEFAULT_RAW_MEDHALT_DIR,
    output_path: Path = DEFAULT_CALIBRATION_DIR / "safety_calibration.jsonl",
    num_samples: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate fixed Med-HALT safety calibration records and a validation report."""
    rng = random.Random(seed)
    loaded = load_medhalt_rows(input_dir)
    configs = sorted(loaded)
    if not configs:
        raise FileNotFoundError(
            f"No Med-HALT reasoning configs found under {input_dir}. Run download_data.py first."
        )

    quota = max(1, num_samples // len(configs))
    sampled: list[tuple[str, dict[str, Any]]] = []
    for config in configs:
        rows = loaded[config]
        take = min(quota, len(rows))
        chosen = rng.sample(rows, take) if take < len(rows) else list(rows)
        for row in chosen:
            sampled.append((config, row))

    while len(sampled) < num_samples:
        remaining_pool = [(config, row) for config in configs for row in loaded[config]]
        sampled.append(rng.choice(remaining_pool))

    rng.shuffle(sampled)
    sampled = sampled[:num_samples]

    records = [
        {
            "id": idx,
            "task_type": task_type,
            "source": "medhalt",
            "prompt": build_safety_prompt(example),
            "original_sample": example,
            "expected_label": str(
                example.get("answer", example.get("output", example.get("label", f"[{task_type}]")))
            ).strip()
            or f"[{task_type}]",
        }
        for idx, (task_type, example) in enumerate(sampled)
    ]
    save_jsonl(records, output_path)

    validation = validate_safety_records(records)
    validation_path = output_path.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {
        "safety_path": str(output_path),
        "validation_path": str(validation_path),
        **validation,
    }


def build_mixed_sft_records(
    cot_path: Path,
    safety_path: Path,
    output_path: Path = DEFAULT_CALIBRATION_DIR / "mixed_sft.jsonl",
) -> dict[str, Any]:
    """Combine CoT and safety calibration prompts into a single SFT corpus."""
    cot_records = load_jsonl(cot_path)
    safety_records = load_jsonl(safety_path)

    mixed_records: list[dict[str, Any]] = []
    for idx, record in enumerate(cot_records + safety_records):
        mixed_records.append(
            {
                "id": idx,
                "source": record.get("source", "unknown"),
                "task_type": record.get("task_type", record.get("variant", "unknown")),
                "prompt": record["prompt"],
            }
        )
    save_jsonl(mixed_records, output_path)
    return {
        "mixed_path": str(output_path),
        "mixed_records": len(mixed_records),
    }
