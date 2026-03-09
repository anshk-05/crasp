"""
scripts/generate_safety_dataset.py
────────────────────────────────────
Build 128 safety calibration sequences from the Med-HALT benchmark.

Med-HALT tests a model's ability to avoid medical hallucinations.  During
CRASP's activation capture phase (Phase 3), the neurons that activate while
the model processes these safety-critical prompts form the *safety saliency
map* — the set of weights that CRASP refuses to prune regardless of sparsity
target.

Task types consumed (Med-HALT configs)
--------------------------------------
  • reasoning_fake — fabricated diseases / treatments / drug combos
  • reasoning_nota — "none of the above" reasoning traps
  • reasoning_FCT  — false clinical triples

The script samples examples proportionally from each available config,
targeting 128 total by default.

Output
------
  data/calibration/safety_calibration.jsonl
  Each line: {"id", "prompt", "task_type", "original_sample", "expected_label"}

Usage
-----
    python scripts/generate_safety_dataset.py

    python scripts/generate_safety_dataset.py \\
        --input-dir  data/raw/medhalt \\
        --output-path data/calibration/safety_calibration.jsonl \\
        --num-samples 128 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from datasets import load_from_disk, Dataset, concatenate_datasets

# ── Prompt template (swap by editing this constant) ──────────────────────────

SAFETY_TEMPLATE: str = """\
You are a medical safety evaluator reviewing a clinical claim.
Your task is to determine whether the medical information below is accurate
and safe, or whether it is fabricated, incorrect, or potentially harmful.

{context_block}Question: {question}

{options_block}Instructions:
- If the claim contains fabricated conditions, non-existent drugs, or
  incorrect medical facts, clearly state that it is INACCURATE.
- If the claim could lead to patient harm, explicitly flag the safety risk.
- Otherwise confirm the information is accurate.

Answer: {answer}\
"""

# ── Constants ─────────────────────────────────────────────────────────────────
# Med-HALT uses HF *configs*, not splits.  Each config directory may contain
# sub-split directories (train, test, all) written by download_data.py.
MEDHALT_CONFIGS = ["reasoning_fake", "reasoning_nota", "reasoning_FCT"]

DEFAULT_INPUT_DIR = Path("data/raw/medhalt")
DEFAULT_OUTPUT_PATH = Path("data/calibration/safety_calibration.jsonl")
DEFAULT_NUM_SAMPLES = 128
DEFAULT_SEED = 42

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    """Initialise root logger with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _format_options(options: dict[str, str] | list[str] | None) -> str:
    """
    Render answer choices as a lettered list, or return an empty string if
    the example has no discrete options (free-form tasks).

    Parameters
    ----------
    options:
        Either a dict ``{"A": "text", …}``, a list ``["text", …]``, or
        ``None`` for free-form tasks.

    Returns
    -------
    Formatted ``"Options:\\nA) …\\n"`` block, or ``""`` if no options.
    """
    if not options:
        return ""
    if isinstance(options, dict):
        lines = [f"{k}) {v}" for k, v in sorted(options.items())]
    else:
        lines = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
    return "Options:\n" + "\n".join(lines) + "\n\n"


def _extract_fields(example: dict[str, Any]) -> tuple[str, str, str, str]:
    """
    Normalise a Med-HALT example into (context, question, options_block, answer).

    Med-HALT field names vary across task types.  This function handles the
    common naming variants seen in the dataset.

    Returns
    -------
    Four strings: context, question, formatted_options_block, answer.
    """
    # Context / background passage (not always present)
    context = example.get("context", example.get("input", example.get("passage", "")))
    context = context.strip() if context else ""

    # Question text
    question = example.get("question", example.get("prompt", "")).strip()

    # Answer / expected output
    answer = example.get("answer", example.get("output", example.get("label", ""))).strip()

    # Options block
    options = example.get("options", example.get("choices", None))
    options_block = _format_options(options)

    return context, question, options_block, answer


def build_safety_prompt(example: dict[str, Any]) -> str:
    """
    Format a single Med-HALT example using the safety evaluation template.

    Parameters
    ----------
    example:
        A single Med-HALT record (field names vary by task type).

    Returns
    -------
    A fully formatted safety prompt string.
    """
    context, question, options_block, answer = _extract_fields(example)

    context_block = f"Clinical context:\n{context}\n\n" if context else ""

    return SAFETY_TEMPLATE.format(
        context_block=context_block,
        question=question,
        options_block=options_block,
        answer=answer,
    )


def _infer_expected_label(example: dict[str, Any], task_type: str) -> str:
    """
    Derive a human-readable expected label for the calibration record.

    For hallucination tasks the model should *refuse* or *identify* the false
    claim.  We store the raw answer field from the dataset alongside a
    normalised ``expected_label`` for downstream scoring.

    Parameters
    ----------
    example:
        Raw Med-HALT example.
    task_type:
        The config name (e.g. ``"reasoning_fake"``).

    Returns
    -------
    A short label string.
    """
    raw = example.get("answer", example.get("output", example.get("label", "")))
    if raw:
        return str(raw).strip()
    # Fallback: annotate with task type so downstream can infer
    return f"[{task_type}]"


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_medhalt_config(input_dir: Path, config: str) -> Dataset | None:
    """
    Load a single Med-HALT config from disk.

    After ``download_data.py`` saves Med-HALT, each config lives under
    ``input_dir/<config>/`` and may contain sub-split directories
    (``train/``, ``test/``, ``all/``).  This function loads and concatenates
    all available sub-splits into a single Dataset.

    Parameters
    ----------
    input_dir:
        Root directory containing Med-HALT Arrow files
        (e.g. ``data/raw/medhalt``).
    config:
        Config name, e.g. ``"reasoning_fake"``.

    Returns
    -------
    HuggingFace ``Dataset``, or ``None`` if the config is not found locally.
    """
    config_path = input_dir / config

    if not config_path.exists():
        logger.warning(
            "Med-HALT config '%s' not found at %s — skipping", config, config_path
        )
        return None

    # The config directory may contain sub-split dirs (train/, test/, all/)
    # or may itself be a single Arrow dataset directory.
    sub_dirs = [d for d in config_path.iterdir() if d.is_dir()]

    if not sub_dirs:
        # Try loading the config directory itself as a dataset
        try:
            ds = load_from_disk(str(config_path))
            logger.info("Loaded Med-HALT '%s': %d examples", config, len(ds))
            return ds
        except Exception:
            logger.warning(
                "Med-HALT config '%s' exists but could not be loaded — skipping",
                config,
            )
            return None

    # Load each sub-split and concatenate
    datasets = []
    for sub_dir in sorted(sub_dirs):
        try:
            ds = load_from_disk(str(sub_dir))
            logger.debug(
                "  Loaded %s/%s: %d examples", config, sub_dir.name, len(ds)
            )
            datasets.append(ds)
        except Exception as exc:
            logger.warning(
                "  Could not load %s/%s: %s — skipping",
                config, sub_dir.name, exc,
            )

    if not datasets:
        logger.warning(
            "Med-HALT config '%s' has no loadable sub-splits — skipping", config
        )
        return None

    combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    logger.info("Loaded Med-HALT '%s': %d examples (from %d sub-splits)",
                config, len(combined), len(datasets))
    return combined


# ── Sampling ──────────────────────────────────────────────────────────────────

def balanced_sample(
    splits: dict[str, Dataset],
    total: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Sample *total* examples evenly across available splits.

    Each split receives a quota of ``ceil(total / num_splits)`` examples,
    capped at the split's actual size.  If a split is undersized, the
    remaining quota is redistributed to other splits.

    Parameters
    ----------
    splits:
        Mapping of config name → HuggingFace Dataset.
    total:
        Target total number of examples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of (config_name, example_dict) tuples, shuffled.
    """
    rng = random.Random(seed)
    num_splits = len(splits)
    quota = math.ceil(total / num_splits)

    sampled: list[tuple[str, dict[str, Any]]] = []
    leftover = 0

    split_items = list(splits.items())

    for i, (split_name, ds) in enumerate(split_items):
        available = len(ds)
        n = min(quota + leftover, available)
        leftover = max(0, quota - available)  # carry deficit forward

        indices = rng.sample(range(available), n)
        for idx in indices:
            sampled.append((split_name, ds[idx]))

        logger.info(
            "Sampled %d / %d from '%s'",
            n,
            available,
            split_name,
        )

    # If total is still short (all splits were undersized), take what we have
    sampled = sampled[:total]
    rng.shuffle(sampled)
    logger.info("Total sampled: %d / %d requested", len(sampled), total)
    return sampled


# ── Record construction ───────────────────────────────────────────────────────

def build_calibration_records(
    samples: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Convert (task_type, raw_example) pairs into calibration records.

    Parameters
    ----------
    samples:
        List of ``(task_type, example_dict)`` tuples from ``balanced_sample``.

    Returns
    -------
    List of dicts with keys:
      ``id``, ``prompt``, ``task_type``, ``original_sample``,
      ``expected_label``.
    """
    records = []
    for i, (task_type, example) in enumerate(samples):
        records.append(
            {
                "id": i,
                "prompt": build_safety_prompt(example),
                "task_type": task_type,
                "original_sample": {
                    k: v
                    for k, v in example.items()
                    # Omit heavy embedding fields if present
                    if not isinstance(v, list) or len(v) < 50
                },
                "expected_label": _infer_expected_label(example, task_type),
            }
        )
    return records


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write *records* to a JSON Lines file at *output_path*.

    Creates parent directories automatically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), output_path)


def log_sequence_stats(records: list[dict[str, Any]]) -> None:
    """Log min/max/mean word counts and task-type breakdown."""
    lengths = [len(r["prompt"].split()) for r in records]
    logger.info(
        "Prompt lengths: min=%d  max=%d  mean=%.0f words",
        min(lengths),
        max(lengths),
        sum(lengths) / len(lengths),
    )

    by_type: dict[str, int] = {}
    for r in records:
        by_type[r["task_type"]] = by_type.get(r["task_type"], 0) + 1
    for task, count in sorted(by_type.items()):
        logger.info("  %-35s  %d examples", task, count)


# ── Public API (importable) ───────────────────────────────────────────────────

def generate(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """
    Build safety calibration sequences from Med-HALT and save to disk.

    Parameters
    ----------
    input_dir:
        Directory containing the downloaded Med-HALT Arrow files.
    output_path:
        Destination path for the safety JSON Lines file.
    num_samples:
        Total number of calibration examples to generate.
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    List of safety calibration record dicts.

    Raises
    ------
    RuntimeError if no Med-HALT configs could be loaded.
    """
    # Load available configs
    loaded: dict[str, Dataset] = {}
    for config in MEDHALT_CONFIGS:
        ds = load_medhalt_config(input_dir, config)
        if ds is not None:
            loaded[config] = ds

    if not loaded:
        raise RuntimeError(
            f"No Med-HALT configs found in {input_dir}. "
            "Run scripts/download_data.py first."
        )

    samples = balanced_sample(loaded, num_samples, seed)
    records = build_calibration_records(samples)
    save_jsonl(records, output_path)
    log_sequence_stats(records)

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate safety calibration sequences from Med-HALT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing downloaded Med-HALT Arrow files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination .jsonl path for the safety calibration file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Total number of calibration examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure_logging(args.log_level)
    generate(
        input_dir=args.input_dir,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
    )