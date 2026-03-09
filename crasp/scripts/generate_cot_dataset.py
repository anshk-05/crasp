"""
scripts/generate_cot_dataset.py
────────────────────────────────
Build 128 chain-of-thought (CoT) calibration sequences from the MedQA
training split.

These prompts are fed through LLaMA-2 during activation capture (Phase 3).
The neurons that activate while the model processes these CoT sequences form
the *clinical reasoning saliency map* that CRASP uses to protect medically
critical weights during pruning.

Two variants are produced:
  • cot_calibration.jsonl   — Full CoT prompt (primary calibration set)
  • plain_calibration.jsonl — Same questions, no CoT scaffold (ablation ctrl)

Usage
-----
    python scripts/generate_cot_dataset.py

    python scripts/generate_cot_dataset.py \\
        --input-dir  data/raw/medqa \\
        --output-path data/calibration/cot_calibration.jsonl \\
        --num-samples 128 \\
        --seed 42

    # Also emit the plain (no-CoT) ablation file
    python scripts/generate_cot_dataset.py --with-plain
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import load_from_disk, Dataset

# ── Prompt templates (swap by editing these constants) ───────────────────────

COT_TEMPLATE: str = """\
You are a medical expert answering a USMLE-style clinical question.
Think step by step through the clinical reasoning before selecting your answer.

Question: {question}

Options:
{options}

Reasoning:
1. Identify the key clinical findings or mechanism.
2. Recall relevant pathophysiology, pharmacology, or clinical guidelines.
3. Evaluate each option systematically, ruling out incorrect choices.
4. State your final answer and briefly explain why it is correct.

Answer: {answer}\
"""

PLAIN_TEMPLATE: str = """\
You are a medical expert answering a USMLE-style clinical question.

Question: {question}

Options:
{options}

Answer: {answer}\
"""

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = Path("data/raw/medqa")
DEFAULT_OUTPUT_PATH = Path("data/calibration/cot_calibration.jsonl")
DEFAULT_PLAIN_OUTPUT_PATH = Path("data/calibration/plain_calibration.jsonl")
DEFAULT_NUM_SAMPLES = 128
DEFAULT_SEED = 42

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    """Initialise root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _format_options(options: dict[str, str]) -> str:
    """
    Render the options dict as a lettered list.

    Parameters
    ----------
    options:
        Mapping of option key → option text, e.g. ``{"A": "Aspirin", ...}``.

    Returns
    -------
    A multi-line string like ``"A) Aspirin\\nB) Ibuprofen\\n…"``.
    """
    return "\n".join(f"{key}) {text}" for key, text in sorted(options.items()))


def _get_answer_string(example: dict[str, Any]) -> str:
    """
    Extract the correct answer key + text from a MedQA example.

    MedQA has two relevant fields:
      • ``answer_idx`` — the letter key (e.g. ``"A"``)
      • ``answer``     — the text of the correct option

    Returns a string like ``"A) Aspirin"`` or falls back to just the text.
    """
    key = example.get("answer_idx", "")
    text = example.get("answer", "")
    if key and text:
        return f"{key}) {text}"
    return text or key


def build_cot_prompt(example: dict[str, Any]) -> str:
    """
    Format a single MedQA example using the CoT prompt template.

    Parameters
    ----------
    example:
        A single MedQA record with fields ``question``, ``options``,
        ``answer_idx``, and ``answer``.

    Returns
    -------
    A fully formatted CoT prompt string.
    """
    return COT_TEMPLATE.format(
        question=example["question"],
        options=_format_options(example["options"]),
        answer=_get_answer_string(example),
    )


def build_plain_prompt(example: dict[str, Any]) -> str:
    """
    Format a single MedQA example using the plain (no-CoT) template.

    Used as an ablation control: same questions, no reasoning scaffold.
    """
    return PLAIN_TEMPLATE.format(
        question=example["question"],
        options=_format_options(example["options"]),
        answer=_get_answer_string(example),
    )


def load_medqa_train(input_dir: Path) -> Dataset:
    """
    Load the MedQA train split from disk.

    Parameters
    ----------
    input_dir:
        Directory containing the Arrow dataset saved by ``download_data.py``
        (expects a ``train/`` subdirectory).

    Returns
    -------
    HuggingFace ``Dataset`` for the training split.

    Raises
    ------
    FileNotFoundError if the expected path does not exist.
    """
    train_path = input_dir / "train"
    if not train_path.exists():
        raise FileNotFoundError(
            f"MedQA train split not found at {train_path}. "
            "Run scripts/download_data.py first."
        )
    logger.debug("Loading MedQA train split from %s", train_path)
    ds = load_from_disk(str(train_path))
    logger.info("Loaded MedQA train split: %d examples", len(ds))
    return ds


def sample_examples(
    dataset: Dataset,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Randomly sample ``num_samples`` examples from *dataset*.

    Parameters
    ----------
    dataset:
        HuggingFace Dataset to sample from.
    num_samples:
        Number of examples to draw.  Capped at ``len(dataset)``.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of example dicts.
    """
    rng = random.Random(seed)
    all_examples = [dataset[i] for i in range(len(dataset))]
    n = min(num_samples, len(all_examples))
    sampled = rng.sample(all_examples, n)
    logger.info("Sampled %d / %d examples (seed=%d)", n, len(all_examples), seed)
    return sampled


def build_calibration_records(
    examples: list[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    """
    Convert raw MedQA examples into calibration records.

    Parameters
    ----------
    examples:
        List of raw MedQA example dicts.
    variant:
        Either ``"cot"`` or ``"plain"``.  Controls which template is used.

    Returns
    -------
    List of dicts with keys ``id``, ``prompt``, ``original_question``,
    ``correct_answer``, and ``variant``.
    """
    build_fn = build_cot_prompt if variant == "cot" else build_plain_prompt
    records = []
    for i, ex in enumerate(examples):
        records.append(
            {
                "id": i,
                "prompt": build_fn(ex),
                "original_question": ex["question"],
                "correct_answer": _get_answer_string(ex),
                "answer_idx": ex.get("answer_idx", ""),
                "variant": variant,
            }
        )
    return records


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write *records* to a JSON Lines file at *output_path*.

    Creates parent directories automatically.

    Parameters
    ----------
    records:
        List of serialisable dicts.
    output_path:
        Destination ``.jsonl`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), output_path)


def log_sequence_stats(records: list[dict[str, Any]], label: str) -> None:
    """Log min/max/mean word counts for a set of calibration records."""
    lengths = [len(r["prompt"].split()) for r in records]
    logger.info(
        "%s prompt lengths: min=%d  max=%d  mean=%.0f words",
        label,
        min(lengths),
        max(lengths),
        sum(lengths) / len(lengths),
    )


# ── Public API (importable) ───────────────────────────────────────────────────

def generate(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    with_plain: bool = False,
    plain_output_path: Path = DEFAULT_PLAIN_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """
    Build CoT calibration sequences from MedQA and save to disk.

    Parameters
    ----------
    input_dir:
        Directory containing the downloaded MedQA Arrow files.
    output_path:
        Destination path for the CoT JSON Lines file.
    num_samples:
        Number of calibration examples to generate.
    seed:
        Random seed for reproducible sampling.
    with_plain:
        If ``True``, also save a plain (no-CoT) ablation file.
    plain_output_path:
        Destination path for the plain variant file (used when
        ``with_plain=True``).

    Returns
    -------
    List of CoT calibration record dicts.
    """
    dataset = load_medqa_train(input_dir)
    examples = sample_examples(dataset, num_samples, seed)

    # CoT variant
    cot_records = build_calibration_records(examples, variant="cot")
    save_jsonl(cot_records, output_path)
    log_sequence_stats(cot_records, "CoT")

    # Plain variant (ablation)
    if with_plain:
        plain_records = build_calibration_records(examples, variant="plain")
        save_jsonl(plain_records, plain_output_path)
        log_sequence_stats(plain_records, "Plain")

    return cot_records


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CoT calibration sequences from MedQA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing downloaded MedQA Arrow files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination .jsonl path for the CoT calibration file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of calibration examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--with-plain",
        action="store_true",
        help="Also generate a plain (no-CoT) ablation file.",
    )
    parser.add_argument(
        "--plain-output-path",
        type=Path,
        default=DEFAULT_PLAIN_OUTPUT_PATH,
        help="Destination .jsonl path for the plain ablation file.",
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
        with_plain=args.with_plain,
        plain_output_path=args.plain_output_path,
    )
