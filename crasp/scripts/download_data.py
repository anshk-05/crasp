"""
scripts/download_data.py
────────────────────────
Download MedQA and Med-HALT datasets from HuggingFace Hub and save them
locally as Arrow/Parquet files for offline use.

Usage
-----
    # Download both datasets to the default location
    python scripts/download_data.py

    # Override output directory
    python scripts/download_data.py --output-dir /workspace/crasp-vol/data/raw

    # Force re-download even if files already exist
    python scripts/download_data.py --force-redownload

    # Verbose logging
    python scripts/download_data.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset, DatasetDict
from tqdm import tqdm

# ── Dataset identifiers (change here to swap sources) ────────────────────────
MEDQA_HF_ID = "GBaker/MedQA-USMLE-4-options"
MEDHALT_HF_ID = "medhalt/Med-HALT"

# Splits to pull from each dataset
MEDQA_SPLITS = ["train", "test"]

# Med-HALT uses HF *configs*, not splits.  Each config contains its own
# train/test splits.  The three reasoning-oriented configs are the most
# relevant for CRASP's hallucination evaluation.
MEDHALT_CONFIGS = ["reasoning_fake", "reasoning_nota", "reasoning_FCT"]

# Retry config for transient network failures
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    """Set up root logger with a human-readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_with_retries(
    hf_id: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
    delay: float = RETRY_DELAY_SECONDS,
) -> DatasetDict:
    """
    Call ``datasets.load_dataset`` with automatic retries on failure.

    Parameters
    ----------
    hf_id:
        HuggingFace dataset identifier (e.g. ``"GBaker/MedQA-USMLE-4-options"``).
    config:
        Optional dataset configuration / subset name (required by some
        multi-config datasets such as Med-HALT).
    split:
        Optional specific split name.  ``None`` downloads all splits.
    max_retries:
        Maximum number of attempts before re-raising the last exception.
    delay:
        Seconds to wait between attempts.

    Returns
    -------
    DatasetDict (or Dataset if ``split`` is specified).
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                "Loading %s config=%s split=%s (attempt %d/%d)",
                hf_id, config, split, attempt, max_retries,
            )
            kwargs: dict = {"path": hf_id, "trust_remote_code": True}
            if config is not None:
                kwargs["name"] = config
            if split is not None:
                kwargs["split"] = split
            return load_dataset(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s (config=%s): %s",
                attempt,
                max_retries,
                hf_id,
                config,
                exc,
            )
            if attempt < max_retries:
                logger.info("Retrying in %ds…", delay)
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {hf_id} (config={config}) after {max_retries} attempts"
    ) from last_exc


def _log_dataset_stats(dataset: DatasetDict, name: str) -> None:
    """Log row counts and column names for each split in *dataset*."""
    logger.info("── %s ──", name)
    for split_name, ds in dataset.items():
        logger.info(
            "  split=%-30s  rows=%-8d  columns=%s",
            split_name,
            len(ds),
            list(ds.column_names),
        )


def _already_downloaded(output_dir: Path, names: list[str]) -> bool:
    """Return True if all expected subdirectories exist under *output_dir*."""
    return all((output_dir / name).exists() for name in names)


# ── Per-dataset download functions ────────────────────────────────────────────

def download_medqa(output_dir: Path, force: bool = False) -> DatasetDict:
    """
    Download MedQA (USMLE-style 4-option MCQ) from HuggingFace Hub.

    The dataset is saved in Arrow format under ``output_dir/medqa/<split>/``.

    Parameters
    ----------
    output_dir:
        Root directory for raw data (``data/raw`` by default).
    force:
        If ``True``, re-download even if files already exist locally.

    Returns
    -------
    DatasetDict with all available splits.
    """
    dest = output_dir / "medqa"

    if not force and _already_downloaded(dest, MEDQA_SPLITS):
        logger.info(
            "MedQA already present at %s — skipping (use --force-redownload to override)",
            dest,
        )
        from datasets import load_from_disk
        return DatasetDict(
            {split: load_from_disk(str(dest / split)) for split in MEDQA_SPLITS}
        )

    logger.info("Downloading MedQA from %s …", MEDQA_HF_ID)
    dataset = _load_with_retries(MEDQA_HF_ID)

    dest.mkdir(parents=True, exist_ok=True)

    splits_to_save = [s for s in MEDQA_SPLITS if s in dataset]
    with tqdm(splits_to_save, desc="Saving MedQA splits", unit="split") as pbar:
        for split in pbar:
            split_path = dest / split
            pbar.set_postfix(split=split, rows=len(dataset[split]))
            dataset[split].save_to_disk(str(split_path))
            logger.debug(
                "Saved %s split (%d rows) → %s",
                split, len(dataset[split]), split_path,
            )

    _log_dataset_stats(
        DatasetDict({s: dataset[s] for s in splits_to_save}), "MedQA"
    )
    return dataset


def download_medhalt(output_dir: Path, force: bool = False) -> DatasetDict:
    """
    Download Med-HALT (Medical Hallucination Test) from HuggingFace Hub.

    Med-HALT is a *multi-config* dataset on HuggingFace.  Each config
    (e.g. ``reasoning_fake``, ``reasoning_nota``, ``reasoning_FCT``) is
    downloaded separately and saved under
    ``output_dir/medhalt/<config>/<split>/``.

    Parameters
    ----------
    output_dir:
        Root directory for raw data.
    force:
        If ``True``, re-download even if files already exist locally.

    Returns
    -------
    DatasetDict whose keys are ``<config>/<split>`` for every downloaded
    config and split combination.
    """
    dest = output_dir / "medhalt"

    if not force and _already_downloaded(dest, MEDHALT_CONFIGS):
        logger.info("Med-HALT already present at %s — skipping", dest)
        from datasets import load_from_disk

        result: dict = {}
        for cfg in MEDHALT_CONFIGS:
            cfg_dir = dest / cfg
            for split_dir in cfg_dir.iterdir():
                if split_dir.is_dir():
                    key = f"{cfg}/{split_dir.name}"
                    result[key] = load_from_disk(str(split_dir))
        return DatasetDict(result)

    logger.info("Downloading Med-HALT from %s …", MEDHALT_HF_ID)

    saved: dict = {}
    with tqdm(MEDHALT_CONFIGS, desc="Downloading Med-HALT configs", unit="config") as pbar:
        for cfg in pbar:
            pbar.set_postfix(config=cfg)
            try:
                dataset = _load_with_retries(MEDHALT_HF_ID, config=cfg)

                # dataset may be a DatasetDict (with train/test) or a single
                # Dataset if there is only one split for this config.
                if isinstance(dataset, DatasetDict):
                    for split_name, ds in dataset.items():
                        split_path = dest / cfg / split_name
                        split_path.mkdir(parents=True, exist_ok=True)
                        ds.save_to_disk(str(split_path))
                        key = f"{cfg}/{split_name}"
                        saved[key] = ds
                        logger.debug(
                            "Saved %s/%s (%d rows) → %s",
                            cfg, split_name, len(ds), split_path,
                        )
                else:
                    # Single Dataset (no sub-splits)
                    split_path = dest / cfg / "all"
                    split_path.mkdir(parents=True, exist_ok=True)
                    dataset.save_to_disk(str(split_path))
                    key = f"{cfg}/all"
                    saved[key] = dataset
                    logger.debug(
                        "Saved %s/all (%d rows) → %s",
                        cfg, len(dataset), split_path,
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Could not download Med-HALT config '%s': %s", cfg, exc
                )

    if not saved:
        raise RuntimeError("No Med-HALT configs could be downloaded.")

    result = DatasetDict(saved)
    _log_dataset_stats(result, "Med-HALT")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download MedQA and Med-HALT datasets for CRASP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory where raw datasets will be saved.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download datasets even if they already exist locally.",
    )
    parser.add_argument(
        "--dataset",
        choices=["medqa", "medhalt", "both"],
        default="both",
        help="Which dataset(s) to download.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run downloads."""
    args = parse_args()
    _configure_logging(args.log_level)

    logger.info("Output directory: %s", args.output_dir.resolve())
    logger.info("Force re-download: %s", args.force_redownload)

    if args.dataset in ("medqa", "both"):
        download_medqa(args.output_dir, force=args.force_redownload)

    if args.dataset in ("medhalt", "both"):
        download_medhalt(args.output_dir, force=args.force_redownload)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()