"""scripts/run_wanda.py
─────────────────────────────────────────────────────────────────────────────
Run Wanda unstructured pruning at multiple sparsity levels and compare two
calibration strategies — the standard C4 corpus and the project's Medical CoT
dataset — against the raw (unpruned) baseline.

For each (calibration, sparsity) combination the script:
  1.  Loads a fresh copy of the model.
  2.  Builds the calibration dataloader (C4 or Medical CoT).
  3.  Prunes in-place with Wanda (src.wanda_loader.run_wanda_pruning).
  4.  Evaluates with CRASPEvaluator.from_model() — no checkpoint saved.
  5.  Loads the raw baseline JSON and computes retention scores.
  6.  Writes an individual result JSON to --output-dir.

After all runs it prints a comparison table and writes a summary JSON.

Usage
-----
# Full run — both calibrations, all three sparsity levels:
    python scripts/run_wanda.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sparsity 0.20 0.25 0.30 \\
        --calibrations both \\
        --device cuda

# Smoke test (few samples, fast):
    python scripts/run_wanda.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sparsity 0.20 \\
        --calibrations cot \\
        --num-samples 8 \\
        --num-eval-samples 20 \\
        --device cuda \\
        --verbose

Prerequisites
-------------
* Raw baseline must exist in --baseline-dir  (run scripts/run_baselines.py first)
* Medical CoT calibration file must exist    (run scripts/generate_cot_dataset.py)
* Wanda must be cloned to vendors/wanda/     (git clone https://github.com/locuslab/wanda vendors/wanda)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from dataclasses import asdict
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Resolve project root and inject src/ ──────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval_harness import CRASPEvaluator
from src.metrics import CRASPMetrics, compute_retention_report
from src.wanda_loader import get_medical_cot_loaders, run_wanda_pruning

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_SPARSITY = [0.20, 0.25, 0.30]
DEFAULT_SEQLEN = 2048
DEFAULT_NUM_SAMPLES = 128     # calibration samples
DEFAULT_BATCH_SIZE = 8
DEFAULT_BASELINE_DIR = _PROJECT_ROOT / "results" / "baselines"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "wanda"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _safe_model_name(model_name: str) -> str:
    """Convert a model ID to a filesystem-safe string."""
    return model_name.replace("/", "_").replace("\\", "_")


def _timestamp() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_baseline(baseline_dir: Path, model_name: str) -> Optional[Path]:
    """Find the most recent raw baseline JSON for the given model."""
    safe = _safe_model_name(model_name)
    candidates = sorted(baseline_dir.glob(f"raw_{safe}_*.json"), reverse=True)
    if candidates:
        return candidates[0]
    # Fallback: search case-insensitively
    for p in sorted(baseline_dir.glob("raw_*.json"), reverse=True):
        if safe.lower() in p.name.lower():
            return p
    return None


def _load_baseline_metrics(baseline_path: Path) -> dict:
    """Load a baseline result JSON and return it as a plain dict."""
    with baseline_path.open() as fh:
        return json.load(fh)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved results to %s", path)


def _print_table(rows: list[dict], baseline: dict) -> None:
    """Print a formatted comparison table to stdout."""
    headers = [
        "Calibration", "Sparsity", "Clinical Acc", "Safety Score",
        "Clin. Retention", "Safety Retention", "Mean Retention",
    ]
    table = []

    # Baseline row
    table.append([
        "(raw baseline)", "0%",
        f"{baseline['clinical_accuracy']:.4f}",
        f"{baseline['safety_score']:.4f}",
        "—", "—", "—",
    ])

    for r in rows:
        ret = r.get("retention") or {}
        table.append([
            r["calibration"],
            f"{r['sparsity']:.0%}",
            f"{r['clinical_accuracy']:.4f}",
            f"{r['safety_score']:.4f}",
            f"{ret.get('clinical_retention', 0):.4f}" if ret else "—",
            f"{ret.get('safety_retention', 0):.4f}" if ret else "—",
            f"{ret.get('mean_retention', 0):.4f}" if ret else "—",
        ])

    print("\n" + tabulate(table, headers=headers, tablefmt="github") + "\n")


# ── Core per-run logic ─────────────────────────────────────────────────────────


def _run_single(
    model_name: str,
    calibration: str,
    sparsity: float,
    num_samples: int,
    seqlen: int,
    batch_size: int,
    device: str,
    num_eval_samples: Optional[int],
) -> dict:
    """Load, prune, and evaluate one (calibration × sparsity) combination.

    Returns a plain dict with the result fields plus ``calibration`` and
    ``sparsity`` keys for the summary table.
    """
    logger.info(
        "=== Run: calibration=%s  sparsity=%.0f%% ===", calibration, sparsity * 100
    )

    # ── 1. Load fresh model ───────────────────────────────────────────────────
    logger.info("Loading model %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # ── 2. Build calibration dataloader ──────────────────────────────────────
    if calibration == "cot":
        logger.info("Building Medical CoT calibration dataloader …")
        dataloader = get_medical_cot_loaders(
            nsamples=num_samples,
            seed=42,
            seqlen=seqlen,
            tokenizer=tokenizer,
        )
    else:  # c4
        logger.info("Building C4 calibration dataloader …")
        # Use Wanda's own C4 loader.
        wanda_path = str(_PROJECT_ROOT / "vendors" / "wanda")
        if wanda_path not in sys.path:
            sys.path.insert(0, wanda_path)
        from lib.data import get_loaders as wanda_get_loaders  # type: ignore[import]
        dataloader, _ = wanda_get_loaders(
            "c4",
            nsamples=num_samples,
            seed=42,
            seqlen=seqlen,
            tokenizer=tokenizer,
        )

    # ── 3. Prune in-place ─────────────────────────────────────────────────────
    logger.info("Pruning at %.0f%% sparsity …", sparsity * 100)
    run_wanda_pruning(
        model=model,
        dataloader=dataloader,
        sparsity_ratio=sparsity,
        nsamples=num_samples,
        seqlen=seqlen,
        device=device,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Evaluating pruned model …")
    evaluator = CRASPEvaluator.from_model(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=seqlen,
    )

    if num_eval_samples is not None:
        # Monkey-patch the dataset size for smoke tests.
        # We wrap evaluate_all() to pass num_samples through; since
        # CRASPEvaluator.evaluate_medqa() accepts num_samples, we call
        # the individual methods instead.
        medqa_result = evaluator.evaluate_medqa(num_samples=num_eval_samples)
        medhalt_result = evaluator.evaluate_medhalt(num_samples=num_eval_samples)
        metrics = CRASPMetrics(
            clinical_accuracy=medqa_result["clinical_accuracy"],
            safety_score=medhalt_result["safety_score"],
            safety_breakdown=medhalt_result["safety_breakdown"],
            retention=None,
            model_name=model_name,
            sparsity=sparsity,
        )
    else:
        metrics = evaluator.evaluate_all()
        metrics.sparsity = sparsity

    # ── 5. Cleanup ────────────────────────────────────────────────────────────
    del model, evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result: dict = asdict(metrics)
    result["calibration"] = calibration
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Wanda pruning at multiple sparsity levels with Medical CoT "
            "and/or C4 calibration and compare against the raw baseline."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path. (default: %(default)s)",
    )
    parser.add_argument(
        "--sparsity",
        nargs="+",
        type=float,
        default=DEFAULT_SPARSITY,
        metavar="RATIO",
        help="One or more sparsity ratios, e.g. 0.20 0.25 0.30. (default: %(default)s)",
    )
    parser.add_argument(
        "--calibrations",
        choices=["c4", "cot", "both"],
        default="both",
        help="Which calibration dataset(s) to run. (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory containing raw baseline JSON files. (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write result JSON files. (default: %(default)s)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of calibration samples for pruning. (default: %(default)s)",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit evaluation to N samples per benchmark (useful for smoke tests). "
            "Omit to use the full test sets."
        ),
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=DEFAULT_SEQLEN,
        help="Calibration sequence length in tokens. (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Evaluation batch size. (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device. (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        args.device = "cpu"

    # Log system info
    logger.info("Model       : %s", args.model)
    logger.info("Sparsity    : %s", args.sparsity)
    logger.info("Calibrations: %s", args.calibrations)
    logger.info("Device      : %s", args.device)
    if torch.cuda.is_available():
        logger.info("GPU         : %s", torch.cuda.get_device_name(0))

    # ── Load baseline ─────────────────────────────────────────────────────────
    baseline_path = _find_baseline(args.baseline_dir, args.model)
    if baseline_path is None:
        logger.error(
            "No raw baseline found for '%s' in %s. "
            "Run: python scripts/run_baselines.py --model %s",
            args.model,
            args.baseline_dir,
            args.model,
        )
        return 1
    baseline = _load_baseline_metrics(baseline_path)
    logger.info("Loaded baseline from %s", baseline_path)
    logger.info(
        "Baseline — clinical: %.4f  safety: %.4f",
        baseline["clinical_accuracy"],
        baseline["safety_score"],
    )

    # ── Determine calibration list ────────────────────────────────────────────
    cal_list: list[str]
    if args.calibrations == "both":
        cal_list = ["c4", "cot"]
    else:
        cal_list = [args.calibrations]

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    ts = _timestamp()

    for calibration in cal_list:
        for sparsity in args.sparsity:
            try:
                result = _run_single(
                    model_name=args.model,
                    calibration=calibration,
                    sparsity=sparsity,
                    num_samples=args.num_samples,
                    seqlen=args.seqlen,
                    batch_size=args.batch_size,
                    device=args.device,
                    num_eval_samples=args.num_eval_samples,
                )
            except Exception:
                logger.exception(
                    "Run failed: calibration=%s sparsity=%.2f", calibration, sparsity
                )
                continue

            # ── Compute retention against raw baseline ────────────────────────
            raw_scores = {
                "clinical_accuracy": baseline["clinical_accuracy"],
                "safety_score": baseline["safety_score"],
            }
            pruned_scores = {
                "clinical_accuracy": result["clinical_accuracy"],
                "safety_score": result["safety_score"],
            }
            retention_report = compute_retention_report(raw_scores, pruned_scores)
            result["retention"] = {
                "clinical_retention": retention_report.clinical_retention,
                "safety_retention": retention_report.safety_retention,
                "mean_retention": retention_report.mean_retention,
            }

            # ── Save individual result ────────────────────────────────────────
            safe_model = _safe_model_name(args.model)
            sparsity_tag = f"sp{int(sparsity * 100):02d}"
            fname = f"wanda_{calibration}_{sparsity_tag}_{safe_model}_{ts}.json"
            _save_json(result, args.output_dir / fname)

            all_results.append(result)

    if not all_results:
        logger.error("No runs completed successfully.")
        return 1

    # ── Print comparison table ────────────────────────────────────────────────
    _print_table(all_results, baseline)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "baseline": baseline,
        "runs": all_results,
        "timestamp": ts,
    }
    summary_path = args.output_dir / f"summary_{_safe_model_name(args.model)}_{ts}.json"
    _save_json(summary, summary_path)
    logger.info("Summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
