"""scripts/run_baselines.py
─────────────────────────────────────────────────────────────────────────────
CLI entry point for CRASP Phase 2 baseline evaluations.

Evaluates a raw base model (and optionally a SFT variant) on MedQA and
Med-HALT, saves each run's :class:`~src.metrics.CRASPMetrics` as a JSON file,
and prints a comparison table to stdout.

If both ``--model`` and ``--sft-model`` are provided, the script also computes
and logs the retention of the SFT model relative to the raw baseline — giving
an early signal of how much SFT affected clinical and safety performance.

Results JSON files written to ``{output_dir}/`` are loadable by Phase 3
pruning scripts to set the retention denominator.

Usage
-----
  # Evaluate raw model only (full test sets):
  python scripts/run_baselines.py --model meta-llama/Llama-3.1-8B

  # Evaluate raw + SFT with a 50-sample smoke test:
  python scripts/run_baselines.py \\
      --model  meta-llama/Llama-3.1-8B \\
      --sft-model  /workspace/checkpoints/llama3-medft \\
      --num-samples 50 \\
      --output-dir results/baselines \\
      --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure the project root is on sys.path so ``src`` is importable ───────────
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval_harness import CRASPEvaluator
from src.metrics import CRASPMetrics, compute_retention_report, metrics_to_json

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR: str = "results/baselines"
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_DEVICE: str = "cuda"

# ── Logging setup ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure root logger level and format.

    Args:
        verbose: If ``True``, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


# ── System info ───────────────────────────────────────────────────────────────


def _log_system_info() -> None:
    """Log GPU name, VRAM, CUDA version, and torch version at INFO level."""
    import torch

    logger.info("── System Info ─────────────────────────────────────────────")
    logger.info("  torch version  : %s", torch.__version__)
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_idx)
        vram_total = torch.cuda.get_device_properties(device_idx).total_memory
        vram_gb = vram_total / 1024 ** 3
        logger.info("  GPU            : %s", gpu_name)
        logger.info("  VRAM total     : %.1f GB", vram_gb)
        logger.info("  CUDA version   : %s", torch.version.cuda)
    else:
        logger.warning("  No CUDA GPU detected — running on CPU.")
    logger.info("────────────────────────────────────────────────────────────")


# ── Results I/O ───────────────────────────────────────────────────────────────


def _save_metrics(
    metrics: CRASPMetrics,
    output_dir: Path,
    prefix: str,
) -> Path:
    """Write a :class:`~src.metrics.CRASPMetrics` object to a timestamped JSON.

    Args:
        metrics: Metrics to serialise.
        output_dir: Directory to write into (created if absent).
        prefix: Filename prefix — ``"raw"`` or ``"sft"``.

    Returns:
        The :class:`~pathlib.Path` of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise model name for use in a filename.
    safe_model = metrics.model_name.replace("/", "_").replace("\\", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{prefix}_{safe_model}_{ts}.json"
    out_path = output_dir / filename

    out_path.write_text(metrics_to_json(metrics), encoding="utf-8")
    logger.info("Results saved → %s", out_path)
    return out_path


def _save_comparison(
    raw_metrics: CRASPMetrics,
    sft_metrics: CRASPMetrics,
    output_dir: Path,
) -> Path:
    """Write a combined comparison JSON for two evaluated models.

    The file contains both metrics dicts and the retention breakdown so that
    Phase 3 can ingest it directly.

    Args:
        raw_metrics: Metrics for the raw baseline model.
        sft_metrics: Metrics for the SFT variant.
        output_dir: Output directory (created if absent).

    Returns:
        Path of the written comparison file.
    """
    import json as _json
    from dataclasses import asdict

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"comparison_{ts}.json"

    retention = compute_retention_report(
        raw_scores={
            "clinical_accuracy": raw_metrics.clinical_accuracy,
            "safety_score": raw_metrics.safety_score,
        },
        pruned_scores={
            "clinical_accuracy": sft_metrics.clinical_accuracy,
            "safety_score": sft_metrics.safety_score,
        },
    )

    payload = {
        "raw": asdict(raw_metrics),
        "sft": asdict(sft_metrics),
        "retention": asdict(retention),
        "generated_at": ts,
    }
    out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Comparison JSON saved → %s", out_path)
    return out_path


# ── Summary table ─────────────────────────────────────────────────────────────


def _print_summary_table(
    raw_metrics: CRASPMetrics,
    sft_metrics: CRASPMetrics | None = None,
) -> None:
    """Print a formatted comparison table to stdout using ``tabulate``.

    Args:
        raw_metrics: Metrics for the raw baseline model (always present).
        sft_metrics: Metrics for the SFT variant.  If ``None``, the table
            shows only the raw row.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning(
            "tabulate not installed — skipping formatted table.  "
            "Install with: pip install tabulate"
        )
        _print_summary_plain(raw_metrics, sft_metrics)
        return

    headers = ["Model", "Clinical Acc (%)", "Safety (%)", "Retention (%)"]
    rows = []

    # Raw baseline row.
    raw_name = Path(raw_metrics.model_name).name or raw_metrics.model_name
    rows.append(
        [
            raw_name,
            f"{raw_metrics.clinical_accuracy * 100:.1f}",
            f"{raw_metrics.safety_score * 100:.1f}",
            "— (reference)",
        ]
    )

    # SFT row (optional).
    if sft_metrics is not None:
        retention = compute_retention_report(
            raw_scores={
                "clinical_accuracy": raw_metrics.clinical_accuracy,
                "safety_score": raw_metrics.safety_score,
            },
            pruned_scores={
                "clinical_accuracy": sft_metrics.clinical_accuracy,
                "safety_score": sft_metrics.safety_score,
            },
        )
        sft_name = Path(sft_metrics.model_name).name or sft_metrics.model_name
        rows.append(
            [
                sft_name,
                f"{sft_metrics.clinical_accuracy * 100:.1f}",
                f"{sft_metrics.safety_score * 100:.1f}",
                f"{retention.mean_retention * 100:.1f}",
            ]
        )

    print("\n" + tabulate(rows, headers=headers, tablefmt="fancy_grid") + "\n")


def _print_summary_plain(
    raw_metrics: CRASPMetrics,
    sft_metrics: CRASPMetrics | None,
) -> None:
    """Plain-text fallback summary when tabulate is unavailable.

    Args:
        raw_metrics: Baseline metrics.
        sft_metrics: Optional SFT metrics.
    """
    print("\n── Baseline Results ─────────────────────────────────────────")
    print(f"  Raw model  : {raw_metrics.model_name}")
    print(f"    Clinical accuracy : {raw_metrics.clinical_accuracy * 100:.1f} %")
    print(f"    Safety score      : {raw_metrics.safety_score * 100:.1f} %")

    if sft_metrics is not None:
        retention = compute_retention_report(
            raw_scores={
                "clinical_accuracy": raw_metrics.clinical_accuracy,
                "safety_score": raw_metrics.safety_score,
            },
            pruned_scores={
                "clinical_accuracy": sft_metrics.clinical_accuracy,
                "safety_score": sft_metrics.safety_score,
            },
        )
        print(f"  SFT model  : {sft_metrics.model_name}")
        print(f"    Clinical accuracy : {sft_metrics.clinical_accuracy * 100:.1f} %")
        print(f"    Safety score      : {sft_metrics.safety_score * 100:.1f} %")
        print(f"    Mean retention    : {retention.mean_retention * 100:.1f} %")

    print("─────────────────────────────────────────────────────────────\n")


# ── Argument parsing ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for ``run_baselines.py``.

    Returns:
        Configured :class:`~argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="run_baselines",
        description=(
            "CRASP Phase 2 — evaluate a base model (and optional SFT variant) "
            "on MedQA and Med-HALT, then save JSON results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL_ID",
        help=(
            "HuggingFace model ID or local path for the raw base model "
            "(e.g. 'meta-llama/Llama-3.1-8B')."
        ),
    )
    parser.add_argument(
        "--sft-model",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a supervised fine-tuned checkpoint to "
            "evaluate alongside the raw model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Directory where result JSON files are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit evaluation to the first N samples per benchmark.  "
            "Use a small number (e.g. 50) for quick smoke tests.  "
            "Defaults to the full test sets."
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cuda", "cpu"],
        help="Target device for inference.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Path to configs/eval_config.yaml (overrides batch-size if set).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Entry point for the baseline evaluation runner.

    Args:
        argv: Argument list to parse.  Defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        Exit code (``0`` on success, ``1`` on error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    _log_system_info()

    output_dir = Path(args.output_dir)
    raw_metrics: CRASPMetrics | None = None
    sft_metrics: CRASPMetrics | None = None

    # ── Evaluate raw base model ────────────────────────────────────────────────
    logger.info("Evaluating raw base model: %s", args.model)
    try:
        raw_evaluator = CRASPEvaluator(
            model_name_or_path=args.model,
            device=args.device,
            batch_size=args.batch_size,
            eval_config_path=args.config,
        )
        raw_metrics = raw_evaluator.evaluate_all()
        raw_evaluator.cleanup()
    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        logger.error("Run 'python scripts/download_data.py' first.")
        return 1
    except RuntimeError as exc:
        logger.error("Model load failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during raw model evaluation: %s", exc)
        return 1

    _save_metrics(raw_metrics, output_dir, prefix="raw")

    # ── Evaluate SFT model (optional) ─────────────────────────────────────────
    if args.sft_model is not None:
        logger.info("Evaluating SFT model: %s", args.sft_model)
        try:
            sft_evaluator = CRASPEvaluator(
                model_name_or_path=args.sft_model,
                device=args.device,
                batch_size=args.batch_size,
                eval_config_path=args.config,
            )
            sft_metrics = sft_evaluator.evaluate_all()
            sft_evaluator.cleanup()
        except FileNotFoundError as exc:
            logger.error("Dataset not found: %s", exc)
            return 1
        except RuntimeError as exc:
            logger.error("SFT model load failed: %s", exc)
            return 1
        except Exception as exc:
            logger.exception("Unexpected error during SFT model evaluation: %s", exc)
            return 1

        _save_metrics(sft_metrics, output_dir, prefix="sft")

    # ── Compute and log retention (if both models evaluated) ──────────────────
    if raw_metrics is not None and sft_metrics is not None:
        retention = compute_retention_report(
            raw_scores={
                "clinical_accuracy": raw_metrics.clinical_accuracy,
                "safety_score": raw_metrics.safety_score,
            },
            pruned_scores={
                "clinical_accuracy": sft_metrics.clinical_accuracy,
                "safety_score": sft_metrics.safety_score,
            },
        )
        logger.info(
            "SFT vs Raw retention — clinical: %.2f %%  safety: %.2f %%  "
            "mean: %.2f %%",
            retention.clinical_retention * 100,
            retention.safety_retention * 100,
            retention.mean_retention * 100,
        )
        _save_comparison(raw_metrics, sft_metrics, output_dir)

    # ── Print summary table ────────────────────────────────────────────────────
    if raw_metrics is not None:
        _print_summary_table(raw_metrics, sft_metrics)

    logger.info("run_baselines.py complete.  Results in: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
