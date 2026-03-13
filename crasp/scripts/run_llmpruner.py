"""scripts/run_llmpruner.py
─────────────────────────────────────────────────────────────────────────────
Run LLM-Pruner structural pruning at multiple sparsity levels and compare
against the raw baseline.

Unlike Wanda and SparseGPT (unstructured weight zeroing), LLM-Pruner removes
entire attention heads and MLP channels, producing a genuinely smaller model.
This means the pruned model must be saved to disk and reloaded for evaluation.

The script calls LLM-Pruner's hf_prune.py via subprocess (block-wise Taylor
importance, no LoRA recovery), saves the checkpoint, loads it, evaluates with
CRASPEvaluator, then optionally deletes the checkpoint.

Usage
-----
# Full run — all three sparsity levels:
    python scripts/run_llmpruner.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sparsity 0.20 0.25 0.30 \\
        --device cuda

# Single sparsity:
    python scripts/run_llmpruner.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sparsity 0.20 \\
        --device cuda

# Keep checkpoints after evaluation (large — ~8 GB each):
    python scripts/run_llmpruner.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sparsity 0.20 0.25 0.30 \\
        --keep-checkpoints \\
        --device cuda

Prerequisites
-------------
* LLM-Pruner cloned to vendors/llm-pruner/
    git clone https://github.com/horseee/LLM-Pruner vendors/llm-pruner
* Raw baseline must exist in --baseline-dir
    python scripts/run_baselines.py --model meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from tabulate import tabulate
from transformers import AutoTokenizer

# ── Resolve project root and inject src/ ──────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval_harness import CRASPEvaluator
from src.metrics import CRASPMetrics, compute_retention_report

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_SPARSITY = [0.20, 0.25, 0.30]
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EXAMPLES = 10       # calibration examples for Taylor importance
DEFAULT_BASELINE_DIR = _PROJECT_ROOT / "results" / "baselines"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "llmpruner"
_LLM_PRUNER_DIR = _PROJECT_ROOT / "vendors" / "llm-pruner"

# LLM-Pruner writes logs + checkpoints to log/ relative to its own directory.
_LLM_PRUNER_LOG_DIR = _LLM_PRUNER_DIR / "log"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_baseline(baseline_dir: Path, model_name: str) -> Optional[Path]:
    safe = _safe_model_name(model_name)
    candidates = sorted(baseline_dir.glob(f"raw_{safe}_*.json"), reverse=True)
    if candidates:
        return candidates[0]
    for p in sorted(baseline_dir.glob("raw_*.json"), reverse=True):
        if safe.lower() in p.name.lower():
            return p
    return None


def _load_baseline_metrics(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved results to %s", path)


def _print_table(rows: list[dict], baseline: dict) -> None:
    headers = [
        "Sparsity", "Clinical Acc", "Safety Score",
        "Clin. Retention", "Safety Retention", "Mean Retention",
    ]
    table = [[
        "0% (raw baseline)",
        f"{baseline['clinical_accuracy']:.4f}",
        f"{baseline['safety_score']:.4f}",
        "—", "—", "—",
    ]]
    for r in rows:
        ret = r.get("retention") or {}
        table.append([
            f"{r['sparsity']:.0%}",
            f"{r['clinical_accuracy']:.4f}",
            f"{r['safety_score']:.4f}",
            f"{ret.get('clinical_retention', 0):.4f}" if ret else "—",
            f"{ret.get('safety_retention', 0):.4f}" if ret else "—",
            f"{ret.get('mean_retention', 0):.4f}" if ret else "—",
        ])
    print("\n" + tabulate(table, headers=headers, tablefmt="github") + "\n")


# ── LLM-Pruner subprocess helpers ─────────────────────────────────────────────


def _prune_with_llmpruner(
    model_name: str,
    sparsity: float,
    log_name: str,
    device: str,
    num_examples: int,
    verbose: bool,
) -> Path:
    """Call hf_prune.py via subprocess and return path to the saved checkpoint.

    LLM-Pruner saves the checkpoint as a torch.save() dict containing
    ``{'model': model, 'tokenizer': tokenizer}`` under::

        vendors/llm-pruner/log/{log_name}_taylor_{sparsity}/

    Args:
        model_name:   HuggingFace model ID.
        sparsity:     Pruning ratio (0.0–1.0).
        log_name:     Unique run identifier used to build the log directory.
        device:       CUDA/CPU device string.
        num_examples: Number of calibration examples for Taylor importance.
        verbose:      Stream subprocess output to console.

    Returns:
        Path to the ``.pt`` checkpoint file produced by LLM-Pruner.

    Raises:
        FileNotFoundError: If the LLM-Pruner repo is not found.
        RuntimeError:      If hf_prune.py exits with a non-zero code, or no
                           checkpoint file is found after the run.
    """
    hf_prune_script = _LLM_PRUNER_DIR / "hf_prune.py"
    if not hf_prune_script.exists():
        raise FileNotFoundError(
            f"LLM-Pruner not found at {_LLM_PRUNER_DIR}. "
            "Run: git clone https://github.com/horseee/LLM-Pruner vendors/llm-pruner"
        )

    cmd = [
        sys.executable,
        str(hf_prune_script),
        "--base_model", model_name,
        "--save_ckpt_log_name", log_name,
        "--pruning_ratio", str(sparsity),
        "--block_wise",
        "--pruner_type", "taylor",
        "--taylor", "param_first",
        "--device", device,
        "--eval_device", device,
        "--num_examples", str(num_examples),
        "--seed", "42",
        "--save_model",
    ]

    logger.info("Running LLM-Pruner: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(_LLM_PRUNER_DIR),
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.STDOUT,
    )
    if result.returncode != 0:
        if not verbose and result.stdout:
            logger.error("LLM-Pruner output:\n%s", result.stdout.decode(errors="replace"))
        raise RuntimeError(
            f"hf_prune.py exited with code {result.returncode} "
            f"(sparsity={sparsity:.0%})"
        )

    # ── Locate checkpoint ─────────────────────────────────────────────────────
    # LLM-Pruner saves to: log/{log_name}_taylor_{sparsity}/
    log_dir = _LLM_PRUNER_LOG_DIR / f"{log_name}_taylor_{sparsity}"
    if not log_dir.exists():
        # Ratio may be formatted differently (e.g. 0.2 vs 0.20) — glob broadly.
        candidates = sorted(_LLM_PRUNER_LOG_DIR.glob(f"{log_name}_taylor_*"))
        if candidates:
            log_dir = candidates[-1]
        else:
            raise RuntimeError(
                f"Could not find LLM-Pruner log dir under {_LLM_PRUNER_LOG_DIR} "
                f"for log_name='{log_name}'"
            )

    ckpt_files = list(log_dir.glob("*.pt")) + list(log_dir.glob("*.bin"))
    if not ckpt_files:
        raise RuntimeError(
            f"No checkpoint file (.pt or .bin) found in {log_dir}. "
            "Check that --save_model was honoured by hf_prune.py."
        )

    ckpt_path = ckpt_files[0]
    logger.info("Checkpoint saved at %s", ckpt_path)
    return ckpt_path


def _load_llmpruner_checkpoint(
    ckpt_path: Path,
    device: str,
) -> tuple:
    """Load a LLM-Pruner torch checkpoint and return (model, tokenizer).

    LLM-Pruner saves via ``torch.save({'model': model, 'tokenizer': tokenizer}, path)``
    rather than the standard HuggingFace ``save_pretrained()`` format.
    """
    logger.info("Loading LLM-Pruner checkpoint from %s …", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ── Core per-run logic ─────────────────────────────────────────────────────────


def _run_single(
    model_name: str,
    sparsity: float,
    num_examples: int,
    batch_size: int,
    device: str,
    num_eval_samples: Optional[int],
    keep_checkpoint: bool,
    verbose: bool,
) -> dict:
    logger.info("=== Run: sparsity=%.0f%% ===", sparsity * 100)

    ts = _timestamp()
    log_name = f"crasp_{_safe_model_name(model_name).split('_')[-1]}_sp{int(sparsity*100):02d}_{ts}"

    # ── 1. Prune and save ─────────────────────────────────────────────────────
    ckpt_path = _prune_with_llmpruner(
        model_name=model_name,
        sparsity=sparsity,
        log_name=log_name,
        device=device,
        num_examples=num_examples,
        verbose=verbose,
    )

    # ── 2. Load checkpoint ────────────────────────────────────────────────────
    model, tokenizer = _load_llmpruner_checkpoint(ckpt_path, device)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Evaluating pruned model …")
    evaluator = CRASPEvaluator.from_model(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    if num_eval_samples is not None:
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

    # ── 4. Cleanup ────────────────────────────────────────────────────────────
    del model, evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not keep_checkpoint:
        shutil.rmtree(ckpt_path.parent, ignore_errors=True)
        logger.info("Deleted checkpoint directory %s", ckpt_path.parent)

    result: dict = asdict(metrics)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM-Pruner structural pruning at multiple sparsity levels "
            "and compare against the raw baseline."
        )
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path. (default: %(default)s)",
    )
    parser.add_argument(
        "--sparsity", nargs="+", type=float, default=DEFAULT_SPARSITY, metavar="RATIO",
        help="One or more pruning ratios, e.g. 0.20 0.25 0.30. (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR,
        help="Directory containing raw baseline JSON files. (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write result JSON files. (default: %(default)s)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES,
        help=(
            "Calibration examples for Taylor importance estimation. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--num-eval-samples", type=int, default=None, metavar="N",
        help="Limit evaluation to N samples per benchmark (smoke tests).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Evaluation batch size. (default: %(default)s)",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Compute device for both pruning and evaluation. (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-checkpoints", action="store_true",
        help=(
            "Keep pruned model checkpoints after evaluation. "
            "Warning: each checkpoint is ~8 GB."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        args.device = "cpu"

    logger.info("Model       : %s", args.model)
    logger.info("Sparsity    : %s", args.sparsity)
    logger.info("Device      : %s", args.device)
    if torch.cuda.is_available():
        logger.info("GPU         : %s", torch.cuda.get_device_name(0))

    # ── Load baseline ─────────────────────────────────────────────────────────
    baseline_path = _find_baseline(args.baseline_dir, args.model)
    if baseline_path is None:
        logger.error(
            "No raw baseline found for '%s' in %s. "
            "Run: python scripts/run_baselines.py --model %s",
            args.model, args.baseline_dir, args.model,
        )
        return 1
    baseline = _load_baseline_metrics(baseline_path)
    logger.info("Loaded baseline from %s", baseline_path)
    logger.info(
        "Baseline — clinical: %.4f  safety: %.4f",
        baseline["clinical_accuracy"], baseline["safety_score"],
    )

    all_results: list[dict] = []
    ts = _timestamp()

    for sparsity in args.sparsity:
        try:
            result = _run_single(
                model_name=args.model,
                sparsity=sparsity,
                num_examples=args.num_examples,
                batch_size=args.batch_size,
                device=args.device,
                num_eval_samples=args.num_eval_samples,
                keep_checkpoint=args.keep_checkpoints,
                verbose=args.verbose,
            )
        except Exception:
            logger.exception("Run failed: sparsity=%.2f", sparsity)
            continue

        # ── Compute retention ─────────────────────────────────────────────────
        retention_report = compute_retention_report(
            raw_scores={
                "clinical_accuracy": baseline["clinical_accuracy"],
                "safety_score": baseline["safety_score"],
            },
            pruned_scores={
                "clinical_accuracy": result["clinical_accuracy"],
                "safety_score": result["safety_score"],
            },
        )
        result["retention"] = {
            "clinical_retention": retention_report.clinical_retention,
            "safety_retention": retention_report.safety_retention,
            "mean_retention": retention_report.mean_retention,
        }

        # ── Save individual result ────────────────────────────────────────────
        safe_model = _safe_model_name(args.model)
        sparsity_tag = f"sp{int(sparsity * 100):02d}"
        fname = f"llmpruner_{sparsity_tag}_{safe_model}_{ts}.json"
        _save_json(result, args.output_dir / fname)
        all_results.append(result)

    if not all_results:
        logger.error("No runs completed successfully.")
        return 1

    _print_table(all_results, baseline)

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
