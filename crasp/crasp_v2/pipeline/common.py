"""Shared model, config, and tokenization helpers for the CRASP v2 CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crasp_v2.config import load_config, resolve_paths
from crasp_v2.data.io import load_jsonl
from crasp_v2.pipeline.artifacts import PhaseArtifact, load_phase_artifact
from crasp_v2.pruning.runtime import AttentionHeadMasker


def load_config_bundle(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load config and resolve all root-relative paths."""
    config = load_config(config_path)
    return config, resolve_paths(config)


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_stage_artifact(stage_dir: Path) -> PhaseArtifact:
    """Load the phase metadata file from a stage directory."""
    return load_phase_artifact(stage_dir / "phase_artifact.json")


def load_mask_map(mask_path: Path) -> dict[str, list[int]]:
    """Load a serialized layer mask map."""
    return json.loads(mask_path.read_text(encoding="utf-8"))


def load_tokenizer(model_name: str, cache_dir: Path | None = None) -> Any:
    """Load a tokenizer lazily."""
    from transformers import AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(model_name: str, model_cfg: dict[str, Any], cache_dir: Path | None = None) -> Any:
    """Load the base causal LM lazily."""
    import torch  # type: ignore[import]
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    dtype_name = str(model_cfg.get("dtype", "bfloat16")).lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map.get(dtype_name, torch.bfloat16),
        device_map=model_cfg.get("device_map", "auto"),
        cache_dir=cache_dir,
    )
    model.eval()
    return model


def _apply_adapter_if_present(model: Any, adapter_path: str | None) -> Any:
    if not adapter_path:
        return model
    from peft import PeftModel  # type: ignore[import]

    return PeftModel.from_pretrained(model, adapter_path)


def load_model_for_stage(
    stage: str,
    model_cfg: dict[str, Any],
    checkpoint_root: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[Any, Any, list[AttentionHeadMasker], str]:
    """Reconstruct a model, tokenizer, and active maskers for a given stage.

    `stage` may be a HuggingFace model name or a stage directory containing
    `phase_artifact.json`.
    """
    stage_path = Path(stage)
    if not stage_path.exists():
        tokenizer = load_tokenizer(stage, cache_dir=cache_dir)
        model = load_base_model(stage, model_cfg, cache_dir=cache_dir)
        return model, tokenizer, [], stage

    artifact = read_stage_artifact(stage_path)
    parent_ref = artifact.parent or artifact.base_model
    if parent_ref and not Path(parent_ref).is_absolute() and (stage_path / parent_ref).exists():
        parent_ref = str(stage_path / parent_ref)
    model, tokenizer, maskers, resolved_name = load_model_for_stage(
        parent_ref,
        model_cfg=model_cfg,
        checkpoint_root=checkpoint_root,
        cache_dir=cache_dir,
    )
    adapter_ref = artifact.adapter_path
    if adapter_ref and not Path(adapter_ref).is_absolute():
        adapter_ref = str(stage_path / adapter_ref)
    model = _apply_adapter_if_present(model, adapter_ref)
    if artifact.mask_path:
        mask_path = Path(artifact.mask_path)
        if not mask_path.is_absolute():
            mask_path = stage_path / mask_path
        masker = AttentionHeadMasker(model, load_mask_map(mask_path))
        masker.apply()
        maskers.append(masker)
    return model, tokenizer, maskers, resolved_name


def serialize_json(payload: dict[str, Any], path: Path) -> Path:
    """Write JSON to disk with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_token_batches(
    records_path: Path,
    tokenizer: Any,
    seq_len: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Tokenize JSONL prompt records into fixed-length calibration batches."""
    import torch  # type: ignore[import]

    records = load_jsonl(records_path)
    combined_text = "\n\n".join(str(record["prompt"]) for record in records)
    all_ids = tokenizer(
        combined_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    usable_tokens = (all_ids.shape[1] // seq_len) * seq_len
    all_ids = all_ids[:, :usable_tokens]
    if usable_tokens == 0:
        raise ValueError(f"Calibration corpus at {records_path} is shorter than seq_len={seq_len}.")

    batches: list[dict[str, Any]] = []
    chunks = all_ids.reshape(-1, seq_len)
    for start in range(0, chunks.shape[0], batch_size):
        batch = chunks[start : start + batch_size]
        attention_mask = torch.ones_like(batch)
        batches.append(
            {
                "input_ids": batch,
                "attention_mask": attention_mask,
            }
        )
    return batches


def load_training_texts(paths: list[Path]) -> list[str]:
    """Load prompt text from one or more JSONL files."""
    texts: list[str] = []
    for path in paths:
        for record in load_jsonl(path):
            texts.append(str(record["prompt"]))
    return texts
