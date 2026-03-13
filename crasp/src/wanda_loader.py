"""src/wanda_loader.py
─────────────────────────────────────────────────────────────────────────────
Wanda pruning integration for CRASP.

Provides two public functions:

  get_medical_cot_loaders  — builds a Wanda-compatible calibration dataloader
                             from the project's Medical CoT JSONL file instead
                             of the standard C4 corpus.

  run_wanda_pruning        — executes Wanda unstructured pruning in-place on a
                             loaded model given an explicit dataloader, without
                             re-downloading C4.  Mirrors the logic of
                             ``prune_wanda()`` in vendors/wanda/lib/prune.py
                             but decouples data loading from pruning.

Usage
-----
    from src.wanda_loader import get_medical_cot_loaders, run_wanda_pruning

    dataloader = get_medical_cot_loaders(
        nsamples=128, seed=42, seqlen=2048, tokenizer=tokenizer
    )
    run_wanda_pruning(
        model=model,
        dataloader=dataloader,
        sparsity_ratio=0.20,
        nsamples=128,
        seqlen=2048,
        device=device,
    )
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Project root — two levels up from this file (src/ → crasp/)
_PROJECT_ROOT: Path = Path(__file__).parent.parent
_DEFAULT_COT_PATH: Path = _PROJECT_ROOT / "data" / "calibration" / "cot_calibration.jsonl"
_WANDA_LIB_PATH: Path = _PROJECT_ROOT / "vendors" / "wanda"


# ── Public helpers ─────────────────────────────────────────────────────────────


def get_c4_loaders(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a Wanda-compatible C4 calibration dataloader with the current datasets API.

    Wanda's vendored ``lib/data.py`` uses the obsolete ``'allenai--c4'`` config
    name which no longer exists.  This reimplements the same logic with the
    correct ``'en'`` config name supported by ``datasets >= 2.x``.

    Returns the same ``list[(inp, tar)]`` format as :func:`get_medical_cot_loaders`.
    """
    from datasets import load_dataset  # type: ignore[import]

    traindata = load_dataset(
        "allenai/c4",
        "en",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    random.seed(seed)
    dataloader: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(nsamples):
        # Keep sampling until we find a document long enough for one seqlen chunk.
        while True:
            i = random.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break
        start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, start : start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))

    logger.info("Built C4 dataloader: %d samples × %d tokens", nsamples, seqlen)
    return dataloader


def get_medical_cot_loaders(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    cot_path: Optional[Path] = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build a Wanda-compatible calibration dataloader from Medical CoT prompts.

    Reads ``data/calibration/cot_calibration.jsonl``, shuffles with ``seed``,
    takes ``nsamples`` entries, tokenises each ``prompt`` field, concatenates
    all tokens into a single long sequence, then slices it into fixed-length
    chunks of ``seqlen`` tokens — exactly the same strategy used by Wanda's
    ``get_c4()`` loader.

    Args:
        nsamples:  Number of calibration samples (chunks) to return.
                   Must match the ``nsamples`` passed to :func:`run_wanda_pruning`.
        seed:      Random seed controlling which JSONL entries are selected.
        seqlen:    Sequence length in tokens per calibration chunk (e.g. 2048).
        tokenizer: HuggingFace tokenizer for the model being pruned.
        cot_path:  Optional override for the JSONL file path.

    Returns:
        A list of ``nsamples`` ``(inp, tar)`` tuples where
        ``inp.shape == tar.shape == (1, seqlen)``.  The label tensor has
        ``-100`` for all positions except the last (matching Wanda's format).

    Raises:
        FileNotFoundError: If the CoT JSONL file does not exist.
        ValueError:        If there are not enough tokens to fill ``nsamples``
                           chunks of length ``seqlen``.
    """
    path = Path(cot_path) if cot_path is not None else _DEFAULT_COT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Medical CoT calibration file not found at {path}. "
            "Run: python scripts/generate_cot_dataset.py"
        )

    # ── Load and shuffle JSONL entries ────────────────────────────────────────
    entries: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                entries.append(obj["prompt"])

    random.seed(seed)
    random.shuffle(entries)
    logger.info("Loaded %d Medical CoT entries from %s", len(entries), path)

    # ── Tokenise and concatenate ───────────────────────────────────────────────
    # We join all prompts, tokenise once, then slice — identical approach to
    # Wanda's C4 loader which also concatenates before slicing.
    combined_text = "\n\n".join(entries)
    all_ids: torch.Tensor = tokenizer(
        combined_text, return_tensors="pt", add_special_tokens=False
    ).input_ids  # shape: (1, total_tokens)

    total_tokens = all_ids.shape[1]
    required_tokens = nsamples * seqlen
    if total_tokens < required_tokens:
        # Tile the corpus rather than failing.  Repeating medical CoT text still
        # provides domain-specific calibration signal, which is the whole point.
        reps = (required_tokens // total_tokens) + 1
        all_ids = all_ids.repeat(1, reps)
        logger.warning(
            "CoT corpus (%d tokens) shorter than required (%d = %d×%d). "
            "Tiling %dx to fill calibration buffer.",
            total_tokens, required_tokens, nsamples, seqlen, reps,
        )
    total_tokens = all_ids.shape[1]  # refresh after potential tiling

    # ── Slice into fixed-length chunks ────────────────────────────────────────
    dataloader: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(nsamples):
        start = i * seqlen
        end = start + seqlen
        inp = all_ids[:, start:end]           # (1, seqlen)
        tar = inp.clone()
        tar[:, :-1] = -100                    # label only last token (Wanda convention)
        dataloader.append((inp, tar))

    logger.info(
        "Built Medical CoT dataloader: %d samples × %d tokens", nsamples, seqlen
    )
    return dataloader


def run_wanda_pruning(
    model,
    dataloader: list[tuple[torch.Tensor, torch.Tensor]],
    sparsity_ratio: float,
    nsamples: int,
    seqlen: int,
    device: str = "cuda",
    use_variant: bool = False,
) -> None:
    """Run Wanda unstructured pruning in-place with an explicit dataloader.

    Replicates the core logic of ``prune_wanda()`` from
    ``vendors/wanda/lib/prune.py`` but accepts a pre-built ``dataloader``
    instead of internally calling ``get_loaders("c4", ...)``.  This allows
    any calibration data (Medical CoT, C4, etc.) to be used without modifying
    Wanda's source code.

    The model's weights are zeroed out in-place according to the computed
    importance scores.  No copy is made.

    Args:
        model:          A loaded HuggingFace causal LM (already on ``device``).
        dataloader:     List of ``(input_ids, labels)`` tuples in Wanda format,
                        as returned by :func:`get_medical_cot_loaders` or
                        Wanda's own ``get_c4()``.
        sparsity_ratio: Fraction of weights to zero out per linear layer,
                        e.g. ``0.20`` for 20% sparsity.
        nsamples:       Number of calibration samples (must match
                        ``len(dataloader)``).
        seqlen:         Token sequence length used in the dataloader.
        device:         Device string (``"cuda"`` or ``"cpu"``).
        use_variant:    If ``True``, use the cumulative-sum Wanda variant
                        (binary-searches for threshold) instead of the
                        default top-k approach.

    Raises:
        ImportError: If the Wanda library is not found at ``vendors/wanda/``.
    """
    # ── Inject vendors/wanda into sys.path ───────────────────────────────────
    wanda_path = str(_WANDA_LIB_PATH)
    if wanda_path not in sys.path:
        sys.path.insert(0, wanda_path)

    try:
        from lib.layerwrapper import WrappedGPT  # type: ignore[import]
        from lib.prune import find_layers, return_given_alpha  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Could not import Wanda library from vendors/wanda/. "
            "Run: git clone https://github.com/locuslab/wanda vendors/wanda"
        ) from exc

    # ── Attach seqlen so prepare_calibration_input can read it ───────────────
    model.seqlen = seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── Capture first-layer inputs (embedding outputs) ────────────────────────
    # We re-implement prepare_calibration_input with a dynamic nsamples
    # instead of the hardcoded 128 in Wanda's version.
    layers = model.model.layers

    embed_device = device
    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        embed_device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=embed_device)
    inps.requires_grad = False
    cache: dict = {"i": 0, "attention_mask": None, "position_ids": None}

    import torch.nn as nn

    class _Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError  # short-circuit the forward pass

    layers[0] = _Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(embed_device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    # ── Layer-wise pruning ────────────────────────────────────────────────────
    logger.info(
        "Starting Wanda pruning: sparsity=%.0f%%, layers=%d",
        sparsity_ratio * 100,
        len(layers),
    )

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # Move tensors to the layer's device for multi-GPU layouts.
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            layer_dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(layer_dev)
            outs = outs.to(layer_dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_dev)
            if position_ids is not None:
                position_ids = position_ids.to(layer_dev)

        # Attach activation-capture wrappers.
        wrapped_layers: dict = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def _add_batch(name):
            def _hook(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return _hook

        handles = [
            subset[name].register_forward_hook(_add_batch(name))
            for name in wrapped_layers
        ]

        # Forward pass to collect activation statistics.
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        # Compute importance scores and apply mask.
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if use_variant:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                tmp_metric = torch.cumsum(sort_res[0], dim=1)
                sum_before = W_metric.sum(dim=1)
                alpha = 0.4
                alpha_hist = [0.0, 0.8]
                W_mask, cur_sparsity = return_given_alpha(
                    alpha, sort_res, W_metric, tmp_metric, sum_before
                )
                while (
                    torch.abs(cur_sparsity - sparsity_ratio) > 0.001
                    and (alpha_hist[1] - alpha_hist[0]) >= 0.001
                ):
                    if cur_sparsity > sparsity_ratio:
                        alpha_new = (alpha + alpha_hist[0]) / 2.0
                        alpha_hist[1] = alpha
                    else:
                        alpha_new = (alpha + alpha_hist[1]) / 2.0
                        alpha_hist[0] = alpha
                    alpha = alpha_new
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                n_prune = int(W_metric.shape[1] * sparsity_ratio)
                indices = sort_res[1][:, :n_prune]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

            logger.debug(
                "Layer %d / %s — actual sparsity: %.4f",
                i,
                name,
                W_mask.float().mean().item(),
            )

        # Propagate outputs to next layer's inputs.
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logger.info("Wanda pruning complete.")
