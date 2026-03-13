"""src/sparsegpt_loader.py
─────────────────────────────────────────────────────────────────────────────
SparseGPT pruning integration for CRASP.

Provides:
  run_sparsegpt_pruning  — executes SparseGPT second-order pruning in-place
                           on a loaded model given an explicit dataloader.

The calibration dataloaders (Medical CoT and C4) are shared with the Wanda
integration — import them from src.wanda_loader:

    from src.wanda_loader import get_c4_loaders, get_medical_cot_loaders

Usage
-----
    from src.sparsegpt_loader import run_sparsegpt_pruning
    from src.wanda_loader import get_medical_cot_loaders

    dataloader = get_medical_cot_loaders(
        nsamples=128, seed=42, seqlen=2048, tokenizer=tokenizer
    )
    run_sparsegpt_pruning(
        model=model,
        dataloader=dataloader,
        sparsity_ratio=0.20,
        nsamples=128,
        seqlen=2048,
        device=device,
    )
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).parent.parent
_WANDA_LIB_PATH: Path = _PROJECT_ROOT / "vendors" / "wanda"


def run_sparsegpt_pruning(
    model,
    dataloader: list[tuple[torch.Tensor, torch.Tensor]],
    sparsity_ratio: float,
    nsamples: int,
    seqlen: int,
    device: str = "cuda",
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> None:
    """Run SparseGPT unstructured pruning in-place with an explicit dataloader.

    SparseGPT uses second-order (Hessian-based) information to minimise the
    layer-wise reconstruction error when zeroing weights, giving better
    accuracy retention than magnitude-based methods at the same sparsity.

    Replicates ``prune_sparsegpt()`` from ``vendors/wanda/lib/prune.py`` but
    accepts a pre-built ``dataloader`` instead of hardcoding C4, matching the
    same interface as ``run_wanda_pruning()`` in ``src/wanda_loader.py``.

    Args:
        model:          A loaded HuggingFace causal LM (already on ``device``).
        dataloader:     List of ``(input_ids, labels)`` tuples, as returned by
                        ``get_medical_cot_loaders()`` or ``get_c4_loaders()``.
        sparsity_ratio: Fraction of weights to prune per linear layer,
                        e.g. ``0.20`` for 20% sparsity.
        nsamples:       Number of calibration samples (must match
                        ``len(dataloader)``).
        seqlen:         Token sequence length used in the dataloader.
        device:         Device string (``"cuda"`` or ``"cpu"``).
        blocksize:      Column block size for the SparseGPT Hessian update.
                        128 is the standard value from the paper.
        percdamp:       Hessian diagonal damping as a fraction of the mean
                        diagonal.  Prevents numerical issues on near-zero
                        diagonal entries.

    Raises:
        ImportError: If the Wanda library is not found at ``vendors/wanda/``.
    """
    wanda_path = str(_WANDA_LIB_PATH)
    if wanda_path not in sys.path:
        sys.path.insert(0, wanda_path)

    try:
        from lib.sparsegpt import SparseGPT  # type: ignore[import]
        from lib.prune import find_layers   # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Could not import SparseGPT from vendors/wanda/. "
            "Run: git clone https://github.com/locuslab/wanda vendors/wanda"
        ) from exc

    model.seqlen = seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # Determine device for embedding layer (may differ under device_map="auto").
    embed_device = device
    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        embed_device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=embed_device
    )
    inps.requires_grad = False
    cache: dict = {"i": 0, "attention_mask": None, "position_ids": None}

    # ── Capture first-layer inputs via a short-circuit Catcher ────────────────
    class _Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

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

    logger.info(
        "Starting SparseGPT pruning: sparsity=%.0f%%, layers=%d, blocksize=%d",
        sparsity_ratio * 100,
        len(layers),
        blocksize,
    )

    # ── Layer-wise pruning ────────────────────────────────────────────────────
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # Move tensors to the correct device for multi-GPU layouts.
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            layer_dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(layer_dev)
            outs = outs.to(layer_dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_dev)
            if position_ids is not None:
                position_ids = position_ids.to(layer_dev)

        # Wrap each linear sub-layer to accumulate Hessian statistics.
        gpts: dict = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def _add_batch(name):
            def _hook(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return _hook

        handles = [
            subset[name].register_forward_hook(_add_batch(name))
            for name in gpts
        ]

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        # Prune using the accumulated Hessian.
        for name in gpts:
            logger.debug("SparseGPT layer %d / %s", i, name)
            gpts[name].fasterprune(
                sparsity_ratio,
                prune_n=0,
                prune_m=0,
                blocksize=blocksize,
                percdamp=percdamp,
            )
            gpts[name].free()

        # Propagate updated outputs to the next layer's inputs.
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logger.info("SparseGPT pruning complete.")
