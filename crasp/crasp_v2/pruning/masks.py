"""Pure pruning-plan helpers."""

from __future__ import annotations

from typing import Any


def rank_heads_for_pruning(
    combined_report: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return a deterministic head pruning order.

    Heads that were never activated are always ranked first, then by ascending
    combined score, then by layer/head index for determinism.
    """
    ranked: list[dict[str, Any]] = []
    for layer_name, layer_heads in combined_report.items():
        for head_idx, payload in layer_heads.items():
            ranked.append(
                {
                    "layer_name": layer_name,
                    "head_idx": int(head_idx),
                    "combined_score": float(payload["combined_score"]),
                    "never_activated": bool(payload["never_activated"]),
                }
            )
    ranked.sort(
        key=lambda item: (
            not item["never_activated"],
            item["combined_score"],
            item["layer_name"],
            item["head_idx"],
        )
    )
    return ranked


def build_layer_mask_map(
    total_heads_per_layer: dict[str, int],
    pruned_heads: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Build binary keep masks from the selected pruned heads."""
    masks = {
        layer_name: [1 for _ in range(num_heads)]
        for layer_name, num_heads in total_heads_per_layer.items()
    }
    for head in pruned_heads:
        layer_name = str(head["layer_name"])
        head_idx = int(head["head_idx"])
        if layer_name in masks and 0 <= head_idx < len(masks[layer_name]):
            masks[layer_name][head_idx] = 0
    return masks


def count_pruned_heads(layer_masks: dict[str, list[int]]) -> int:
    """Count total zeroed heads in a mask map."""
    return sum(mask.count(0) for mask in layer_masks.values())
