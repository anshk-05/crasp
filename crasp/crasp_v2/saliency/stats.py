"""Pure saliency aggregation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass
class HeadActivationStats:
    """Aggregated head-level activation statistics."""

    total_activation: float = 0.0
    max_activation: float = 0.0
    seen_batches: int = 0
    active_batches: int = 0

    @property
    def mean_activation(self) -> float:
        return self.total_activation / self.seen_batches if self.seen_batches else 0.0

    @property
    def coverage(self) -> float:
        return self.active_batches / self.seen_batches if self.seen_batches else 0.0

    @property
    def never_activated(self) -> bool:
        return self.active_batches == 0

    def to_dict(self) -> dict[str, float | int | bool]:
        payload = asdict(self)
        payload["mean_activation"] = self.mean_activation
        payload["coverage"] = self.coverage
        payload["never_activated"] = self.never_activated
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, float | int | bool]) -> "HeadActivationStats":
        """Reconstruct stats from serialized JSON."""
        return cls(
            total_activation=float(payload.get("total_activation", 0.0)),
            max_activation=float(payload.get("max_activation", 0.0)),
            seen_batches=int(payload.get("seen_batches", 0)),
            active_batches=int(payload.get("active_batches", 0)),
        )


def update_head_stats(
    registry: dict[str, dict[int, HeadActivationStats]],
    layer_name: str,
    per_head_values: Iterable[float],
    activation_epsilon: float = 1e-8,
) -> None:
    """Update stats for a single batch worth of per-head values."""
    layer_stats = registry.setdefault(layer_name, {})
    for head_idx, value in enumerate(per_head_values):
        stats = layer_stats.setdefault(head_idx, HeadActivationStats())
        stats.total_activation += float(value)
        stats.max_activation = max(stats.max_activation, float(value))
        stats.seen_batches += 1
        if float(value) > activation_epsilon:
            stats.active_batches += 1


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    return [(value - vmin) / (vmax - vmin) for value in values]


def combine_head_scores(
    reasoning_stats: dict[int, HeadActivationStats],
    safety_stats: dict[int, HeadActivationStats],
    reasoning_weight: float = 0.5,
    safety_weight: float = 0.5,
) -> dict[int, dict[str, float | bool]]:
    """Combine reasoning and safety saliency into a comparable head ranking."""
    head_ids = sorted(set(reasoning_stats) | set(safety_stats))
    reasoning_values = [reasoning_stats.get(idx, HeadActivationStats()).mean_activation for idx in head_ids]
    safety_values = [safety_stats.get(idx, HeadActivationStats()).mean_activation for idx in head_ids]

    reasoning_norm = _normalize(reasoning_values)
    safety_norm = _normalize(safety_values)

    combined: dict[int, dict[str, float | bool]] = {}
    for pos, head_idx in enumerate(head_ids):
        reasoning = reasoning_stats.get(head_idx, HeadActivationStats())
        safety = safety_stats.get(head_idx, HeadActivationStats())
        combined[head_idx] = {
            "reasoning_mean_activation": reasoning.mean_activation,
            "reasoning_coverage": reasoning.coverage,
            "safety_mean_activation": safety.mean_activation,
            "safety_coverage": safety.coverage,
            "reasoning_score": reasoning_norm[pos],
            "safety_score": safety_norm[pos],
            "combined_score": (reasoning_weight * reasoning_norm[pos]) + (safety_weight * safety_norm[pos]),
            "never_activated": reasoning.never_activated and safety.never_activated,
        }
    return combined


def combine_serialized_layer_reports(
    reasoning_layer: dict[int | str, dict[str, float | int | bool]],
    safety_layer: dict[int | str, dict[str, float | int | bool]],
    reasoning_weight: float = 0.5,
    safety_weight: float = 0.5,
) -> dict[int, dict[str, float | bool]]:
    """Combine two serialized layer reports without manual reconstruction."""
    reasoning_stats = {
        int(head_idx): HeadActivationStats.from_dict(payload)
        for head_idx, payload in reasoning_layer.items()
    }
    safety_stats = {
        int(head_idx): HeadActivationStats.from_dict(payload)
        for head_idx, payload in safety_layer.items()
    }
    return combine_head_scores(
        reasoning_stats=reasoning_stats,
        safety_stats=safety_stats,
        reasoning_weight=reasoning_weight,
        safety_weight=safety_weight,
    )
