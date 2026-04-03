"""Model traversal and activation capture for head-level saliency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stats import HeadActivationStats, update_head_stats


@dataclass
class AttentionModuleSpec:
    """Metadata for a prunable attention module."""

    layer_name: str
    module: Any
    o_proj: Any
    num_heads: int
    head_dim: int


def discover_attention_modules(model: Any) -> list[AttentionModuleSpec]:
    """Find attention modules that expose `o_proj`, `num_heads`, and `head_dim`."""
    modules: list[AttentionModuleSpec] = []
    for name, module in model.named_modules():
        o_proj = getattr(module, "o_proj", None)
        num_heads = getattr(module, "num_heads", None)
        head_dim = getattr(module, "head_dim", None)
        if o_proj is None or num_heads is None or head_dim is None:
            continue
        modules.append(
            AttentionModuleSpec(
                layer_name=name,
                module=module,
                o_proj=o_proj,
                num_heads=int(num_heads),
                head_dim=int(head_dim),
            )
        )
    return modules


class AttentionHeadCollector:
    """Collect per-head activation saliency from attention output projections."""

    def __init__(self, model: Any, activation_epsilon: float = 1e-8) -> None:
        self.model = model
        self.activation_epsilon = activation_epsilon
        self.specs = discover_attention_modules(model)
        self.handles: list[Any] = []
        self.stats: dict[str, dict[int, HeadActivationStats]] = {}

    def _make_hook(self, spec: AttentionModuleSpec):
        def _hook(_module, inputs):
            tensor = inputs[0]
            shape = getattr(tensor, "shape", None)
            if shape is None or len(shape) != 3:
                return
            batch_size, seq_len, hidden_size = shape
            expected_hidden = spec.num_heads * spec.head_dim
            if hidden_size != expected_hidden:
                return

            reshaped = tensor.reshape(batch_size, seq_len, spec.num_heads, spec.head_dim)
            per_head = reshaped.abs().mean(dim=(0, 1, 3)).detach().cpu().tolist()
            update_head_stats(
                self.stats,
                spec.layer_name,
                per_head,
                activation_epsilon=self.activation_epsilon,
            )

        return _hook

    def attach(self) -> None:
        """Attach all pre-hooks."""
        for spec in self.specs:
            self.handles.append(spec.o_proj.register_forward_pre_hook(self._make_hook(spec)))

    def remove(self) -> None:
        """Remove all active hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def collect(self, batches: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float | int | bool]]]:
        """Run forward passes and collect saliency statistics."""
        import torch  # type: ignore[import]

        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        self.attach()
        try:
            for batch in batches:
                batch = {
                    key: value.to(model_device) if hasattr(value, "to") else value
                    for key, value in batch.items()
                }
                with torch.no_grad():
                    self.model(**batch)
        finally:
            self.remove()

        return {
            layer_name: {
                head_idx: head_stats.to_dict()
                for head_idx, head_stats in layer_stats.items()
            }
            for layer_name, layer_stats in self.stats.items()
        }
