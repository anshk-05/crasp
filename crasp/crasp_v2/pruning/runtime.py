"""Runtime head masking for structured CRASP pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crasp_v2.saliency.modeling import AttentionModuleSpec, discover_attention_modules


@dataclass
class AppliedHeadMask:
    """Tracks an active mask hook attached to an attention block."""

    layer_name: str
    handle: Any


class AttentionHeadMasker:
    """Apply per-layer head masks via `o_proj` forward pre-hooks."""

    def __init__(self, model: Any, layer_masks: dict[str, list[int]]) -> None:
        self.model = model
        self.layer_masks = layer_masks
        self.specs: list[AttentionModuleSpec] = discover_attention_modules(model)
        self.handles: list[AppliedHeadMask] = []

    def _make_hook(self, spec: AttentionModuleSpec):
        def _hook(_module, inputs):
            import torch  # type: ignore[import]

            tensor = inputs[0]
            mask = self.layer_masks.get(spec.layer_name)
            if mask is None:
                return
            batch_size, seq_len, hidden_size = tensor.shape
            expected_hidden = spec.num_heads * spec.head_dim
            if hidden_size != expected_hidden:
                return
            mask_tensor = torch.tensor(
                mask,
                dtype=tensor.dtype,
                device=tensor.device,
            ).view(1, 1, spec.num_heads, 1)
            masked = tensor.reshape(batch_size, seq_len, spec.num_heads, spec.head_dim) * mask_tensor
            return (masked.reshape(batch_size, seq_len, hidden_size),)

        return _hook

    def apply(self) -> None:
        """Attach all configured head masks."""
        for spec in self.specs:
            if spec.layer_name not in self.layer_masks:
                continue
            handle = spec.o_proj.register_forward_pre_hook(self._make_hook(spec))
            self.handles.append(AppliedHeadMask(layer_name=spec.layer_name, handle=handle))

    def remove(self) -> None:
        """Remove all active mask hooks."""
        for applied in self.handles:
            applied.handle.remove()
        self.handles.clear()
