"""Artifact metadata helpers for CRASP v2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PhaseArtifact:
    """Serializable metadata for a pipeline phase output."""

    phase: str
    created_at: str
    model_name: str
    base_model: str
    parent: str | None = None
    adapter_path: str | None = None
    mask_path: str | None = None
    metrics_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        phase: str,
        model_name: str,
        base_model: str,
        parent: str | None = None,
        adapter_path: str | None = None,
        mask_path: str | None = None,
        metrics_path: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "PhaseArtifact":
        return cls(
            phase=phase,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            base_model=base_model,
            parent=parent,
            adapter_path=adapter_path,
            mask_path=mask_path,
            metrics_path=metrics_path,
            extra=extra or {},
        )


def save_phase_artifact(artifact: PhaseArtifact, output_dir: Path) -> Path:
    """Write the phase metadata file into the target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase_artifact.json"
    path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")
    return path


def load_phase_artifact(path: Path) -> PhaseArtifact:
    """Load phase metadata from disk."""
    return PhaseArtifact(**json.loads(path.read_text(encoding="utf-8")))
