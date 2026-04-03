"""Load YAML or JSON config files for CRASP v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load a configuration file from YAML or JSON."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML config files.") from exc
    return yaml.safe_load(text)


def resolve_paths(config: dict[str, Any], root_dir: Path | None = None) -> dict[str, Path]:
    """Resolve workspace paths from config."""
    root_dir = root_dir or Path.cwd()
    paths = config.get("paths", {})
    volume_root = Path(paths.get("volume_root", root_dir))

    def _resolve(value: str, fallback: str) -> Path:
        raw = Path(paths.get(value, fallback))
        return raw if raw.is_absolute() else volume_root / raw

    return {
        "volume_root": volume_root,
        "model_cache": _resolve("model_cache", "models"),
        "data_dir": _resolve("data_dir", "data"),
        "checkpoint_dir": _resolve("checkpoint_dir", "checkpoints"),
        "results_dir": _resolve("results_dir", "results"),
    }
