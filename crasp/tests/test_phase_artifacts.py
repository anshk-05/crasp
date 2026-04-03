from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crasp_v2.pipeline.artifacts import PhaseArtifact, load_phase_artifact, save_phase_artifact


class PhaseArtifactTests(unittest.TestCase):
    def test_phase_artifact_round_trip(self) -> None:
        artifact = PhaseArtifact.create(
            phase="post_prune",
            model_name="meta-llama/Llama-3.1-8B",
            base_model="meta-llama/Llama-3.1-8B",
            parent="/tmp/post_sft",
            mask_path="mask.json",
            metrics_path="metrics.json",
            extra={"pruned_heads": 12},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_phase_artifact(artifact, Path(tmpdir))
            loaded = load_phase_artifact(path)
        self.assertEqual(loaded.phase, "post_prune")
        self.assertEqual(loaded.extra["pruned_heads"], 12)


if __name__ == "__main__":
    unittest.main()
