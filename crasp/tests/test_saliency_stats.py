from __future__ import annotations

import unittest

from crasp_v2.saliency.stats import combine_serialized_layer_reports, update_head_stats


class SaliencyStatsTests(unittest.TestCase):
    def test_update_head_stats_creates_non_empty_layer_report(self) -> None:
        registry = {}
        update_head_stats(registry, "model.layers.0.self_attn", [0.2, 0.0, 0.4])
        update_head_stats(registry, "model.layers.0.self_attn", [0.1, 0.0, 0.3])
        self.assertIn("model.layers.0.self_attn", registry)
        self.assertEqual(len(registry["model.layers.0.self_attn"]), 3)
        self.assertFalse(registry["model.layers.0.self_attn"][0].never_activated)
        self.assertTrue(registry["model.layers.0.self_attn"][1].never_activated)

    def test_combined_scores_prioritize_never_activated_heads(self) -> None:
        reasoning = {
            0: {"total_activation": 1.0, "max_activation": 0.6, "seen_batches": 2, "active_batches": 2},
            1: {"total_activation": 0.0, "max_activation": 0.0, "seen_batches": 2, "active_batches": 0},
        }
        safety = {
            0: {"total_activation": 0.8, "max_activation": 0.5, "seen_batches": 2, "active_batches": 2},
            1: {"total_activation": 0.0, "max_activation": 0.0, "seen_batches": 2, "active_batches": 0},
        }
        combined = combine_serialized_layer_reports(reasoning, safety)
        self.assertIn(0, combined)
        self.assertIn(1, combined)
        self.assertTrue(combined[1]["never_activated"])
        self.assertLess(combined[1]["combined_score"], combined[0]["combined_score"])


if __name__ == "__main__":
    unittest.main()
