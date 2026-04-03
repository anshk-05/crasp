from __future__ import annotations

import unittest

from crasp_v2.pruning.masks import build_layer_mask_map, count_pruned_heads, rank_heads_for_pruning


class PruningMaskTests(unittest.TestCase):
    def test_rank_heads_for_pruning_prefers_never_activated(self) -> None:
        ranking = rank_heads_for_pruning(
            {
                "layer0": {
                    0: {"combined_score": 0.8, "never_activated": False},
                    1: {"combined_score": 0.0, "never_activated": True},
                },
                "layer1": {
                    0: {"combined_score": 0.2, "never_activated": False},
                },
            }
        )
        self.assertEqual(ranking[0]["layer_name"], "layer0")
        self.assertEqual(ranking[0]["head_idx"], 1)
        self.assertTrue(ranking[0]["never_activated"])

    def test_build_layer_mask_map_and_count(self) -> None:
        masks = build_layer_mask_map(
            total_heads_per_layer={"layer0": 3, "layer1": 2},
            pruned_heads=[
                {"layer_name": "layer0", "head_idx": 1},
                {"layer_name": "layer1", "head_idx": 0},
            ],
        )
        self.assertEqual(masks["layer0"], [1, 0, 1])
        self.assertEqual(masks["layer1"], [0, 1])
        self.assertEqual(count_pruned_heads(masks), 2)


if __name__ == "__main__":
    unittest.main()
