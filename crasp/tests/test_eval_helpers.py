from __future__ import annotations

import unittest

from crasp_v2.eval.helpers import (
    bootstrap_mean_ci,
    compute_retention,
    extract_answer_letter,
    prediction_diagnostics,
)


class EvalHelperTests(unittest.TestCase):
    def test_extract_answer_letter_handles_explicit_prefix(self) -> None:
        letter = extract_answer_letter("Answer: C because of the mechanism", {"A", "B", "C", "D"})
        self.assertEqual(letter, "C")

    def test_extract_answer_letter_handles_bare_letter(self) -> None:
        letter = extract_answer_letter("b", {"A", "B", "C", "D"})
        self.assertEqual(letter, "B")

    def test_compute_retention_uses_raw_baseline_reference(self) -> None:
        retention = compute_retention(
            raw_clinical=0.6,
            raw_safety=0.2,
            new_clinical=0.54,
            new_safety=0.1,
        )
        self.assertAlmostEqual(retention["clinical_retention"], 0.9)
        self.assertAlmostEqual(retention["safety_retention"], 0.5)
        self.assertAlmostEqual(retention["mean_retention"], 0.7)

    def test_prediction_diagnostics_counts_invalid_outputs(self) -> None:
        diagnostics = prediction_diagnostics(["A", "B", "X", "A"], {"A", "B", "C", "D"})
        self.assertEqual(diagnostics["prediction_distribution"], {"A": 2, "B": 1, "X": 1})
        self.assertEqual(diagnostics["invalid_predictions"], 1)
        self.assertAlmostEqual(diagnostics["invalid_prediction_rate"], 0.25)

    def test_bootstrap_mean_ci_is_deterministic_and_centered(self) -> None:
        ci = bootstrap_mean_ci([True, True, False, False], num_resamples=200, seed=123)
        self.assertAlmostEqual(ci["mean"], 0.5)
        self.assertLessEqual(ci["lower"], ci["mean"])
        self.assertGreaterEqual(ci["upper"], ci["mean"])
        self.assertEqual(ci, bootstrap_mean_ci([True, True, False, False], num_resamples=200, seed=123))


if __name__ == "__main__":
    unittest.main()
