from __future__ import annotations

import unittest

from crasp_v2.eval.helpers import compute_retention, extract_answer_letter


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


if __name__ == "__main__":
    unittest.main()
