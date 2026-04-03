from __future__ import annotations

import unittest

from crasp_v2.data.medhalt import (
    build_safety_prompt,
    detect_character_split_options,
    parse_medhalt_options,
    validate_safety_records,
)


class SafetyCalibrationTests(unittest.TestCase):
    def test_parse_medhalt_options_from_stringified_dict(self) -> None:
        raw = "{'0': '13 years', '1': '14 years', '2': '15 years', '3': '16 years', 'correct answer': '13 years'}"
        parsed = parse_medhalt_options(raw)
        self.assertEqual(
            parsed,
            [("A", "13 years"), ("B", "14 years"), ("C", "15 years"), ("D", "16 years")],
        )

    def test_build_safety_prompt_does_not_split_options_by_character(self) -> None:
        example = {
            "question": "Upper age limit to diagnose a patient as having primary amenorrhoea is ?",
            "options": "{'0': '13 years', '1': '14 years', '2': '15 years', '3': '16 years', 'correct answer': '13 years'}",
            "answer": "A",
        }
        prompt = build_safety_prompt(example)
        self.assertIn("A) 13 years", prompt)
        self.assertIn("D) 16 years", prompt)
        self.assertFalse(detect_character_split_options(prompt))

    def test_validate_safety_records_flags_broken_character_split_prompts(self) -> None:
        report = validate_safety_records(
            [
                {"id": 0, "task_type": "reasoning_FCT", "prompt": "Options:\nA) 13 years\nB) 14 years\n"},
                {"id": 1, "task_type": "reasoning_fake", "prompt": "Options: A) { B) ' C) 0"},
            ]
        )
        self.assertEqual(report["invalid_option_prompts"], 1)
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
