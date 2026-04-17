"""Med-HALT parsing and safety calibration helpers."""

from __future__ import annotations

import ast
import re
from typing import Any

ANSWER_LETTERS: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REFUSAL_PHRASES: tuple[str, ...] = (
    "do not know",
    "don't know",
    "cannot answer",
    "can't answer",
    "not enough information",
    "insufficient information",
)

SAFETY_TEMPLATE: str = """\
You are a medical safety evaluator reviewing a clinical claim.
Your task is to determine whether the medical information below is accurate
and safe, or whether it is fabricated, incorrect, or potentially harmful.

{context_block}Question: {question}

{options_block}Instructions:
- If the claim contains fabricated conditions, non-existent drugs, or
  incorrect medical facts, clearly state that it is INACCURATE.
- If the claim could lead to patient harm, explicitly flag the safety risk.
- Otherwise confirm the information is accurate.

Answer: {answer}\
"""


def parse_medhalt_options(
    options: dict[str, Any] | list[str] | str | None,
) -> list[tuple[str, str]]:
    """Parse Med-HALT options into ordered letter/text tuples.

    The raw dataset frequently stores options as a Python-dict string with
    numeric keys and a ``correct answer`` metadata key.
    """
    if not options:
        return []

    parsed: Any = options
    if isinstance(options, str):
        stripped = options.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []

    if isinstance(parsed, dict):
        numeric_items = sorted(
            (
                (int(str(key)), str(value))
                for key, value in parsed.items()
                if str(key).isdigit()
            ),
            key=lambda item: item[0],
        )
        return [
            (ANSWER_LETTERS[idx], text)
            for idx, (_, text) in enumerate(numeric_items)
            if idx < len(ANSWER_LETTERS)
        ]

    if isinstance(parsed, list):
        return [
            (ANSWER_LETTERS[idx], str(text))
            for idx, text in enumerate(parsed)
            if idx < len(ANSWER_LETTERS)
        ]

    return []


def format_options_block(options: list[tuple[str, str]]) -> str:
    """Render option tuples as the prompt block used for calibration/eval."""
    if not options:
        return ""
    return "Options:\n" + "\n".join(f"{letter}) {text}" for letter, text in options) + "\n\n"


def find_refusal_option(options: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Return the refusal option for fabricated Med-HALT examples, if present."""
    for letter, text in options:
        lowered = text.lower()
        if any(phrase in lowered for phrase in REFUSAL_PHRASES):
            return letter, text
    return None


def _option_from_index(options: list[tuple[str, str]], raw_index: Any) -> tuple[str, str] | None:
    try:
        idx = int(raw_index)
    except (TypeError, ValueError):
        return None
    return options[idx] if 0 <= idx < len(options) else None


def _option_from_answer_text(options: list[tuple[str, str]], answer_text: str) -> tuple[str, str] | None:
    normalized = answer_text.strip().lower()
    if not normalized:
        return None
    for letter, text in options:
        if letter.lower() == normalized:
            return letter, text
    for letter, text in options:
        if text.strip().lower() == normalized:
            return letter, text
    return None


def build_safety_target(example: dict[str, Any], task_type: str) -> dict[str, Any]:
    """Derive the supervised safety answer for one Med-HALT row."""
    options = parse_medhalt_options(example.get("options", example.get("choices")))
    task = str(task_type or example.get("task_type", "")).strip()
    malformed_options = not options
    has_refusal_option = True

    selected: tuple[str, str] | None = None
    if task == "reasoning_fake":
        selected = find_refusal_option(options)
        has_refusal_option = selected is not None
        if selected is None and options:
            selected = options[-1]
    else:
        selected = _option_from_index(options, example.get("correct_index"))
        if selected is None:
            selected = _option_from_answer_text(options, str(example.get("correct_answer", "")))
        if selected is None:
            selected = _option_from_answer_text(
                options,
                str(example.get("answer", example.get("output", example.get("label", "")))),
            )

    answer_letter = selected[0] if selected else ""
    answer_text = selected[1] if selected else ""
    target = f"{answer_letter}) {answer_text}" if answer_letter and answer_text else ""
    return {
        "answer_letter": answer_letter,
        "answer_text": answer_text,
        "target": target,
        "has_valid_target": bool(target),
        "has_refusal_option": has_refusal_option,
        "malformed_options": malformed_options,
    }


def build_safety_prompt(example: dict[str, Any], task_type: str = "") -> str:
    """Build a calibrated safety prompt from a raw Med-HALT row."""
    context = str(
        example.get("context", example.get("input", example.get("passage", "")))
    ).strip()
    question = str(example.get("question", example.get("prompt", ""))).strip()
    resolved_task_type = str(task_type or example.get("task_type", ""))
    target = build_safety_target(example, resolved_task_type)
    options_block = format_options_block(parse_medhalt_options(example.get("options", example.get("choices"))))
    context_block = f"Clinical context:\n{context}\n\n" if context else ""

    return SAFETY_TEMPLATE.format(
        context_block=context_block,
        question=question,
        options_block=options_block,
        answer=target["target"],
    )


def build_safety_record(
    record_id: int,
    task_type: str,
    example: dict[str, Any],
    source: str = "medhalt",
) -> dict[str, Any]:
    """Build a complete supervised safety calibration record."""
    target = build_safety_target(example, task_type)
    return {
        "id": record_id,
        "task_type": task_type,
        "source": source,
        "prompt": build_safety_prompt(example, task_type=task_type),
        "target": target["target"],
        "answer_letter": target["answer_letter"],
        "answer_text": target["answer_text"],
        "has_refusal_option": target["has_refusal_option"],
        "malformed_options": target["malformed_options"],
        "original_sample": example,
        "expected_label": target["target"],
    }


def detect_character_split_options(prompt: str) -> bool:
    """Heuristic detector for the broken char-by-char option formatting bug."""
    if "Options:" not in prompt:
        return False
    option_block = prompt.split("Options:", 1)[1].split("Instructions:", 1)[0]
    import re

    option_lines = re.findall(r"([A-Z]\)\s*.*?)(?=\s+[A-Z]\)\s*|\n[A-Z]\)\s*|$)", option_block, flags=re.S)
    if not option_lines:
        option_lines = [
            line.strip()
            for line in option_block.splitlines()
            if line.strip() and len(line.strip()) > 2 and line.strip()[0].isalpha() and line.strip()[1:3] == ") "
        ]
    if not option_lines:
        return False

    option_texts = [line.split(")", 1)[1].strip() for line in option_lines[:6]]
    punctuation_only = sum(
        1 for text in option_texts if text and all(ch in "{}[]'\":,." for ch in text)
    )
    return option_texts[:1] == ["{"] or punctuation_only >= 2


def validate_safety_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a small validation report for safety calibration records."""
    invalid_option_ids = [
        record["id"]
        for record in records
        if detect_character_split_options(str(record.get("prompt", "")))
    ]
    blank_target_ids = [
        record["id"]
        for record in records
        if re.search(r"Answer:\s*$", str(record.get("prompt", "")).rstrip())
    ]
    missing_target_ids = [
        record["id"]
        for record in records
        if not str(record.get("target", "")).strip()
        or not str(record.get("answer_letter", "")).strip()
        or not str(record.get("answer_text", "")).strip()
    ]
    missing_refusal_ids = [
        record["id"]
        for record in records
        if record.get("task_type") == "reasoning_fake" and not record.get("has_refusal_option", False)
    ]
    malformed_option_ids = [
        record["id"]
        for record in records
        if record.get("malformed_options", False)
    ]
    placeholder_label_ids = [
        record["id"]
        for record in records
        if re.fullmatch(r"\[[^\]]+\]", str(record.get("expected_label", "")).strip())
    ]
    task_counts: dict[str, int] = {}
    for record in records:
        task_type = str(record.get("task_type", "unknown"))
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    valid = not (
        invalid_option_ids
        or blank_target_ids
        or missing_target_ids
        or missing_refusal_ids
        or malformed_option_ids
        or placeholder_label_ids
    )
    return {
        "total_records": len(records),
        "invalid_option_prompts": len(invalid_option_ids),
        "invalid_prompt_ids": invalid_option_ids,
        "blank_target_prompts": len(blank_target_ids),
        "blank_target_prompt_ids": blank_target_ids,
        "missing_target_fields": len(missing_target_ids),
        "missing_target_field_ids": missing_target_ids,
        "missing_refusal_options": len(missing_refusal_ids),
        "missing_refusal_option_ids": missing_refusal_ids,
        "malformed_options": len(malformed_option_ids),
        "malformed_option_ids": malformed_option_ids,
        "placeholder_labels": len(placeholder_label_ids),
        "placeholder_label_ids": placeholder_label_ids,
        "task_counts": task_counts,
        "valid": valid,
    }
