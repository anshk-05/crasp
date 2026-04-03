"""MedQA prompt builders used by CRASP v2."""

from __future__ import annotations

from typing import Any

COT_TEMPLATE: str = """\
You are a medical expert answering a USMLE-style clinical question.
Think step by step through the clinical reasoning before selecting your answer.

Question: {question}

Options:
{options}

Reasoning:
1. Identify the key clinical findings or mechanism.
2. Recall relevant pathophysiology, pharmacology, or clinical guidelines.
3. Evaluate each option systematically, ruling out incorrect choices.
4. State your final answer and briefly explain why it is correct.

Answer: {answer}\
"""

PLAIN_TEMPLATE: str = """\
You are a medical expert answering a USMLE-style clinical question.

Question: {question}

Options:
{options}

Answer: {answer}\
"""


def _format_options(options: dict[str, str]) -> str:
    """Render option mapping as a lettered block."""
    return "\n".join(f"{key}) {text}" for key, text in sorted(options.items()))


def _get_answer_string(example: dict[str, Any]) -> str:
    """Extract a canonical answer string from a MedQA example."""
    key = str(example.get("answer_idx", "")).strip()
    text = str(example.get("answer", "")).strip()
    if key and text:
        return f"{key}) {text}"
    return text or key


def build_cot_prompt(example: dict[str, Any]) -> str:
    """Build the clinical chain-of-thought calibration prompt."""
    return COT_TEMPLATE.format(
        question=example["question"],
        options=_format_options(example["options"]),
        answer=_get_answer_string(example),
    )


def build_plain_prompt(example: dict[str, Any]) -> str:
    """Build the plain no-CoT control prompt."""
    return PLAIN_TEMPLATE.format(
        question=example["question"],
        options=_format_options(example["options"]),
        answer=_get_answer_string(example),
    )
