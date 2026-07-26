# answer_parser.py
import re
from typing import Iterable, Optional


def extract_choice_answer(text: str, choices: Iterable[str] = ("A", "B", "C", "D", "E")) -> Optional[str]:
    """
    Extract a multiple-choice answer from an LLM response.

    Supported examples:
    - "Final answer: A"
    - "Answer: (A)"
    - "The correct answer is A"
    - "The correct option is (B)"
    - "A"
    - "(C)"
    - "The answer is D"

    Returns:
        The extracted option letter, or None if no valid answer is found.
    """
    if text is None:
        return None

    choices = tuple(c.upper() for c in choices)
    choice_class = "".join(re.escape(c) for c in choices)
    raw = str(text).strip()
    normalized = re.sub(r"\s+", " ", raw)

    if not normalized:
        return None

    direct = normalized.upper().strip()
    if direct in choices:
        return direct

    paren_direct = re.fullmatch(rf"\(?\s*([{choice_class}])\s*\)?[.\s]*", direct)
    if paren_direct:
        return paren_direct.group(1).upper()

    priority_patterns = [
        rf"(?:final\s*answer|final|answer|correct\s*answer|correct\s*option|selected\s*option)\s*(?:is|:)?\s*\(?\s*([{choice_class}])\s*\)?",
        rf"(?:the\s+answer\s+is|the\s+correct\s+answer\s+is|the\s+correct\s+option\s+is)\s*\(?\s*([{choice_class}])\s*\)?",
        rf"(?:option|choice)\s*\(?\s*([{choice_class}])\s*\)?",
        rf"(?:answer|correct\s+answer|selected)\s*(?:is|:)?\s*\(?\s*([{choice_class}])\s*\)?",
    ]

    for pattern in priority_patterns:
        matches = re.findall(pattern, normalized, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        m = re.fullmatch(rf"\(?\s*([{choice_class}])\s*\)?[.\s]*", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    fallback = re.findall(rf"\(([{choice_class}])\)", normalized, flags=re.IGNORECASE)
    if fallback:
        return fallback[-1].upper()

    return None


if __name__ == "__main__":
    examples = [
        "A",
        "(B)",
        "Final answer: C",
        "Answer: (D)",
        "The correct answer is (A).",
        "The answer is B",
    ]

    for item in examples:
        print(item, "->", extract_choice_answer(item))