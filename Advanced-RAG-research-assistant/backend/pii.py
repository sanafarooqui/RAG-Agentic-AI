import re
from typing import Callable, Optional

# Structured PII only. Free-text names are not masked in general: research questions name authors
# ("What did Vaswani propose?") and masking them would break retrieval. Names are masked only when
# the user introduces themselves.
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)"
    r"|(?<![\w.])\+\d{1,3}(?:[\s.-]?\d{2,4}){3,4}(?!\d)"
)
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_NAME = re.compile(r"\b((?i:my name is|call me|i am called|i'm called))\s+[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2}")


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _valid_ip(value: str) -> bool:
    return all(int(part) <= 255 for part in value.split("."))


# Order matters: cards and SSNs before phones, so their digit groups are not read as phone numbers
_RULES: list[tuple[str, re.Pattern, Optional[Callable[[str], bool]]]] = [
    ("EMAIL", _EMAIL, None),
    ("SSN", _SSN, None),
    ("CREDIT_CARD", _CARD, _luhn),
    ("PHONE", _PHONE, None),
    ("IP_ADDRESS", _IPV4, _valid_ip),
]


def mask_text(text: str) -> str:
    for label, pattern, valid in _RULES:
        text = pattern.sub(
            lambda m, label=label, valid=valid: f"[{label}]" if valid is None or valid(m.group()) else m.group(),
            text,
        )
    return _NAME.sub(lambda m: f"{m.group(1)} [NAME]", text)


def mask_history(history: list[dict]) -> list[dict]:
    return [{**m, "content": mask_text(m["content"])} for m in history]
