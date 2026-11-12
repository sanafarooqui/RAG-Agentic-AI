import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from backend.pii import mask_history, mask_text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("email me at jordan.blake@example.com please", "email me at [EMAIL] please"),
        ("SSN 078-05-1120", "SSN [SSN]"),
        ("phone 415-555-0137", "phone [PHONE]"),
        ("call (415) 555-0137 or +1 415.555.0137", "call [PHONE] or [PHONE]"),
        ("+44 20 7946 0958", "[PHONE]"),
        ("card 4111 1111 1111 1111 exp 09/29", "card [CREDIT_CARD] exp 09/29"),
        ("card 4111-1111-1111-1111", "card [CREDIT_CARD]"),
        ("server at 192.168.1.20", "server at [IP_ADDRESS]"),
        ("My name is Jordan Blake, summarize it", "My name is [NAME], summarize it"),
        ("call me Priya", "call me [NAME]"),
        (
            "My name is Jordan Blake, email jordan.blake@example.com, phone 415-555-0137 and SSN 078-05-1120.",
            "My name is [NAME], email [EMAIL], phone [PHONE] and SSN [SSN].",
        ),
    ],
)
def test_masks_pii(text, expected):
    assert mask_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "How many identical layers are stacked in the Transformer's encoder and decoder?",
        "Training took 100,000 steps at about 0.4 seconds each, roughly 12 hours.",
        "Transformer (big) reached 28.4 BLEU, 41.8 BLEU and 2.3 · 10^19 FLOPs on 8 P100 GPUs.",
        "Adam with beta1 = 0.9, beta2 = 0.98, epsilon = 10^-9 and warmup_steps = 4000.",
        "What does arXiv:1706.03762v7 say about WMT 2014 English-to-German with 4.5 million pairs?",
        "Single-head attention: 5.29 PPL, 24.9 BLEU, 65 million parameters (213 million for big).",
        "What did Vaswani et al. propose, and how does Shazeer's mixture of experts compare?",
        "I am interested in how the Transformer handles long sequences.",
        "A 16 digit number that fails Luhn: 1234 5678 9012 3456",
        "Version 1.2.3.999 and 300.1.1.1 are not addresses",
    ],
)
def test_leaves_research_text_alone(text):
    assert mask_text(text) == text


def test_masking_is_idempotent():
    once = mask_text("My name is Jordan Blake, jordan@example.com, 415-555-0137")
    assert mask_text(once) == once


def test_mask_history_keeps_roles_and_masks_content():
    history = [{"role": "user", "content": "I'm at jordan@example.com"}, {"role": "assistant", "content": "ok"}]
    assert mask_history(history) == [
        {"role": "user", "content": "I'm at [EMAIL]"},
        {"role": "assistant", "content": "ok"},
    ]
