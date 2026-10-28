"""Deterministic input guardrails for the SupportScope remit."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailDecision:
    blocked: bool
    status: str
    message: str = ""


# Intentionally explicit patterns: this is a lightweight first line of defense,
# not a complete safety classifier. The limitations are documented in README.md.
UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(phish(?:ing)?|steal (?:customer|someone's|user) credentials|steal passwords?)\b", re.I),
    re.compile(r"\b(hack|break into)\b.{0,50}\b(account|password|system|customer|login)\b", re.I),
    re.compile(r"\b(bypass|evade|disable)\b.{0,40}\b(authentication|login|security check|payment verification|account security)\b", re.I),
    re.compile(r"\b(fake|forge|fabricate)\b.{0,40}\b(receipt|refund claim|proof of purchase|invoice|identity)\b", re.I),
    re.compile(r"\b(reveal|print|show|expose)\b.{0,50}\b(system prompt|api key|secret key|hidden instructions)\b", re.I),
    re.compile(r"\bignore\b.{0,40}\b(previous|all|your)\b.{0,30}\b(instructions|rules|safety)\b", re.I),
)

UNSAFE_REFUSAL = (
    "I can’t help with credential theft, fraud, security bypasses, or requests to expose private system instructions. "
    "I can help with legitimate store-support questions, such as payment safety, returns, or account-related support."
)
OUT_OF_SCOPE_REFUSAL = (
    "I’m SupportScope AI, a customer-support assistant for the fictional Northstar Store. "
    "I can answer questions about shipping, returns, refunds, cancellations, damaged items, warranties, payments, and store support. "
    "I don’t have reliable information to answer this request outside that scope."
)


def check_input(text: str) -> GuardrailDecision:
    """Reject malformed or known unsafe requests before retrieval or API use."""
    if not text or not text.strip():
        return GuardrailDecision(True, "invalid_input", "Please enter a question.")
    if len(text) > 1500:
        return GuardrailDecision(
            True,
            "invalid_input",
            "Please shorten your question to 1,500 characters or fewer.",
        )
    for pattern in UNSAFE_PATTERNS:
        if pattern.search(text):
            return GuardrailDecision(True, "unsafe_refusal", UNSAFE_REFUSAL)
    return GuardrailDecision(False, "allowed")
