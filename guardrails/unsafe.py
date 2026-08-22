"""Unsafe-input filter, applied to the transcript before retrieval runs.

Scope, stated honestly: this is a fast lexical first line, not a complete
moderation system. It runs in ~0.1 ms on the critical path and catches the
clearly-unsafe asks a public voice demo actually receives. It is deliberately
transparent -- you can read exactly what it blocks -- rather than a black-box
classifier, and it fails toward answering, because over-blocking a Hindi Q&A
corpus is its own kind of failure.

Patterns cover Devanagari, English, and romanized Hindi, since Sarvam returns
Devanagari but users code-switch and the STT transliterates loanwords.
"""
import re
from typing import Optional, Tuple

# Each category maps to a list of regex fragments. Matching is substring-based
# on a normalized (lowercased, whitespace-collapsed) transcript.
_CATEGORIES = {
    "weapons_explosives": [
        r"\bbomb\b", r"\bexplosive", r"\bdetonat", r"\bgrenade\b", r"\bied\b",
        r"\bammonium nitrate\b", r"\bpipe bomb\b",
        r"बम\s*(बनान|कैसे)", r"विस्फोटक", r"धमाका\s*कैसे",
    ],
    "self_harm": [
        r"\bkill myself\b", r"\bsuicide\b", r"\bself[- ]harm\b",
        r"\bhow to die\b", r"\bend my life\b",
        r"आत्महत्या", r"खुदकुशी", r"खुद को (मार|नुकसान)",
    ],
    "violence": [
        r"\bhow to (kill|murder|poison)\b", r"\bhurt someone\b",
        r"\buntraceable poison\b",
        r"किसी को (मारना|कैसे मार)", r"जहर\s*(देना|कैसे)", r"हत्या\s*कैसे",
    ],
    "illicit_drugs": [
        r"\b(make|synthesi[sz]e|cook)\s+(meth|cocaine|heroin|mdma)\b",
        r"\bmethamphetamine synthesis\b",
        r"ड्रग्स\s*(बनान|कैसे बना)", r"नशीली दवा\s*बनान",
    ],
    "csam": [
        r"\bchild (porn|sexual)", r"\bminor.{0,15}\bsexual\b", r"\bcsam\b",
    ],
    "cyber_attack": [
        r"\bhack (into|someone|his|her|their)\b", r"\bsteal (password|credit card)",
        r"\bransomware\b", r"\bddos attack\b", r"\bkeylogger\b",
        r"हैक\s*(करना|कैसे)", r"पासवर्ड\s*चुरा",
    ],
}

_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _CATEGORIES.items()
}

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def check(query: str) -> Tuple[bool, Optional[str]]:
    """Return (is_unsafe, category)."""
    norm = _normalize(query)
    if not norm:
        return False, None
    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(norm):
                return True, category
    return False, None
