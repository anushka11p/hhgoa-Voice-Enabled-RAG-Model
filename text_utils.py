"""Tokenization and sentence splitting shared by generation and guardrails.

Deliberately dependency-free and regex-based. A proper Hindi NLP stack
(stanza, indic-nlp) would be more accurate, but each of these functions sits
on the per-request latency path where the whole budget is 200 ms, and the
accuracy we need -- "which sentence best answers this?" and "do these tokens
appear in the source?" -- is well served by regex.
"""
import re
from typing import List

# Devanagari danda / double danda terminate Hindi sentences; keep the Latin
# terminators too since the corpus mixes in English fragments and numerals.
_SENT_SPLIT = re.compile(r"(?<=[।॥.!?])\s+")

# Word characters across Devanagari and Latin, plus digits. The Devanagari
# range deliberately skips U+0964/U+0965 (danda, double danda): those are
# sentence punctuation, and letting them into tokens produces "रखें।" != "रखें"
# and silently breaks every overlap comparison.
_TOKEN = re.compile(r"[\u0900-\u0963\u0966-\u097F]+|[A-Za-z]+|\d+(?:[.,]\d+)*")

# High-frequency Hindi function words plus common English ones. Removing these
# stops the overlap scores from being dominated by grammatical glue.
_STOPWORDS = {
    # Hindi
    "का", "के", "की", "को", "में", "से", "है", "हैं", "था", "थी", "थे", "और",
    "पर", "यह", "वह", "एक", "कि", "जो", "ने", "हो", "भी", "तो", "ही", "कर",
    "गया", "गई", "गए", "करने", "किया", "लिए", "साथ", "तक", "या", "नहीं",
    "क्या", "कौन", "कब", "कहाँ", "कैसे", "क्यों", "कितने", "कितना", "कुछ",
    "अपने", "इस", "उस", "होता", "होती", "होते", "रहा", "रही", "रहे", "करता",
    "करती", "करते", "जाता", "जाती", "जाते", "बाद", "द्वारा", "वाले", "वाली",
    # English
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and",
    "or", "for", "on", "at", "by", "with", "from", "as", "that", "this",
    "it", "be", "been", "has", "have", "had", "what", "which", "who", "how",
    "when", "where", "why", "does", "do", "did", "can", "will", "would",
}


def split_sentences(text: str) -> List[str]:
    """Split into sentences on Devanagari danda or Latin terminators."""
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def tokenize(text: str) -> List[str]:
    """Lowercased word tokens. Casing is a no-op for Devanagari but matters
    for the Latin loanwords and unit strings mixed into the corpus."""
    return [t.lower() for t in _TOKEN.findall(text or "")]


def content_tokens(text: str) -> List[str]:
    """Tokens with stopwords and single characters removed."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def overlap_ratio(candidate: str, source: str) -> float:
    """Fraction of the candidate's content tokens that appear in the source.

    This is the grounding primitive: it asks "is every meaningful word in this
    answer actually present in the passage we cited?" It is asymmetric on
    purpose -- the source is allowed to say far more than the answer does.
    """
    cand = content_tokens(candidate)
    if not cand:
        return 0.0
    src = set(content_tokens(source))
    return sum(1 for t in cand if t in src) / len(cand)


def keyword_coverage(query: str, sentence: str) -> float:
    """Fraction of the query's content tokens present in the sentence."""
    q = content_tokens(query)
    if not q:
        return 0.0
    s = set(content_tokens(sentence))
    return sum(1 for t in q if t in s) / len(q)
