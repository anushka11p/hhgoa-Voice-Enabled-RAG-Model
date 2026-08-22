"""Text utilities must handle Devanagari, not just Latin script."""
from text_utils import (content_tokens, keyword_coverage, overlap_ratio,
                        split_sentences, tokenize)


def test_splits_on_devanagari_danda():
    text = "ब्रसेल्स स्प्राउट्स को फ्रिज में रखें। तीन दिनों में उपयोग करें।"
    assert len(split_sentences(text)) == 2


def test_splits_on_latin_terminators():
    assert len(split_sentences("First one. Second one! Third one?")) == 3


def test_danda_is_not_part_of_a_token():
    # Regression: the Devanagari block includes U+0964, so a naive range put
    # the danda inside tokens and broke every overlap comparison.
    assert "रखें" in tokenize("फ्रिज में रखें।")
    assert "रखें।" not in tokenize("फ्रिज में रखें।")


def test_stopwords_are_dropped():
    toks = content_tokens("यह एक बहुत अच्छी किताब है")
    assert "है" not in toks and "एक" not in toks
    assert "किताब" in toks


def test_overlap_ratio_is_asymmetric():
    src = "ब्रसेल्स स्प्राउट्स को फ्रिज में तीन दिन रखें"
    assert overlap_ratio("तीन दिन", src) == 1.0
    assert overlap_ratio("हवाई जहाज़ का टिकट", src) < 0.5


def test_empty_inputs_do_not_crash():
    assert split_sentences("") == []
    assert tokenize("") == []
    assert overlap_ratio("", "abc") == 0.0
    assert keyword_coverage("", "abc") == 0.0
