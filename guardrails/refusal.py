"""Builds the user-facing refusal messages.

Each gate gets its own wording. A user who asked something off-topic and a
user whose question we simply could not find an answer for are in different
situations, and a single generic "I can't help with that" tells them nothing
about whether rephrasing would help.
"""
import config

_MESSAGES = {
    "hi": {
        "unsafe": "यह प्रश्न असुरक्षित सामग्री से संबंधित है, इसलिए मैं इसका उत्तर नहीं दे सकता।",
        "off_topic": "यह प्रश्न मेरे उपलब्ध दस्तावेज़ों के दायरे से बाहर है, इसलिए मेरे पास इसका उत्तर नहीं है।",
        "low_confidence": "मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
        "ungrounded": "मुझे इसका विश्वसनीय उत्तर नहीं मिला, इसलिए मैं अनुमान नहीं लगाऊँगा।",
        "empty": "मुझे कोई प्रश्न सुनाई नहीं दिया। कृपया दोबारा बोलें।",
    },
    "en": {
        "unsafe": "I can't help with that — the question involves unsafe content.",
        "off_topic": "That's outside what the indexed documents cover, so I don't have an answer for it.",
        "low_confidence": "I don't have enough information to answer that.",
        "ungrounded": "I couldn't find a reliable answer in the sources, so I won't guess.",
        "empty": "I didn't catch a question. Please try speaking again.",
    },
}


def message(stage: str) -> str:
    table = _MESSAGES.get(config.LANG, _MESSAGES["en"])
    return table.get(stage, config.REFUSAL_TEXT)
