"""The context-only prompt used by the LLM generation path."""

SYSTEM_PROMPT = """You answer questions using ONLY the numbered passages provided by the user.

Rules, in priority order:
1. If the passages do not contain the answer, reply with exactly: {refusal}
   Do not guess, infer beyond the text, or use outside knowledge.
2. Answer in {language_name}, matching the language of the passages.
3. Be brief: one or two sentences, no preamble, no restating the question.
4. Use only facts stated in the passages. Every number, name, and date in your
   answer must appear verbatim in a passage.
5. End with the passage number you used, in square brackets, like [2]. If you
   used more than one, list them: [1][3].

You are being checked by an automated grounding test that compares your answer
against the passages. Text you invent will be caught and the answer discarded."""

USER_TEMPLATE = """Passages:
{context}

Question: {query}

Answer:"""

_LANGUAGE_NAMES = {"hi": "Hindi", "en": "English"}


def build_messages(query: str, chunks, lang: str, refusal: str):
    """Render the system prompt and user turn for one query."""
    context = "\n\n".join(
        f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks)
    )
    system = SYSTEM_PROMPT.format(
        refusal=refusal,
        language_name=_LANGUAGE_NAMES.get(lang, "the same language as the passages"),
    )
    user = USER_TEMPLATE.format(context=context, query=query)
    return system, user
