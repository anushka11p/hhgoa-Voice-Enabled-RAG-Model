"""Stage 0 - build the corpus and the evaluation set.

Pulls the ai4bharat Indic MS MARCO parquet for the configured language,
flattens it into flat passage documents, dedupes, and writes two artefacts:

  data/processed/cleaned_corpus.json  - the passages we index
  data/processed/eval_queries.json    - query -> relevant passage pairs

The second file is what makes the dataset double as a retrieval-quality eval
set and a latency benchmark query set: every row already pairs a real query
with the passage a human marked relevant, so we never have to hand-label.

Run:  python -m data.build_corpus
"""
import json
import re
import sys

import config


def _clean(text: str) -> str:
    """Collapse whitespace and strip the odd control character."""
    if not isinstance(text, str):
        return ""
    text = text.replace("​", " ").replace("\xa0", " ")
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build(max_rows: int = None) -> tuple[list[dict], list[dict]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    max_rows = max_rows or config.MAX_CORPUS_ROWS

    print(f"Downloading {config.DATASET_REPO} :: {config.DATASET_FILE} ...")
    path = hf_hub_download(
        config.DATASET_REPO, config.DATASET_FILE, repo_type="dataset"
    )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} raw rows. Columns: {list(df.columns)}")

    corpus: list[dict] = []
    eval_rows: list[dict] = []
    # Dedupe passages by text -- the same passage can be paired with more than
    # one query, and indexing it twice would corrupt the recall denominator.
    seen_passages: dict[str, str] = {}

    for idx, row in df.head(max_rows).iterrows():
        query = _clean(row.get("query", ""))
        passage = _clean(row.get("passage", ""))
        answer = _clean(row.get("answer", ""))
        if not query or not passage:
            continue

        query_id = str(row.get("query_id", f"q{idx}"))

        if passage in seen_passages:
            passage_id = seen_passages[passage]
        else:
            # The source parquet leaves passage_id blank, so mint a stable one.
            passage_id = f"{query_id}_p{len(corpus)}"
            seen_passages[passage] = passage_id
            corpus.append(
                {
                    "passage_id": passage_id,
                    "query_id": query_id,
                    "language": str(row.get("language", config.LANG)),
                    "source": "IndicMSMARCO",
                    "source_query": query,
                    "passage_text": passage,
                    "is_selected": bool(row.get("is_selected", False)),
                }
            )

        eval_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "relevant_passage_ids": [passage_id],
                "reference_answer": answer,
                "query_type": str(row.get("query_type", "")),
            }
        )

    return corpus, eval_rows


def main() -> int:
    corpus, eval_rows = build()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    config.CORPUS_PATH.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    eval_path = config.DATA_DIR / "eval_queries.json"
    eval_path.write_text(
        json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    avg_len = sum(len(d["passage_text"]) for d in corpus) / max(len(corpus), 1)
    print(f"\nUnique passages : {len(corpus)}")
    print(f"Eval queries    : {len(eval_rows)}")
    print(f"Avg passage len : {avg_len:.0f} chars")
    print(f"Corpus -> {config.CORPUS_PATH}")
    print(f"Eval   -> {eval_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
