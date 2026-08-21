from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

class BM25Retriever:
    def __init__(self):
        """
        Initializes the Sparse (Lexical) Retrieval track.
        """
        self.corpus_meta = []
        self.bm25_model = None

    def fit(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 statistical lexical index from chunk texts.
        
        Why this works:
        While Dense networks are great at abstractions (meaning), they often fail at 
        explicit string lookups (like querying for "RX-78-2 Gundam" or specific IDs). 
        BM25 evaluates exact token frequency vs inverted document frequency, meaning it 
        is brilliant at hard keyword matching.
        """
        print("Fitting BM25 lexical index...")
        self.corpus_meta = chunks
        
        # Basic tokenization: lowercase and space split
        # Pluggable with NLTK or Spacy tokenizers if needed later
        tokenized_corpus = [chunk.get("chunk_text", "").lower().split(" ") for chunk in chunks]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        print("BM25 lexical index initialized fully.")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Scores the query using BM25 and returns top exact keyword matches.
        """
        if not self.bm25_model:
            raise ValueError("BM25Retriever needs to be fitted with data first!")
            
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        # Sort indices by highest score globally (unlike L2 FAISS where lowest is better)
        top_k_indices = doc_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = doc_scores[idx]
            if score > 0: # Only return actual lexical matches
                meta = self.corpus_meta[idx]
                results.append((meta.get("chunk_id", ""), float(score), meta))
                
        return results
