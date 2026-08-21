from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes our Cross-Encoder MS MARCO rerank stage.
        
        Why this works:
        FAISS computes L2 distances between two PRE-CALCULATED vector states. (Bi-Encoder logic)
        A Cross-Encoder doesn't pre-calculate arrays; it actively computes the logical mapping natively 
        between the raw textual Query and the raw textual Passage in one unified inference step!
        It generates a highly accurate, rigorous relevance relationship score that destroys FAISS/BM25 
        accuracy ceilings but is intensely computationally expensive.
        Therefore, we only use it strategically to reorder the *final Top 20* items passed by RRF.
        """
        print(f"Loading Cross-Encoder Reranker: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Reranker framework loaded successfully!")

    def rerank(self, query: str, preliminary_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-scores RRF outputs against the actual query rigorously.
        """
        if not preliminary_results:
            return []
            
        print(f"Reranking top {len(preliminary_results)} candidates via Cross-Encoder neural inspection...")
        
        # Package text formats mathematically for the Hugging Face predict interface
        # The cross encoder must accept a matrix of explicitly matching Query-to-Passage tuples.
        cross_inp = [[query, doc.get("chunk_text", "")] for doc in preliminary_results]
        
        # Execute active cross-correlation neural matrix processing
        scores = self.model.predict(cross_inp)
        
        # Bind the highly accurate explicit relevance scores back to our payload
        for idx, doc in enumerate(preliminary_results):
            doc["relevance_score"] = float(scores[idx])
            
        # Execute rigorous downward scalar sorting
        sorted_results = sorted(preliminary_results, key=lambda x: x["relevance_score"], reverse=True)
        
        # Only hand back the absolute top-K (Typically Top 5) items to present to the user/LLM
        return sorted_results[:top_k]
