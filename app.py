import os
import json
from typing import List, Dict, Any

# Local Module Imports
from embeddings.embed import EmbeddingGenerator
from vectordb.faiss_store import FaissStore
from retrieval.dense import dense_retrieval
from retrieval.bm25 import BM25Retriever
from retrieval.rrf import compute_rrf
from retrieval.reranker import CrossEncoderReranker

class VoiceRAGPipeline:
    def __init__(self, data_path: str):
        """
        Initializes the entire monolithic RAG pipeline.
        In a production server endpoint, this class is instantiated once on server boot!
        """
        print("--- Booting up Voice RAG Pipeline Infrastructure ---")
        
        # 1. Load the structured embedded corpus
        self.chunks = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
                
        # 2. Init Embedder model
        self.embedder = EmbeddingGenerator()
        
        # 3. Init VectorDB (FAISS) 
        self.faiss_store = FaissStore()
        print("Attaching vectors to FAISS L2 graph...")
        # Note: real systems load from pre-built faiss bin files to save boot time!
        if self.chunks and "embedding" in self.chunks[0]:
            self.faiss_store.build_index(self.chunks)
        
        # 4. Init Sparse BM25
        self.bm25 = BM25Retriever()
        if self.chunks:
            self.bm25.fit(self.chunks)
            
        # 5. Init Top-K AI Reranker
        self.reranker = CrossEncoderReranker()
        print("\n--- Pipeline Fully Operational ---\n")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        The Master Endpoint Function.
        Execution: Query -> Dense Retrieval -> BM25 -> Reciprocal Rank Fusion -> Cross Encoder Reranker -> Top-5 Format.
        """
        out_results = []
        if not self.chunks:
            print("Warning: Corpus empty or not embedded. Skipping...")
            return out_results
            
        print(f"Processing Voice/Text Query: '{query}'")
        
        # A. Dense Phase (Get Top 20)
        dense_res = dense_retrieval(query, self.embedder, self.faiss_store, top_k=20)
        
        # B. Sparse Phase (Get Top 20)
        bm25_res = self.bm25.search(query, top_k=20)
        
        # C. RRF Fusion (Harmonize Dense + Sparse mathematical graphs)
        fused_results = compute_rrf(dense_res, bm25_res, top_n=20)
        
        # D. Rerank Phase (Top 5 explicitly rescored against query string via Neural Network)
        final_top5 = self.reranker.rerank(query, fused_results, top_k=5)
        
        # Format explicitly to the final dictionary schema
        for doc in final_top5:
            out_results.append({
                "text": doc.get("chunk_text", doc.get("passage_text", "")),
                "score": round(doc.get("relevance_score", 0.0), 4),
                "passage_id": doc.get("passage_id", "unknown"),
                "source": doc.get("source", "MSMARCO")
            })
            
        return out_results

if __name__ == "__main__":
    # Test API Endpoint locally in terminal
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Requirements: 
    # For this to actually hit results, you need a corpus with physical "chunk_text" and "embedding" fields.
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_corpus.json")
    
    # Init Pipeline Framework
    pipeline = VoiceRAGPipeline(data_path)
    
    # End-User targeted simulated search
    query_string = "How heavy is the Earth?"
    final_output = pipeline.retrieve(query_string)
    
    print(f"\n================== FINAL EXPORT API FOR: '{query_string}' ==================")
    print(json.dumps(final_output, indent=4))
    print("=========================================================================")
