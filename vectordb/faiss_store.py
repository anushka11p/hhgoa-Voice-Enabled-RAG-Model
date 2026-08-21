import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple

class FaissStore:
    def __init__(self, vector_dim: int = 384):
        """
        Initializes the Vector Database using faiss-cpu.
        FAISS (Facebook AI Similarity Search) is extremely fast for nearest-neighbor queries.
        We initialize IndexFlatL2 for exact L2 (Euclidean) distance matching.
        """
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # We also need to map integer vector IDs to string passages natively
        self.metadata = []

    def build_index(self, embedded_chunks: List[Dict[str, Any]]):
        """
        Iterates over generated chunk dictionaries, extracting the dense embedding arrays 
        and storing their respective metadata into an aligned lookup array.
        """
        print(f"Building FAISS vector index with {len(embedded_chunks)} records...")
        
        if not embedded_chunks:
            return
            
        vectors = []
        for chunk in embedded_chunks:
            # FAISS enforces np.float32 
            vec = np.array(chunk["embedding"], dtype=np.float32)
            vectors.append(vec)
            
            # Save chunk metadata, but drop the dense array so we don't duplicate memory
            meta_copy = chunk.copy()
            meta_copy.pop("embedding", None)
            self.metadata.append(meta_copy)
            
        # Add to index internally
        vector_matrix = np.vstack(vectors)
        self.index.add(vector_matrix)
        print(f"Index successfully built! Total vectors tracked: {self.index.ntotal}")

    def save_index(self, directory_path: str, index_name: str = "faiss_index"):
        """
        Serializes the math matrix to disk (.bin), combined with the text lookups (.json), 
        ensuring fast zero-compute boot up during inference deployment.
        """
        os.makedirs(directory_path, exist_ok=True)
        
        index_path = os.path.join(directory_path, f"{index_name}.bin")
        meta_path = os.path.join(directory_path, f"{index_name}_meta.json")
        
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
            
        print(f"Saved database bundle accurately to: {directory_path}")

    def load_index(self, directory_path: str, index_name: str = "faiss_index"):
        """
        Reconstructs the FAISS memory block perfectly using saved context.
        """
        index_path = os.path.join(directory_path, f"{index_name}.bin")
        meta_path = os.path.join(directory_path, f"{index_name}_meta.json")
        
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        print(f"Loaded internal FAISS store! Total searchable records: {self.index.ntotal}")

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Conducts nearest-neighbor math search tracking against the single query.
        Returns the explicit matching JSON records sorted by mathematical relevance.
        """
        q_vec = np.array([query_embedding], dtype=np.float32)
        
        # FAISS search outputs exact array distances + local list indexes natively
        distances, indices = self.index.search(q_vec, top_k)
        
        results = []
        for j, idx in enumerate(indices[0]):
            # -1 happens structurally if `top_k` requested is greater than `ntotal` vectors
            if idx == -1: 
                continue
                
            dist = distances[0][j]
            result_meta = self.metadata[idx]
            
            # Lower distance in IndexFlatL2 perfectly dictates higher similarity!
            results.append((float(dist), result_meta))
            
        return results

if __name__ == "__main__":
    # Interactive Developer Test Block
    print("Testing FAISS Module mechanics...")
    store = FaissStore(vector_dim=384) # Miniature dimensional spoof
    
    fake_data = []
    # Using np random matrix to act as 'word vectors'
    for i in range(5):
        fake_data.append({
            "chunk_id": f"chunk_{i}",
            "passage_text": f"Mock passage text context {i}",
            "embedding": np.random.rand(384).tolist()
        })
        
    store.build_index(fake_data)
    
    # Validate physical saving/loading
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_db")
    store.save_index(test_dir, "test")
    store.load_index(test_dir, "test")
    
    print("\n--- Running Similarity Search Simulation ---")
    mock_q = np.random.rand(384).tolist()
    res = store.similarity_search(mock_q, top_k=2)
    for dist, meta in res:
        print(f"Distance Score: {dist:.4f} \t Source: {meta['passage_text']}")
