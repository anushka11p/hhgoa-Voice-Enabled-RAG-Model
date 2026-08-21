import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the sentence-transformers embedding model.
        
        Why this model? 
        'all-MiniLM-L6-v2' is widely adopted as an excellent baseline 
        for semantic search because it perfectly balances immense speed 
        and relatively low memory usage with good similarity search quality.
        It generates embeddings of 384 dimensions.
        """
        print(f"Loading Embedding Model: {model_name}...")
        self.model_name = model_name
        # The first time this is run, Hugging Face will download the model mapping weights.
        self.model = SentenceTransformer(self.model_name)
        print("Embedding model loaded successfully.")

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of dictionaries (where each holds 'chunk_text' and metadata),
        generates vector embeddings for the text, and stores them back into the dictionary.
        """
        if not chunks:
            return []
            
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text list for processing in bulk.
        # Deep learning models process batches significantly faster than single records!
        texts = [chunk.get("chunk_text", "") for chunk in chunks]
        
        # .encode() automatically builds a dense numerical representation (vector) of the text
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Bind the dense vectors directly back to the original dictionary
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            new_chunk = chunk.copy()
            # Convert standard numpy array to a pure Python list so that 
            # if we wanted to dump this to raw JSON elsewhere, it works seamlessly.
            new_chunk["embedding"] = embeddings[i].tolist()
            embedded_chunks.append(new_chunk)
            
        print("Done binding embeddings to chunk metadata!")
        return embedded_chunks

if __name__ == "__main__":
    # Internal module testing block
    # We can fake some chunks to prove that vectors generate accurately.
    sample_chunks = [
        {
            "passage_id": "p0",
            "chunk_id": "p0_chunk0",
            "source": "MSMARCO_TEST",
            "chunk_text": "This is a basic standalone sentence being converted to an embedding."
        },
        {
            "passage_id": "p0",
            "chunk_id": "p0_chunk1",
            "source": "MSMARCO_TEST",
            "chunk_text": "Neural networks mathematically translate text into vectors."
        }
    ]
    
    print("\n--- Testing Module 3: Embedding Generator ---")
    embedder = EmbeddingGenerator()
    embedded_data = embedder.generate_embeddings(sample_chunks)
    
    print("\nTest Complete!")
    print(f"Vector Dimensions per Chunk: {len(embedded_data[0]['embedding'])} floats.")
    print("Sample Metadata Output Keys:", list(embedded_data[0].keys()))
    print("Snapshot of Vector (first 5 floats):", embedded_data[0]['embedding'][:5])
