import os
import json
import re
from typing import List, Dict, Any

def get_semantic_chunks(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Sentence-aware (Semantic) chunking strategy.
    
    Why it works:
    Instead of slicing cleanly at X characters mid-word, semantic chunking ensures we only 
    split at natural sentence boundaries (e.g., period, question mark). This guarantees that
    the resulting chunk maintains theoretical "wholeness". Vector embedding models 
    produce far better similarity scores when they ingest whole, un-chopped concepts.
    """
    chunked_data = []
    
    for doc in documents:
        text = doc.get("passage_text", "")
        
        # Simple heuristic constraint for English sentence splitting
        # Split on '.', '!', '?' followed by a space
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        current_chunk = ""
        chunks = []
        
        # Group sentences without violating chunk_size boundaries
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add leftover chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        for i, chunk in enumerate(chunks):
            # Retain metadata exactly as parent
            chunk_doc = doc.copy()
            chunk_doc["chunk_text"] = chunk
            chunk_doc["chunk_id"] = f"{doc.get('passage_id', 'unknown')}_chunk{i}"
            chunked_data.append(chunk_doc)
            
    return chunked_data

if __name__ == "__main__":
    # Testing Script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_corpus.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        test_docs = docs[:3]
        chunks = get_semantic_chunks(test_docs, chunk_size=200, chunk_overlap=0)
        
        print(f"Generated {len(chunks)} semantic chunks from {len(test_docs)} docs.")
        print("Sample Chunk:", json.dumps(chunks[0], indent=2))
    else:
        print("Run Module 1 first to generate cleaned_corpus.json!")
