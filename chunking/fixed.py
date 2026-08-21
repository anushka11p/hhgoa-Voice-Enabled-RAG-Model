import os
import json
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Dict, Any

def get_fixed_chunks(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Fixed-size character chunking strategy.
    
    Why it works:
    This strategy simply splits text into chunks of an exact number of characters.
    It is extremely fast and ensures uniformity in chunk sizes. It works best when 
    the text structure doesn't matter much or when we strictly need uniform sizes 
    to maximize vector database storage efficiency.
    """
    splitter = CharacterTextSplitter(
        separator="", 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunked_data = []
    for doc in documents:
        text = doc.get("passage_text", "")
        # Create text chunks
        text_chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(text_chunks):
            # Return chunks while retaining all previous metadata
            chunk_doc = doc.copy()
            chunk_doc["chunk_text"] = chunk
            chunk_doc["chunk_id"] = f"{doc.get('passage_id', 'unknown')}_chunk{i}"
            chunked_data.append(chunk_doc)
            
    return chunked_data

if __name__ == "__main__":
    # Test script functionality
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_corpus.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        # Test on the first 3 documents
        test_docs = docs[:3]
        chunks = get_fixed_chunks(test_docs, chunk_size=50, chunk_overlap=10)
        
        print(f"Generated {len(chunks)} fixed-size chunks from {len(test_docs)} docs.")
        print("Sample Chunk:", json.dumps(chunks[0], indent=2))
    else:
        print("Run Module 1 first to generate cleaned_corpus.json!")
