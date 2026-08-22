import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def get_recursive_chunks(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Recursive Character Splitter chunking strategy.
    
    Why it works:
    This strategy recursively tries to split on paragraph boundaries ("\\n\\n"), then spaces, 
    and finally individual characters. It is generally the most robust general-purpose 
    chunker because it respects natural language transitions while adhering to the 
    maximum chunk size constraint.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "\u0964 ", "\u0964", ". ", " ", ""]
    )
    
    chunked_data = []
    
    for doc in documents:
        text = doc.get("passage_text", "")
        # Break text recursively based on natural separators
        text_chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(text_chunks):
            # Pass along existing metadata
            chunk_doc = doc.copy()
            chunk_doc["chunk_text"] = chunk
            chunk_doc["chunk_id"] = f"{doc.get('passage_id', 'unknown')}_chunk{i}"
            chunked_data.append(chunk_doc)
            
    return chunked_data

if __name__ == "__main__":
    # Test script functionality
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_corpus.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        test_docs = docs[:3]
        chunks = get_recursive_chunks(test_docs, chunk_size=100, chunk_overlap=20)
        
        print(f"Generated {len(chunks)} recursive chunks from {len(test_docs)} docs.")
        print("Sample Chunk:", json.dumps(chunks[0], indent=2))
    else:
        print("Run Module 1 first to generate cleaned_corpus.json!")
