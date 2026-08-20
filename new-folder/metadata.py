import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def get_metadata_chunks(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Metadata-aware chunking strategy.
    
    Why it works:
    Standard embeddings focus exclusively on passage text. However, often the exact 
    ID, source name, and context are highly relevant for nuanced vector searches. 
    This strategy injects "Source" and "Language" metadata directly into the chunk 
    text. Consequently, the embedding model bakes this metadata into the mathematical vector.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunked_data = []
    
    for doc in documents:
        # 1. Prepend metadata straight into passage string!
        metadata_header = f"Source: {doc.get('source', 'Unknown')} | Language: {doc.get('language', 'en')}\n"
        enhanced_text = metadata_header + doc.get("passage_text", "")
        
        # 2. Split mathematically
        text_chunks = splitter.split_text(enhanced_text)
        
        for i, chunk in enumerate(text_chunks):
            # 3. Retain base tags alongside new enriched chunk
            chunk_doc = doc.copy()
            chunk_doc["chunk_text"] = chunk
            chunk_doc["chunk_id"] = f"{doc.get('passage_id', 'unknown')}_chunk{i}"
            chunked_data.append(chunk_doc)
            
    return chunked_data

if __name__ == "__main__":
    # Quick Test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_corpus.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        test_docs = docs[:3]
        chunks = get_metadata_chunks(test_docs, chunk_size=150, chunk_overlap=30)
        
        print(f"Generated {len(chunks)} metadata-infused chunks from {len(test_docs)} docs.")
        print("Sample Chunk:", json.dumps(chunks[0], indent=2))
    else:
        print("Run Module 1 first to generate cleaned_corpus.json!")
