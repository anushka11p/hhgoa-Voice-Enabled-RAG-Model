from typing import List, Dict, Any, Tuple

def dense_retrieval(query: str, embedder, faiss_store, top_k: int = 20) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Performs standard Dense Retrieval track.
    
    Why this works:
    Dense vectors capture the semantic *meaning* of a query rather than explicit keyword 
    matching. For example, "where to buy cheap fuel" can natively match "affordable gasoline station" 
    because the neural network places their meanings closely in space.
    
    Execution:
    Converts query text to vector using our instantiated text embedder, 
    then shoots that coordinate array directly into the FAISS store index.
    
    Returns: List of tuples (chunk_id, distance_score, full_metadata) sorted by closest match.
    """
    # 1. Convert incoming raw text to semantic vector array
    query_vector = embedder.model.encode([query])[0].tolist()
    
    # 2. Extract math similarity using faiss optimized L2 Euclidean limits
    raw_results = faiss_store.similarity_search(query_vector, top_k=top_k)
    
    # 3. Format output pipeline cleanly so it maps perfectly for RRF later
    standardized_results = []
    
    # Note: In FAISS L2, a LOWER distance score is mathematically BETTER.
    # The rank order in the list is returned correctly by FAISS (index 0 is best).
    for score, meta in raw_results:
        standardized_results.append((meta.get("chunk_id", ""), float(score), meta))
        
    return standardized_results
