from typing import List, Dict, Any, Tuple

def compute_rrf(dense_results: List[Tuple[str, float, Dict[str, Any]]], 
                bm25_results: List[Tuple[str, float, Dict[str, Any]]], 
                k: int = 60,
                top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF).
    
    Why this works:
    Dense Retrieval outputs Euclidean distances. Sparse BM25 outputs frequency logits. 
    These mathematical scales are entirely incompatible; you can't legitimately add them.
    RRF solves this entirely without parameters! It awards points exclusively based 
    on the *rank index* mathematically returning: `1 / (k + rank_position)`.
    If a document places high in BOTH lexical AND semantic tracks, its RRF score spikes,
    meaning we flawlessly blend keyword intent mapping and abstract semantic meaning automatically!
    """
    rrf_scores = {}
    meta_map = {}
    
    # 1. Process Dense Ranks (These lists MUST arrive pre-sorted from best to worst)
    for rank, (chunk_id, score, meta) in enumerate(dense_results, start=1):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0.0
            meta_map[chunk_id] = meta
        # RRF formula execution
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        
    # 2. Process BM25 Ranks
    for rank, (chunk_id, score, meta) in enumerate(bm25_results, start=1):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0.0
            meta_map[chunk_id] = meta
        # RRF formula execution
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        
    # 3. Sort by aggregated RRF score explicitly (Higher score = better rank)
    sorted_fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 4. Bind unified ranking score arrays and compile final merged list
    final_output = []
    for chunk_id, fused_score in sorted_fused[:top_n]:
        final_doc = meta_map[chunk_id].copy()
        final_doc["rrf_score"] = float(fused_score)
        final_output.append(final_doc)
        
    return final_output
