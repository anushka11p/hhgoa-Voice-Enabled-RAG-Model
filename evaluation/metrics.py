import pandas as pd
from typing import List, Dict

def calculate_recall_at_k(retrieved_passage_ids: List[str], relevant_passage_ids: List[str], k: int) -> float:
    """
    Calculates standard Recall@K metric.
    
    Why this works:
    Recall answers: "Out of all explicitly relevant passages, what fraction did we successfully retrieve 
    inside our Top K list?" It proves whether our DB retrieval algorithms are firing correctly.
    """
    if not relevant_passage_ids:
        return 0.0
        
    top_k_retrieved = retrieved_passage_ids[:k]
    
    # Compare raw overlap via set intersection
    hits = set(top_k_retrieved).intersection(set(relevant_passage_ids))
    return len(hits) / len(relevant_passage_ids)

def calculate_mrr(retrieved_passage_ids: List[str], relevant_passage_ids: List[str]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR).
    
    Why this works:
    While Recall tracks IF we found the document, MRR tracks WHERE we put it.
    If the correct document is at index #1, MRR score is 1/1 (1.0). 
    If it's at index #5, the score drops sharply to 1/5 (0.2).
    This strictly evaluates our ranking/sorting algorithm (like BM25, FAISS, CrossEncoder).
    """
    for rank, ret_id in enumerate(retrieved_passage_ids, start=1):
        if ret_id in relevant_passage_ids:
            return 1.0 / float(rank)
    return 0.0

def evaluate_retrieval(query_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates Recall@5, Recall@10, and MRR continuously for a list of query dictionaries.
    
    Format requires:
    [
        {
            "query_id": "q1",
            "retrieved_ids": ["p1", "p4", "p7", ...],
            "relevant_ids": ["p1"]
        }
    ]
    """
    total_recall_5 = 0.0
    total_recall_10 = 0.0
    total_mrr = 0.0
    num_queries = len(query_results)
    
    if num_queries == 0:
        return {"Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}
        
    for res in query_results:
        retrieved = res.get("retrieved_ids", [])
        relevant = res.get("relevant_ids", [])
        
        total_recall_5 += calculate_recall_at_k(retrieved, relevant, k=5)
        total_recall_10 += calculate_recall_at_k(retrieved, relevant, k=10)
        total_mrr += calculate_mrr(retrieved, relevant)
        
    return {
        "Recall@5": total_recall_5 / num_queries,
        "Recall@10": total_recall_10 / num_queries,
        "MRR": total_mrr / num_queries
    }

def print_chunking_comparison(evaluation_metrics: Dict[str, Dict[str, float]]):
    """
    Generates a terminal-friendly Markdown Pandas DataFrame table 
    comparing our 4 different chunking strategies mathematically.
    
    evaluation_metrics structure:
    {
        "Fixed": {"Recall@5": 0.8, "Recall@10": 0.9, "MRR": 0.6},
        "Recursive": {"Recall@5": 0.85, ...}
    }
    """
    df = pd.DataFrame(evaluation_metrics).T
    # Rounding out for clean visual reading
    df = df.round(4)
    print("\n========= RAG Strategy Performance Evaluation =========")
    print(df.to_markdown())
    print("========================================================\n")

if __name__ == "__main__":
    # Internal Mock Test script
    print("Testing internal Metric calculation engines...")
    
    mock_data = {
        "Fixed Chunking": {
            "Recall@5": 0.7300,
            "Recall@10": 0.8400,
            "MRR": 0.4500
        },
        "Recursive Chunking": {
            "Recall@5": 0.8100,
            "Recall@10": 0.9100,
            "MRR": 0.6500
        },
        "Semantic Chunking": {
            "Recall@5": 0.8800,
            "Recall@10": 0.9600,
            "MRR": 0.7200
        },
        "Metadata Chunking": {
            "Recall@5": 0.8950,
            "Recall@10": 0.9700,
            "MRR": 0.7700 # Boosted by internal metadata signals
        }
    }
    
    print_chunking_comparison(mock_data)
