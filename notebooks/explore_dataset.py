import os
import json
from datasets import load_dataset

def load_and_clean_data(output_path: str, max_samples: int = 100):
    """
    Loads the MS MARCO dataset via Hugging Face datasets.
    Explores its structure, flattens the nested passages, retains essential 
    metadata (query_id, passage_id, language, source), and saves a cleaned JSON subset.
    """
    print("Loading MS MARCO dataset (v1.1) in stream mode...")
    
    # We use streaming=True so we don't have to download the very large entire dataset initially.
    # This is highly recommended for exploration.
    try:
        dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    cleaned_records = []
    
    print("\n================ Exploring Dataset Structure & Sample Field ================")
    for idx, row in enumerate(dataset):
        # On the first iteration, let's explore and explain the schema
        if idx == 0:
            print("Fields available in raw dataset:", list(row.keys()))
            print("\nSample row (raw JSON representation):")
            print(json.dumps({k: v for k, v in row.items() if k != 'passages'}, indent=2))
            print("  'passages': { ... nested block containing passage_text and is_selected ... }")
            
            print("\n--- Field Explanations ---")
            print("- query_id: Unique numerical identifier for the user's question.")
            print("- query: The actual text string of the query.")
            print("- answers: A list of human-provided answers.")
            print("- passages: A nested dictionary containing 'is_selected', 'passage_text', and 'url'.")
            print("            This requires flattening because our vector DB needs individual flat documents.")
            print("- query_type: The categorization of the query intent.")
            print("=========================================================================\n")
            print("Process started: Flattening nested passages into individual documents...")
            
        # Extract fields
        query_id = row.get("query_id", f"q_{idx}")
        # nested passage structure in MS MARCO
        passages = row.get("passages", {})
        
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        # Flattening: iterate through every nested passage
        for p_idx, p_text in enumerate(passage_texts):
            # Create a unique, deterministic ID for the passage chunk
            passage_id = f"{query_id}_p{p_idx}"
            
            # Creating our standardized schema document
            cleaned_record = {
                "query_id": str(query_id),
                "passage_id": str(passage_id),
                "language": "en",        # Filtering to one language (English)
                "source": "MSMARCO",     # Preserving the source metadata
                "passage_text": p_text,
                "is_selected": is_selected[p_idx] if p_idx < len(is_selected) else 0
            }
            cleaned_records.append(cleaned_record)
            
            if len(cleaned_records) >= max_samples:
                break
                
        if len(cleaned_records) >= max_samples:
            break

    print(f"\nSuccessfully extracted and flattened {len(cleaned_records)} passage documents.")
    
    # Save the cleaned dataset to the target path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_records, f, indent=4)
        
    print(f"Cleaned dataset beautifully saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # Define absolute paths dynamically based on the project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # We will save to data/processed
    processed_data_path = os.path.join(project_root, "data", "processed", "cleaned_corpus.json")
    
    # Fetch 100 samples for fast exploration & demonstration
    load_and_clean_data(processed_data_path, max_samples=100)
