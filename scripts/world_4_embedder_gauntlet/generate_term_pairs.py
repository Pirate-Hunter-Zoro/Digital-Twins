import os
import sys
import json
import pandas as pd
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.llm.query_llm import query_llm

def get_synonym_from_llm(term: str, category: str) -> str:
    """
    Uses an LLM to generate a single, plausible synonym for a given medical term.
    """
    prompt = f"""
    You are a medical terminologist. Provide a single, common, and plausible alternative name or synonym for the following medical {category}.
    Do not use the original term in your answer. Provide only the synonym.

    Term: "{term}"
    Synonym:
    """
    synonym = query_llm(prompt, max_tokens=50)
    # Clean up the response to ensure it's just the term
    return synonym.strip().replace('"', '')

def main():
    print("--- 🚀 Kicking off the Synonym Invention Machine! 🚀 ---")

    # --- Path setup ---
    data_dir = project_root / "data"
    output_path = data_dir / "term_pairs_by_category.json"
    
    # Define input frequency files
    frequency_files = {
        "diagnosis": data_dir / "diagnosis_concept_counts.json",
        "procedure": data_dir / "procedure_concept_counts.json",
        "medication": data_dir / "medication_concept_counts.json",
    }

    all_pairs = defaultdict(list)
    
    for category, file_path in frequency_files.items():
        print(f"\n--- Processing category: {category} ---")
        with open(file_path, 'r') as f:
            concept_data = json.load(f)

        # Group terms by their medical code, now case-insensitively!
        code_to_terms = defaultdict(set)
        for item in concept_data:
            code = item.get("code")
            # ✨ The Magnificent Fix! Convert term to lowercase! ✨
            term = item.get("term", "").lower()
            if code and term:
                code_to_terms[code].add(term)

        print(f"Found {len(code_to_terms)} unique codes for {category}.")
        
        # --- Generate pairs from codes ---
        code_based_pairs = set()
        for code, terms in code_to_terms.items():
            if len(terms) > 1:
                # Create all possible pairs of unique terms for a given code
                for t1, t2 in combinations(sorted(list(terms)), 2):
                    # Add the pair in a consistent order to avoid duplicates
                    code_based_pairs.add(tuple(sorted((t1, t2))))

        all_pairs[category].extend(list(code_based_pairs))
        print(f"Generated {len(code_based_pairs)} pairs based on shared medical codes.")

        # --- Generate LLM-augmented pairs ---
        terms_in_pairs = {term for pair in code_based_pairs for term in pair}
        
        # Find all unique terms that were left out
        all_unique_terms = {item.get("term", "").lower() for item in concept_data if item.get("term")}
        lonely_terms = all_unique_terms - terms_in_pairs
        
        print(f"Found {len(lonely_terms)} lonely terms. Asking the LLM for synonyms...")
        
        llm_pairs_generated = 0
        for term in lonely_terms:
            try:
                synonym = get_synonym_from_llm(term, category).lower()
                # Ensure synonym is not empty and is different from the original term
                if synonym and synonym != term:
                    new_pair = tuple(sorted((term, synonym)))
                    all_pairs[category].append(new_pair)
                    llm_pairs_generated += 1
                    if llm_pairs_generated % 20 == 0:
                        print(f"  - Generated {llm_pairs_generated} pairs with the LLM...")
            except Exception as e:
                print(f"  - Could not generate synonym for '{term}': {e}")
                continue
        print(f"Generated a total of {llm_pairs_generated} LLM-augmented pairs for {category}.")


    # --- Save the final, de-duplicated list of pairs ---
    final_output = {}
    for category, pairs_list in all_pairs.items():
        # Use a set to ensure all pairs are unique!
        final_output[category] = sorted(list(set(pairs_list)))
        print(f"Total unique pairs for {category}: {len(final_output[category])}")

    print(f"\n💾 Saving all pairs to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n🎉 Glorious success! All term pairs have been generated!")

if __name__ == "__main__":
    main()