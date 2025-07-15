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
    print("--- 🚀 Kicking off the Synonym Invention Machine V3.0! 🚀 ---")

    # --- Path setup ---
    data_dir = project_root / "data"
    output_path = data_dir / "term_pairs_by_category.json"

    # ✨ NEW: Check for existing output to avoid re-computation! ✨
    if os.path.exists(output_path):
        print(f"✅ Hooray! Term pairs file already exists at {output_path}. Nothing to do here!")
        return
    
    # --- ✨ NEW: Point to the CSV frequency files! ✨ ---
    frequency_files = {
        "diagnosis": data_dir / "diagnosis_frequency.csv",
        "procedure": data_dir / "procedure_frequency.csv",
        "medication": data_dir / "medication_frequency.csv",
    }
    rxnorm_table_path = data_dir / "RXNorm_Table-25_06_17-v1.csv"

    # --- ✨ NEW: Load the RxNorm mapping table! ✨ ---
    print(f"📂 Loading RxNorm mapping from: {rxnorm_table_path}")
    rxnorm_df = pd.read_csv(rxnorm_table_path)
    # Create a dictionary for easy lookups! {medication_code: rxnorm_code}
    rxnorm_map = pd.Series(rxnorm_df['RxNorm_Code'].values, index=rxnorm_df['Medication_Code']).to_dict()

    all_pairs = defaultdict(list)
    
    for category, file_path in frequency_files.items():
        print(f"\n--- Processing category: {category} ---")
        df = pd.read_csv(file_path)

        # ✨ The Magnificent Fix! All terms start as lowercase! ✨
        df['term'] = df.iloc[:, 0].str.lower()
        df['code'] = df.iloc[:, 1]

        # --- Generate pairs from codes ---
        code_based_pairs = set()
        
        if category == "medication":
            # ✨ NEW 3-Step Medication Logic! ✨
            # Add RxNorm codes to our dataframe
            df['rxnorm_code'] = df['code'].map(rxnorm_map)

            # 1. Pair by RxNorm
            rxnorm_to_terms = df.dropna(subset=['rxnorm_code']).groupby('rxnorm_code')['term'].apply(set)
            for terms in rxnorm_to_terms:
                if len(terms) > 1:
                    for t1, t2 in combinations(sorted(list(terms)), 2):
                        code_based_pairs.add(tuple(sorted((t1, t2))))
            print(f"Generated {len(code_based_pairs)} pairs based on shared RxNorm codes.")

            # 2. Pair by original medication code for the leftovers
            terms_in_rxnorm_pairs = {term for pair in code_based_pairs for term in pair}
            df_remaining = df[~df['term'].isin(terms_in_rxnorm_pairs)]
            
            epic_code_to_terms = df_remaining.groupby('code')['term'].apply(set)
            epic_pairs_generated = 0
            for terms in epic_code_to_terms:
                if len(terms) > 1:
                    for t1, t2 in combinations(sorted(list(terms)), 2):
                        new_pair = tuple(sorted((t1, t2)))
                        if new_pair not in code_based_pairs:
                            code_based_pairs.add(new_pair)
                            epic_pairs_generated += 1
            print(f"Generated {epic_pairs_generated} new pairs based on shared original medication codes.")
        
        else: # Standard logic for diagnoses and procedures
            code_to_terms = df.groupby('code')['term'].apply(set)
            for terms in code_to_terms:
                if len(terms) > 1:
                    for t1, t2 in combinations(sorted(list(terms)), 2):
                        code_based_pairs.add(tuple(sorted((t1, t2))))
            print(f"Generated {len(code_based_pairs)} pairs based on shared codes for {category}.")

        all_pairs[category].extend(list(code_based_pairs))

        # --- 3. Generate LLM-augmented pairs for lonely terms ---
        terms_in_pairs = {term for pair in all_pairs[category] for term in pair}
        all_unique_terms = set(df['term'].unique())
        lonely_terms = all_unique_terms - terms_in_pairs
        
        print(f"Found {len(lonely_terms)} lonely terms. Asking the LLM for synonyms...")
        
        llm_pairs_generated = 0
        for term in lonely_terms:
            try:
                synonym = get_synonym_from_llm(term, category).lower()
                if synonym and synonym != term:
                    new_pair = tuple(sorted((term, synonym)))
                    all_pairs[category].append(new_pair)
                    llm_pairs_generated += 1
            except Exception as e:
                print(f"  - Could not generate synonym for '{term}': {e}", file=sys.stderr)
                continue
        print(f"Generated a total of {llm_pairs_generated} LLM-augmented pairs for {category}.")

    # --- Save the final, de-duplicated list of pairs ---
    final_output = {}
    for category, pairs_list in all_pairs.items():
        # Re-format for the original structure: list of dicts
        formatted_pairs = [{"term": p[0], "counterpart": p[1], "category": category} for p in set(pairs_list)]
        final_output[category] = formatted_pairs
        print(f"Total unique pairs for {category}: {len(final_output[category])}")

    # For the final output, let's flatten it into one big list of dicts
    flat_final_list = [item for sublist in final_output.values() for item in sublist]

    print(f"\n💾 Saving all {len(flat_final_list)} pairs to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(flat_final_list, f, indent=4)
        
    print("\n🎉 Glorious success! All term pairs have been generated!")

if __name__ == "__main__":
    main()