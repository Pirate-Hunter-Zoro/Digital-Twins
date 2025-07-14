# scripts/world_4_embedder_gauntlet/generate_term_pairs.py (Case-Insensitive SUPER-UPGRADE!)
import pandas as pd
import json
from itertools import combinations
from pathlib import Path
import time
import os
import sys

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our magnificent LLM query function!
from scripts.common.llm.query_llm import query_llm, get_llm_client

# --- Pairing Functions (Now with case-insensitivity!) ---

def create_code_based_pairs(all_terms_list, code_groups, category):
    """
    Generic function to create pairs based on shared codes, now with
    advanced case-insensitive logic! So smart!
    """
    pairs = []
    used_terms_lower = set()

    for code, descriptions in code_groups.items():
        # Get unique descriptions, ignoring case
        unique_descs = list(pd.Series(descriptions).str.lower().unique())
        
        # Map lowercased back to an original capitalization
        desc_map = {d.lower(): d for d in reversed(descriptions)}
        original_case_descs = [desc_map[d_lower] for d_lower in unique_descs]

        if len(original_case_descs) > 1:
            for p in combinations(original_case_descs, 2):
                term1_lower = p[0].lower()
                term2_lower = p[1].lower()
                
                # Only create a pair if BOTH terms haven't been used yet
                if term1_lower not in used_terms_lower and term2_lower not in used_terms_lower:
                    pairs.append({'term': p[0], 'counterpart': p[1], 'category': category, 'type': 'code_based'})
                    used_terms_lower.add(term1_lower)
                    used_terms_lower.add(term2_lower)
                    
    # Find all terms that were actually used
    final_used_terms = {term for pair in pairs for term in [pair['term'], pair['counterpart']]}
    return pairs, final_used_terms

# --- The LLM-Powered Synonym Invention Machine ---

def generate_llm_pairs_for_lonely_terms(all_terms, used_terms, category_name):
    """
    For lonely terms in ANY category, ask our big LLM friend to invent a new friend!
    """
    lonely_terms = sorted(list(set(all_terms) - used_terms))
    if not lonely_terms:
        return []

    print(f"🤖 Found {len(lonely_terms)} lonely '{category_name}' terms. Powering up the Synonym Invention Machine!")
    
    new_pairs = []
    cache_file = Path(f"data/llm_synonym_cache_{category_name}.json")
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
        
    for i, term in enumerate(lonely_terms):
        print(f"  ({i+1}/{len(lonely_terms)}) Finding a friend for: '{term}'")
        
        if term in cache:
            counterpart = cache[term]
            print(f"    -> Found in cache: '{counterpart}'")
        else:
            prompt = (
                f"You are a medical terminologist. Your task is to provide a single, common clinical synonym "
                f"or a slightly rephrased version of the following medical {category_name} term. "
                "Do not explain your reasoning. Just provide the alternative term.\n\n"
                f"TERM: \"{term}\"\n\n"
                "SYNONYM:"
            )
            counterpart = query_llm(prompt, max_tokens=50, temperature=0.5).strip().replace('"', '')
            cache[term] = counterpart
            print(f"    -> LLM generated: '{counterpart}'")
            time.sleep(1)

        if counterpart and counterpart.lower() != term.lower():
            new_pairs.append({'term': term, 'counterpart': counterpart, 'category': category_name, 'type': 'llm_generated'})

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"✨ Created {len(new_pairs)} new LLM-generated '{category_name}' pairs!")
    return new_pairs

# --- Main Execution ---
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'data'
    
    # --- Diagnoses ---
    print("\n--- Processing Diagnoses ---")
    diag_df = pd.read_csv(data_dir / 'diagnosis_frequency.csv')
    diag_df.dropna(subset=['Code', 'Description'], inplace=True)
    all_diag_terms = diag_df['Description'].unique().tolist()
    diag_code_groups = diag_df.groupby('Code')['Description'].apply(list)
    diag_pairs, used_diag_terms = create_code_based_pairs(all_diag_terms, diag_code_groups, 'diagnosis')

    # --- Procedures ---
    print("\n--- Processing Procedures ---")
    proc_df = pd.read_csv(data_dir / 'procedure_frequency.csv')
    proc_df.dropna(subset=['CPT_Procedure_Code', 'CPT_Procedure_Description'], inplace=True)
    all_proc_terms = proc_df['CPT_Procedure_Description'].unique().tolist()
    proc_code_groups = proc_df.groupby('CPT_Procedure_Code')['CPT_Procedure_Description'].apply(list)
    proc_pairs, used_proc_terms = create_code_based_pairs(all_proc_terms, proc_code_groups, 'procedure')

    # --- Medications ---
    print("\n--- Processing Medications ---")
    med_df = pd.read_csv(data_dir / 'medication_frequency.csv')
    rxnorm_df = pd.read_csv(data_dir / 'RXNorm_Table-25_06_17-v1.csv')
    all_med_terms = med_df['MedSimpleGenericName'].unique().tolist()
    merged_df = pd.merge(med_df, rxnorm_df, on='MedicationEpicID')
    filtered_df = merged_df[merged_df['RXNORM_TYPE'].isin(['Ingredient', 'Brand Name', 'Precise Ingredient', 'MED_ONLY'])]
    med_code_groups = filtered_df.groupby('RXNORM_CODE')['Name'].unique().apply(list)
    med_pairs, used_med_terms = create_code_based_pairs(all_med_terms, med_code_groups, 'medication')

    # --- LLM Generation for Lonely Terms ---
    print("\n--- Generating LLM pairs for all lonely terms ---")
    get_llm_client() # Initialize the client once!
    llm_diag_pairs = generate_llm_pairs_for_lonely_terms(all_diag_terms, used_diag_terms, 'diagnosis')
    llm_proc_pairs = generate_llm_pairs_for_lonely_terms(all_proc_terms, used_proc_terms, 'procedure')
    llm_med_pairs = generate_llm_pairs_for_lonely_terms(all_med_terms, used_med_terms, 'medication')

    all_pairs = diag_pairs + proc_pairs + med_pairs + llm_diag_pairs + llm_proc_pairs + llm_med_pairs
    
    output_filename = data_dir / 'term_pairs_final_clean.json'
    with open(output_filename, 'w') as f:
        json.dump(all_pairs, f, indent=4)
        
    print(f"\n--- Grand Summary ---")
    print(f"Diagnosis Pairs  : {len(diag_pairs)} (code) + {len(llm_diag_pairs)} (LLM)")
    print(f"Procedure Pairs  : {len(proc_pairs)} (code) + {len(llm_proc_pairs)} (LLM)")
    print(f"Medication Pairs : {len(med_pairs)} (code) + {len(llm_med_pairs)} (LLM)")
    print(f"--------------------")
    print(f"TOTAL Pairs      : {len(all_pairs)}")
    print(f"\nALL DONE! The final, clean dataset is saved to {output_filename}!")