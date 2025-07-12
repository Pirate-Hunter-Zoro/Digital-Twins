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

# --- Pairing Functions (Modified to return all terms and used terms) ---

def create_diagnosis_pairs(diagnosis_file):
    """Creates code-based pairs for diagnoses."""
    df = pd.read_csv(diagnosis_file)
    df.dropna(subset=['Code', 'Description'], inplace=True)
    all_terms = df['Description'].unique().tolist()
    code_groups = df.groupby('Code')['Description'].apply(list)
    pairs = []
    used_terms = set()
    for code, descriptions in code_groups.items():
        if len(descriptions) > 1:
            for p in combinations(descriptions, 2):
                pairs.append({'term': p[0], 'counterpart': p[1], 'category': 'diagnosis', 'type': 'code_based'})
                used_terms.add(p[0])
                used_terms.add(p[1])
    return pairs, used_terms, all_terms

def create_procedure_pairs(procedure_file):
    """Creates code-based pairs for procedures."""
    df = pd.read_csv(procedure_file)
    df.dropna(subset=['CPT_Procedure_Code', 'CPT_Procedure_Description'], inplace=True)
    all_terms = df['CPT_Procedure_Description'].unique().tolist()
    code_groups = df.groupby('CPT_Procedure_Code')['CPT_Procedure_Description'].apply(list)
    pairs = []
    used_terms = set()
    for code, descriptions in code_groups.items():
        if len(descriptions) > 1:
            for p in combinations(descriptions, 2):
                pairs.append({'term': p[0], 'counterpart': p[1], 'category': 'procedure', 'type': 'code_based'})
                used_terms.add(p[0])
                used_terms.add(p[1])
    return pairs, used_terms, all_terms

def create_medication_pairs(med_freq_file, rxnorm_file):
    """Creates code-based pairs for medications."""
    med_df = pd.read_csv(med_freq_file)
    # --- THIS IS THE FIX! ---
    # We were looking for 'Name', but the real column is 'MedSimpleGenericName'!
    all_med_terms = med_df['MedSimpleGenericName'].unique().tolist()
    
    merged_df = pd.merge(med_df, rxnorm_df, on='MedicationEpicID')
    # The 'Name' column here comes from the rxnorm_df and is correct!
    filtered_df = merged_df[merged_df['RXNORM_TYPE'].isin(['Ingredient', 'Brand Name', 'Precise Ingredient', 'MED_ONLY'])]
    code_groups = filtered_df.groupby('RXNORM_CODE')['Name'].unique().apply(list)
    pairs = []
    used_terms = set()
    for code, names in code_groups.items():
        if len(names) > 1:
            for p in combinations(names, 2):
                term1 = ' '.join(p[0].split(' ')[:2])
                term2 = ' '.join(p[1].split(' ')[:2])
                if term1.lower() != term2.lower():
                    pairs.append({'term': p[0], 'counterpart': p[1], 'category': 'medication', 'type': 'code_based'})
                    used_terms.add(p[0])
                    used_terms.add(p[1])
    unique_pairs = [dict(t) for t in {tuple(d.items()) for d in pairs}]
    return unique_pairs, used_terms, all_med_terms


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
    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parents[1]
    data_dir = project_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_file = data_dir / 'diagnosis_frequency.csv'
    procedure_file = data_dir / 'procedure_frequency.csv'
    med_freq_file = data_dir / 'medication_frequency.csv'
    rxnorm_file = data_dir / 'RXNorm_Table-25_06_17-v1.csv'
    
    get_llm_client()
    
    print("--- Step 1: Generating all code-based pairs ---")
    diag_pairs, used_diag, all_diag = create_diagnosis_pairs(diagnosis_file)
    proc_pairs, used_proc, all_proc = create_procedure_pairs(procedure_file)
    med_pairs, used_med, all_med = create_medication_pairs(med_freq_file, rxnorm_file)
    
    print("\n--- Step 2: Generating LLM-based pairs for ALL lonely terms ---")
    llm_diag_pairs = generate_llm_pairs_for_lonely_terms(all_diag, used_diag, 'diagnosis')
    llm_proc_pairs = generate_llm_pairs_for_lonely_terms(all_proc, used_proc, 'procedure')
    llm_med_pairs = generate_llm_pairs_for_lonely_terms(all_med, used_med, 'medication')

    all_pairs = diag_pairs + proc_pairs + med_pairs + llm_diag_pairs + llm_proc_pairs + llm_med_pairs
    
    output_filename = data_dir / 'term_pairs_fully_generated.json'
    with open(output_filename, 'w') as f:
        json.dump(all_pairs, f, indent=4)
        
    print(f"\n--- Grand Summary ---")
    print(f"Diagnosis Pairs  : {len(diag_pairs)} (code) + {len(llm_diag_pairs)} (LLM)")
    print(f"Procedure Pairs  : {len(proc_pairs)} (code) + {len(llm_proc_pairs)} (LLM)")
    print(f"Medication Pairs : {len(med_pairs)} (code) + {len(llm_med_pairs)} (LLM)")
    print(f"--------------------")
    print(f"TOTAL Pairs      : {len(all_pairs)}")
    print(f"\nALL DONE! All pairs saved to {output_filename}! Now the experiment is complete and magnificent!")