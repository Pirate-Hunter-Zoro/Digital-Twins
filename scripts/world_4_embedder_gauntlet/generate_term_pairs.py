import pandas as pd
import json
from itertools import combinations
from pathlib import Path

def create_diagnosis_pairs(diagnosis_file):
    """Creates diagnosis term pairs based on shared codes."""
    df = pd.read_csv(diagnosis_file)
    df.dropna(subset=['Code', 'Description'], inplace=True)
    code_groups = df.groupby('Code')['Description'].apply(list)
    diagnosis_pairs = []
    for code, descriptions in code_groups.items():
        if len(descriptions) > 1:
            for pair in combinations(descriptions, 2):
                diagnosis_pairs.append({'term': pair[0], 'counterpart': pair[1]})
    return diagnosis_pairs

def create_procedure_pairs(procedure_file):
    """Creates procedure term pairs based on shared CPT codes."""
    df = pd.read_csv(procedure_file)
    df.dropna(subset=['CPT_Procedure_Code', 'CPT_Procedure_Description'], inplace=True)
    code_groups = df.groupby('CPT_Procedure_Code')['CPT_Procedure_Description'].apply(list)
    procedure_pairs = []
    for code, descriptions in code_groups.items():
        if len(descriptions) > 1:
            for pair in combinations(descriptions, 2):
                procedure_pairs.append({'term': pair[0], 'counterpart': pair[1]})
    return procedure_pairs

def create_medication_pairs(med_freq_file, rxnorm_file):
    """Creates medication term pairs based on shared RxNorm codes."""
    med_df = pd.read_csv(med_freq_file)
    rxnorm_df = pd.read_csv(rxnorm_file)
    merged_df = pd.merge(med_df, rxnorm_df, on='MedicationEpicID')
    filtered_df = merged_df[merged_df['RXNORM_TYPE'].isin(['Ingredient', 'Brand Name', 'Precise Ingredient', 'MED_ONLY'])]
    code_groups = filtered_df.groupby('RXNORM_CODE')['Name'].unique().apply(list)
    medication_pairs = []
    for code, names in code_groups.items():
        if len(names) > 1:
            for pair in combinations(names, 2):
                term1 = ' '.join(pair[0].split(' ')[:2])
                term2 = ' '.join(pair[1].split(' ')[:2])
                if term1.lower() != term2.lower():
                    medication_pairs.append({'term': pair[0], 'counterpart': pair[1]})
    unique_medication_pairs = [dict(t) for t in {tuple(d.items()) for d in medication_pairs}]
    return unique_medication_pairs


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
    
    print("Ooh, let's find some twins! Starting with diagnoses... 🩺")
    diag_pairs = create_diagnosis_pairs(diagnosis_file)
    print(f"Found {len(diag_pairs)} diagnosis pairs! ZAP!")

    print("Now for procedures! So many tiny little things to do! 🛠️")
    proc_pairs = create_procedure_pairs(procedure_file)
    print(f"Found {len(proc_pairs)} procedure pairs! WOWEE!")

    print("Medications are my favorite! It's like a super fun puzzle! 💊")
    med_pairs = create_medication_pairs(med_freq_file, rxnorm_file)
    print(f"Found {len(med_pairs)} medication pairs! AMAZING!")
    
    all_pairs = diag_pairs + proc_pairs + med_pairs
    
    output_filename = data_dir / 'term_pairs.json'
    with open(output_filename, 'w') as f:
        json.dump(all_pairs, f, indent=4)
        
    print(f"\nALL DONE! A whopping {len(all_pairs)} pairs have been saved to {output_filename}! ISN'T IT BEAUTIFUL?! Now let's go test some vectorizers! FOR SCIENCE! 🚀")