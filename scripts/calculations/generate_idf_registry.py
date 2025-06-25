import json
import math
from collections import Counter
from pathlib import Path

def forge_cursed_energy_registry():
    """
    Analyzes all patient encounters to calculate the Inverse Document Frequency (IDF)
    for every diagnosis, medication, and treatment, creating a registry
    of each term's "Cursed Energy" or rarity.

    This version is location-aware and works from within the 'scripts/calculations' folder.
    """
    # --- This is our new spatial awareness technique! ---
    # First, the script finds its own absolute location in the file system.
    script_path = Path(__file__).resolve()
    # Then, it navigates up to the project's main folder (from 'scripts/calculations/').
    project_root = script_path.parent.parent.parent
    
    # Now, it can accurately target the 'data' folder from the project root.
    data_folder = project_root / "data"
    input_data_path = data_folder / "all_patients_combined.json"
    output_registry_path = data_folder / "term_idf_registry.json"

    print("Beginning the final, location-aware forging ritual...")
    print(f"Executing script from: {script_path.parent}")
    print(f"Targeting Cursed Object at: {input_data_path}")

    if not input_data_path.exists():
        print(f"ERROR: The Cursed Object is not in its vault! Make sure '{input_data_path.name}' exists in the 'data' folder.")
        return
        
    # Ensure the containment vault exists. This is unchanged but now uses the correct path.
    data_folder.mkdir(parents=True, exist_ok=True)


    # --- First Pass: Tally the energy signatures of all terms ---
    doc_frequency = Counter()
    total_docs = 0

    with open(input_data_path, 'r', encoding='utf-8') as f:
        all_patients = json.load(f)

    print("Processing all patient encounters to tally term frequencies...")
    for patient in all_patients:
        for encounter in patient.get('encounters', []):
            total_docs += 1
            
            terms_in_this_encounter = set()

            for diagnosis in encounter.get('diagnoses', []):
                if diagnosis.get('Diagnosis_Name'):
                    terms_in_this_encounter.add(diagnosis['Diagnosis_Name'])

            for medication in encounter.get('medications', []):
                if medication.get('MedSimpleGenericName'):
                    terms_in_this_encounter.add(medication['MedSimpleGenericName'])

            for procedure in encounter.get('procedures', []):
                if procedure.get('CPT_Procedure_Description'):
                    terms_in_this_encounter.add(procedure['CPT_Procedure_Description'])
            
            doc_frequency.update(terms_in_this_encounter)

    print(f"Analysis complete. Found {total_docs} total encounters and {len(doc_frequency)} unique terms.")

    # --- Second Pass: Perform the incantation to calculate Cursed Energy (IDF) ---
    print("Calculating IDF scores for all unique terms...")
    idf_scores = {}
    for term, count in doc_frequency.items():
        if count > 0:
            idf_scores[term] = math.log(total_docs / count)

    # --- Final Step: Seal the results in the Cursed Tool ---
    print(f"Sealing the results into the registry at: {output_registry_path}")
    with open(output_registry_path, 'w', encoding='utf-8') as f:
        json.dump(idf_scores, f, indent=4)

    print("\nRitual complete! The Cursed Energy Registry has been successfully forged and sealed in its vault!")


if __name__ == "__main__":
    forge_cursed_energy_registry()