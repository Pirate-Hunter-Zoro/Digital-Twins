import os
import sys
import json
import argparse
import pickle
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.llm.llm_helper import get_narrative
from scripts.common.llm.query_llm import query_llm

# --- Biomni Integration! ---
try:
    from biomni.agent import A1
    agent = A1()
    BIOMNI_AVAILABLE = True
    print("🤖 Biomni agent initialized successfully!")
except ImportError:
    print("⚠️ Biomni package not found. Proceeding without Biomni-powered narrative enhancements.")
    BIOMNI_AVAILABLE = False
except Exception as e:
    print(f"🔬 Biomni agent failed to initialize: {e}. Proceeding without enhancements.")
    BIOMNI_AVAILABLE = False


def get_enhanced_narrative_with_biomni(visits):
    """
    Generates a narrative enhanced with biological pathway data from Biomni.
    """
    basic_narrative = get_narrative(visits) # Your existing function
    if not BIOMNI_AVAILABLE:
        return basic_narrative
    try:
        key_diagnosis = None
        for visit in reversed(visits):
            if visit.get("diagnoses"):
                key_diagnosis = visit["diagnoses"][0].get("Diagnosis_Name")
                if key_diagnosis:
                    break
        if key_diagnosis:
            pathway_info = agent.go(f"Find KEGG pathway for {key_diagnosis}")
            return f"{basic_narrative} The primary diagnosis, {key_diagnosis}, is associated with the {pathway_info}."
    except Exception as e:
        print(f"Biomni call failed: {e}")
    return basic_narrative

def generate_prediction_prompt(patient_dict, patient_lookup, neighbors_data):
    """
    Generates the full prompt for the LLM, including neighbor context.
    """
    config = get_global_config()
    patient_id = patient_dict["patient_id"]
    # We're predicting for the visit AFTER the number of visits used for context
    context_visits = patient_dict["visits"][:config.num_visits]
    
    # 1. Get the patient's own narrative
    patient_narrative = get_enhanced_narrative_with_biomni(context_visits)
    
    # 2. Get neighbor narratives if we have them!
    neighbor_narratives_str = "No neighbor data was available for this patient."
    if neighbors_data:
        neighbor_keys = neighbors_data.get((patient_id, len(context_visits) - 1), [])
        
        neighbor_narratives = []
        for (neighbor_id, neighbor_vidx), _, _ in neighbor_keys[:config.num_neighbors]:
            neighbor_patient_dict = patient_lookup.get(neighbor_id)
            if neighbor_patient_dict:
                neighbor_visits = neighbor_patient_dict["visits"][:neighbor_vidx + 1]
                neighbor_narrative = get_enhanced_narrative_with_biomni(neighbor_visits)
                neighbor_narratives.append(f"[Neighbor: {neighbor_id}]\n{neighbor_narrative}")
        
        if neighbor_narratives:
            neighbor_narratives_str = "\n\n".join(neighbor_narratives)

    # 3. Construct the final, magnificent prompt
    prompt = f"""You are a medical expert generating a prediction for a patient's next visit.

    [Patient History]
    {patient_narrative}

    [Similar Patient Cases (Neighbors)]
    {neighbor_narratives_str}

    [Instruction]
    Based on the patient's history and the histories of similar patients, predict the diagnoses, medications, and treatments for the patient's NEXT clinical visit.
    Return the prediction as a structured JSON object.

    [Output Format Example]
    ```json
    {{
    "predicted_visit": {{
        "diagnoses": ["Example Diagnosis"],
        "medications": ["Example Medication"],
        "treatments": ["Example Treatment"]
    }}
    }}

    """
    return prompt.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate patient predictions for World 1.")
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", required=True)
    parser.add_argument("--num_visits", type=int, required=True)
    parser.add_argument("--num_patients", type=int, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--model_name", required=True, help="Name of the LLM to use for prediction.")
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name,
    )
    config = get_global_config()

    # --- Build our hyper-structured paths ---
    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method
    input_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}" / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}"
    output_dir = input_dir # We'll save the output in the same place!
    os.makedirs(output_dir, exist_ok=True)

    neighbors_path = input_dir / "neighbors.pkl"
    output_path = output_dir / "patient_predictions.json"

    # --- Load Data (with smart neighbor handling!) ---
    print("📂 Loading patient data...")
    patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patient_data}

    neighbors_data = None
    if os.path.exists(neighbors_path):
        print(f"✅ Found and loaded neighbors file from: {neighbors_path}")
        with open(neighbors_path, "rb") as f:
            neighbors_data = pickle.load(f)
    else:
        print(f"⚠️ Neighbors file not found at {neighbors_path}. Proceeding without neighbor data.")

    # --- Prediction Loop ---
    all_predictions = {}
    for patient in patient_data:
        # We need at least num_visits history to predict the one AFTER
        if len(patient.get("visits", [])) <= config.num_visits:
            continue
            
        print(f"🧠 Generating prediction for patient: {patient['patient_id']}")
        
        prompt = generate_prediction_prompt(patient, patient_lookup, neighbors_data)
        
        # This is where we would use a function like your force_valid_prediction
        # For now, we'll just use the basic query_llm
        prediction_json_str = query_llm(prompt, max_tokens=1024) 
        
        try:
            # A simple way to extract JSON from a markdown block
            json_match = prediction_json_str.split("```json")[1].split("```")[0]
            prediction = json.loads(json_match)
            all_predictions[patient['patient_id']] = prediction
        except (IndexError, json.JSONDecodeError) as e:
            print(f"⚠️ Could not parse prediction for patient {patient['patient_id']}. Error: {e}")

    # --- Save Results ---
    print(f"\n💾 Saving {len(all_predictions)} predictions to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print("✅ Done!")

if __name__ == "__main__":
    main()