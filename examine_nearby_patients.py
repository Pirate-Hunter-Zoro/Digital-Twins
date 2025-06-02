import json
import pickle
from generate_patients import load_patient_data
from generate_narrative import get_narrative

debug = False

# Load patient data
patients = load_patient_data()
patient_lookup = {p["patient_id"]: p for p in patients}

# Load prediction outputs
with open("patient_output_results.json", "r") as f:
    patient_output_results = json.load(f)

# Load prompts
with open("all_prompts.json", "r") as f:
    all_prompts = json.load(f)

# Load precomputed neighbors
with open("nearest_neighbors.pkl", "rb") as f:
    nearest_neighbors = pickle.load(f)

def inspect_visit(patient_id: str, visit_idx: int, k: int = 5, output_path="examine_output.txt") -> None:
    output = []

    key = (patient_id, visit_idx)
    neighbors = nearest_neighbors.get(key)
    if not neighbors:
        output.append(f"⚠️ No neighbors found for {key}\n")
    else:
        n = len(neighbors)
        if 2 * k > n:
            output.append(f"⚠️ Not enough neighbors (have {n}, need {2*k})\n")
        else:
            prompt_key = f"{patient_id}_{visit_idx}"
            prompt = all_prompts.get(prompt_key)
            if prompt is None:
                output.append(f"⚠️ No prompt found for {prompt_key}\n")
            else:
                if debug:
                    output.append(f"\n📝 Prompt for {patient_id} visit {visit_idx}:\n{prompt}\n")

                output.append("🔍 Closest Sequence of Visits from Other Patients (later is closer):")
                for i in range(k - 1, -1, -1):
                    (neighbor_pid, neighbor_vidx), similarity = neighbors[i]
                    narrative = get_narrative(patient_lookup[neighbor_pid]["visits"], neighbor_vidx)
                    output.append(f"  {k - i}. ID: ({neighbor_pid}, {neighbor_vidx}), Cosine Similarity: {similarity:.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")

                output.append("🧭 Farthest Sequence of Visits from Other Patients (later is farther):")
                for i in range(n - k, n):
                    (neighbor_pid, neighbor_vidx), similarity = neighbors[i]
                    narrative = get_narrative(patient_lookup[neighbor_pid]["visits"], neighbor_vidx)
                    output.append(f"  {i - (n - k) + 1}. ID: ({neighbor_pid}, {neighbor_vidx}), Cosine Similarity: {similarity:.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")

                target_narrative = get_narrative(patient_lookup[patient_id]["visits"], visit_idx)
                output.append(f"\n🎯 Target Patient Narrative (up to Visit {visit_idx}):\n{target_narrative}\n")

                results = patient_output_results.get(patient_id, [])
                if visit_idx < len(results):
                    result = results[visit_idx]
                    output.append(f"🔮 Predicted: {result['predicted']}")
                    output.append(f"✅ Actual: {result['actual']}")
                    output.append(f"📊 Scores: {result.get('scores', {})}")
                else:
                    output.append("⚠️ No prediction result found for this visit.")

    with open(output_path, "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    inspect_visit("P0000001", 3, 2)  # Output will be written to 'examine_output.txt' by default
