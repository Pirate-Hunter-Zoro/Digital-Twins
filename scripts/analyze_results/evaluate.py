import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

def evaluate_prediction_by_category(predicted: dict, actual: dict) -> dict[str, float]:
    scores = {}
    all_keys = ["diagnoses", "medications", "treatments"]
    for key in all_keys:
        pred_set = predicted.get(key, set())
        actual_set = actual.get(key, set())
        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        scores[key] = intersection / union if union else 0.0
    scores["overall"] = sum(scores.values()) / len(all_keys)
    return scores