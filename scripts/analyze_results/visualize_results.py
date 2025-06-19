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

import json
import numpy as np
from matplotlib import pyplot as plt
import os
from scripts.config import get_global_config

def visualize_results():
    # Create directory for saving plots if it doesn't exist
    if not os.path.exists('jaccard_scores'):
        os.makedirs('jaccard_scores', exist_ok=True)

    # Load the patient results file
    # The current main.py produces patient_results_{...}.json which is a dict of patient_id -> single_visit_result
    results_file_path = f"real_data/patient_results_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json"
    with open(results_file_path, 'r') as file:
        data = json.load(file) # 'data' will be a dictionary: {patient_id: {result_dict}}

        # For visualization, we'll plot the scores for each patient.
        # This current setup only plots one set of scores (for num_visits-1 visit) per patient.
        # To plot "scores over time" as in original, main.py would need to generate multiple results per patient.

        # Let's plot average scores or individual patient scores for the single predicted visit
        # We need to decide what 'over time' means if only one visit is predicted.
        # It's better to visualize individual patient scores if only one visit is predicted.

        # Refactoring visualize_results to plot individual patient scores or overall averages.
        # Let's assume for now, it plots individual patient scores.
        
        # We need to iterate through the data.items() but 'visits' no longer exists.
        # It's patient_id, result_dict
        for patient_id, result_dict in data.items():
            scores = result_dict['scores']
            
            # Since we have scores for a single visit (num_visits-1), plotting over time is tricky.
            # Let's plot a bar chart for this patient's scores, or just print them to output.
            # To match the spirit of "scores over time", main.py needs to predict multiple visits.

            # For now, let's simplify and just log the scores or plot a summary if plotting over time isn't feasible with this input.
            # If visualize_results is expected to plot over time, main.py must be changed to accumulate a list of results.
            
            # Let's assume for now the user expects some aggregated plot or a list of scores.
            # The current setup provides single scores for the predicted visit.
            
            # We can create a DataFrame from all patients' scores to plot them as distributions or overall average.
            # Or just print for now.

            # To plot as "scores over time", the 'result' object in main.py's 'all_results' should contain
            # a list of these 'result' dicts, one for each visit_idx predicted for that patient.
        
            # For a quick fix, let's create a dummy DataFrame that will allow the plot code to run.
            # But the plot itself won't be "over time" for individual patients based on current main.py output.

            # Plot for each patient (single visit scores)
            plt.figure(figsize=(8, 5))
            labels = ['Diagnoses', 'Medications', 'Treatments']
            values = [scores.get('diagnoses', 0.0), scores.get('medications', 0.0), scores.get('treatments', 0.0)]
            x = np.arange(len(labels))
            
            plt.bar(x, values)
            plt.ylabel('Jaccard Score')
            plt.xticks(x, labels)
            plt.title(f'Prediction Scores for Patient {patient_id} (Visit {result_dict["visit_idx"]})')
            plt.ylim(0, 1) # Jaccard scores are between 0 and 1
            plt.tight_layout()
            plt.savefig(f'real_data/jaccard_scores/patient_{patient_id}_scores.png')
            plt.close()

    print("Visualization completed and saved in 'jaccard_scores' directory (Individual patient scores).")

if __name__ == "__main__":
    # Ensure config setup is done here for standalone run
    from scripts.parser import parse_data_args
    from scripts.config import setup_config
    args = parse_data_args()

    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    
    visualize_results()