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
import pandas as pd
from matplotlib import pyplot as plt
import os
from scripts.config import get_global_config

def visualize_results():
    # Create directory for saving plots if it doesn't exist
    if not os.path.exists('jaccard_scores'):
        os.makedirs('jaccard_scores', exist_ok=True)

    with open(f"real_data/patient_results_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json", 'r') as file:
        data = json.load(file)

        for patient_id, visits in data.items():
            df = pd.DataFrame([
                {
                    'visit_idx': v['visit_idx'],
                    'diagnoses_score': v['scores']['diagnoses'],
                    'medications_score': v['scores']['medications'],
                    'treatments_score': v['scores']['treatments'],
                }
                for v in visits
            ])

            # Sort by visit index
            df = df.sort_values(by='visit_idx')

            # Plot the scores
            plt.figure(figsize=(10, 6))
            plt.plot(df['visit_idx'], df['diagnoses_score'], label='Diagnoses', marker='o')
            plt.plot(df['visit_idx'], df['medications_score'], label='Medications', marker='o')
            plt.plot(df['visit_idx'], df['treatments_score'], label='Treatments', marker='o')

            plt.xlabel('Visit Index')
            plt.ylabel('Jaccard Score')
            plt.title(f'Prediction Scores Over Time for Patient {patient_id}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'real_data/jaccard_scores/patient_{patient_id}_scores.png')
            plt.close()

if __name__ == "__main__":
    visualize_results()
    print("Visualization completed and saved in 'jaccard_scores' directory.")