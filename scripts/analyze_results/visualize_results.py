import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns # For enhanced plotting
import numpy as np # For np.mean
from scripts.config import setup_config, get_global_config
from scripts.parser import parse_data_args


def visualize_results():
    args = parse_data_args() # Parse arguments for config setup
    
    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    global_config = get_global_config()

    # Define the path to the patient results file
    results_file_path = f"real_data/patient_results_{global_config.num_patients}_{global_config.num_visits}_{global_config.vectorizer_method}_{global_config.distance_metric}.json"
    results_full_path = os.path.join(project_root, results_file_path)

    if not os.path.exists(results_full_path):
        raise FileNotFoundError(f"Patient results file not found at: {results_full_path}. Please ensure main.py has run successfully.")

    print(f"Loading patient results from: {results_full_path}")
    with open(results_full_path, 'r') as file:
        data = json.load(file) # 'data' will be a dictionary: {patient_id: {result_dict}}

    if not data:
        print("No patient results found in the file. Cannot generate visualization.")
        return

    # --- Collect all Jaccard scores by category ---
    diagnoses_scores = []
    medications_scores = []
    treatments_scores = []
    overall_scores = []
    
    # We'll also try to get the visit_idx if it's consistent across all patients
    # (It should be, as it's num_visits-1 from config)
    representative_visit_idx = None

    for patient_id, result_dict in data.items():
        scores = result_dict.get('scores', {}) # Get scores, default to empty dict if missing
        
        diagnoses_scores.append(scores.get('diagnoses', 0.0))
        medications_scores.append(scores.get('medications', 0.0))
        treatments_scores.append(scores.get('treatments', 0.0))
        overall_scores.append(scores.get('overall', 0.0))

        if representative_visit_idx is None:
            representative_visit_idx = result_dict.get('visit_idx', 'N/A')

    # --- Calculate average Jaccard score for each category ---
    avg_diagnoses = np.mean(diagnoses_scores) if diagnoses_scores else 0.0
    avg_medications = np.mean(medications_scores) if medications_scores else 0.0
    avg_treatments = np.mean(treatments_scores) if treatments_scores else 0.0
    avg_overall = np.mean(overall_scores) if overall_scores else 0.0

    # Prepare data for the bar chart
    categories = ['Diagnoses', 'Medications', 'Treatments', 'Overall']
    average_scores = [avg_diagnoses, avg_medications, avg_treatments, avg_overall]

    # --- Generate and Save the Bar Chart ---
    print("\n--- Generating Average Jaccard Scores Bar Chart ---")

    plt.figure(figsize=(10, 6)) # Create a new figure
    sns.barplot(x=categories, y=average_scores, palette='viridis') # Use seaborn for nice bar plot

    plt.ylabel('Average Jaccard Score')
    plt.title(f'Average Prediction Scores by Category\n(for Visit Index {representative_visit_idx}, {len(data)} Patients)')
    plt.ylim(0, 1) # Jaccard scores are between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Define the output directory for plots
    plot_output_dir = os.path.join(project_root, "real_data", "jaccard_score_plots")
    os.makedirs(plot_output_dir, exist_ok=True) # Ensure directory exists

    plot_filename = f"avg_jaccard_scores_{global_config.num_patients}_{global_config.num_visits}_{global_config.vectorizer_method}_{global_config.distance_metric}.png"
    plot_full_path = os.path.join(plot_output_dir, plot_filename)

    plt.savefig(plot_full_path)
    plt.close() # Close the plot to free memory

    print(f"Average Jaccard Scores plot saved to: {plot_full_path}")
    print("----------------------------------------")

if __name__ == "__main__":
    visualize_results()