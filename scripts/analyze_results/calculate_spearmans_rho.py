import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# Assuming this script is in 'project_root/scripts/analyze_results/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import json
import numpy as np
from scipy.stats import spearmanr
from scripts.config import setup_config, get_global_config
from scripts.parser import parse_data_args
from matplotlib import pyplot as plt # NEW: Import pyplot
import seaborn as sns # NEW: Import seaborn for enhanced plotting


def calculate_and_print_spearmans_rho():
    # --- Setup Project Configuration ---
    args = parse_data_args()

    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        representation_method=args.representation_method,
    )
    global_config = get_global_config()

    # Define the path to the correlation data file
    correlation_data_file = f"data/llm_mahalanobis_correlation_{global_config.num_patients}_{global_config.num_visits}_{global_config.vectorizer_method}_{global_config.distance_metric}.json"

    # Ensure the output directory is correct relative to the project_root
    correlation_data_path = os.path.join(project_root, correlation_data_file)

    if not os.path.exists(correlation_data_path):
        raise FileNotFoundError(f"Correlation data file not found at: {correlation_data_path}. Please ensure examine_nearby_patients.py has run successfully.")

    print(f"Loading correlation data from: {correlation_data_path}")
    with open(correlation_data_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("No data found in the correlation file. Cannot compute Spearman's Rho.")
        return

    # Extract the lists of scores and distances
    llm_scores = [d["avg_llm_relevance_score"] for d in data]
    mahalanobis_distances = [d["mahalanobis_distance"] for d in data]

    # Ensure there's enough data for correlation
    if len(llm_scores) < 2 or len(mahalanobis_distances) < 2:
        print("Not enough data points (need at least 2) to compute correlation.")
        return

    # Compute Spearman's Rho
    correlation_rho, pvalue_rho = spearmanr(llm_scores, mahalanobis_distances)

    print(f"\n--- Spearman's Rho Correlation Results ---")
    print(f"Number of data points: {len(llm_scores)}")
    print(f"Spearman's Rho: {correlation_rho:.4f}")
    print(f"P-value: {pvalue_rho:.4f}")
    print(f"Interpretation: {'' if pvalue_rho > 0.05 else 'Statistically significant (p < 0.05). '}")
    print(f"A correlation close to 1 indicates strong positive monotonic relationship.")
    print(f"A correlation close to -1 indicates strong negative monotonic relationship.")
    print(f"A correlation close to 0 indicates no monotonic relationship.")
    print(f"(Remember, a *negative* correlation is expected if higher relevance corresponds to *lower* distance).")
    print(f"----------------------------------------")

    # --- NEW: Generate and Save the Plot ---
    print("\n--- Generating Correlation Plot ---")
    
    # Create a DataFrame for seaborn plotting
    plot_df = pd.DataFrame({
        'LLM_Relevance_Score': llm_scores,
        'Mahalanobis_Distance': mahalanobis_distances
    })

    plt.figure(figsize=(10, 7)) # Create a new figure
    
    # Use seaborn.regplot for scatter plot with a regression line
    # lowess=True fits a locally weighted regression (non-linear, good for monotonic trends)
    # scatter_kws controls scatter plot properties (e.g., transparency)
    # line_kws controls line properties
    sns.regplot(
        x='LLM_Relevance_Score',
        y='Mahalanobis_Distance',
        data=plot_df,
        scatter_kws={'alpha': 0.6},
        line_kws={'color': 'red'},
        lowess=True # Use local regression for non-linear monotonic trend
    )

    plt.title(f'LLM Relevance Score vs. Mahalanobis Distance\nSpearman\'s ρ = {correlation_rho:.4f}, p = {pvalue_rho:.4f}')
    plt.xlabel('Average LLM Relevance Score (0-9)')
    plt.ylabel('Mahalanobis Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Define the output path for the plot
    plot_output_dir = os.path.join(project_root, "data", "correlation_plots")
    os.makedirs(plot_output_dir, exist_ok=True) # Ensure directory exists

    plot_filename = f"llm_mahalanobis_correlation_plot_{global_config.num_patients}_{global_config.num_visits}_{global_config.vectorizer_method}_{global_config.distance_metric}.png"
    plot_full_path = os.path.join(plot_output_dir, plot_filename)

    plt.savefig(plot_full_path)
    plt.close() # Close the plot to free memory

    print(f"Correlation plot saved to: {plot_full_path}")
    print("----------------------------------------")

if __name__ == "__main__":
    # Ensure config setup is done here for standalone run
    # (parse_data_args already does setup_config inside this script context)
    calculate_and_print_spearmans_rho()