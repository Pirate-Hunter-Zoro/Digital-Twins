import sys
import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.config import get_global_config, setup_config
from scripts.parser import parse_data_args


def load_results_to_dataframe(results_path: str) -> pd.DataFrame:
    """
    A technique to summon our raw JSON data into a powerful
    Pandas DataFrame Cursed Tool for easier manipulation.
    """
    print(f"Loading patient results from: {results_path}")
    with open(results_path, 'r') as file:
        data = json.load(file)

    if not data:
        print("No patient results found in the file.")
        return pd.DataFrame()

    # Transform the nested dictionary into a flat list for the DataFrame
    records = []
    for patient_id, result_dict in data.items():
        scores = result_dict.get('scores', {})
        record = {
            'patient_id': patient_id,
            'visit_idx': result_dict.get('visit_idx'),
            'diagnoses': scores.get('diagnoses', 0.0),
            'medications': scores.get('medications', 0.0),
            'treatments': scores.get('treatments', 0.0),
            'overall': scores.get('overall', 0.0)
        }
        records.append(record)
    
    return pd.DataFrame(records)


def plot_score_distribution(df: pd.DataFrame, output_dir: str, config_str: str):
    """
    Technique 1: The Six Eyes Analysis.
    Instead of a simple average, we perceive the full distribution of scores
    using a box plot to see median, spread, and outliers.
    """
    if df.empty:
        return

    print("--- Generating Overall Score Distribution Plot (Box Plot) ---")
    
    # Melt the DataFrame from wide to long format for Seaborn
    df_melted = df.melt(id_vars=['patient_id'], value_vars=['diagnoses', 'medications', 'treatments', 'overall'],
                        var_name='category', value_name='score')

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='category', y='score', data=df_melted, palette='viridis')
    sns.stripplot(x='category', y='score', data=df_melted, color=".25", size=4, jitter=True, alpha=0.6)

    plt.ylabel('Weighted Similarity Score')
    plt.xlabel('Prediction Category')
    plt.title(f'Distribution of Prediction Scores by Category\n({len(df)} Patients, {config_str})')
    # IMPORTANT: We removed the y-axis limit. Our new score is so powerful,
    # it is not capped! This allows us to see its true power level.
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # The file is no longer named "jaccard"
    plot_filename = f"overall_score_distribution_{config_str}.png"
    plot_full_path = os.path.join(output_dir, plot_filename)

    plt.savefig(plot_full_path)
    plt.close()
    print(f"Overall distribution plot saved to: {plot_full_path}")


def plot_individual_reports(df: pd.DataFrame, output_dir: str, config_str: str):
    """
    Technique 2: Individual Case File Creation.
    We create a separate scroll (plot) for each exorcism (patient prediction)
    for granular analysis.
    """
    if df.empty:
        return

    # Create a dedicated vault for the individual reports
    reports_vault_path = os.path.join(output_dir, f"individual_reports_{config_str}")
    os.makedirs(reports_vault_path, exist_ok=True)
    
    print(f"\n--- Generating Individual Patient Reports in: {reports_vault_path} ---")

    for index, row in df.iterrows():
        patient_id = row['patient_id']
        scores_df = pd.DataFrame({
            'Category': ['Diagnoses', 'Medications', 'Treatments', 'Overall'],
            'Score': [row['diagnoses'], row['medications'], row['treatments'], row['overall']]
        })

        plt.figure(figsize=(8, 5))
        sns.barplot(x='Category', y='Score', data=scores_df, palette='plasma')
        
        plt.ylabel('Weighted Similarity Score')
        plt.title(f'Prediction Scores for Patient: {patient_id}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        report_filename = f"report_{patient_id}.png"
        report_full_path = os.path.join(reports_vault_path, report_filename)
        plt.savefig(report_full_path)
        plt.close()

    print(f"Saved {len(df)} individual patient reports.")


def visualize_results():
    """
    The main Domain Expansion controller. It summons the data,
    then unleashes multiple analysis and visualization techniques.
    """
    args = parse_data_args()
    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    global_config = get_global_config()

    config_str = f"{global_config.num_patients}_{global_config.num_visits}_{global_config.vectorizer_method}_{global_config.distance_metric}"
    results_file_path = f"real_data/patient_results_{config_str}.json"
    results_full_path = os.path.join(project_root, results_file_path)

    try:
        df = load_results_to_dataframe(results_full_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}. Please ensure main.py has run successfully.") from e

    if df.empty:
        print("No data to visualize.")
        return

    # Define the main output directory, now with a generic name
    plot_output_dir = os.path.join(project_root, "real_data", "prediction_score_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Unleash our techniques!
    plot_score_distribution(df, plot_output_dir, config_str)
    plot_individual_reports(df, plot_output_dir, config_str)

    print("\n----------------------------------------")
    print("Visualization Domain Expansion complete!")
    print("----------------------------------------")


if __name__ == "__main__":
    visualize_results()