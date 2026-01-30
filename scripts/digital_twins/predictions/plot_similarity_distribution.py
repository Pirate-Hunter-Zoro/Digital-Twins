import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import ast

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])

def main():
    print("Loading TRD evaluation result files for similarity distribution plot...", flush=True)
    csv_result_files = RESULTS_DIR.glob("trd_evaluation_results_*.csv")
    dfs_to_merge = []
    for result_file in csv_result_files:
        dfs_to_merge.append(pd.read_csv(result_file))
    merged_results = pd.concat(dfs_to_merge, ignore_index=True)
    
    # Create list of all 'near' scores and all 'random' scores
    merged_results['nearest_scores'] = merged_results['nearest_scores'].apply(ast.literal_eval) # Convert to list
    merged_results['random_scores'] = merged_results['random_scores'].apply(ast.literal_eval)
    nearest_scores = []
    random_scores = []
    for near_score_list, random_score_list in zip(merged_results['nearest_scores'], merged_results['random_scores']):
        nearest_scores.extend(near_score_list)
        random_scores.extend(random_score_list)
            
    # Now we plot
    print("Plotting similarity score distributions...", flush=True)
    plt.figure(figsize=(10, 6))
    plt.hist(nearest_scores, bins=100, alpha=0.5, label="Nearest (Cosine) Neighbors", color="blue", density=True)
    plt.hist(random_scores, bins=100, alpha=0.5, label="Random Neighbors", color="orange", density=True)
    plt.xlabel("LLM Judge Similarity Score")
    plt.ylabel("Density")
    plt.title("Distribution of Similarity Scores: Nearest vs Random Neighbors")
    plt.legend()
    plt.savefig(RESULTS_DIR / "similarity_score_distribution.png")
    plt.close()
    print("Similarity distribution plot saved!", flush=True)
    
if __name__ == "__main__":
    main()