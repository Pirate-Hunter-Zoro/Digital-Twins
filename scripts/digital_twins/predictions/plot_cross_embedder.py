import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.utils import VectorSource
from scripts.data_loading.ablation_registry import ABLATIONS

ARTIFACTS_DIR = Path(os.environ['ARTIFACTS_DIR'])
VLLM_MODEL_NAME = os.environ['VLLM_MODEL_NAME']

ABLATION_SPECS = ["permute_psych_history", "permute_med_burden"]
ABLATION_NAMES = []
for abl in ABLATION_SPECS:
    for abl_dict in ABLATIONS:
        if abl_dict['id'] == abl:
            ABLATION_NAMES.append(abl_dict['display'])
            break
    
EMBEDDERS = ["bge-small-en-v1.5", "bge-en-icl", "Qwen-Qwen3-Embedding-4B", "Qwen-Qwen3-Embedding-8B"]
SHORT_NAME_EMBS = ["bge-small", "bge-en-icl", "Qwen3-4B", "Qwen3-8B"]
ENCODING_DIRS = [ARTIFACTS_DIR / embedder_model_name / VLLM_MODEL_NAME for embedder_model_name in EMBEDDERS]

def main():
    for dir in ENCODING_DIRS:
        if not ((dir / f"classical_ml_results_{VectorSource.EMBEDDED.name}.json").exists() and (dir / "ablation_summary.csv").exists()):
            raise FileNotFoundError(f"Cannot run cross-embedder display as {str(dir)} is missing some of its result files...")
    
    scores = np.zeros(shape=(len(ENCODING_DIRS), 3))
    error_bar_lengths = np.zeros(shape=(2, len(ENCODING_DIRS)))
    for i, dir in enumerate(ENCODING_DIRS):
        with open(dir / f"classical_ml_results_{VectorSource.EMBEDDED.name}.json", 'r') as f:
            ml_results = json.load(f)
            roc_score, roc_score_ci_low, roc_score_ci_high = ml_results['logistic_regression']['roc_score'],\
                ml_results['logistic_regression']['roc_score_ci_low'],\
                    ml_results['logistic_regression']['roc_score_ci_high']
            scores[i] = np.array([roc_score_ci_low, roc_score, roc_score_ci_high])
            error_bar_lengths[0, i] = roc_score - roc_score_ci_low # How far each error bar should dip below dot
            error_bar_lengths[1, i] = roc_score_ci_high - roc_score # How far each error bar should rise above dot
    
    # For logistic logression only, over all the different embedding models and specified ablations, see the roc difference
    lr_ablation_deltas = np.zeros(shape=(len(ENCODING_DIRS),len(ABLATION_SPECS)))
    lr_ablation_delta_ci_lows = np.zeros_like(lr_ablation_deltas)
    lr_ablation_delta_ci_highs = np.zeros_like(lr_ablation_deltas)
    for i, dir in enumerate(ENCODING_DIRS):
        ablation_df = pd.read_csv(dir / "ablation_summary.csv")
        lr_for_ablations = ablation_df[(ablation_df['classifier'] == 'logistic_regression') & ablation_df['spec_id'].isin(ABLATION_SPECS)]
        lr_for_ablations = lr_for_ablations.set_index('spec_id')
        for j, abl_id in enumerate(ABLATION_SPECS):
            lr_ablation_deltas[i, j] = lr_for_ablations.loc[abl_id, 'delta_roc_score']
            lr_ablation_delta_ci_lows[i, j] = lr_for_ablations.loc[abl_id, 'delta_roc_score_ci_low']
            lr_ablation_delta_ci_highs[i, j] = lr_for_ablations.loc[abl_id, 'delta_roc_score_ci_high']
    
    # Plot cross-embedder ablation results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11,6))
    left_ax, right_ax = axes[0], axes[1]
    
    # Left plot is AUC score confidence interval over all embedders
    positions = np.arange(len(EMBEDDERS))
    # scores[:, 1] stores the height of each dot for each error bar - the actual observed ROC AUC score
    left_ax.errorbar(scores[:, 1], positions, fmt='o', xerr=error_bar_lengths, capsize=5)
    left_ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    left_ax.set_xlabel("Embedded LR ROC AUC")
    left_ax.set_yticks(positions, SHORT_NAME_EMBS)
    left_ax.set_title("(A) Discrimination")
    left_ax.invert_yaxis()
    
    # Right plot is similar, but now multiple bars for each ablation
    n = len(ABLATION_SPECS)
    cluster_width = 0.8
    bar_width = cluster_width / n
    for j, abl_name in enumerate(ABLATION_NAMES):
        abl_error_bar_lengths = np.zeros(shape=(2, len(ENCODING_DIRS)))
        # Similar to confidence interval plot in left plot, but now it is multiple such intervals for one embedding model - one for each ablation
        for i in range(len(ENCODING_DIRS)):
            abl_error_bar_lengths[0, i] = lr_ablation_deltas[i, j] - lr_ablation_delta_ci_lows[i, j]
            abl_error_bar_lengths[1, i] = lr_ablation_delta_ci_highs[i, j] - lr_ablation_deltas[i, j]
        # Offset positions spread across entire right plot for this particular ablation
        right_ax.errorbar(lr_ablation_deltas[:, j], positions + (j - (n-1)/2)*bar_width, fmt='o', xerr=abl_error_bar_lengths, capsize=5, label=abl_name)
    right_ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    right_ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    right_ax.set_xlabel("Δ ROC AUC (ablated − baseline)")
    right_ax.set_yticks(positions, SHORT_NAME_EMBS) 
    right_ax.set_title("(B) Semantic-feature ablation")
    right_ax.invert_yaxis()
    
    fig.tight_layout(rect=(0,0,0.85,1))
    fig.savefig(ARTIFACTS_DIR / f"cross_embedder_robustness_{VectorSource.EMBEDDED.name}.png")
    plt.close(fig)
    
if __name__=="__main__":
    main()