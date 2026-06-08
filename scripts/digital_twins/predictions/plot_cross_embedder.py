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
            error_bar_lengths[0, i] = roc_score - roc_score_ci_low
            error_bar_lengths[1, i] = roc_score_ci_high - roc_score
    
    lr_ablation_deltas = np.zeros(shape=(len(ENCODING_DIRS),len(ABLATION_SPECS)))
    for i, dir in enumerate(ENCODING_DIRS):
        ablation_df = pd.read_csv(dir / "ablation_summary.csv")
        lr_for_ablations = ablation_df[(ablation_df['classifier'] == 'logistic_regression') & ablation_df['spec_id'].isin(ABLATION_SPECS)]
        lr_for_ablations = lr_for_ablations.set_index('spec_id')
        
    
if __name__=="__main__":
    main()