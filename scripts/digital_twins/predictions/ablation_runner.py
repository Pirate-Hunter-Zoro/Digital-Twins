import os
from pathlib import Path
import subprocess

from scripts.data_loading.ablation_registry import ABLATIONS

def main():
    baseline_narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])
    baseline_results_dir = Path(os.environ['RESULTS_DIR'])
    for spec in ABLATIONS:
        spec_id = spec["id"]
        ablation_narrative_dir = baseline_narratives_dir / spec_id
        os.environ['NARRATIVES_DIR'] = str(ablation_narrative_dir)
        ablation_embeddings_dir = baseline_embeddings_dir / spec_id
        os.environ['EMBEDDINGS_DIR'] = str(ablation_embeddings_dir)
        os.makedirs(ablation_embeddings_dir, exist_ok=True)
        ablation_results_dir = baseline_results_dir / spec_id
        os.environ['RESULTS_DIR'] = str(ablation_results_dir)
        os.makedirs(ablation_results_dir, exist_ok=True)
        
        print(f"Ablating on {spec}...", flush=True)
        print(f"Narratives: {os.environ['NARRATIVES_DIR']}", flush=True)
        print(f"Embeddings: {os.environ['EMBEDDINGS_DIR']}", flush=True)
        print(f"Results: {os.environ['RESULTS_DIR']}", flush=True)
        subprocess.run(
            ["python", '-m', 'scripts.digital_twins.embeddings.forge_embeddings'],
            check=True,
        )

if __name__=="__main__":
    main()