import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.ablation_registry import ABLATIONS

def main():
    baseline_narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])
    
    for spec in ABLATIONS:
        spec_id = spec["id"]
        ablation_narrative_dir = baseline_narratives_dir / spec_id
        os.environ['NARRATIVES_DIR'] = str(ablation_narrative_dir)
        ablation_embeddings_dir = baseline_embeddings_dir / spec_id
        os.environ['EMBEDDINGS_DIR'] = str(ablation_embeddings_dir)
        os.makedirs(ablation_embeddings_dir, exist_ok=True)
        
        print(f"Ablating on {spec}...", flush=True)
        print(f"Narratives: {os.environ['NARRATIVES_DIR']}", flush=True)
        print(f"Embeddings: {os.environ['EMBEDDINGS_DIR']}", flush=True)
        
        # With all the proper .env changes made, forge the embeddings with the given ablation
        subprocess.run(
            ["python", '-m', 'scripts.embedder_investigation.embeddings.forge_embeddings'],
            check=True,
        )
        
if __name__=="__main__":
    main()