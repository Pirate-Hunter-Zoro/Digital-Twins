from pathlib import Path
import os
import json

from scripts.pipeline.causal import run_one

from dotenv import load_dotenv
load_dotenv()

results_path = Path(os.environ['ARTIFACTS_DIR']) / 'causal_pipeline'
os.makedirs(results_path, exist_ok=True)
# Should already exist
results = list(results_path.glob("results_*.json"))
records = []
for res in results:
    with open(res, 'r') as f:
        records.append(json.load(res))