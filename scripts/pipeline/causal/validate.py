import os
import json
from pathlib import Path
import numpy as np

from scripts.pipeline.causal.core import (
    load_encoded_data,
    fit_causal_forest
)
from scripts.pipeline.causal.treatment_registry import TREATMENT_REGISTRY

from dotenv import load_dotenv
load_dotenv()

seed = int(os.environ['SEED'])
train_matrix, test_matrix, train_labels, test_labels = load_encoded_data()

# Create reverse specs from the existing specs
reverse_specs = [
    {
        "key": f"{spec['reference_arm'].lower()}_vs_{spec['comparison_arm'].lower()}",
        "display_name": f"{spec['reference_arm']} vs {spec['comparison_arm']}",
        "reference_arm": spec['comparison_arm'],
        "comparison_arm": spec['reference_arm']
    }
    for spec in TREATMENT_REGISTRY
]