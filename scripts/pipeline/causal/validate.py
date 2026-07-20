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

pairs = []
for i, (spec, reverse_spec) in enumerate(zip(TREATMENT_REGISTRY, reverse_specs)):
    _, cate_fwd, _, X_fwd, _, T_fwd = fit_causal_forest(spec, train_matrix, test_matrix, train_labels, seed)
    _, cate_rev, _, X_rev, _, T_rev = fit_causal_forest(reverse_spec, train_matrix, test_matrix, train_labels, seed)
    assert cate_fwd.shape == cate_rev.shape, f"Received forward CATE shape {cate_fwd.shape}, reverse CATE shape {cate_rev.shape}"
    assert X_fwd.index.equals(X_rev.index), f"Test patients are not identical or are not in identical order for forward and reverse specs..."
    assert np.array_equal(T_rev, 1 - T_fwd), f"Treatments not perfectly complementary for forward and reverse specs..."
    pairs.append((spec['key'], cate_fwd.ravel(), cate_rev.ravel()))
    print(f"Spec {spec['key']}: Mean forward CATE = {cate_fwd.mean()}, Mean reverse CATE = {cate_rev.mean()}", flush=True)