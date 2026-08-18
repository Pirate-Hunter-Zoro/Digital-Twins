import os
import json
from pathlib import Path
import numpy as np
from scipy.stats import linregress

from scripts.pipeline.causal.core import (
    load_encoded_data,
    fit_causal_forest,
    contrast_output_dir,
    plot_cate_overlay,
)
from scripts.shared.treatment_registry import TREATMENT_REGISTRY

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
    pairs.append((spec['key'], spec['display_name'], reverse_spec['display_name'],
                  cate_fwd.ravel(), cate_rev.ravel()))
    print(f"Spec {spec['key']}: Mean forward CATE = {cate_fwd.mean()}, Mean reverse CATE = {cate_rev.mean()}", flush=True)

# Overlay 1: all three forward contrasts on one axis, each mean tagged in the legend
results_root = Path(os.environ['ARTIFACTS_DIR']) / 'causal_pipeline'
os.makedirs(results_root, exist_ok=True)
plot_cate_overlay(
    [(display_name, cate_fwd) for (_, display_name, _, cate_fwd, _) in pairs],
    title="CATE distributions across contrasts",
    save_path=results_root / "CATE_histograms_all_contrasts.png",
)
# Overlay 2: per contrast, the forward CATE against its sign-flipped reverse mirror
for (key, fwd_name, rev_name, cate_fwd, cate_rev) in pairs:
    plot_cate_overlay(
        [(fwd_name, cate_fwd), (rev_name, cate_rev)],
        title=f"{fwd_name}: forward vs reverse mirror",
        save_path=contrast_output_dir(key) / "CATE_histogram_mirror.png",
    )

# We should hope that for all pairs, we have strong negative correlation between the CATE and reverse CATE
results_path = results_root
results = {}
for (key, _fwd_name, _rev_name, cate_fwd, cate_rev) in pairs:
    corr_result = linregress(cate_fwd, cate_rev)
    pearson_r = float(corr_result.rvalue)
    m = float(corr_result.slope)
    b = float(corr_result.intercept)
    mean_sum = float(np.mean(cate_fwd + cate_rev))
    max_sum = float(np.max(np.abs(cate_fwd + cate_rev)))
    results[key] = {
        'pearson_r': pearson_r, 
        'm': m, 
        'b': b, 
        'mean_sum': mean_sum, 
        'max_sum': max_sum
    }
    
# Transitivity check
needed_values = {
    "snri_vs_ssri": 0,
    "bupropion_vs_ssri": 0,
    "bupropion_vs_snri": 0,
}
first = 'bupropion_vs_ssri'
second = 'snri_vs_ssri'
to_predict = "bupropion_vs_snri"
# bupropion_vs_snri ~ bupropion_vs_ssri - snri_vs_ssri, both of which reduce to bup - snri treatment effect
for spec in TREATMENT_REGISTRY:
    key = spec['key']
    with open(results_path / key / "results.json", 'r') as f:
        spec_res = json.load(f)
        assert spec_res['passed_overlap'], f"Error: spec {key} did not pass overlap conditions"
        ate_est = spec_res['ate_res']['ate']
        needed_values[key] = ate_est
predicted = needed_values[first] - needed_values[second]
actual = needed_values[to_predict]
gap = actual - predicted
results['transitivity'] = {
    **needed_values,
    f'predicted_cate ({to_predict})': predicted,
    f'actual_cate ({to_predict})': actual,
    'abs_gap': abs(gap)
}
    
with open(results_path / 'validation_report.json', 'w') as f:
    json.dump(results, f, indent=4)