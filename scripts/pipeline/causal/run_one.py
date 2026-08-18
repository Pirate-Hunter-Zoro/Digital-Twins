import os
import numpy as np
import json

from scripts.shared.treatment_registry import TREATMENT_REGISTRY
from scripts.pipeline.causal.core import (
    load_encoded_data,
    build_treatment,
    fit_and_evaluate,
    contrast_output_dir,
)

from dotenv import load_dotenv
load_dotenv()

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
spec_dict = TREATMENT_REGISTRY[task_id]

# No matter which treatment we test, the training and testing split stays the same
train_matrix, test_matrix, train_labels, test_labels = load_encoded_data()

keep_mask_train, compar_flag_train = build_treatment(spec_dict, train_matrix)
compar_flag_train = compar_flag_train[keep_mask_train]
keep_mask_test, compar_flag_test = build_treatment(spec_dict, test_matrix)
compar_flag_test = compar_flag_test[keep_mask_test]

# See the number of patients that apply to the comparison treatment
combined_compar_flag_mask = np.concatenate([compar_flag_train, compar_flag_test])
total = combined_compar_flag_mask.size
compar_count = combined_compar_flag_mask.sum()
arm_count = total - compar_count
minority_arm_n = min(arm_count, compar_count)
record = {
    "total": int(total),
    "compar_count": int(compar_count),
    "arm_count": int(arm_count),
    "minority_arm_n": int(minority_arm_n)
}

result = fit_and_evaluate(spec_dict, train_matrix, test_matrix, train_labels, test_labels)
if result is None:
    metrics = {
        'key': spec_dict['key'],
        'display_name': spec_dict['display_name'],
        'passed_overlap': False,
        **record
    }
else:
    # Otherwise we passed the overlap requirement - balance in both treatments
    metrics = {
        'key': spec_dict['key'],
        'display_name': spec_dict['display_name'],
        'passed_overlap': True,
        **record,
        **result
    }
with open(contrast_output_dir(spec_dict['key']) / "results.json", 'w') as f:
    json.dump(metrics, f, indent=4)