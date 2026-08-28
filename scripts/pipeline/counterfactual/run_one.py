"""One pairwise contrast of the counterfactual (T-learner) treatment-selection pipeline.

Mirrors scripts/pipeline/causal/run_one.py: the SLURM array index picks the contrast out of
the shared TREATMENT_REGISTRY, core.py stays a library with no main, and everything for the
contrast is written under ARTIFACTS_DIR/counterfactual_pipeline/<key>/.

Point estimates come from a single fit on the full data. The intervals come from
bootstrap_effect, which refits all three models inside every draw, and it is run TWICE --
once per scheme -- because the two are nested rather than rival:

  estimation : training rows resample, the test frame is held fixed.
  total      : training rows AND test rows resample.

Artifacts written:
  effect_results.json                        point estimates, trim report, 8 CI keys, failure counts
  model_grades.json                          gradeable (non-counterfactual) metrics per arm
  per_patient_risks.csv                      the full-data risk frame
  effect_histogram.png                       per-patient effects, one number per patient
  bootstrap_effect_histogram_<scheme>.png    per-patient effects pooled over every draw
  bootstrap_draws.csv                        the per-draw averages the intervals are cut from
  ate_sampling_distribution_<estimand>_<scheme>.png
                                             sampling distribution of the AVERAGE effect,
                                             with the reported 95% CI shaded
"""

import os
import json

import pandas as pd

from scripts.shared.treatment_registry import TREATMENT_REGISTRY
from scripts.pipeline.counterfactual.core import (
    SCHEME_ESTIMATION,
    SCHEME_TOTAL,
    build_eligible_populations,
    estimate_once,
    summarize_effect,
    grade_arm_models,
    bootstrap_effect,
    plot_effect_distribution,
    plot_bootstrap_effect_distribution,
    plot_ate_sampling_distribution,
    contrast_output_dir,
)

from dotenv import load_dotenv
load_dotenv()

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
spec_dict = TREATMENT_REGISTRY[task_id]
save_dir = contrast_output_dir(spec_dict['key'])

population = build_eligible_populations(spec_dict)

# The full-data pass. Everything reported as a point estimate comes off this frame, so the
# numbers do not move because the bootstrap was added.
risk_df = estimate_once(population)
point_estimates = summarize_effect(risk_df)
grades = grade_arm_models(risk_df)

# Scheme A: model-estimation uncertainty only. Scheme B: that plus the sampling variability
# of the population averaged over. B's band should contain A's.
estimation_cis, estimation_effects, estimation_draws = bootstrap_effect(
    population, resample_test=False, scheme=SCHEME_ESTIMATION
)
total_cis, total_effects, total_draws = bootstrap_effect(
    population, resample_test=True, scheme=SCHEME_TOTAL
)

results = {
    'key': spec_dict['key'],
    'display_name': spec_dict['display_name'],
    **point_estimates,
    **estimation_cis,
    **total_cis,
}

with open(save_dir / "effect_results.json", 'w') as f:
    json.dump(results, f, indent=4)
with open(save_dir / "model_grades.json", 'w') as f:
    json.dump(grades, f, indent=4, default=float)

risk_df.to_csv(save_dir / "per_patient_risks.csv", index_label="patient_id")

plot_effect_distribution(spec_dict, risk_df, save_dir)
for scheme, pooled in ((SCHEME_ESTIMATION, estimation_effects), (SCHEME_TOTAL, total_effects)):
    plot_bootstrap_effect_distribution(
        spec_dict, pooled, point_estimates['ate_trimmed'], scheme, save_dir
    )

# The per-draw averages: the object the confidence interval is percentiles of. Persisted as
# well as plotted, so any later figure or re-cut of the interval costs a read rather than a
# 2 x N_BOOTSTRAP refit. One column per (field, scheme); columns may differ in length when a
# scheme loses draws to degenerate replicates, which is why this is built from a dict of
# Series rather than a 2-D array.
draw_columns = {}
for scheme, draws in ((SCHEME_ESTIMATION, estimation_draws), (SCHEME_TOTAL, total_draws)):
    for field, values in draws.items():
        draw_columns[f"{field}_{scheme}"] = pd.Series(values)
pd.DataFrame(draw_columns).to_csv(save_dir / "bootstrap_draws.csv", index_label="draw")

# Four sampling-distribution figures per contrast: the headline hard-trimmed estimand and
# the overlap-weighted sensitivity one, each under both bootstrap schemes.
for scheme, cis, draws in (
    (SCHEME_ESTIMATION, estimation_cis, estimation_draws),
    (SCHEME_TOTAL, total_cis, total_draws),
):
    for estimand in ('ate_trimmed', 'ate_overlap_weighted'):
        plot_ate_sampling_distribution(
            spec_dict,
            draws[estimand],
            point_estimates[estimand],
            cis[f"{estimand}_ci_low_{scheme}"],
            cis[f"{estimand}_ci_high_{scheme}"],
            scheme,
            estimand,
            save_dir,
        )
