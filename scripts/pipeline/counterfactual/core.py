import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from scripts.shared.utils import (
    load_trd_set,
    load_feature_matrix,
    get_AD_mappings
)
from scripts.pipeline.predictions.create_train_test_split import create_train_test_split
from scripts.pipeline.predictions.classical_ml import make_classifier
from scripts.pipeline.predictions.trd_prediction_computation import compute_metrics
from scripts.shared.plots import N_BOOTSTRAP

PROB_FLOOR = 0.1
PROB_CEILING = 0.9

@dataclass
class EligiblePopulations:
    ref_arm_train_matrix: pd.DataFrame
    ref_arm_train_labels: np.ndarray
    comp_arm_train_matrix: pd.DataFrame
    comp_arm_train_labels: np.ndarray
    # When scoring/testing, we don't need to break patients up by which medication arm they belong to - that will only come into play when testing the models on the patients with the same respective medication arm
    eligible_test_matrix: pd.DataFrame
    eligible_test_labels: np.ndarray
    # In the test patients, flag for each one part of the comparison arm
    test_comparison_flag: np.ndarray
    
def build_eligible_populations(spec_dict: dict) -> EligiblePopulations:
    """Break up the eligible patient population into training and testing populations and keep track of their features and TRD labels

    Args:
        spec_dict (dict): Specifies the reference and comparison arms

    Returns:
        EligiblePopulations: dataclass object containing all relevant information on the population
    """
    train_ids, test_ids = create_train_test_split()
    train_matrix, test_matrix = load_feature_matrix(train_ids), load_feature_matrix(test_ids)
    # Maps ALL patients to their respective medication arm
    mappings = get_AD_mappings()
    train_arms = train_matrix.index.map(mappings)
    train_arms = pd.Series(train_arms)
    test_arms = test_matrix.index.map(mappings)
    test_arms = pd.Series(test_arms)
    
    # Grab the reference and comparison arm markers
    ref_arm, compar_arm = spec_dict['reference_arm'], spec_dict['comparison_arm']
    
    # See which patients are in the reference/comparator arms
    train_keep_mask = train_arms.isin([ref_arm, compar_arm]).to_numpy()
    compar_flag_train = (train_arms == compar_arm).astype(int).to_numpy()
    test_keep_mask = test_arms.isin([ref_arm, compar_arm]).to_numpy()
    compar_flag_test = (test_arms == compar_arm).astype(int).to_numpy()
    
    # Load TRD flags
    trd_patients = load_trd_set()
    
    # Apply filtering
    kept_train_matrix = train_matrix[train_keep_mask]
    kept_compar_flag_train = compar_flag_train[train_keep_mask]
    kept_test_matrix = test_matrix[test_keep_mask]
    kept_compar_flag_test = compar_flag_test[test_keep_mask]
    
    kept_train_y, kept_test_y = np.array([int(id in trd_patients) for id in kept_train_matrix.index]),\
        np.array([int(id in trd_patients) for id in kept_test_matrix.index])
        
    return EligiblePopulations(
        ref_arm_train_matrix=kept_train_matrix[kept_compar_flag_train == 0],
        ref_arm_train_labels=kept_train_y[kept_compar_flag_train == 0],
        comp_arm_train_matrix=kept_train_matrix[kept_compar_flag_train == 1],
        comp_arm_train_labels=kept_train_y[kept_compar_flag_train == 1],
        eligible_test_matrix=kept_test_matrix,
        eligible_test_labels=kept_test_y,
        test_comparison_flag=kept_compar_flag_test
    )
    
def score_counterfactual_risks(population: EligiblePopulations) -> pd.DataFrame:
    """Returns predicted probabilities given each treatment group over the entire population

    Args:
        population (EligiblePopulations): Broken up into treatment groups, and train/test groups

    Returns:
        pd.DataFrame: Resulting risk scores in both scenarios of each medication being taken
    """
    ref_pipeline = make_classifier(LogisticRegression(max_iter=1000))
    ref_pipeline.fit(population.ref_arm_train_matrix, population.ref_arm_train_labels)
    comp_pipeline = make_classifier(LogisticRegression(max_iter=1000))
    comp_pipeline.fit(population.comp_arm_train_matrix, population.comp_arm_train_labels)
    
    # Second column of predictions for positive probability
    ref_world_probs = ref_pipeline.predict_proba(population.eligible_test_matrix)[:, 1]
    comp_world_probs = comp_pipeline.predict_proba(population.eligible_test_matrix)[:, 1]
    
    return pd.DataFrame(
        {
            'risk_ref': ref_world_probs,
            'risk_comp': comp_world_probs,
            'trd_label': population.eligible_test_labels,
            'is_comparison': population.test_comparison_flag
        }
    ).set_index(population.eligible_test_matrix.index)
    
def grade_arm_models(risk_scores: pd.DataFrame) -> dict:
    """For the risk scores which are gradable (not counterfactuals), grade them

    Args:
        risk_scores (pd.DataFrame): Counterfactual and non-counterfactual risk scores

    Returns:
        dict: Performance metrics for non-counterfactual risk scores
    """
    is_comp_mask = risk_scores['is_comparison'] == 1
    is_ref_mask = ~is_comp_mask
    non_counterfactual_comp_scores = risk_scores['risk_comp'][is_comp_mask]
    non_counterfactual_ref_scores = risk_scores['risk_ref'][is_ref_mask]
    comp_flags = risk_scores['trd_label'][is_comp_mask]
    ref_flags = risk_scores['trd_label'][is_ref_mask]
    
    # Now that we have broken up the non-counterfactual risk scores with their flags, grade them
    comp_scores = compute_metrics(comp_flags, non_counterfactual_comp_scores)
    comp_scores['n_gradable'] = int(is_comp_mask.sum())
    comp_scores['n_events'] = int(comp_flags.sum())
    ref_scores = compute_metrics(ref_flags, non_counterfactual_ref_scores)
    ref_scores['n_gradable'] = int(is_ref_mask.sum())
    ref_scores['n_events'] = int(ref_flags.sum())
    
    # Create weighted calibration slopes and intercepts
    bin_edges = np.linspace(0.0, 1.0, 11)
    binned_predictions_comp = np.digitize(non_counterfactual_comp_scores, bin_edges) - 1
    binned_predictions_ref = np.digitize(non_counterfactual_ref_scores, bin_edges) - 1
    # Any bin assignment greater than the maximum allowed bin gets subtracted by 1 - which would only matter if a prediction were exactly 1.0 - unlikely but we'll be pedantic
    overflow_mask_comp = binned_predictions_comp >= bin_edges.shape[0]-1
    binned_predictions_comp[overflow_mask_comp] = binned_predictions_comp[overflow_mask_comp] - 1
    overflow_mask_ref = binned_predictions_ref >= bin_edges.shape[0]-1
    binned_predictions_ref[overflow_mask_ref] = binned_predictions_ref[overflow_mask_ref] - 1
    
    # In order to weight calibration by bin sizes, we need to compute the bins of risk scores and find each of their counts
    comp_bins = []
    ref_bins = []
    for b_idx in range(bin_edges.shape[0]-1):
        bin_low, bin_high = bin_edges[b_idx], bin_edges[b_idx+1]
        in_bin_comp = binned_predictions_comp == b_idx
        if in_bin_comp.any():
            mean_bin_prediction = non_counterfactual_comp_scores[in_bin_comp].mean()
            observed_fraction = comp_flags[in_bin_comp].mean()
            bin_count = int(in_bin_comp.sum())
            comp_bins.append({
                "bin_low": bin_low,
                "bin_high": bin_high,
                "n": bin_count,
                "mean_predicted": mean_bin_prediction,
                "observed_fraction": observed_fraction,
            })
            
        in_bin_ref = binned_predictions_ref == b_idx
        if in_bin_ref.any():
            mean_bin_prediction = non_counterfactual_ref_scores[in_bin_ref].mean()
            observed_fraction = ref_flags[in_bin_ref].mean()
            bin_count = int(in_bin_ref.sum())
            ref_bins.append({
                "bin_low": bin_low,
                "bin_high": bin_high,
                "n": bin_count,
                "mean_predicted": mean_bin_prediction,
                "observed_fraction": observed_fraction,
            })  
    ref_scores = ref_scores | count_weighted_slope(pd.DataFrame(ref_bins))
    ref_scores['bins'] = ref_bins
    comp_scores = comp_scores | count_weighted_slope(pd.DataFrame(comp_bins))
    comp_scores['bins'] = comp_bins
    
    return {
        'reference': ref_scores,
        'comparison': comp_scores
    }
    
def count_weighted_slope(bin_table: pd.DataFrame) -> dict:
    """Given the bins that normally go into the slope calculation for a calibration curve, return the slope and intercept when each bin is weighted by its size

    Args:
        bin_table (pd.DataFrame): Calibration bins

    Returns:
        dict: Resulting weighted calibration slope and intercept
    """
    model = LinearRegression()
    model.fit(bin_table['mean_predicted'].to_numpy().reshape(-1,1), bin_table['observed_fraction'], bin_table['n']) # Third argument is sample weight
    return {
        "weighted_cal_slope": float(model.coef_[0]),
        "weighted_cal_intercept": float(model.intercept_)
    }
    
def attach_propensity(population: EligiblePopulations, risk_frame: pd.DataFrame) -> pd.DataFrame:
    """For each patient, determine their probability of being in the comparison arm, and whether that lands in a reasonable interval

    Args:
        population (EligiblePopulations): Population broken up into the two medication arms
        risk_frame (pd.DataFrame): Inputted risk scores for the given counterfactual treatment

    Returns:
        pd.DataFrame: Risk dataframe with propensity scores appended
    """
    train_matrix = pd.concat([population.ref_arm_train_matrix, population.comp_arm_train_matrix])
    arm_target = np.concat([np.zeros(len(population.ref_arm_train_labels)), np.ones(len(population.comp_arm_train_labels))])
    classifier_pipeline = make_classifier(LogisticRegression(max_iter=1000))
    classifier_pipeline.fit(train_matrix, arm_target)
    arm_probs = classifier_pipeline.predict_proba(population.eligible_test_matrix)[:, 1]
    risk_frame['propensity'] = arm_probs
    risk_frame['in_prob_interval'] = risk_frame['propensity'].between(PROB_FLOOR, PROB_CEILING, inclusive='neither')
    return risk_frame

def estimate_effect(risk_df: pd.DataFrame) -> dict:
    """Calculate treatment effect estimate given the risk scores and propensity scores for each patient

    Args:
        risk_df (pd.DataFrame): Containing risk scores with propensity scores

    Returns:
        dict: Report of how many patients fell out of each arm due to unreasonable propensity score, average effect estimate over patients falling in specified propensity range AND weighted average effect estimate over all patients
        (NOTE - effect is comparison risk score minus reference risk score, so positive means the first-named arm raises P(TRD))
    """
    # Per-patient contrast. Comparison minus reference, so a positive value means
    # the first-named (comparison) arm raises P(TRD).
    per_patient_effect = (risk_df['risk_comp'] - risk_df['risk_ref']).to_numpy()
    # Probability of each patient being in the comparison arm.
    propensity = risk_df['propensity'].to_numpy()
    # Flag for whether that probability landed inside the band; its negation is the trimmed set.
    in_prob_interval = risk_df['in_prob_interval'].to_numpy()
    trimmed = ~in_prob_interval
    is_comparison = (risk_df['is_comparison'] == 1).to_numpy()
    is_reference = ~is_comparison

    # Trim report, broken out by arm: trimming heavily from one arm and barely from
    # the other localizes where overlap fails, which a pooled count hides.
    ref_arm_n = int(is_reference.sum())
    comp_arm_n = int(is_comparison.sum())
    ref_trimmed_count = int((is_reference & trimmed).sum())
    comp_trimmed_count = int((is_comparison & trimmed).sum())
    
    # Max weighting occurs at equal treatment probability
    propensity_weights = propensity * (1 - propensity)
    ate_trimmed = float(per_patient_effect[in_prob_interval].mean())
    ate_weighted = float(np.average(per_patient_effect, weights=propensity_weights))
    n_rows = len(risk_df)
    generator = np.random.default_rng(seed=int(os.environ['SEED']))

    trim_report = {
        "n_eligible": int(len(risk_df)),
        "reference_arm_n": ref_arm_n,
        "comparison_arm_n": comp_arm_n,
        "reference_trimmed_count": ref_trimmed_count,
        "comparison_trimmed_count": comp_trimmed_count,
        "reference_trimmed_share": float(ref_trimmed_count / ref_arm_n) if ref_arm_n else float("nan"),
        "comparison_trimmed_share": float(comp_trimmed_count / comp_arm_n) if comp_arm_n else float("nan"),
        # Observed extremes over the WHOLE column, in-band and out: how far the
        # propensity model actually reached, not where the band was drawn.
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
    }

    # One patient-resampling bootstrap serving BOTH estimates. Sharing the draw is
    # deliberate: the two then differ only by the averaging rule, not by sampling
    # noise, which is what makes "do hard and soft agree" a meaningful question.
    sample_indices = generator.integers(low=0, high=n_rows, size=(N_BOOTSTRAP, n_rows))
    boot_trimmed = np.full(shape=(N_BOOTSTRAP,), fill_value=np.nan)
    boot_weighted = np.full(shape=(N_BOOTSTRAP,), fill_value=np.nan)
    for i in range(N_BOOTSTRAP):
        draw = sample_indices[i]
        drawn_effect = per_patient_effect[draw]
        drawn_in_band = in_prob_interval[draw]
        drawn_weights = propensity_weights[draw]
        if drawn_in_band.any():
            # We did pull some samples which had probability within the interval
            boot_trimmed[i] = drawn_effect[drawn_in_band].mean()
        if drawn_weights.sum() > 0:
            # We did pull some samples with propensity scores greater than zero
            boot_weighted[i] = np.average(drawn_effect, weights=drawn_weights)

    trimmed_ci_low, trimmed_ci_high = np.nanpercentile(boot_trimmed, [2.5, 97.5])
    weighted_ci_low, weighted_ci_high = np.nanpercentile(boot_weighted, [2.5, 97.5])

    return {
        **trim_report,
        # HEADLINE: hard-trimmed average. Estimand is nameable in a sentence --
        # "patients whose propensity fell inside the band" -- and the band matches
        # the causal package's, keeping the two triangulating estimators comparable.
        "ate_trimmed": ate_trimmed,
        "ate_trimmed_ci_low": float(trimmed_ci_low),
        "ate_trimmed_ci_high": float(trimmed_ci_high),
        # SENSITIVITY: same per-patient contrasts re-averaged under overlap weights
        # (Li, Morgan & Zaslavsky 2018). Smooth analogue, no cliff at the floor.
        "ate_overlap_weighted": ate_weighted,
        "ate_overlap_weighted_ci_low": float(weighted_ci_low),
        "ate_overlap_weighted_ci_high": float(weighted_ci_high),
    }

def plot_effect_distribution(spec_dict: dict, risk_df: pd.DataFrame, save_dir: Path) -> None:
    """Render the marginal distribution of the per-patient treatment-effect contrasts for one contrast.

    The T-learner counterpart to causal/core.py's plot_cate_distribution, and deliberately drawn the
    same way so the two triangulating estimators can be read side by side: same 1st/99th percentile
    x-clip, same dashed zero line, same dashed mean line with the value called out in a box on the
    axes. Restricted to the patients INSIDE the overlap band, because that trimmed subset is the
    headline estimand -- drawing the trimmed patients too would show a spread no reported number
    describes. Purely a side-effect plot, no returned metric.

    Args:
        spec_dict (dict): The pairwise contrast spec (its 'key', 'display_name').
        risk_df (pd.DataFrame): Risk frame carrying risk_ref, risk_comp and in_prob_interval.
        save_dir (Path): Directory to write the figure into.
    """
    per_patient_effect = (risk_df['risk_comp'] - risk_df['risk_ref']).to_numpy()
    in_band = risk_df['in_prob_interval'].to_numpy()
    effects = per_patient_effect[in_band]
    mean_effect = float(effects.mean())

    fig, ax = plt.subplots()
    ax.hist(effects, bins=50, range=tuple(np.percentile(effects, [1, 99])))
    ax.axvline(x=0, color='green', linestyle='--', label="No effect")
    ax.axvline(x=mean_effect, color='red', linestyle='--', label=f"Average effect ({mean_effect:.4f})")
    # Print the mean directly on the plot at the red line, so the ATE is readable off the
    # figure itself and not only from the legend (and unambiguous when the mean sits close
    # to the zero line). Placed at mid-height to clear the upper-right legend.
    y_top = ax.get_ylim()[1]
    ax.text(
        mean_effect, y_top * 0.55, f" ATE = {mean_effect:.4f}",
        color='red', ha='left', va='center', fontweight='bold', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', alpha=0.85),
    )
    ax.set_xlabel("Effect on P(TRD): comparison arm minus reference arm")
    ax.set_ylabel("Number of patients")
    ax.set_title(spec_dict['display_name'])
    ax.legend(loc='upper right')
    fig.savefig(save_dir / "effect_histogram.png")
    plt.close(fig)