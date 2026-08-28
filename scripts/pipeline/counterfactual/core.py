import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from joblib import Parallel, delayed, parallel_config
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

# The two bootstrap schemes, named once here so the artifact keys, the figure
# filenames and the write-up all read from the same strings.
#   'estimation' -- resample the TRAINING rows only, average over the fixed test
#                   frame. Captures model-estimation uncertainty alone.
#   'total'      -- resample the training rows AND the test rows. Adds the
#                   sampling variability of the population being averaged over.
# 'estimation' is nested inside 'total'; they are not rival schemes.
SCHEME_ESTIMATION = "estimation"
SCHEME_TOTAL = "total"
BOOTSTRAP_SCHEMES = (SCHEME_ESTIMATION, SCHEME_TOTAL)


def contrast_output_dir(key: str, subdir: str = None) -> Path:
    """Return (and create) the per-contrast output directory for one pairwise contrast.

    Single source of truth for the counterfactual_pipeline on-disk layout, and the exact
    mirror of causal/core.py's helper of the same name so the two packages' outputs sit
    side by side under ARTIFACTS_DIR.

    Args:
        key (str): The contrast key (spec_dict['key']), e.g. 'bupropion_vs_snri'.
        subdir (str, optional): A per-family subfolder under the contrast dir. Defaults
            to None (the contrast root).

    Returns:
        Path: The created directory.
    """
    out = Path(os.environ['ARTIFACTS_DIR']) / 'counterfactual_pipeline' / key
    if subdir is not None:
        out = out / subdir
    os.makedirs(out, exist_ok=True)
    return out

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
    
def pooled_training_frame(population: EligiblePopulations) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Glue the two training arms back into one frame, reference arm first.

    The concatenation ORDER is load-bearing: the arm flag and the label vector are built
    to match it position for position, so anything that re-splits the pool (the propensity
    target, the bootstrap resampler) can do so by the flag alone and cannot silently
    disagree with a second, separately written concatenation. This is the construction
    that used to be inlined in attach_propensity.

    Args:
        population (EligiblePopulations): Population broken up into the two medication arms.

    Returns:
        tuple: (pooled training matrix, arm flag as 0 for reference / 1 for comparison,
            pooled TRD labels) -- all three aligned by position.
    """
    train_matrix = pd.concat([population.ref_arm_train_matrix, population.comp_arm_train_matrix])
    arm_flag = np.concatenate([
        np.zeros(len(population.ref_arm_train_labels), dtype=int),
        np.ones(len(population.comp_arm_train_labels), dtype=int)
    ])
    train_labels = np.concatenate([population.ref_arm_train_labels, population.comp_arm_train_labels])
    return train_matrix, arm_flag, train_labels


def attach_propensity(population: EligiblePopulations, risk_frame: pd.DataFrame) -> pd.DataFrame:
    """For each patient, determine their probability of being in the comparison arm, and whether that lands in a reasonable interval

    Args:
        population (EligiblePopulations): Population broken up into the two medication arms
        risk_frame (pd.DataFrame): Inputted risk scores for the given counterfactual treatment

    Returns:
        pd.DataFrame: Risk dataframe with propensity scores appended
    """
    train_matrix, arm_target, _ = pooled_training_frame(population)
    classifier_pipeline = make_classifier(LogisticRegression(max_iter=1000))
    classifier_pipeline.fit(train_matrix, arm_target)
    arm_probs = classifier_pipeline.predict_proba(population.eligible_test_matrix)[:, 1]
    risk_frame['propensity'] = arm_probs
    risk_frame['in_prob_interval'] = risk_frame['propensity'].between(PROB_FLOOR, PROB_CEILING, inclusive='neither')
    return risk_frame

def summarize_effect(risk_df: pd.DataFrame) -> dict:
    """Collapse one scored, propensity-annotated risk frame into the trim report and the two average effects.

    This is the ESTIMATOR, and nothing else: no draws, no intervals, no randomness. It is
    called once on the full-data frame for the point estimates and once per bootstrap draw
    on that draw's frame, which is what makes the interval an interval of this estimator
    rather than of a frozen vector.

    Args:
        risk_df (pd.DataFrame): Output of attach_propensity -- carries risk_ref, risk_comp,
            is_comparison, propensity and in_prob_interval.

    Returns:
        dict: The per-arm trim report, plus 'ate_trimmed' (hard-trimmed average over the
            in-band patients) and 'ate_overlap_weighted' (the same per-patient contrasts
            re-averaged under overlap weights). Either average is float('nan') when its
            denominator is degenerate -- no in-band patient, or zero total weight.
            (NOTE - effect is comparison risk score minus reference risk score, so positive
            means the first-named arm raises P(TRD))
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
    
    # Max weighting occurs at equal treatment probability.
    propensity_weights = propensity * (1 - propensity)
    # Both guards matter only inside the bootstrap, where a draw can land with nothing in
    # the band; on the full data they never fire. A degenerate draw becomes nan and is
    # dropped by the nanpercentile reduction rather than poisoning the interval.
    ate_trimmed = float(per_patient_effect[in_prob_interval].mean()) if in_prob_interval.any() else float("nan")
    ate_weighted = (
        float(np.average(per_patient_effect, weights=propensity_weights))
        if propensity_weights.sum() > 0 else float("nan")
    )

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
    return {
        **trim_report,
        # HEADLINE: hard-trimmed average. Estimand is nameable in a sentence --
        # "patients whose propensity fell inside the band" -- and the band matches
        # the causal package's, keeping the two triangulating estimators comparable.
        "ate_trimmed": ate_trimmed,
        # SENSITIVITY: same per-patient contrasts re-averaged under overlap weights
        # (Li, Morgan & Zaslavsky 2018). Smooth analogue, no cliff at the floor.
        # NOTE the name: the local is ate_weighted but the persisted key has always
        # been ate_overlap_weighted, and nothing downstream catches a wrong string.
        "ate_overlap_weighted": ate_weighted,
    }


def in_band_effects(risk_df: pd.DataFrame) -> np.ndarray:
    """Per-patient treatment-effect contrasts for the in-band patients only.

    The same subset plot_effect_distribution draws and the same subset ate_trimmed
    averages, extracted once so the point-estimate histogram and the pooled bootstrap
    histogram cannot end up describing different populations.

    Args:
        risk_df (pd.DataFrame): Output of attach_propensity.

    Returns:
        np.ndarray: 1-D array of comparison-minus-reference contrasts, in-band rows only.
    """
    per_patient_effect = (risk_df['risk_comp'] - risk_df['risk_ref']).to_numpy()
    return per_patient_effect[risk_df['in_prob_interval'].to_numpy()]


def resample_populations(population: EligiblePopulations, generator: np.random.Generator, resample_test: bool) -> EligiblePopulations:
    """Draw one bootstrap replicate of the eligible populations.

    Training rows are resampled POOLED ACROSS ARMS -- the two arms are glued back together
    by pooled_training_frame, one with-replacement draw is taken over the whole pool, and
    the replicate is re-split by the arm flag it carried through the draw. Arm sizes
    therefore wobble from draw to draw, which is the intended behaviour: the split between
    arms is itself a feature of the sample, not a design constant.

    What is NOT resampled, ever, is the train/test split. create_train_test_split /
    test_patient_ids.txt is shared with the classical-ML and neighbour pipelines, so
    "pooled" means pooled across arms, within train and within test separately.

    Rows are selected with .iloc, positionally. Label-based .loc would expand a repeated
    patient ID combinatorially and silently change the replicate's size.

    Args:
        population (EligiblePopulations): The full eligible population to resample from.
        generator (np.random.Generator): Source of the draw. One generator per draw, seeded
            by a child of the SEED sequence, so the result does not depend on the order in
            which parallel workers happen to finish.
        resample_test (bool): False for the 'estimation' scheme -- training rows resample,
            the test frame is passed through untouched. True for the 'total' scheme -- the
            test matrix, its labels and its comparison flag are all sliced by ONE shared
            index set, so a test patient's covariates, outcome and arm stay together.

    Returns:
        EligiblePopulations: The replicate, same dataclass shape as the input.
    """
    train_matrix, arm_flag, train_labels = pooled_training_frame(population)
    n_train = len(train_matrix)
    draw = generator.integers(low=0, high=n_train, size=n_train)
    drawn_matrix = train_matrix.iloc[draw]
    drawn_labels = train_labels[draw]
    drawn_flag = arm_flag[draw]
    is_reference = drawn_flag == 0
    is_comparison = ~is_reference

    if resample_test:
        n_test = len(population.eligible_test_matrix)
        test_draw = generator.integers(low=0, high=n_test, size=n_test)
        test_matrix = population.eligible_test_matrix.iloc[test_draw]
        test_labels = population.eligible_test_labels[test_draw]
        test_flag = population.test_comparison_flag[test_draw]
    else:
        test_matrix = population.eligible_test_matrix
        test_labels = population.eligible_test_labels
        test_flag = population.test_comparison_flag

    return EligiblePopulations(
        ref_arm_train_matrix=drawn_matrix[is_reference],
        ref_arm_train_labels=drawn_labels[is_reference],
        comp_arm_train_matrix=drawn_matrix[is_comparison],
        comp_arm_train_labels=drawn_labels[is_comparison],
        eligible_test_matrix=test_matrix,
        eligible_test_labels=test_labels,
        test_comparison_flag=test_flag
    )


def estimate_once(population: EligiblePopulations) -> pd.DataFrame:
    """Run the whole estimation pipeline over one population: fit, score, propensity, trim flag.

    Three calls, in the order the design requires. Because the propensity model is refitted
    here rather than reused, in_prob_interval is recomputed too and the trimmed population
    moves from draw to draw. That is deliberate: the trimming rule is part of the estimator,
    so its variability belongs inside the band, not outside it.

    Args:
        population (EligiblePopulations): Full data or one bootstrap replicate.

    Returns:
        pd.DataFrame: The scored, propensity-annotated risk frame, ready for summarize_effect.
    """
    risk_frame = score_counterfactual_risks(population)
    return attach_propensity(population, risk_frame)


def _bootstrap_draw(population: EligiblePopulations, seed_sequence: np.random.SeedSequence, resample_test: bool) -> dict:
    """One bootstrap replicate, end to end. Returns None if the replicate was degenerate.

    A pooled draw can hand an arm zero patients, or every patient the same TRD label, and
    sklearn raises ValueError rather than fitting. bupropion_vs_snri is the contrast at risk
    -- it is the smallest, at ~2,210 eligible. A failed replicate is dropped and counted,
    never silently absorbed into the interval.

    Args:
        population (EligiblePopulations): The population to resample from.
        seed_sequence (np.random.SeedSequence): This draw's own seed, spawned from SEED.
        resample_test (bool): Whether the test rows resample too. See resample_populations.

    Returns:
        dict: 'ate_trimmed', 'ate_overlap_weighted', the two per-arm trimmed shares, and
            'effects' (this replicate's in-band per-patient contrasts). None on failure.
    """
    try:
        replicate = resample_populations(population, np.random.default_rng(seed_sequence), resample_test)
        risk_frame = estimate_once(replicate)
    except ValueError:
        return None
    summary = summarize_effect(risk_frame)
    return {
        "ate_trimmed": summary['ate_trimmed'],
        "ate_overlap_weighted": summary['ate_overlap_weighted'],
        "reference_trimmed_share": summary['reference_trimmed_share'],
        "comparison_trimmed_share": summary['comparison_trimmed_share'],
        "effects": in_band_effects(risk_frame),
    }


def _bootstrap_chunk(population: EligiblePopulations, seed_sequences: list, resample_test: bool) -> list:
    """Run a contiguous block of draws inside one worker process.

    Draws are chunked rather than dispatched one at a time for a boring but decisive reason:
    a draw takes well under a second, and shipping the whole population to a worker costs
    more than that. Dispatched per draw, the parallel version measured SLOWER than the serial
    one. One chunk per worker-ish amortises the transfer over many draws.

    Chunking cannot change any result -- each draw still uses its own seed from the spawned
    sequence, so the chunk boundaries are invisible in the output.

    Args:
        population (EligiblePopulations): The population to resample from.
        seed_sequences (list): This chunk's np.random.SeedSequence objects.
        resample_test (bool): Whether the test rows resample too.

    Returns:
        list: One _bootstrap_draw result per seed, with None for degenerate replicates.
    """
    return [_bootstrap_draw(population, seed, resample_test) for seed in seed_sequences]


def bootstrap_effect(population: EligiblePopulations, resample_test: bool, scheme: str, n_jobs: int = None) -> tuple[dict, np.ndarray]:
    """Bootstrap the WHOLE pipeline -- fit, score, propensity, trim, average -- N_BOOTSTRAP times.

    This is the change the whole refactor existed for. The previous interval resampled a
    frozen per-patient effect vector, so it measured the sampling variability of a mean and
    omitted model-estimation uncertainty entirely -- the dominant term, since both risk
    columns are themselves predictions from fitted logistic regressions. Here every draw
    refits all three models, so the band contains that term.

    Both averages come out of the SAME replicate, which is what keeps "do the hard-trimmed
    and overlap-weighted answers agree" a meaningful question rather than a comparison of
    two independent noise draws.

    The draws are independent, so they are farmed out with joblib. Each gets its own seed
    spawned from SEED, so the result is identical however many workers run and in whatever
    order they return.

    Args:
        population (EligiblePopulations): The full eligible population.
        resample_test (bool): False for the estimation scheme, True for the total scheme.
        scheme (str): SCHEME_ESTIMATION or SCHEME_TOTAL. Suffixes every returned key.
        n_jobs (int, optional): Worker count. Defaults to SLURM_CPUS_PER_TASK if set, else 1.

    Returns:
        tuple: (dict of percentile-interval keys suffixed with the scheme name, plus
            'n_bootstrap_failures_<scheme>'; and the 1-D array of every in-band per-patient
            contrast from every surviving draw, pooled -- which is what the bootstrap
            effect-distribution figure is drawn from).
    """
    if n_jobs is None:
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    seeds = np.random.SeedSequence(int(os.environ['SEED'])).spawn(N_BOOTSTRAP)
    chunks = np.array_split(np.arange(N_BOOTSTRAP), min(N_BOOTSTRAP, max(n_jobs, 1) * 4))
    # inner_max_num_threads=1 pins BLAS inside each worker. Without it joblib sizes the inner
    # thread pool from the worker count, the reduction order inside lbfgs changes with it, and
    # the intervals move in the last few decimal places when n_jobs changes. Pinned, the run
    # is bit-identical however many cores it is given.
    with parallel_config(backend='loky', inner_max_num_threads=1):
        chunk_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_bootstrap_chunk)(population, [seeds[i] for i in chunk], resample_test)
            for chunk in chunks
        )
    draws = [draw for chunk in chunk_results for draw in chunk]

    survived = [draw for draw in draws if draw is not None]
    n_failures = len(draws) - len(survived)

    def interval(field: str) -> tuple[float, float]:
        values = np.array([draw[field] for draw in survived], dtype=float)
        if values.size == 0 or np.isnan(values).all():
            return float("nan"), float("nan")
        low, high = np.nanpercentile(values, [2.5, 97.5])
        return float(low), float(high)

    trimmed_low, trimmed_high = interval('ate_trimmed')
    weighted_low, weighted_high = interval('ate_overlap_weighted')
    ref_share_low, ref_share_high = interval('reference_trimmed_share')
    comp_share_low, comp_share_high = interval('comparison_trimmed_share')

    pooled_effects = (
        np.concatenate([draw['effects'] for draw in survived])
        if survived else np.array([], dtype=float)
    )

    return {
        f"ate_trimmed_ci_low_{scheme}": trimmed_low,
        f"ate_trimmed_ci_high_{scheme}": trimmed_high,
        f"ate_overlap_weighted_ci_low_{scheme}": weighted_low,
        f"ate_overlap_weighted_ci_high_{scheme}": weighted_high,
        # The trim shares move per draw too, so they get their own bands. The point
        # estimate stays the full-data value in the trim report; these say how firm it is.
        # The lopsided ~2x reference-arm trimming in the SSRI-referenced contrasts is one
        # of the more informative numbers in this output and is entitled to error bars.
        f"reference_trimmed_share_ci_low_{scheme}": ref_share_low,
        f"reference_trimmed_share_ci_high_{scheme}": ref_share_high,
        f"comparison_trimmed_share_ci_low_{scheme}": comp_share_low,
        f"comparison_trimmed_share_ci_high_{scheme}": comp_share_high,
        f"n_bootstrap_failures_{scheme}": int(n_failures),
    }, pooled_effects


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
    effects = in_band_effects(risk_df)
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


def plot_bootstrap_effect_distribution(spec_dict: dict, pooled_effects: np.ndarray, ate: float, scheme: str, save_dir: Path) -> None:
    """Render the per-patient treatment-effect distribution pooled over ALL patients and ALL bootstrap draws.

    The companion to plot_effect_distribution and deliberately drawn the same way, but from a
    different object. That figure shows one number per patient, computed once from models fitted
    on all the data: it is the spread of the estimate. This one pools every in-band patient from
    every surviving replicate, so each patient contributes many contrasts -- one per draw that
    kept them -- and the extra width over the point-estimate histogram is exactly the
    model-estimation uncertainty the old frozen-vector bootstrap could not see.

    Read the two side by side: same centre, wider tails here. If the tails were NOT wider, the
    refit inside the bootstrap did not take.

    The dashed red line is the full-data point estimate, not the pooled mean, so the figure is
    annotated with the number that gets reported. The shaded band is the central 95% of the
    pooled contrasts -- a spread over patients-and-draws, NOT the confidence interval of the
    average, which is narrower by roughly the square root of the sample size and lives in
    effect_results.json.

    Args:
        spec_dict (dict): The pairwise contrast spec (its 'key', 'display_name').
        pooled_effects (np.ndarray): Second element of bootstrap_effect's return.
        ate (float): The full-data point estimate (ate_trimmed) for the reference line.
        scheme (str): SCHEME_ESTIMATION or SCHEME_TOTAL; names the file and titles the plot.
        save_dir (Path): Directory to write the figure into.
    """
    if pooled_effects.size == 0:
        # Every replicate failed. Nothing to draw, and a blank axes would be worse than no file.
        return

    low, high = np.percentile(pooled_effects, [2.5, 97.5])
    fig, ax = plt.subplots()
    ax.hist(pooled_effects, bins=100, range=tuple(np.percentile(pooled_effects, [1, 99])))
    ax.axvspan(low, high, color='orange', alpha=0.15, label="Central 95% of pooled contrasts")
    ax.axvline(x=0, color='green', linestyle='--', label="No effect")
    ax.axvline(x=ate, color='red', linestyle='--', label=f"Point estimate ({ate:.4f})")
    y_top = ax.get_ylim()[1]
    ax.text(
        ate, y_top * 0.55, f" ATE = {ate:.4f}",
        color='red', ha='left', va='center', fontweight='bold', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', alpha=0.85),
    )
    ax.set_xlabel("Effect on P(TRD): comparison arm minus reference arm")
    ax.set_ylabel("Patient-draws")
    ax.set_title(f"{spec_dict['display_name']} -- pooled over {N_BOOTSTRAP} bootstrap draws ({scheme})")
    ax.legend(loc='upper right')
    fig.savefig(save_dir / f"bootstrap_effect_histogram_{scheme}.png")
    plt.close(fig)