import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
from econml.validate import DRTester
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.pipeline.predictions.create_train_test_split import create_train_test_split
from scripts.shared.utils import load_trd_set, load_feature_matrix
from scripts.data_loading.med_definitions import get_med_arm

from dotenv import load_dotenv
load_dotenv()

OVERLAP_FLOOR = 0.1
OVERLAP_CEILING = 1 - OVERLAP_FLOOR

def get_AD_mappings() -> dict[str, str]:
    """For every patient, return which antidepressant arm their anchor date prescription belongs to

    Returns:
        dict[str, str]: Patient ID, AD prescription arm
    """
    med_dates = pd.read_csv(Path(os.environ['MDD_MED_DATE_CSV_PATH'])).set_index('PatientEpicId_SH')
    med_dates = med_dates.sort_values(by='MedStartInstant', ascending=True)
    earliest_mask = ~med_dates.index.duplicated(keep='first') # Indexed by patient ID
    med_dates = med_dates[earliest_mask]
    return med_dates['MedName'].apply(get_med_arm).to_dict()

def passes_overlap(train_treatment_array: np.ndarray) -> bool:
    """Given all of the training population's treatment flags, determine if enough patients were both treated and untreated to warrant CATE analysis

    Args:
        train_treatment_array (np.ndarray): Treatment flags over entire training population

    Returns:
        bool: If proportion of treated patients falls within the valid proportion interval
    """
    p = train_treatment_array.mean()
    return bool(p <= OVERLAP_CEILING and p >= OVERLAP_FLOOR)

def load_encoded_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the encoded train and test patient data frames and their resulting labels

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: train matrix, test matrix, train labels, test labels
    """
    train_ids, test_ids = create_train_test_split()
    trd_ids = load_trd_set()
    X_reduced_train, X_reduced_test = load_feature_matrix(train_ids), load_feature_matrix(test_ids)
    cat_cols = ["Sex", "PreferredLanguage", "MaritalStatus", "Religion", "SmokingStatus", "Race_Ethnicity", "mdd_recurrence", "mdd_severity"]
    X_reduced_full = pd.concat([X_reduced_train, X_reduced_test])
    # Convert all categorical variables to one-hot encodings
    X_encoded_full = pd.get_dummies(X_reduced_full, columns=cat_cols)
    X_encoded_train, X_encoded_test = X_encoded_full.loc[X_reduced_train.index], X_encoded_full.loc[X_reduced_test.index]
    train_y, test_y = np.array([int(id in trd_ids) for id in X_reduced_train.index]),\
        np.array([int(id in trd_ids) for id in X_reduced_test.index])
    return (X_encoded_train, X_encoded_test, train_y, test_y)

def build_treatment(spec_dict: dict, feature_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Assign the active-comparator treatment for one pairwise contrast over a feature matrix.

    Each patient's treatment is the antidepressant CLASS started at their anchor (index)
    prescription. For this contrast, patients whose index class is the comparison arm are
    labelled 1, those whose index class is the reference arm are labelled 0, and every other
    patient (any other arm, or a med get_med_arm does not recognise) is EXCLUDED via the mask.

    Args:
        spec_dict (dict): The contrast spec; supplies the two arms to compare via its
            'reference_arm' (labelled 0) and 'comparison_arm' (labelled 1) keys.
        feature_matrix (pd.DataFrame): Patients to assign, indexed by patient_id (the index is
            what maps to each patient's anchor-medication class).

    Returns:
        tuple[np.ndarray, np.ndarray]: (keep_mask, compar_flag), both aligned to feature_matrix's
            rows. keep_mask is a boolean array, True for patients whose index class is the
            reference or comparison arm. compar_flag is a 0/1 array, 1 for the comparison arm and
            0 for the reference arm; it is only meaningful where keep_mask is True.
    """
    ref_arm, compar_arm = spec_dict['reference_arm'], spec_dict['comparison_arm']
    patient_anchor_med_arm_map = get_AD_mappings()
    arms = feature_matrix.index.map(patient_anchor_med_arm_map)
    arms = pd.Series(arms)
    # We only want the patients whose antidepressant anchor date medication arm is either the reference arm or comparison arm
    keep_mask = arms.isin([ref_arm, compar_arm]).to_numpy()
    compar_flag = (arms == compar_arm).astype(int).to_numpy()
    return (keep_mask, compar_flag)

def fit_causal_forest(spec_dict: dict, train_matrix: pd.DataFrame, test_matrix: pd.DataFrame, y_train: np.ndarray, seed: int=None) -> tuple[CausalForestDML, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit one active-comparator pairwise causal forest and return everything the eval surface needs.

    The treatment is which antidepressant CLASS was started at the anchor prescription. For this
    contrast the cohort is filtered to only the two arms named in spec_dict via build_treatment's
    keep-mask; the reference arm becomes T=0 and the comparison arm T=1, and every other patient is
    excluded. All feature columns are retained (the old per-candidate source_cols drop is gone in the
    pairwise design). Overlap is checked WITHIN the kept pair on the training labels: if it fails the
    OVERLAP_FLOOR/CEILING band the contrast is skipped and None is returned before the fit.

    Args:
        spec_dict (dict): The pairwise contrast spec; supplies the two arms via its 'reference_arm'
            (T=0) and 'comparison_arm' (T=1) keys.
        train_matrix (pd.DataFrame): Full encoded train patients, indexed by patient_id.
        test_matrix (pd.DataFrame): Full encoded test patients, indexed by patient_id.
        y_train (np.ndarray): Binary TRD outcome for the FULL train population, row-aligned to
            train_matrix (subset to the kept pair internally before fitting).
        seed (int, optional): Random seed. Defaults to None, in which case the SEED env var is read.

    Returns:
        tuple[CausalForestDML, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray] | None:
            None if the kept pair fails the overlap band. Otherwise (fitted_forest, cate_test,
            X_fit_train, X_fit_test, T_train, T_test), all filtered to the two-arm pair: X_fit_*
            keep every feature column; T_train/T_test are the 0/1 comparison-arm flags for the kept
            train/test patients; cate_test is the per-patient CATE over the kept test rows.
    """
    if seed is None:
        seed = int(os.environ['SEED'])
    model_y = RandomForestRegressor(random_state=seed, n_jobs=-1)
    model_t = RandomForestClassifier(random_state=seed, n_jobs=-1)
    n_estimators = 1000
    
    # Train data
    train_keep_mask, train_comparison_flag = build_treatment(spec_dict, train_matrix)
    X_fit_train = train_matrix[train_keep_mask]
    T_train = train_comparison_flag[train_keep_mask]
    
    # Test data
    test_keep_mask, test_comparison_flag = build_treatment(spec_dict, test_matrix)
    X_fit_test = test_matrix[test_keep_mask]
    T_test = test_comparison_flag[test_keep_mask]
    
    if not passes_overlap(T_train):
        return None
    
    causal_forest = CausalForestDML(n_estimators=n_estimators, model_y=model_y, model_t=model_t, random_state=seed, n_jobs=-1, discrete_treatment=True)
    # Fit with the train output, and whether this treatment was used on each patient in the training set
    y_train = y_train[train_keep_mask]
    causal_forest.fit(y_train, T_train, X=X_fit_train, W=X_fit_train)
    cate_test = causal_forest.effect(X_fit_test)
    return causal_forest, cate_test, X_fit_train, X_fit_test, T_train, T_test

def fit_dr_tester(fitted_forest: CausalForestDML, X_fit_train: pd.DataFrame, X_fit_test: pd.DataFrame, treatments_train: np.ndarray, treatments_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, seed: int=None) -> DRTester:
    """Generate and fit a doubly-robust tester to evaluate the CATE values associated with the given fitted causal random forest

    Args:
        fitted_forest (CausalForestDML): Causal random forest fitted on training data
        X_fit_train (pd.DataFrame): Matrix of training patients forest was fitted on
        X_fit_test (pd.DataFrame): Matrix of testing patients forest estimated cate values for
        treatments_train (np.ndarray): Binary treatment flags for the training population
        treatments_test (np.ndarray): Binary treatment flags for the testing population
        y_train (np.ndarray): Binary outcome flags for training population
        y_test (np.ndarray): Binary outcome flags for testing population
        seed (int, optional): Random seed for reproducibility. Defaults to .env variable.

    Returns:
        DRTester: Resulting fitted tester
    """
    if seed is None:
        seed = int(os.environ['SEED'])
    tester = DRTester(
        model_regression=RandomForestRegressor(random_state=seed), # One regressor to estimate mu1/mu0, probabilities of being TRD given receiving treatment and not receiving treatment, over all patients
        model_propensity=RandomForestClassifier(random_state=seed), # Estimates e(x), probability of being treated
        cate=fitted_forest, # Already-fitted forest
        cv=5, # Cross validation sweep
    )
    # Estimate m_1(x), m_2(x), e(x)
    # TODO - Clusters seem to show up - are they treatment groups? Try to investigate
    tester.fit_nuisance(
        Xval=X_fit_test.to_numpy(), # Patients with the treatment attribute removed and all others kept
        Dval=treatments_test, # Binary treatment flags
        yval=y_test, # TRD flags
        Xtrain=X_fit_train.to_numpy(),
        Dtrain=treatments_train,
        ytrain=y_train
    )
    return tester

def contrast_output_dir(key: str, subdir: str=None) -> Path:
    """Return (and create) the per-contrast output directory for one pairwise contrast.

    Single source of truth for the causal_pipeline on-disk layout: everything for a
    contrast lives under ARTIFACTS_DIR/causal_pipeline/<key>/, with optional per-family
    subfolders (e.g. 'subgroup_ate', 'cate_vs_feature'). Global artifacts
    (leaderboard.csv, validation_report.json) stay at the causal_pipeline root and do
    not use this helper.

    Args:
        key (str): The contrast key (spec_dict['key']), e.g. 'bupropion_vs_snri'.
        subdir (str, optional): A per-family subfolder under the contrast dir. Defaults
            to None (the contrast root).

    Returns:
        Path: The created directory.
    """
    out = Path(os.environ['ARTIFACTS_DIR']) / 'causal_pipeline' / key
    if subdir is not None:
        out = out / subdir
    os.makedirs(out, exist_ok=True)
    return out

def evaluate_calibration(spec_dict: dict,
                         tester: DRTester,
                         X_fit_train: pd.DataFrame,
                         X_fit_test: pd.DataFrame,
                         save_dir: Path,
                         seed: int=None
                         ) -> float:
    """Compute doubly robust treatment effect scores and compare them to CATE estimates from causal random forest

    Args:
        spec_dict (dict): Specified treatment along with its column names
        tester (DRTester): Fitted doubly robust tester which can plot correlation between its DR values and CATE values
        X_fit_train (pd.DataFrame): Train patient matrix (treatment columns dropped)
        X_fit_test (pd.DataFrame): Test patient matrix (treatment columns dropped)
        save_dir (Path): Location to save plots
        seed (int): Random seed for reproducibility - defaults to .env variable

    Returns:
        float: Resulting R^2 calibration
    """
    if seed is None:
        seed = int(os.environ['SEED'])
    
    # Evaluate calibration of fitted tester
    cal_result = tester.evaluate_cal(
        Xval=X_fit_test,
        Xtrain=X_fit_train,
        n_groups=10,
    )
    cal_result.plot_cal(tmt=1).figure.savefig(save_dir / "calibration.png")
    return float(cal_result.cal_r_squared[0])

def evaluate_blp(spec_dict: dict, tester: DRTester, cate_test: np.ndarray, X_fit_test: pd.DataFrame, save_dir: Path) -> dict:
    """Evaluate best linear fit of the input DRTester's doubly-robust estimates versus the CATE values associated with the causal random forest in question

    Args:
        spec_dict (dict): Specified treatment with its columns
        tester (DRTester): Pre-fitted doubly robust tester with its DR values
        cate_test (np.ndarray): Forest's CATE values
        X_fit_test (pd.DataFrame): Testing patient matrix
        save_dir (Path): Path to save BLP scatterplot
        
    Returns:
        dict: Resulting slope, error, and pvalue associated with correlation
    """
    blp_res = tester.evaluate_blp(
        Xval=X_fit_test,
    )
    param, err, dr_pval = float(blp_res.params[0]), float(blp_res.errs[0]), float(blp_res.pvals[0])
    # Obtain dr estimates pertaining to test patients
    dr_vals = tester.dr_val_[:, 0]
    assert cate_test.shape == dr_vals.shape, f"Expected shapes of CATE values and DR values to match, but received {cate_test.shape} and {dr_vals.shape} respectively..."
    dr_rho, dr_rho_pval = spearmanr(cate_test, dr_vals)
    fitted_line = np.polyfit(cate_test, dr_vals, deg=1) # Returns slope and intercept
    assert np.isclose(fitted_line[0], param), f"Inconsistent slope values from BLP: {param} and numpy polyfit: {fitted_line[0]}..."
    fig, ax = plt.subplots()
    x_range = np.array([np.min(cate_test), np.max(cate_test)])
    ax.scatter(cate_test, dr_vals, alpha=0.5, s=0.15)
    ax.plot(x_range, np.polyval(fitted_line, x_range), color='red', label=f"slope={fitted_line[0]:.3f}")
    ax.axhline(y=0, linestyle='--', color='green')
    ax.set_ylim(np.percentile(dr_vals, [1, 99]))
    ax.set_xlabel("Predicted CATE")
    ax.set_ylabel("DR pseudo-outcome")
    ax.set_title(spec_dict["display_name"])
    ax.legend()
    fig.savefig(save_dir / "BLP_scatter.png")
    plt.close(fig)
    return {
        'blp_est': param,
        'blp_se': err,
        'blp_pval': dr_pval,
        'dr_spearman_rho': float(dr_rho),
        'dr_spearman_pval': float(dr_rho_pval)
    }
    
def evaluate_uplift(spec_dict: dict, tester: DRTester, X_fit_train: pd.DataFrame, X_fit_test: pd.DataFrame, save_dir: Path) -> dict:
    """Uplift evaluation creating a cumulative sum analysis of doubly robust estimates over patients sorted by decreasing CATE values
    
    Conceptually: 
    TOC - sort patients by CATE value in descending order, x-axis is the top-k of said patients, y-axis is that respective top-k's group's average doubly-robust value MINUS the overall average doubly-robust value
    e.g. - TOC(q) = (avg DR in top-q) − (avg DR over everyone, or ATE)
    BY DEFINITION this means TOC(1)=0
    
    Qini - Same x-axis
Qini(q) = q * (avg DR in top-q − ATE)

    Args:
        spec_dict (dict): Specified treatment with its columns
        tester (DRTester): Pre-fitted doubly robust tester
        X_fit_train (pd.DataFrame): Training patient matrix
        X_fit_test (pd.DataFrame): Testing patient matrix
        save_dir (Path): Save directory for resulting .png outputs

    Returns:
        dict: Qini and AUTOC/TOC results
    """
    qini_result = tester.evaluate_uplift(
        metric='qini',
        Xval=X_fit_test,
        Xtrain=X_fit_train
    )
    qini_ax = qini_result.plot_uplift(tmt=1)
    qini_ax.figure.savefig(save_dir / "qini.png", bbox_inches='tight')
    plt.close(qini_ax.figure)
    
    toc_result = tester.evaluate_uplift(
        metric='toc',
        Xval=X_fit_test,
        Xtrain=X_fit_train
    )
    toc_ax = toc_result.plot_uplift(tmt=1)
    toc_ax.figure.savefig(save_dir / "toc.png", bbox_inches='tight')
    plt.close(toc_ax.figure)
    
    return {
        "qini_est": float(qini_result.params[0]),
        "qini_se": float(qini_result.errs[0]),
        "qini_pval": float(qini_result.pvals[0]),
        "autoc_est": float(toc_result.params[0]),
        "autoc_se": float(toc_result.errs[0]),
        "autoc_pval": float(toc_result.pvals[0])
    }
    
def evaluate_shap_moderators(fitted_forest: CausalForestDML, X_fit_test: pd.DataFrame, top_k: int=5) -> list[tuple[str, float]]:
    """For a given treatment, determine the SHAP value for each feature with respect to the fitted forest's CATE values

    Args:
        fitted_forest (CausalForestDML): Causal random forest estimator
        X_fit_test (pd.DataFrame): Test patient matrix
        top_k (int, optional): How many top SHAP-values are to be returned. Defaults to 5.

    Returns:
        list[tuple[str, float]]: Most important featuers paired with their SHAP values
    """
    shap_vals = fitted_forest.shap_values(X=X_fit_test.astype('float64'), feature_names=list(X_fit_test.columns))
    for outer_dict in shap_vals.values():
        # Dictionary with only one entry whose value is what what we care about
        for shap_values in outer_dict.values():
            # That value is once again a dictionary with only one entry whos value we care about
            feature_importances = np.abs(shap_values.values).mean(axis=0).tolist() # One non-negative number per feature
            paired_features_with_importance = [(name, imp) for name, imp in zip(list(X_fit_test.columns), feature_importances)] # Order was preserved since we passed in the same feature name to the 'shap_values' call
            paired_features_with_importance.sort(key=lambda x: -x[1]) # sort by importance - decreasing order
            return paired_features_with_importance[:top_k]
        
def evaluate_subgroup_ate(spec_dict: dict, cate_test: np.ndarray, X_fit_test: pd.DataFrame, save_dir: Path, features: list[str]=['pre_anchor_history_days', 'AgeInYears', 'in_patient_days', 'num_encounters', 'num_emergency']) -> dict[str, dict[str, float]]:
    """For the given treatment, find the correlations between a given list of features and the CATE values

    Args:
        spec_dict (dict): Specified treatment
        cate_test (np.ndarray): Resulting CATE values
        X_fit_test (pd.DataFrame): Test patient matrix
        save_dir (Path): Location to save plots
        features (list[str], optional): Features whose correlation should be judged. Defaults to ['pre_anchor_history_days', 'AgeInYears', 'in_patient_days', 'num_encounters', 'num_emergency'].

    Returns:
        dict[str, dict[str, float]]: For each feature, resulting 'spearman_rho' and 'spearman_pval' correlation results
    """
    quartiles = 4
    subgroup_dir = save_dir / 'subgroup_ate'
    cate_vs_dir = save_dir / 'cate_vs_feature'
    os.makedirs(subgroup_dir, exist_ok=True)
    os.makedirs(cate_vs_dir, exist_ok=True)
    correlations = {}
    for feature in features:
        correlations[feature] = {}
        X = X_fit_test[feature]
        quartile_bins = pd.qcut(X, q=quartiles, duplicates='drop')
        y = cate_test
        cate_per_quartile = pd.Series(y, index=X_fit_test.index).groupby(quartile_bins).mean()
        print(f"{spec_dict['display_name']} CATE per quartile:")
        print(cate_per_quartile)
        print("\n")
        cate_per_quartile.name = 'mean_cate'
        cate_per_quartile.to_csv(subgroup_dir / f"subgroup_ATE_{feature}.csv")
        
        # Create quartile-binned histogram of values
        fig, ax = plt.subplots()
        ax.bar(cate_per_quartile.index.astype(str), cate_per_quartile.values)
        ax.axhline(y=0, color='green', linestyle='--', label="No effect")
        ax.axhline(y=cate_test.mean(), color='red', linestyle='--', label="Average effect")
        ax.set_xlabel(f"{feature} quartile")
        ax.set_ylabel("Mean CATE on P(TRD)")
        ax.set_title(spec_dict['display_name'])
        ax.legend()
        fig.savefig(subgroup_dir / f"subgroup_ATE_{feature}.png")
        plt.close(fig)
        
        # Now create raw correlation plot
        correlation, p_val = spearmanr(X, y)
        correlations[feature]['spearman_rho'] = float(correlation)
        correlations[feature]['spearman_pval'] = float(p_val)
        fitted_line = np.polyfit(X, y, deg=1)
        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.5, s=0.15)
        
        x_range = np.array([np.min(X), np.max(X)])
        ax.plot(x_range, np.polyval(fitted_line, x_range), color='red', label=f"slope={fitted_line[0]:.3f}")
        ax.axhline(y=0, linestyle='--', color='green')
        ax.set_ylim(np.percentile(y, [1, 99]))
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("CATE on P(TRD)")
        p_display = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.set_title(
            f"{spec_dict['display_name']}\n"
            fr"Spearman $\rho$ = {correlation:.3f}   ({p_display})",
            fontsize=11,
            fontweight='bold',
            pad=10,
        )
        ax.legend()
        fig.savefig(cate_vs_dir / f"CATE_vs_{feature}.png", bbox_inches='tight')
        plt.close(fig)
        
    return correlations

def plot_cate_distribution(spec_dict: dict, cate_test: np.ndarray, save_dir: Path) -> None:
    """Render the marginal distribution of the forest's per-patient CATE values for one contrast.

    A companion to the BLP / subgroup plots: shows whether the forest found real spread in the
    treatment effect or just a tight spike at the ATE. Purely a side-effect plot, no returned metric.
    The x-axis is clipped to the 1st/99th percentile so a handful of extreme CATE values cannot
    stretch the axis into a useless spike (same outlier guard evaluate_blp/evaluate_subgroup_ate use).

    Args:
        spec_dict (dict): The pairwise contrast spec (its 'key', 'display_name').
        cate_test (np.ndarray): Per-patient CATE values over the kept test rows.
        save_dir (Path): Directory to write the figure into.
    """
    mean_cate = float(cate_test.mean())
    fig, ax = plt.subplots()
    ax.hist(cate_test, bins=50, range=tuple(np.percentile(cate_test, [1, 99])))
    ax.axvline(x=0, color='green', linestyle='--', label="No effect")
    ax.axvline(x=mean_cate, color='red', linestyle='--', label=f"Average effect ({mean_cate:.4f})")
    # Print the mean value directly on the plot at the red line, so the ATE is readable
    # off the figure itself and not only from the legend (and unambiguous when the mean
    # sits close to the zero line). Placed at mid-height to clear the upper-right legend.
    y_top = ax.get_ylim()[1]
    ax.text(
        mean_cate, y_top * 0.55, f" ATE = {mean_cate:.4f}",
        color='red', ha='left', va='center', fontweight='bold', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', alpha=0.85),
    )
    ax.set_xlabel("CATE on P(TRD)")
    ax.set_ylabel("Number of patients")
    ax.set_title(spec_dict['display_name'])
    ax.legend(loc='upper right')
    fig.savefig(save_dir / "CATE_histogram.png")
    plt.close(fig)

def fit_and_evaluate(spec_dict: dict, train_matrix: pd.DataFrame, test_matrix: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray, seed: int=None) -> dict:
    """Fit one pairwise contrast's causal forest and run the full evaluation surface over it.

    Orchestrator for a single active-comparator contrast: fits the forest (which self-filters to
    the two-arm pair and enforces overlap), builds ONE shared doubly-robust tester, and runs all
    five evals against it. The full-population y_train/y_test are re-selected down to the kept pair
    by indexing on the already-filtered X, so they align row-for-row with T_train/T_test before the
    tester sees them. Figures for every eval are written under ARTIFACTS_DIR/causal_pipeline.

    Args:
        spec_dict (dict): The pairwise contrast spec (its 'key', 'display_name', 'reference_arm',
            'comparison_arm').
        train_matrix (pd.DataFrame): Full encoded train patients, indexed by patient_id.
        test_matrix (pd.DataFrame): Full encoded test patients, indexed by patient_id.
        y_train (np.ndarray): Binary TRD outcome for the FULL train population, row-aligned to
            train_matrix.
        y_test (np.ndarray): Binary TRD outcome for the FULL test population, row-aligned to
            test_matrix.
        seed (int, optional): Random seed. Defaults to None, in which case the SEED env var is read.

    Returns:
        dict: None if the contrast failed overlap (fit_causal_forest returned None). Otherwise a
            metrics dict: identity ('key', 'display_name'), 'passed_overlap' True, the calibration
            'cal_r_squared', and the nested BLP / uplift / SHAP-moderator / subgroup-ATE results.
    """
    if seed is None:
        seed = int(os.environ['SEED'])
    result = fit_causal_forest(spec_dict, train_matrix, test_matrix, y_train, seed)
    if result is None: # Overlap failed - nothing to evaluate
        return None 
    forest, cate_test, X_fit_train, X_fit_test, T_train, T_test = result
    # Filter the training and testing y to be only the patients used in the causal random forest who were in one of the two treatment groups specified by the spec_dict
    y_train = pd.Series(y_train, train_matrix.index).loc[X_fit_train.index].to_numpy()
    y_test = pd.Series(y_test, test_matrix.index).loc[X_fit_test.index].to_numpy()
    # Now they align with T_train, T_test
    
    save_dir = contrast_output_dir(spec_dict['key'])
    tester = fit_dr_tester(forest, X_fit_train, X_fit_test, T_train, T_test, y_train, y_test, seed)
    cal_r_squared = evaluate_calibration(spec_dict, tester, X_fit_train, X_fit_test, save_dir, seed)
    blp_res = evaluate_blp(spec_dict, tester, cate_test, X_fit_test, save_dir)
    uplift_res = evaluate_uplift(spec_dict, tester, X_fit_train, X_fit_test, save_dir)
    top_shap_moderators = evaluate_shap_moderators(forest, X_fit_test, top_k=5)
    gate_res = evaluate_subgroup_ate(spec_dict, cate_test, X_fit_test, save_dir)
    plot_cate_distribution(spec_dict, cate_test, save_dir)
    
    average_cate = float(forest.ate(X_fit_test))
    lower, upper = forest.ate_interval(X_fit_test, alpha=0.05) # 95% confidence interval
    lower, upper = float(lower), float(upper)
    ate_res = {
        "ate": average_cate,
        "ate_ci_low": lower,
        "ate_ci_high": upper,
    }
    
    return {
        'key': spec_dict['key'],
        'display_name': spec_dict['display_name'],
        'passed_overlap': True,
        'cal_r_squared': cal_r_squared,
        'blp_res': blp_res,
        'uplift_res': uplift_res,
        'top_shap_moderators': top_shap_moderators,
        'gate_res':  gate_res, # Group Average Treatment effect correlation
        'ate_res': ate_res, # Confidence interval for average treatment effect
    }