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

from dotenv import load_dotenv
load_dotenv()

OVERLAP_FLOOR = 0.05
OVERLAP_CEILING = 1 - OVERLAP_FLOOR

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

def build_treatment(spec_dict: dict, feature_matrix: pd.DataFrame) -> np.ndarray:
    """Determine whether a treatment occurred or did not occur over an entire feature matrix - e.g. for each patient, determine if the given treatment occurred or not

    Args:
        spec_dict (dict): Given treatment specs - specifies the columns relevant to this particular treatment
        feature_matrix (pd.DataFrame): All relevant patients who need treatment flags

    Returns:
        np.ndarray: 0/1 array for whether each patient received the specified treatment
    """
    relevant_cols = spec_dict['source_cols']
    treatment_info = feature_matrix[relevant_cols]
    treatment_flags = treatment_info.sum(axis=1) > 0 # One binary flag per patient
    return treatment_flags.astype(int).to_numpy()

def fit_causal_forest(spec_dict: dict, train_matrix: pd.DataFrame, test_matrix: pd.DataFrame, y_train: np.ndarray, seed: int=None) -> tuple[CausalForestDML, Any, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return fitted causal forest given the specified treatment and the training data, and return it plus all the information needed for calibration analysis

    Args:
        spec_dict (dict): specified treatment with the columns it corresponds to
        train_matrix (pd.DataFrame): dataframe of train patients
        test_matrix (pd.DataFrame): dataframe of test patients
        y_train (np.ndarray): boolean flag of outcome for train patients
        seed (int, optional): random seed. Defaults to None and will be autoset in that case.

    Returns:
        tuple[CausalForestDML, Any, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: Resulting fitted forest, cate_test, X_fit_train (with the treatment removed), X_fit_test, T_train (binary treatment flag of patients), T_test
    """
    if seed is None:
        seed = int(os.environ['SEED'])
    model_y = RandomForestRegressor(random_state=seed, n_jobs=-1)
    model_t = RandomForestClassifier(random_state=seed, n_jobs=-1)
    n_estimators = 1000
    treatments_train = build_treatment(spec_dict, train_matrix)
    treatments_test = build_treatment(spec_dict, test_matrix)
    causal_forest = CausalForestDML(n_estimators=n_estimators, model_y=model_y, model_t=model_t, random_state=seed, n_jobs=-1, discrete_treatment=True)
    # Fit with the train output, and whether this treatment was used on each patient in the training set
    X_fit_train, X_fit_test = train_matrix.drop(columns=spec_dict['source_cols']), test_matrix.drop(columns=spec_dict['source_cols'])
    causal_forest.fit(y_train, treatments_train, X=X_fit_train, W=X_fit_train)
    cate_test = causal_forest.effect(X_fit_test)
    return causal_forest, cate_test, X_fit_train, X_fit_test, treatments_train, treatments_test

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
    tester.fit_nuisance(
        Xval=X_fit_test.to_numpy(), # Patients with the treatment attribute removed and all others kept
        Dval=treatments_test, # Binary treatment flags
        yval=y_test, # TRD flags
        Xtrain=X_fit_train.to_numpy(),
        Dtrain=treatments_train,
        ytrain=y_train
    )
    return tester

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
    cal_result.plot_cal(tmt=1).figure.savefig(save_dir / f"{spec_dict['key']}_calibration.png")
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
    param, err, pval = float(blp_res.params[0]), float(blp_res.errs[0]), float(blp_res.pvals[0])
    # Obtain dr estimates pertaining to test patients
    dr_vals = tester.dr_val_[:, 0]
    assert cate_test.shape == dr_vals.shape, f"Expected shapes of CATE values and DR values to match, but received {cate_test.shape} and {dr_vals.shape} respectively..."
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
    fig.savefig(save_dir / f"BLP_scatter_{spec_dict['key']}.png")
    plt.close(fig)
    return {
        'blp_est': param,
        'blp_se': err,
        'blp_pval': pval
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
    qini_ax.figure.savefig(save_dir / f"qini_{spec_dict['key']}.png", bbox_inches='tight')
    plt.close(qini_ax.figure)
    
    toc_result = tester.evaluate_uplift(
        metric='toc',
        Xval=X_fit_test,
        Xtrain=X_fit_train
    )
    toc_ax = toc_result.plot_uplift(tmt=1)
    toc_ax.figure.savefig(save_dir / f"toc_{spec_dict['key']}.png", bbox_inches='tight')
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
    treatment = spec_dict['key']
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
        cate_per_quartile.to_csv(save_dir / f"subgroup_ATE_{treatment}_{feature}.csv")
        
        # Create quartile-binned histogram of values
        fig, ax = plt.subplots()
        ax.bar(cate_per_quartile.index.astype(str), cate_per_quartile.values)
        ax.axhline(y=0, color='green', linestyle='--', label="No effect")
        ax.axhline(y=cate_test.mean(), color='red', linestyle='--', label="Average effect")
        ax.set_xlabel(f"{feature} quartile")
        ax.set_ylabel("Mean CATE on P(TRD)")
        ax.set_title(spec_dict['display_name'])
        ax.legend()
        fig.savefig(save_dir / f"subgroup_ATE_{treatment}_{feature}.png")
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
        fig.savefig(save_dir / f"CATE_vs_{feature}_{treatment}.png", bbox_inches='tight')
        plt.close(fig)
        
    return correlations