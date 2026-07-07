import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
from econml.validate import DRTester
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt

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