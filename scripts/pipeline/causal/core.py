import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML

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

def fit_causal_forest(spec_dict: dict, train_matrix: pd.DataFrame, test_matrix: pd.DataFrame, y_train: np.ndarray, seed: int=None) -> tuple:
    """Return fitted causal forest given the specified treatment and the training data, and return it plus all the information needed for calibration analysis

    Args:
        spec_dict (dict): specified treatment with the columns it corresponds to
        train_matrix (pd.DataFrame): dataframe of train patients
        test_matrix (pd.DataFrame): dataframe of test patients
        y_train (np.ndarray): boolean flag of outcome for train patients
        seed (int, optional): random seed. Defaults to None and will be autoset in that case.

    Returns:
        tuple: Resulting fitted forest, cate_test, X_fit_train (with the treatment removed), X_fit_test, T_train (binary treatment flag of patients), T_test
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
    