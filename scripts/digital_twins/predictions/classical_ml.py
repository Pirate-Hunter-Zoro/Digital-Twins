import numpy as np
from typing import Tuple
from pathlib import Path
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor

def load_data_set(patient_ids: set[str]) -> Tuple[np.array, np.array]:
    """Load all the patient vectors and find their labels

    Returns:
        Tuple[np.array, np.array]: vectors and labels
    """
    predictor = TRDPredictor()
    all_vector_paths = Path(os.environ['DETERMINISTIC_VECTORS_DIR']).glob("*.npy")
    patient_vector_paths = [p for p in all_vector_paths if p.stem in patient_ids]
    # Sort for the purposes of keeping X and y consistent with each other
    patient_vector_paths.sort(key=lambda x: x.stem)
    X = np.array([np.load(f, allow_pickle=True) for f in patient_vector_paths])
    y = np.array([predictor.get_trd_status(id) for id in sorted(list(patient_ids))])
    print(f"Shape of X: {X.shape}; Shape of y: {y.shape}", flush=True)
    return (X, y)

def make_classifier(model):
    return Pipeline(steps=[\
                    # Replace all 'nan' values with the median value for all values present
                    ("fill", SimpleImputer(strategy="median")),\
                    # Zero mean, unit variance normalization
                    ("scale", StandardScaler()),\
                    ("model", model)\
                ])
    
def evaluate_models(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    """Obtain classification results from varioius ML models on the input data

    Args:
        X_train (np.array): Train observations
        y_train (np.array): Train labels
        X_test (np.array): Test observations
        y_test (np.array): Test labels
    """
    classifiers = {
        "Logistic Regression": make_classifier(LogisticRegression(max_iter=1000, random_state=int(os.environ['SEED']))),
        "Naive Bayes": make_classifier(GaussianNB()),
        "SVM": make_classifier(SVC(probability=True, random_state=int(os.environ['SEED'])))
    }
    # Fit each classifier on the training data
    classifier_predictions = {}
    for name, classifier in classifiers.items():
        classifier.fit(X=X_train, y=y_train)
        predictions = classifier.predict_proba(X=X_test)[:, 1]
        classifier_predictions[name] = predictions