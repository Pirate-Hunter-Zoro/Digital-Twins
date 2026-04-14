import numpy as np
from typing import Tuple
from pathlib import Path
import os
import sqlite3
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.predictions.create_train_test_split import create_train_test_split
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration

)
from scripts.shared.utils import VectorSource

def load_data_set(patient_ids: set[str], source: VectorSource=VectorSource.DETERMINISTIC) -> Tuple[np.array, np.array]:
    """Load all the patient vectors and find their labels

    Returns:
        Tuple[np.array, np.array]: vectors and labels
    """
    predictor = TRDPredictor() # We don't care about ID exclusion - only the ability to flag patients as TRD positive or negative
    if source == VectorSource.DETERMINISTIC:
        all_vector_paths = Path(os.environ['DETERMINISTIC_VECTORS_DIR']).glob("*.npy")
        patient_vector_paths = [p for p in all_vector_paths if p.stem in patient_ids]
        # Sort for the purposes of keeping X and y consistent with each other
        patient_vector_paths.sort(key=lambda x: x.stem)
        X = np.array([np.load(f, allow_pickle=True) for f in patient_vector_paths])
    else:
        embeddings_db_path = Path(os.environ['EMBEDDINGS_DIR']) / 'embeddings.db'
        connection = sqlite3.connect(embeddings_db_path)
        cursor = connection.cursor()
        placeholders = ",".join(["?"] * len(patient_ids))
        cursor.execute(
f"SELECT embedding FROM embeddings WHERE patient_id IN ({placeholders}) ORDER BY patient_id",
            sorted(list(patient_ids)) # Ensures the order is preserved
        )
        X = []
        for row in cursor.fetchall():
            X.append(np.frombuffer(row[0], dtype=np.float32))
        X = np.array(X)
        connection.close()
    y = np.array([predictor.get_trd_status(id) for id in sorted(list(patient_ids))])
    print(f"Shape of X from source {source.name}: {X.shape}; Shape of y: {y.shape}", flush=True)
    return (X, y)

def make_classifier(model):
    return Pipeline(steps=[\
                    # Replace all 'nan' values with the median value for all values present
                    ("fill", SimpleImputer(strategy="median")),\
                    # Zero mean, unit variance normalization
                    ("scale", StandardScaler()),\
                    ("model", model)\
                ])
    
def evaluate_models(X_train: np.array, y_train: np.array, X_test: np.array) -> dict[str, np.array]:
    """Obtain classification results from varioius ML models on the input data

    Args:
        X_train (np.array): Train observations
        y_train (np.array): Train labels
        X_test (np.array): Test observations

    Returns:
        dict[str, np.array]: Probability scores for each model
    """
    """
    Logistic Regression: model__C, model__penalty, model__solver
    Naive Bayes: model__var_smoothing
    SVM: model__C, model__kernel, model__gamma
    Random Forest: model__n_estimators, model__max_depth, model__min_samples_split
    Gradient Boosting: model__n_estimators, model__learning_rate, model__max_depth
    XGBoost: model__n_estimators, model__learning_rate, model__max_depth, model__subsample
    """
    hyperparameters = {
        'logistic_regression': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet', None],
            'model__solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'],
        },
        'naive_bayes': {
            'model__var_smoothing': np.logspace(0, -9, num=100).tolist(),
        },
        'svm': {
            'model__C': [0.1, 1, 10, 100, 1000],
            'model__kernel': ['linear', 'poly', 'rbf'],
            'model__gamma': [1, 0.1, 0.01, 0.001],
        },
        'random_forest': {
            'model__n_estimators': np.arange(100, 500, 100).tolist(),
            'model__max_depth': np.arange(10, 100, 10).tolist(),
            'model__min_samples_split': np.arange(2, 20, 2).tolist(),
            'model__min_samples_leaf': np.arange(1, 10).tolist()
        },
        'gradient_boosting': {
            'model__n_estimators': np.arange(100, 500, 100).tolist(),
            'model__learning_rate': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__max_depth': np.arange(10, 100, 10).tolist(),
        },
        'xgboost': {
            'model__n_estimators': np.arange(100, 1000, 100).tolist(),
            'model__learning_rate': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__max_depth': np.arange(1, 10, 1).tolist(),
            'model__subsample': np.arange(0,1,0.1).tolist(),
        }
    }
    classifiers = {
        "logistic_regression": make_classifier(LogisticRegression(max_iter=1000, random_state=int(os.environ['SEED']))),
        "naive_bayes": make_classifier(GaussianNB()),
        "svm": make_classifier(SVC(probability=True, random_state=int(os.environ['SEED']))),
        "random_forest": make_classifier(RandomForestClassifier(random_state=int(os.environ['SEED']))),
        "gradient_boosting": make_classifier(GradientBoostingClassifier(random_state=int(os.environ['SEED']))),
        "xgboost": make_classifier(XGBClassifier(random_state=int(os.environ['SEED']), eval_metric='logloss'))
    }
    # Fit each classifier on the training data
    classifier_predictions = {}
    for name, classifier in classifiers.items():
        param_grid = hyperparameters[name]
        # GridSearchCV(classifier, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
        classifier.fit(X=X_train, y=y_train)
        predictions = classifier.predict_proba(X=X_test)[:, 1]
        classifier_predictions[name] = predictions
        
    return classifier_predictions

def main():
    # Get training and test split
    (train_ids, test_ids) = create_train_test_split()
    
    for source in VectorSource:
        (train_X, train_y) = load_data_set(train_ids, source=source)
        (test_X, test_y) = load_data_set(test_ids, source=source)
        model_predictions = evaluate_models(train_X, train_y, test_X)
        for model_name, predictions in model_predictions.items():
            plot_receiving_operator_characteristic(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
            plot_precision_recall(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
            plot_calibration(y_true=test_y, y_prob=predictions, mode=f"{model_name}_{source.name}")
        
if __name__=="__main__":
    main()