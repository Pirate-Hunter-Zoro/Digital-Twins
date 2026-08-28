"""Representation parity: does matching the two representations' inputs move the result?

Review item (2026-08-28), the largest one. The narrative and the feature matrix are not
fed the same fields:

  narrative only  three within-patient mean vital signs (dropped at load by
                  load_feature_matrix), a SexualOrientation field absent from
                  CATEGORICAL_FIELDS entirely, the anchor's calendar date, and individual
                  medication/NSAID/hypnotic ingredient names where the feature vector
                  carries only counts.
  feature only    pre_anchor_history_days, which the narrative never rendered.

The mismatch is about the size of the effect under test (~0.008 AUC on the head-to-head
contrast), so it bounds the parity claim. This module decides whether it overturns it,
WITHOUT paying for a full pipeline pass: two classifiers rather than four, one encoder
rather than four, cosine-and-uniform KNN rather than the LLM-judged retrieval grid, and no
ablation re-score.

Which way it cuts is not obvious in advance. The surplus fields sit on the NARRATIVE side,
so removing the imbalance should make the head-to-head null safer while putting the
+0.028 embedded-versus-feature gain for logistic regression at risk.

Three arms, and the third exists only because of a defect found while building this:

  feature_vitals   the feature matrix WITH the vitals and an explicit missingness
                   indicator per vital block, median-imputed inside the fold.
  narrative_control the narrative re-rendered with NO content change. render_narrative
                   used to emit its flag lists in set-iteration order, which varies per
                   process, so the published narratives are not byte-reproducible. The
                   order is sorted now, which makes a re-render deterministic but does not
                   make it match what was embedded. This arm measures what that re-render
                   alone costs, so the parity arm's delta can be attributed to the field
                   rather than to the reshuffle.
  narrative_parity the narrative re-rendered with pre_anchor_history_days added.

Every comparison against the published numbers is a PAIRED bootstrap on the same held-out
patients, reusing paired_auc_delta from the discrimination forest, because the two score
vectors rank the same people and the paired interval is roughly a third the width of the
marginal ones.
"""

import json
import os
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.predictions.classical_ml import HYPERPARAMETERS, make_classifier
from scripts.pipeline.predictions.create_train_test_split import create_train_test_split
from scripts.pipeline.predictions.plot_discrimination_forest import paired_auc_delta
from scripts.pipeline.predictions.trd_prediction_computation import compute_metrics
from scripts.pipeline.review.paths import review_output_dir
from scripts.shared.plots import N_BOOTSTRAP
from scripts.shared.utils import VectorSource, load_feature_matrix, load_trd_set

ANALYSIS_NAME = "parity"

# The two classifiers that carry the published headline: logistic regression is the best
# embedded model and XGBoost the best feature-vector one, so the head-to-head contrast and
# the classifier-by-representation interaction both live in this pair. Random forest and
# gradient boosting would add two rows and roughly triple the fit time.
PARITY_MODELS = ("logistic_regression", "xgboost")

# Narrative arms, keyed by the value NARRATIVE_INCLUDE_HISTORY_LENGTH takes for each.
NARRATIVE_ARMS = {"narrative_control": 0, "narrative_parity": 1}


def parity_dir() -> Path:
    """Root for every parity artifact.

    Returns:
        Path: ARTIFACTS_DIR/review/parity/.
    """
    return review_output_dir(ANALYSIS_NAME)


def narratives_dir(arm: str) -> Path:
    """Where one narrative arm's rendered text lives.

    Args:
        arm (str): 'narrative_control' or 'narrative_parity'.

    Returns:
        Path: The arm's narrative directory.
    """
    return parity_dir() / arm / "narratives"


def embeddings_dir(arm: str) -> Path:
    """Where one narrative arm's embedding database lives.

    Args:
        arm (str): 'narrative_control' or 'narrative_parity'.

    Returns:
        Path: The arm's embeddings directory, holding embeddings.db.
    """
    return parity_dir() / arm / os.environ['EMBEDDER_MODEL_NAME']


def build_classifier(name: str, impute_numeric: bool) -> GridSearchCV:
    """Construct the grid search for one classifier, with the published grid.

    Args:
        name (str): 'logistic_regression' or 'xgboost'.
        impute_numeric (bool): Whether the numeric branch needs a median imputer, which
            it does exactly when the matrix carries the vital signs.

    Returns:
        GridSearchCV: Unfitted searcher over the published hyperparameter grid.
    """
    seed = int(os.environ['SEED'])
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "xgboost":
        model = XGBClassifier(random_state=seed, eval_metric='logloss', n_jobs=1)
    else:
        raise ValueError(f"Unsupported parity classifier: {name!r}")
    return GridSearchCV(
        make_classifier(model, impute_numeric=impute_numeric),
        HYPERPARAMETERS[name],
        scoring='roc_auc',
        cv=5,
        n_jobs=16,
    )


def load_embedded_matrix(arm: str, patient_ids: Iterable[str]) -> pd.DataFrame:
    """Read one arm's embeddings for the given patients, in sorted-id order.

    Row order is sorted(patient_ids), matching load_data_set, so a parity score vector
    lines up row-for-row with the published test_predictions parquet without a join.

    Args:
        arm (str): Narrative arm name.
        patient_ids (Iterable[str]): Patients to load.

    Returns:
        pd.DataFrame: Shape (n_patients, embedding_dim).
    """
    import sqlite3
    ordered = sorted(list(patient_ids))
    connection = sqlite3.connect(embeddings_dir(arm) / "embeddings.db")
    cursor = connection.cursor()
    placeholders = ",".join(["?"] * len(ordered))
    cursor.execute(
        f"SELECT patient_id, embedding FROM embeddings WHERE patient_id IN ({placeholders}) ORDER BY patient_id",
        ordered,
    )
    rows = cursor.fetchall()
    connection.close()
    returned = [row[0] for row in rows]
    if returned != ordered:
        raise ValueError(
            f"{arm}: embeddings cover {len(returned)} of {len(ordered)} requested patients; "
            "the arm's embedding pass is incomplete, so the comparison would silently "
            "run on a different population than the published one."
        )
    return pd.DataFrame(np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows]))


def fit_arm(arm: str, source: VectorSource) -> pd.DataFrame:
    """Fit the parity classifiers for one arm and score the held-out patients.

    Args:
        arm (str): 'feature_vitals' for the feature arm, else a narrative arm name.
        source (VectorSource): FEATURE or EMBEDDED, controlling how the matrix is loaded.

    Returns:
        pd.DataFrame: patient_id, true_label, and one probability column per model, row
            ordered on sorted(test_ids).
    """
    (train_ids, test_ids) = create_train_test_split()
    trd_ids = load_trd_set()
    if source == VectorSource.FEATURE:
        train_X = load_feature_matrix(train_ids, include_vitals=True)
        test_X = load_feature_matrix(test_ids, include_vitals=True)
        impute_numeric = True
    else:
        train_X = load_embedded_matrix(arm, train_ids)
        test_X = load_embedded_matrix(arm, test_ids)
        impute_numeric = False
    train_y = np.array([1 if pid in trd_ids else 0 for pid in sorted(list(train_ids))])
    test_y = np.array([1 if pid in trd_ids else 0 for pid in sorted(list(test_ids))])
    print(f"[{arm}] train {train_X.shape}, test {test_X.shape}", flush=True)

    predictions = pd.DataFrame({'patient_id': sorted(list(test_ids)), 'true_label': test_y})
    metrics = {}
    model_dir = parity_dir() / arm / "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    for name in PARITY_MODELS:
        cache_path = model_dir / f"{name}.joblib"
        if cache_path.exists():
            print(f"[{arm}] loading {name} from cache", flush=True)
            searcher = joblib.load(cache_path)
        else:
            start = time.perf_counter()
            print(f"[{arm}] fitting {name}...", flush=True)
            searcher = build_classifier(name, impute_numeric)
            searcher.fit(X=train_X, y=train_y)
            print(f"[{arm}] {name} done in {time.perf_counter() - start:.1f}s", flush=True)
            joblib.dump(searcher, cache_path)
        probabilities = searcher.predict_proba(X=test_X)[:, 1]
        predictions[name] = probabilities
        metrics[name] = compute_metrics(y_true=test_y, y_prob=probabilities)
        metrics[name]['best_parameters'] = {k: str(v) for k, v in searcher.best_params_.items()}

    arm_dir = parity_dir() / arm
    predictions.to_parquet(arm_dir / "test_predictions.parquet", index=False)
    with open(arm_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    return predictions


def published_predictions(source: VectorSource) -> pd.DataFrame:
    """Load the published per-patient held-out probabilities for one representation.

    Args:
        source (VectorSource): FEATURE or EMBEDDED.

    Returns:
        pd.DataFrame: The published table, row ordered on sorted(test_ids).
    """
    path = Path(os.environ['RESULTS_DIR']) / f"test_predictions_{source.name}.parquet"
    return pd.read_parquet(path)


def compare_against_published(parity: pd.DataFrame, published: pd.DataFrame, label: str) -> list[dict]:
    """Paired ROC AUC deltas, parity minus published, for every shared model column.

    Args:
        parity (pd.DataFrame): Output of fit_arm.
        published (pd.DataFrame): Output of published_predictions.
        label (str): Contrast name recorded on each row.

    Returns:
        list[dict]: One row per model with both AUCs and the paired interval.
    """
    if not parity['patient_id'].equals(published['patient_id']):
        raise ValueError(
            f"{label}: parity and published prediction tables are not row-aligned, so a "
            "paired comparison would be comparing different patients."
        )
    labels = parity['true_label'].to_numpy()
    rng = np.random.default_rng(int(os.environ['SEED']))
    index_matrix = rng.integers(low=0, high=len(labels), size=(N_BOOTSTRAP, len(labels)))
    rows = []
    for name in PARITY_MODELS:
        point, low, high = paired_auc_delta(
            labels,
            parity[name].to_numpy(),
            published[name].to_numpy(),
            index_matrix,
        )
        rows.append({
            'contrast': label,
            'model': name,
            'published_roc': float(roc_auc_score(labels, published[name].to_numpy())),
            'parity_roc': float(roc_auc_score(labels, parity[name].to_numpy())),
            'delta_roc': point,
            'delta_ci_low': low,
            'delta_ci_high': high,
            'excludes_zero': bool(low > 0 or high < 0),
        })
    return rows
