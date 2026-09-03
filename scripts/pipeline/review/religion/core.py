"""Religion sensitivity: does the model lean on a field that is 29.4% unrecorded?

Review item (2026-09-02). The supplied supplementary methods asked for a sensitivity
analysis comparing retaining religion against removing it, and the reason is the shape of
its missingness rather than the field itself. Religion is absent for 29.4% of the cohort,
and that absence is not uniform: 43.8% of 18-29-year-olds have no recorded religion
against 17.5% of those aged 65 or older. Both representations turn "unrecorded" into
something a model can read -- the FEATURE one-hot encoder gives it a level of its own, and
the narrative prints the literal token `Missing` -- so a model can use the field's absence
as an age proxy without ever using the field.

Removing it is therefore a different question from permuting it, which is what the
semantic-feature ablation does. A permutation destroys the link between a patient and
their own value while leaving the pattern of who has one intact; a removal takes both
away. This module removes.

Scope, deliberately the same reduction the parity analysis used: two classifiers rather
than four, one encoder rather than four, no neighbour arm, no ablation re-score. Its only
job is to say whether the published result depends on the field.

  feature_no_religion    the published feature matrix minus the Religion column.
  narrative_no_religion  the narrative re-rendered with the Religion field omitted from
                         the sociodemographics line, then re-embedded.

THE EMBEDDED ARM IS COMPARED AGAINST THE PARITY CONTROL, NOT AGAINST THE PUBLISHED ARM,
and that is the whole reason this analysis is affordable. render_narrative once emitted its
flag lists in set-iteration order, so the published narratives are not byte-reproducible
and a re-render moves the embedded arm on its own -- measured at up to +0.003 AUC for the
fitted classifiers in the parity round. ARTIFACTS_DIR/review/parity/narrative_control is a
re-render with NO content change, already embedded, so the contrast against it is
attributable to the removed field rather than to the rebuild. The contrast against the
published arm is reported too, but it carries the re-render inside it and is the weaker
of the two.

Every comparison is a PAIRED bootstrap on the same held-out patients, reusing
paired_auc_delta from the discrimination forest.
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

ANALYSIS_NAME = "religion"

# The same pair the parity analysis used, and for the same reason: logistic regression is
# the best embedded model and XGBoost the best feature-vector one, so the head-to-head
# contrast and the classifier-by-representation interaction both live in this pair.
RELIGION_MODELS = ("logistic_regression", "xgboost")

FEATURE_ARM = "feature_no_religion"
NARRATIVE_ARM = "narrative_no_religion"

# The parity round's re-render with no content change, already rendered and embedded. The
# embedded arm here is scored against it so the delta is the field rather than the rebuild.
CONTROL_ARM = "narrative_control"


def religion_dir() -> Path:
    """Root for every religion-sensitivity artifact.

    Returns:
        Path: ARTIFACTS_DIR/review/religion/.
    """
    return review_output_dir(ANALYSIS_NAME)


def parity_dir() -> Path:
    """Root of the parity round's artifacts, which hold the control arm.

    Returns:
        Path: ARTIFACTS_DIR/review/parity/.
    """
    return Path(os.environ['ARTIFACTS_DIR']) / "review" / "parity"


def narratives_dir() -> Path:
    """Where the religion-free narratives live.

    Returns:
        Path: The arm's narrative directory.
    """
    return religion_dir() / NARRATIVE_ARM / "narratives"


def embeddings_dir(arm: str) -> Path:
    """Where one narrative arm's embedding database lives.

    The control arm is the parity round's, so its database is looked up under the parity
    root rather than this analysis's.

    Args:
        arm (str): NARRATIVE_ARM or CONTROL_ARM.

    Returns:
        Path: The directory holding embeddings.db.
    """
    root = parity_dir() if arm == CONTROL_ARM else religion_dir()
    return root / arm / os.environ['EMBEDDER_MODEL_NAME']


def build_classifier(name: str) -> GridSearchCV:
    """Construct the grid search for one classifier, with the published grid.

    No imputer anywhere: neither arm here carries the vital signs, so the numeric branch
    never meets a NaN, exactly as in the published run.

    Args:
        name (str): 'logistic_regression' or 'xgboost'.

    Returns:
        GridSearchCV: Unfitted searcher over the published hyperparameter grid.
    """
    seed = int(os.environ['SEED'])
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "xgboost":
        model = XGBClassifier(random_state=seed, eval_metric='logloss', n_jobs=1)
    else:
        raise ValueError(f"Unsupported religion-arm classifier: {name!r}")
    return GridSearchCV(
        make_classifier(model, impute_numeric=False),
        HYPERPARAMETERS[name],
        scoring='roc_auc',
        cv=5,
        n_jobs=16,
    )


def load_embedded_matrix(arm: str, patient_ids: Iterable[str]) -> pd.DataFrame:
    """Read one arm's embeddings for the given patients, in sorted-id order.

    Row order is sorted(patient_ids), matching load_data_set, so a score vector lines up
    row-for-row with the published test_predictions parquet without a join.

    Args:
        arm (str): NARRATIVE_ARM or CONTROL_ARM.
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
            "the arm's embedding pass is incomplete, so the comparison would silently run "
            "on a different population than the published one."
        )
    return pd.DataFrame(np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows]))


def fit_arm(arm: str, source: VectorSource) -> pd.DataFrame:
    """Fit this analysis's classifiers for one arm and score the held-out patients.

    Args:
        arm (str): FEATURE_ARM, NARRATIVE_ARM, or CONTROL_ARM.
        source (VectorSource): FEATURE or EMBEDDED, controlling how the matrix is loaded.

    Returns:
        pd.DataFrame: patient_id, true_label, and one probability column per model, row
            ordered on sorted(test_ids).
    """
    (train_ids, test_ids) = create_train_test_split()
    trd_ids = load_trd_set()
    if source == VectorSource.FEATURE:
        train_X = load_feature_matrix(train_ids, drop_religion=True)
        test_X = load_feature_matrix(test_ids, drop_religion=True)
    else:
        train_X = load_embedded_matrix(arm, train_ids)
        test_X = load_embedded_matrix(arm, test_ids)
    train_y = np.array([1 if pid in trd_ids else 0 for pid in sorted(list(train_ids))])
    test_y = np.array([1 if pid in trd_ids else 0 for pid in sorted(list(test_ids))])
    print(f"[{arm}] train {train_X.shape}, test {test_X.shape}", flush=True)

    predictions = pd.DataFrame({'patient_id': sorted(list(test_ids)), 'true_label': test_y})
    metrics = {}
    arm_dir = religion_dir() / arm
    model_dir = arm_dir / "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    for name in RELIGION_MODELS:
        cache_path = model_dir / f"{name}.joblib"
        if cache_path.exists():
            print(f"[{arm}] loading {name} from cache", flush=True)
            searcher = joblib.load(cache_path)
        else:
            start = time.perf_counter()
            print(f"[{arm}] fitting {name}...", flush=True)
            searcher = build_classifier(name)
            searcher.fit(X=train_X, y=train_y)
            print(f"[{arm}] {name} done in {time.perf_counter() - start:.1f}s", flush=True)
            joblib.dump(searcher, cache_path)
        probabilities = searcher.predict_proba(X=test_X)[:, 1]
        predictions[name] = probabilities
        metrics[name] = compute_metrics(y_true=test_y, y_prob=probabilities)
        metrics[name]['best_parameters'] = {k: str(v) for k, v in searcher.best_params_.items()}

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


def control_predictions() -> pd.DataFrame:
    """Load the parity round's no-content-change re-render predictions.

    Returns:
        pd.DataFrame: The control arm's table, row ordered on sorted(test_ids).

    Raises:
        FileNotFoundError: If the parity round's control arm was never fitted, in which
            case the field-attributable contrast cannot be formed at all.
    """
    path = parity_dir() / CONTROL_ARM / "test_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The embedded religion contrast is defined against the "
            "parity round's re-rendered control arm; without it only the weaker "
            "against-published contrast is available."
        )
    return pd.read_parquet(path)


def compare(arm_predictions: pd.DataFrame, reference: pd.DataFrame, label: str) -> list[dict]:
    """Paired ROC AUC deltas, arm minus reference, for every shared model column.

    Args:
        arm_predictions (pd.DataFrame): Output of fit_arm.
        reference (pd.DataFrame): The table being scored against.
        label (str): Contrast name recorded on each row.

    Returns:
        list[dict]: One row per model with both AUCs and the paired interval.
    """
    if not arm_predictions['patient_id'].equals(reference['patient_id']):
        raise ValueError(
            f"{label}: the two prediction tables are not row-aligned, so a paired "
            "comparison would be comparing different patients."
        )
    labels = arm_predictions['true_label'].to_numpy()
    rng = np.random.default_rng(int(os.environ['SEED']))
    index_matrix = rng.integers(low=0, high=len(labels), size=(N_BOOTSTRAP, len(labels)))
    rows = []
    for name in RELIGION_MODELS:
        point, low, high = paired_auc_delta(
            labels,
            arm_predictions[name].to_numpy(),
            reference[name].to_numpy(),
            index_matrix,
        )
        rows.append({
            'contrast': label,
            'model': name,
            'reference_roc': float(roc_auc_score(labels, reference[name].to_numpy())),
            'no_religion_roc': float(roc_auc_score(labels, arm_predictions[name].to_numpy())),
            'delta_roc': point,
            'delta_ci_low': low,
            'delta_ci_high': high,
            'excludes_zero': bool(low > 0 or high < 0),
        })
    return rows
