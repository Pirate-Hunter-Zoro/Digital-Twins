"""Collect every parity arm's deltas into one table and state what it implies.

Usage:
    python -m scripts.pipeline.review.parity.summarize

Reads whichever arm artifacts exist and writes the single pair of files the write-up
quotes:

  parity_summary.csv     every contrast, model, both AUCs, the paired delta and interval
  parity_summary.json    the same plus the decision the deltas support

Three families of contrast are reported, and they answer different questions.

  within-representation  each arm against its own published counterpart. Reads the
                         *_deltas.csv the arm jobs wrote. These say what the added field
                         did to the arm that received it, which is a manipulation check:
                         a field carrying information should move its own arm.
  head-to-head           the embedded arm against the feature arm, both on matched
                         inputs, per classifier plus best-versus-best. This is the
                         published primary comparison recomputed with the asymmetry
                         closed, and it is the quantity the decision rule names.
  neighbour              nearest-retrieval KNN, each narrative arm against the published
                         run. The arm jobs write ROC scores with no interval; the paired
                         bootstrap is computed here from the per-patient risks.

THE DECISION RULE, as fixed on 2026-08-28 before any arm reported: if no paired interval
on the HEAD-TO-HEAD contrast excludes zero, the published comparison stands as reported
and this analysis is held in reserve. If one does, the full pipeline pass becomes
unavoidable. The within-representation and neighbour contrasts are evidence about why the
head-to-head moved or did not; they are not themselves the trigger, and the supplement's
own justification for the trigger says as much -- it names the donor pairings and frozen
baselines downstream of the NARRATIVE text as what a full pass would have to rebuild,
which a feature-arm result cannot require.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.predictions.plot_discrimination_forest import paired_auc_delta
from scripts.pipeline.review.parity.core import NARRATIVE_ARMS, PARITY_MODELS, parity_dir
from scripts.shared.plots import N_BOOTSTRAP

FEATURE_ARM = "feature_vitals"

# Best embedded model and best feature-vector model in the published run, which is the
# pairing the published headline contrast uses.
BEST_EMBEDDED_MODEL = "logistic_regression"
BEST_FEATURE_MODEL = "xgboost"

# Nearest-retrieval strategies the narrative arms ran. The farthest and subsampled schemes
# and the LLM and combined weightings are switched off in the parity design.
KNN_STRATEGIES = ("UNIFORM", "COSINE")


def _index_matrix(n_rows: int) -> np.ndarray:
    """Bootstrap resample indices, seeded so every contrast here shares the draws.

    Args:
        n_rows (int): Number of held-out patients.

    Returns:
        np.ndarray: Shape (N_BOOTSTRAP, n_rows).
    """
    rng = np.random.default_rng(int(os.environ['SEED']))
    return rng.integers(low=0, high=n_rows, size=(N_BOOTSTRAP, n_rows))


def _predictions(arm: str) -> pd.DataFrame | None:
    """One arm's per-patient held-out probabilities, if the arm has been fitted.

    Args:
        arm (str): Arm directory name.

    Returns:
        pd.DataFrame | None: The parquet the arm job wrote, or None if it is absent.
    """
    path = parity_dir() / arm / "test_predictions.parquet"
    return pd.read_parquet(path) if path.exists() else None


def head_to_head_rows(narrative_arm: str, feature: pd.DataFrame) -> list[dict]:
    """Paired embedded-minus-feature deltas on matched inputs, for one narrative arm.

    Mirrors the published discrimination forest: one contrast per classifier holding the
    classifier fixed and varying only the representation, plus the best-versus-best
    contrast that carries the headline.

    Args:
        narrative_arm (str): 'narrative_control' or 'narrative_parity'.
        feature (pd.DataFrame): The feature arm's predictions.

    Returns:
        list[dict]: One row per contrast, empty if the narrative arm is not fitted yet.
    """
    embedded = _predictions(narrative_arm)
    if embedded is None:
        return []
    if not embedded['patient_id'].equals(feature['patient_id']):
        raise ValueError(
            f"{narrative_arm}: embedded and feature prediction tables are not row-aligned, "
            "so a paired comparison would be comparing different patients."
        )
    labels = embedded['true_label'].to_numpy()
    index_matrix = _index_matrix(len(labels))
    pairings = [(name, name) for name in PARITY_MODELS]
    pairings.append((BEST_EMBEDDED_MODEL, BEST_FEATURE_MODEL))
    rows = []
    for embedded_model, feature_model in pairings:
        headline = embedded_model != feature_model
        point, low, high = paired_auc_delta(
            labels,
            embedded[embedded_model].to_numpy(),
            feature[feature_model].to_numpy(),
            index_matrix,
        )
        rows.append({
            'contrast': f"head_to_head_matched__{narrative_arm}_vs_{FEATURE_ARM}",
            'model': (
                f"{embedded_model}_vs_{feature_model}" if headline else embedded_model
            ),
            'published_roc': float(roc_auc_score(labels, feature[feature_model].to_numpy())),
            'parity_roc': float(roc_auc_score(labels, embedded[embedded_model].to_numpy())),
            'delta_roc': point,
            'delta_ci_low': low,
            'delta_ci_high': high,
            'excludes_zero': bool(low > 0 or high < 0),
            'headline': headline,
        })
    return rows


def _knn_risks(results_dir: Path, strategy: str) -> pd.Series:
    """Per-patient nearest-neighbour risk for one weighting strategy, id-sorted.

    Args:
        results_dir (Path): Directory holding summary_predictions.csv.
        strategy (str): 'UNIFORM' or 'COSINE'.

    Returns:
        pd.Series: Predicted risk indexed by anchor_patient_id, sorted on the index.
    """
    frame = pd.read_csv(results_dir / "summary_predictions.csv")
    selected = frame[
        (frame['neighbor_scheme'] == "NEAREST") & (frame['weighting_strategy'] == strategy)
    ]
    return selected.set_index('anchor_patient_id').sort_index()[['predicted_risk', 'true_label']]


def knn_rows() -> list[dict]:
    """Paired nearest-retrieval KNN deltas, each narrative arm against the published run.

    Returns:
        list[dict]: One row per arm and weighting strategy; empty if the published or the
            arm-level summary_predictions.csv is missing.
    """
    published_dir = Path(os.environ['ARTIFACTS_DIR']) / os.environ['EMBEDDER_MODEL_NAME'] / os.environ['VLLM_MODEL_NAME']
    if not (published_dir / "summary_predictions.csv").exists():
        return []
    rows = []
    for arm in sorted(NARRATIVE_ARMS):
        arm_dir = parity_dir() / arm
        if not (arm_dir / "summary_predictions.csv").exists():
            continue
        for strategy in KNN_STRATEGIES:
            arm_risks = _knn_risks(arm_dir, strategy)
            published_risks = _knn_risks(published_dir, strategy).loc[arm_risks.index]
            labels = arm_risks['true_label'].to_numpy()
            point, low, high = paired_auc_delta(
                labels,
                arm_risks['predicted_risk'].to_numpy(),
                published_risks['predicted_risk'].to_numpy(),
                _index_matrix(len(labels)),
            )
            rows.append({
                'contrast': f"knn_nearest__{arm}_vs_published",
                'model': f"NEAREST_{strategy}",
                'published_roc': float(roc_auc_score(labels, published_risks['predicted_risk'].to_numpy())),
                'parity_roc': float(roc_auc_score(labels, arm_risks['predicted_risk'].to_numpy())),
                'delta_roc': point,
                'delta_ci_low': low,
                'delta_ci_high': high,
                'excludes_zero': bool(low > 0 or high < 0),
                'headline': False,
            })
    return rows


def main():
    save_dir = parity_dir()
    frames = [pd.read_csv(path) for path in sorted(save_dir.glob("*_deltas.csv"))]
    if not frames:
        raise FileNotFoundError(
            f"No *_deltas.csv under {save_dir}; run the arm jobs before summarizing."
        )
    within = pd.concat(frames, ignore_index=True)
    within['headline'] = False

    feature = _predictions(FEATURE_ARM)
    matched = []
    if feature is not None:
        for arm in sorted(NARRATIVE_ARMS):
            matched.extend(head_to_head_rows(arm, feature))
    neighbour = knn_rows()

    table = pd.concat(
        [within] + [pd.DataFrame(rows) for rows in (matched, neighbour) if rows],
        ignore_index=True,
    )
    table.to_csv(save_dir / "parity_summary.csv", index=False)

    knn_scores = {}
    for path in sorted(save_dir.glob("*/knn_results.json")):
        with open(path) as f:
            arm_results = json.load(f)
        knn_scores[path.parent.name] = {
            key: round(value['roc_score'], 4)
            for key, value in arm_results.items()
            if key.startswith("NEAREST") or key.startswith("RANDOM")
        }

    head_to_head = table[table['contrast'].str.startswith("head_to_head_matched__")]
    triggering = head_to_head[head_to_head['excludes_zero'] & head_to_head['headline']]
    complete = feature is not None and len(head_to_head) == 2 * (len(PARITY_MODELS) + 1)
    if not complete:
        decision = (
            "INCOMPLETE -- the head-to-head contrast the decision rule names cannot be "
            "computed until every arm has been fitted; no decision is on the record"
        )
    elif len(triggering) > 0:
        decision = (
            "matched inputs move the head-to-head comparison beyond sampling noise; the "
            "full pipeline pass is required"
        )
    else:
        decision = (
            "no head-to-head contrast on matched inputs moves discrimination beyond "
            "sampling noise; the published comparison stands and this result is held in "
            "reserve"
        )
    summary = {
        'contrasts': table.to_dict(orient='records'),
        'knn_roc_by_arm': knn_scores,
        'models': list(PARITY_MODELS),
        'n_intervals_excluding_zero': int(table['excludes_zero'].sum()),
        'contrasts_excluding_zero': table[table['excludes_zero']][
            ['contrast', 'model', 'delta_roc']
        ].to_dict(orient='records'),
        'decision': decision,
    }
    with open(save_dir / "parity_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    print(table.to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != 'contrasts'}, indent=4), flush=True)


if __name__ == "__main__":
    main()
