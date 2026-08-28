"""Collect every parity arm's deltas into one table and state what it implies.

Usage:
    python -m scripts.pipeline.review.parity.summarize

Reads whichever *_deltas.csv files exist, plus each arm's KNN results when the neighbour
stage has run, and writes the single artifact the write-up quotes:

  parity_summary.csv     every contrast, model, both AUCs, the paired delta and interval
  parity_summary.json    the same plus the decision the deltas support

The decision rule was fixed before the numbers existed: if no paired interval on the
head-to-head contrast excludes zero, the published comparison stands unchanged and the
matched-input result is held in reserve for a reviewer. If one does, the full pipeline pass
becomes unavoidable.
"""

import json
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.parity.core import PARITY_MODELS, parity_dir


def main():
    save_dir = parity_dir()
    frames = [pd.read_csv(path) for path in sorted(save_dir.glob("*_deltas.csv"))]
    if not frames:
        raise FileNotFoundError(
            f"No *_deltas.csv under {save_dir}; run the arm jobs before summarizing."
        )
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(save_dir / "parity_summary.csv", index=False)

    knn = {}
    for arm_dir in sorted(save_dir.glob("*/knn_results.json")):
        with open(arm_dir) as f:
            arm_results = json.load(f)
        knn[arm_dir.parent.name] = {
            key: round(value['roc_score'], 4)
            for key, value in arm_results.items()
            if key.startswith("NEAREST") or key.startswith("RANDOM")
        }

    significant = table[table['excludes_zero']]
    summary = {
        'contrasts': table.to_dict(orient='records'),
        'knn_roc_by_arm': knn,
        'models': list(PARITY_MODELS),
        'n_intervals_excluding_zero': int(len(significant)),
        'contrasts_excluding_zero': significant[['contrast', 'model', 'delta_roc']].to_dict(orient='records'),
        'decision': (
            "matched inputs move discrimination beyond sampling noise; the full pipeline "
            "pass is required"
            if len(significant) > 0 else
            "no matched-input contrast moves discrimination beyond sampling noise; the "
            "published comparison stands and this result is held in reserve"
        ),
    }
    with open(save_dir / "parity_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    print(table.to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != 'contrasts'}, indent=4), flush=True)


if __name__ == "__main__":
    main()
