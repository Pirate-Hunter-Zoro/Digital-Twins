"""Flatten both religion arms into one table and one JSON block.

Usage:
    python -m scripts.pipeline.review.religion.summarize

Reads whichever delta CSVs exist, so it is safe to run after the cheap feature arm alone
and again once the embedded arm lands. Writes religion_summary.csv and
religion_summary.json under ARTIFACTS_DIR/review/religion/, and prints the table.
"""

import json

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.religion.core import religion_dir

SOURCES = ("feature_arm_deltas.csv", "embedded_arm_deltas.csv")


def main():
    root = religion_dir()
    frames = [pd.read_csv(root / name) for name in SOURCES if (root / name).exists()]
    if not frames:
        raise FileNotFoundError(
            f"No delta CSVs under {root}. Run run_feature_arm and/or run_embedded_arm first."
        )
    summary = pd.concat(frames, ignore_index=True)
    summary['verdict'] = summary['excludes_zero'].map({True: "moves", False: "null"})
    summary.to_csv(root / "religion_summary.csv", index=False)
    with open(root / "religion_summary.json", 'w') as f:
        json.dump(summary.to_dict(orient='records'), f, indent=4)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary.to_string(index=False), flush=True)
    print(f"\nWrote {root / 'religion_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
