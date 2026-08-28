"""Render one parity narrative arm: baseline text only, into the arm's own directory.

Usage:
    NARRATIVE_INCLUDE_HISTORY_LENGTH=<0|1> python -m scripts.pipeline.review.parity.render_narratives <arm>

Deliberately NOT scripts.pipeline.narratives.forge_narratives. That driver also writes six
ablated narratives per patient -- 42,579 x 7 files -- and the ablation slate is out of
scope here: no delta is being re-scored, only the baseline representation. It also honours
SCRUB_NARRATIVES and prunes non-cohort files, neither of which applies to a fresh
directory. What is reused is the part that matters: extract_fields and render_narrative,
so the parity text comes off exactly the same renderer as the published text, with
NARRATIVE_INCLUDE_HISTORY_LENGTH the only difference between the two arms.
"""

import multiprocessing
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.deterministic_narrative import (
    INCLUDE_HISTORY_LENGTH,
    extract_fields,
    render_narrative,
)
from scripts.data_loading.load_patient_data import load_patient_data
from scripts.pipeline.review.parity.core import NARRATIVE_ARMS, narratives_dir

RECORD_EVERY = 2000
_OUTPUT_DIR: Path = None


def _render_one(sliced_json: dict) -> str:
    """Render and write one patient's narrative.

    Args:
        sliced_json (dict): The patient's anchored, windowed record.

    Returns:
        str: The patient id, so the parent can count progress.
    """
    patient_id = sliced_json["patient_id"]
    text = render_narrative(extract_fields(sliced_json))
    (_OUTPUT_DIR / f"{patient_id}.md").write_text(text)
    return patient_id


def main():
    global _OUTPUT_DIR
    arm = sys.argv[1]
    if arm not in NARRATIVE_ARMS:
        raise ValueError(f"Unknown parity arm {arm!r}; expected one of {sorted(NARRATIVE_ARMS)}.")
    expected_flag = NARRATIVE_ARMS[arm] == 1
    if INCLUDE_HISTORY_LENGTH != expected_flag:
        raise ValueError(
            f"Arm {arm!r} needs NARRATIVE_INCLUDE_HISTORY_LENGTH={NARRATIVE_ARMS[arm]}, but the "
            f"renderer loaded with INCLUDE_HISTORY_LENGTH={INCLUDE_HISTORY_LENGTH}. Rendering "
            "would silently produce the other arm's text."
        )
    _OUTPUT_DIR = narratives_dir(arm)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print(f"Rendering arm {arm} (history length {'ON' if expected_flag else 'OFF'}) into {_OUTPUT_DIR}", flush=True)

    sliced_jsons = list(load_patient_data())
    print(f"Loaded {len(sliced_jsons):,} anchored patient records.", flush=True)
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render_one, sliced_jsons)):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"  rendered {i + 1:,}", flush=True)
    written = len(list(_OUTPUT_DIR.glob("*.md")))
    print(f"Arm {arm}: {written:,} narratives on disk.", flush=True)


if __name__ == "__main__":
    main()
