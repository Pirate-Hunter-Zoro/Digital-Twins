"""Render the religion-free narratives into this analysis's own directory.

Usage:
    NARRATIVE_DROP_RELIGION=1 python -m scripts.pipeline.review.religion.render_narratives

Deliberately NOT scripts.pipeline.narratives.forge_narratives, for the same reason the
parity renderer is not: that driver also writes six ablated narratives per patient --
42,579 x 7 files -- and no ablation delta is being re-scored here. What is reused is the
part that matters, extract_fields and render_narrative, so this text comes off exactly the
same renderer as the published text with NARRATIVE_DROP_RELIGION the only difference.

The comparison arm is the parity round's narrative_control, which is already on disk and
already embedded, so nothing is rendered for it here.
"""

import multiprocessing
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.deterministic_narrative import (
    DROP_RELIGION,
    extract_fields,
    render_narrative,
)
from scripts.data_loading.load_patient_data import load_patient_data
from scripts.pipeline.review.religion.core import narratives_dir

RECORD_EVERY = 2000
_OUTPUT_DIR: Path = None


def _render_one(sliced_json: dict) -> str:
    """Render and write one patient's religion-free narrative.

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
    if not DROP_RELIGION:
        raise ValueError(
            "This arm needs NARRATIVE_DROP_RELIGION=1, but the renderer loaded with it "
            "off. Rendering would silently produce narratives that still carry religion, "
            "and the arm would be a re-render of the control rather than a sensitivity "
            "analysis."
        )
    _OUTPUT_DIR = narratives_dir()
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print(f"Rendering the religion-free arm into {_OUTPUT_DIR}", flush=True)

    sliced_jsons = list(load_patient_data())
    print(f"Loaded {len(sliced_jsons):,} anchored patient records.", flush=True)
    with multiprocessing.Pool(processes=int(os.environ['NUM_WORKERS_NON_LLM_TASK'])) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render_one, sliced_jsons)):
            if (i + 1) % RECORD_EVERY == 0:
                print(f"  rendered {i + 1:,}", flush=True)
    written = len(list(_OUTPUT_DIR.glob("*.md")))
    print(f"Religion-free arm: {written:,} narratives on disk.", flush=True)

    # A render that still mentions religion is the one failure this job cannot leave
    # undetected, because everything downstream would run and return a clean null.
    sample = sorted(_OUTPUT_DIR.glob("*.md"))[:200]
    offenders = [p.name for p in sample if "Religion:" in p.read_text()]
    if offenders:
        raise ValueError(
            f"{len(offenders)} of {len(sample)} sampled narratives still render a Religion "
            f"field, e.g. {offenders[0]}. The flag did not take effect."
        )
    print(f"Checked {len(sample)} narratives: no Religion field rendered.", flush=True)


if __name__ == "__main__":
    main()
