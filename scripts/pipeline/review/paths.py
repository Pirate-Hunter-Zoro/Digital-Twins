"""Output locations for the 2026-08-28 review-round analyses.

Every analysis in this package writes under ARTIFACTS_DIR/review/<name>/ rather than
into RESULTS_DIR. That separation is the point: these are review-response analyses run
against the frozen published artifacts, and none of them may overwrite a published
number. The repo-side mirror is results/review/<name>/, rsynced by the sbatch jobs the
same way the pipeline mirrors RESULTS_DIR.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def review_output_dir(name: str) -> Path:
    """Create and return the output directory for one review analysis.

    Args:
        name (str): Analysis slug, e.g. 'holdout_representativeness'.

    Returns:
        Path: ARTIFACTS_DIR/review/<name>/, created if absent.
    """
    save_dir = Path(os.environ['ARTIFACTS_DIR']) / "review" / name
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
