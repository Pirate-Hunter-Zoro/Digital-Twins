"""Runner: sample cached pairs, re-judge them phenotype-free, and report the agreement.

Usage (inside the sbatch, with VLLM_URL pointing at one live server):
    python -m scripts.pipeline.review.judge_prompt.run_comparison [n_pairs]

Artifacts, in ARTIFACTS_DIR/review/judge_prompt/:
  judgements_no_phenotype.db     the new judgements, in their own table
  judge_prompt_pairs.csv         per-pair old score, new score, and sub-scores
  judge_prompt_agreement.json    the correlation and agreement statistics
  judge_prompt_agreement.png     old-versus-new hexbin plus the difference distribution
"""

import asyncio
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.judge_prompt.core import (
    ANALYSIS_NAME,
    compare,
    plot_comparison,
    rejudge,
    sample_cached_pairs,
)
from scripts.pipeline.review.paths import review_output_dir

DEFAULT_PAIRS = 5000


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PAIRS
    save_dir = review_output_dir(ANALYSIS_NAME)

    pairs = sample_cached_pairs(n_pairs)
    print(f"Sampled {len(pairs):,} pairs from the canonical cache.", flush=True)

    judged = asyncio.run(rejudge(pairs))
    judged.to_csv(save_dir / "judge_prompt_pairs.csv", index=False)

    statistics = compare(judged)
    with open(save_dir / "judge_prompt_agreement.json", 'w') as f:
        json.dump(statistics, f, indent=4)
    figure_path = plot_comparison(judged, statistics, save_dir)

    print(json.dumps(statistics, indent=4), flush=True)
    print(f"\nWrote {figure_path} and 3 sibling artifacts to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
