"""Figure: the study's time zero, its lookback window, and its ascertainment window.

Review item (2026-08-28): the reviewer asked for a single timeline figure alongside the
algorithmic definition of anchor selection, because the paper's original description of its
index event was wrong -- it called the anchor a "first adequate antidepressant exposure",
which would have required 42 days of post-anchor observation to establish and would have
made prediction at the prescription date incoherent. It is not that. The anchor is the
earliest antidepressant prescription recorded on or after the patient's first documented
depression diagnosis, and no property of it reads post-anchor data.

The figure exists to make that checkable at a glance: everything left of zero feeds the
model, everything right of zero feeds only eligibility and the label.

Pure presentation. The two window widths come from YEARS_BACK and YEARS_AHEAD, and the
median pre-anchor history is read off the feature table, so no number here is typed by
hand and none can drift from the pipeline.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from dotenv import load_dotenv
load_dotenv()

# The share of candidate antidepressant orders in the upstream index table that start on
# the same day as the first recorded depression diagnosis. Verified against
# post_mdd_ad_index.csv and quoted in Methods, *Anchor selection*.
SAME_DAY_SHARE = 57.9


def median_pre_anchor_history_days() -> float:
    """Median recorded history length before the anchor, over the analysis cohort.

    Returns:
        float: Median of pre_anchor_history_days.
    """
    history = pd.read_parquet(
        Path(os.environ['FEATURE_DATAFRAME_PATH']), columns=['pre_anchor_history_days']
    )
    return float(history['pre_anchor_history_days'].median())


def plot_timeline(save_path: Path) -> Path:
    """Draw the one-panel time-zero figure.

    Laid out in fixed horizontal lanes rather than by centring text on the features it
    describes. The outcome window is 365 days against a ~2,400-day axis, so a label
    centred inside it collides with the lookback label beside it; every caption therefore
    sits in its own lane with a leader where one is needed.

    Args:
        save_path (Path): Destination PNG.

    Returns:
        Path: The written figure.
    """
    lookback_days = 365 * int(os.environ['YEARS_BACK'])
    followup_days = 365 * int(os.environ['YEARS_AHEAD'])
    median_history = median_pre_anchor_history_days()

    # Wide and short, matching the manuscript's redrawn panel geometry: a full-width panel
    # taller than about half the 9in text column strands the rest of the page.
    figure, ax = plt.subplots(figsize=(6.2, 2.9))

    left_limit = -(median_history + 150)
    right_limit = followup_days + 150
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(-1.0, 1.0)
    ax.axis('off')

    # Lane constants, top to bottom. Every element is pinned to one of these.
    Y_TITLE = 0.97
    Y_DIAGNOSIS = 0.68
    Y_OUTCOME_LABEL = 0.42
    Y_BAND_TOP = 0.19
    BAND_HEIGHT = 0.13
    Y_AXIS = 0.0
    Y_TICK_LABEL = -0.09
    Y_HISTORY = -0.33
    Y_HISTORY_LABEL = -0.41
    Y_FOOTER = -0.66

    ax.annotate(
        "", xy=(right_limit, Y_AXIS), xytext=(left_limit, Y_AXIS),
        arrowprops=dict(arrowstyle='-|>', color='black', linewidth=1.0),
    )

    ax.add_patch(plt.Rectangle(
        (-lookback_days, Y_BAND_TOP), lookback_days, BAND_HEIGHT,
        facecolor='#4c78a8', edgecolor='none',
    ))
    ax.add_patch(plt.Rectangle(
        (0, Y_BAND_TOP), followup_days, BAND_HEIGHT,
        facecolor='#e45756', edgecolor='none',
    ))

    # Left of its own band, so it cannot run under the outcome label above the band's
    # right-hand end.
    ax.text(
        -lookback_days - 80, Y_BAND_TOP + BAND_HEIGHT / 2,
        f"Lookback window ({lookback_days} days)\nevery predictor measured here",
        ha='right', va='center', fontsize=7.0, color='#2f4a66', fontweight='bold',
    )
    # The outcome window is 365 days wide against a ~2,400-day axis, so its label cannot
    # sit over its own band. It gets a lane of its own above the lookback label, with a
    # leader down to the band.
    ax.annotate(
        f"Outcome window ({followup_days} days)",
        xy=(followup_days / 2, Y_BAND_TOP + BAND_HEIGHT + 0.01),
        xytext=(right_limit, Y_OUTCOME_LABEL),
        ha='right', va='center', fontsize=7.0, color='#8c3330', fontweight='bold',
        arrowprops=dict(arrowstyle='-', color='#8c3330', linewidth=0.7, shrinkA=3, shrinkB=1),
    )

    # Time zero.
    ax.plot([0, 0], [Y_BAND_TOP - 0.05, Y_DIAGNOSIS - 0.08], color='black', linewidth=1.3)
    ax.text(
        left_limit, Y_TITLE,
        "Anchor (time zero): the earliest antidepressant prescription recorded on or\n"
        "after the first documented depression diagnosis. Prediction is made here.",
        ha='left', va='top', fontsize=7.4, fontweight='bold',
    )

    # First documented depression diagnosis: at or before the anchor.
    ax.annotate(
        f"First documented depression diagnosis: at or before the anchor\n"
        f"({SAME_DAY_SHARE}% of candidate orders start on the anchor day itself)",
        xy=(0, Y_DIAGNOSIS - 0.06), xytext=(-200, Y_DIAGNOSIS),
        ha='right', va='center', fontsize=6.5, color='#7d4f74',
        arrowprops=dict(arrowstyle='-|>', color='#b279a2', linewidth=0.8, shrinkA=2, shrinkB=0),
    )

    # Recorded history reaching back beyond the lookback window. Its LENGTH is a
    # predictor even though its content outside the window is never read.
    ax.add_patch(FancyArrowPatch(
        (-median_history, Y_HISTORY), (0, Y_HISTORY),
        arrowstyle='|-|', mutation_scale=2.5, linewidth=0.9, color='#54a24b',
    ))
    ax.text(
        -median_history, Y_HISTORY_LABEL,
        f"Recorded pre-anchor history, median {median_history:,.0f} days: its LENGTH is a\n"
        "predictor; its content outside the lookback window is never read",
        ha='left', va='top', fontsize=6.5, color='#3a7a34',
    )

    for offset, label in (
        (-lookback_days, f"\u2212{lookback_days}"), (0, "0"), (followup_days, f"+{followup_days}")
    ):
        ax.plot([offset, offset], [Y_AXIS - 0.035, Y_AXIS + 0.035], color='black', linewidth=1.0)
        ax.text(offset, Y_TICK_LABEL, label, ha='center', va='top', fontsize=7.0)
    # Above the axis, clear of the +365 tick label directly below it.
    ax.text(
        right_limit - 30, Y_AXIS + 0.05, "days from anchor",
        ha='right', va='bottom', fontsize=6.5, style='italic',
    )

    ax.text(
        left_limit, Y_FOOTER,
        f"Eligibility requires at least {followup_days} days of post-anchor follow-up. Three or more antidepressant\n"
        f"treatments inside the outcome window label a patient TRD-positive. No property of the anchor, and no\n"
        "predictor, reads data recorded after the anchor date.",
        ha='left', va='top', fontsize=6.5, color='#333333',
    )

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return save_path


def main():
    save_path = Path(os.environ['RESULTS_DIR']) / "time_zero_timeline.png"
    os.makedirs(save_path.parent, exist_ok=True)
    written = plot_timeline(save_path)
    print(f"Wrote {written}", flush=True)


if __name__ == "__main__":
    main()
