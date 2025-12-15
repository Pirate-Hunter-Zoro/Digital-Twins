"""Matplotlib plotting helpers (headless).
Used by Stage-3 judge–cosine plots."""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from scripts.patient_embedding.shared.io import ensure_dir

def histogram(x: List[float], title: str, outpath: Path) -> None:
    # Create a new figure
    x = [v for v in x if (isinstance(v, float) and (not math.isnan(v)) and (not math.isinf(v)))]
    if len(x) > 0:
        plt.figure(figsize=(10, 6))
    
        # Plot the histogram
        # 'bins='auto'' lets matplotlib decide the optimal number of bins
        plt.hist(x, bins=100, edgecolor='black', alpha=0.7)
        
        # Set the title and labels
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        # Add a grid for better readability
        plt.grid(axis='y', alpha=0.75)
        
        # Ensure the parent directory exists
        outpath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the figure to the specified Path object
        # bbox_inches='tight' helps prevent the title or labels from being cut off
        plt.savefig(outpath, format='png', bbox_inches='tight')
        
        # Close the plot to free up memory
        plt.close()

def scatter(xs: List[Optional[float]], ys: List[Optional[float]], title: str, outpath: Path,
           xlabel: str, ylabel: str) -> None:
    X, Y = [], []
    for a, b in zip(xs, ys):
        if a is None or b is None:
            continue
        if isinstance(a, float) and math.isnan(a):
            continue
        if isinstance(b, float) and math.isnan(b):
            continue
        X.append(float(a)); Y.append(float(b))

    ensure_dir(outpath.parent)

    # Set up a figure with 2 side axes for histograms
    from matplotlib import gridspec

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(4, 4, figure=fig)
    ax_scatter = fig.add_subplot(gs[1:,:3])   # main scatter
    ax_histx   = fig.add_subplot(gs[0,:3], sharex=ax_scatter)  # top histogram
    ax_histy   = fig.add_subplot(gs[1:,3], sharey=ax_scatter)  # right histogram

    if X and Y:
        ax_scatter.scatter(X, Y, s=8)
    else:
        ax_scatter.scatter([0], [0], s=8)

    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    fig.suptitle(title)

    # histograms
    if X:
        ax_histx.hist(X, bins=50, color="gray")
    if Y:
        ax_histy.hist(Y, bins=50, orientation="horizontal", color="gray")

    # Clean up tick labels on marginal axes
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)