"""Stage 3 entrypoint.
Builds pairs, judges once, computes cosines, plots, and emits discordant bundles.
CLI-only; delegates to stage3.runner.run()."""

#!/usr/bin/env python3
# scripts/patient_embedding/analyze_and_plot.py
from __future__ import annotations
import sys
from pathlib import Path

# Preserve original sys.path behavior
sys.path.append("scripts")
sys.path.append(str(Path(__file__).resolve().parents[1] / "common" / "models"))

from patient_embedding.stage3.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()